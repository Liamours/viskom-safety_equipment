"""
PACT Evaluation — Visual Samples
=================================
Draws side-by-side annotated images showing:
  Left panel  : Ground truth  (GT persons in blue, GT PPE in cyan)
  Right panel : PACT output   (pose landmarks, predicted PPE in orange,
                                compliance label, TP/FP/FN markers)

Color legend
------------
  Blue  box       GT person
  Cyan  box       GT PPE item
  Green circle    Predicted person box matched to GT  (TP person)
  Red   circle    Predicted person box NOT matched    (FP person)
  Orange box      Predicted PPE — TP  (matched GT box)
  Red    box      Predicted PPE — FP  (no GT match)
  Dashed cyan     GT PPE — FN  (missed by detector)

Usage
-----
python sample_pact_eval.py --dataset chv  --split original --n 8
python sample_pact_eval.py --dataset sh17 --split original --n 8
python sample_pact_eval.py --dataset cppe5 --split original --n 8
"""
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compliance.pipeline import PACTPipeline
from compliance.visualizer import draw_pact_result
from landmarks.core import compute_iou

ROOT        = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
RESULTS_DIR = ROOT / "results/phase_2"

IOU_THRESH = 0.50

# ── reuse dataset config from evaluate_pact ──────────────────────────────────
DATASET_CFG: Dict[str, Dict] = {
    "chv": {
        "image_dir":  ROOT / "dataset/CHV/raw/images",
        "label_dir":  ROOT / "dataset/CHV/raw/labels",
        "splits": {
            "original": ROOT / "dataset/CHV/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/CHV/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_chv_8020/weights/best.pt",
        },
        "class_names": {0: "person", 1: "vest", 2: "helmet",
                        3: "helmet", 4: "helmet", 5: "helmet"},
        "person_class_ids": {0},
        "rule": "chv",
    },
    "cppe5": {
        "image_dir":  ROOT / "dataset/CPPE-5/raw/images",
        "label_dir":  ROOT / "dataset/CPPE-5/raw/labels",
        "splits": {
            "original": ROOT / "dataset/CPPE-5/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/CPPE-5/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_cppe_5_8020/weights/best.pt",
        },
        "class_names": {0: "coverall", 1: "face_shield", 2: "gloves",
                        3: "goggles",  4: "mask"},
        "person_class_ids": set(),
        "rule": "cppe5",
    },
    "sh17": {
        "image_dir":  ROOT / "dataset/SH17/raw_640/images",
        "label_dir":  ROOT / "dataset/SH17/raw_640/labels",
        "splits": {
            "original": ROOT / "dataset/SH17/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/SH17/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_sh17_8020/weights/best.pt",
        },
        "class_names": {
            0: "person", 1: "ear", 2: "earmuffs", 3: "face",
            4: "face-guard", 5: "face-mask-medical", 6: "foot", 7: "tools",
            8: "glasses", 9: "gloves", 10: "helmet", 11: "hands",
            12: "head", 13: "medical-suit", 14: "shoes",
            15: "safety-suit", 16: "safety-vest",
        },
        "person_class_ids": {0},
        "rule": "sh17",
    },
}

# ── colors (BGR) ─────────────────────────────────────────────────────────────
C_GT_PERSON  = (200,  80,   0)   # blue-ish
C_GT_PPE     = (200, 200,   0)   # cyan
C_PRED_TP    = (0,   200,   0)   # green  — correctly detected PPE
C_PRED_FP    = (0,     0, 220)   # red    — spurious PPE
C_GT_FN      = (0,   200, 200)   # yellow — missed PPE
C_PERSON_TP  = (0,   200,   0)   # green  — person correctly detected
C_PERSON_FP  = (0,     0, 220)   # red    — spurious person


# ── helpers ───────────────────────────────────────────────────────────────────

def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def load_gt(label_path, img_w, img_h, class_names, person_class_ids):
    gt_persons, gt_ppe = [], []
    if not label_path.exists():
        return gt_persons, gt_ppe
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            bbox = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
            name = class_names.get(cls_id, f"class_{cls_id}")
            entry = {"class_name": name, "bbox": bbox}
            if cls_id in person_class_ids:
                gt_persons.append(entry)
            else:
                gt_ppe.append(entry)
    return gt_persons, gt_ppe


def match_boxes(gt_list, pred_list, iou_thresh=IOU_THRESH):
    """
    Returns (matched_gt_indices, matched_pred_indices).
    Greedy by descending IoU.
    """
    pairs = []
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    matched_gt:   Set[int] = set()
    matched_pred: Set[int] = set()
    for _, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
    return matched_gt, matched_pred


def draw_dashed_rect(canvas, x1, y1, x2, y2, color, thickness=1, dash=8):
    """Draw a dashed rectangle."""
    pts = [(x1,y1,x2,y1), (x2,y1,x2,y2), (x2,y2,x1,y2), (x1,y2,x1,y1)]
    for ax, ay, bx, by in pts:
        dx, dy = bx - ax, by - ay
        length = max(1, int((dx**2 + dy**2) ** 0.5))
        steps  = length // (dash * 2)
        for k in range(steps + 1):
            t0 = min(k * 2 * dash / length, 1.0)
            t1 = min((k * 2 + 1) * dash / length, 1.0)
            p0 = (int(ax + t0 * dx), int(ay + t0 * dy))
            p1 = (int(ax + t1 * dx), int(ay + t1 * dy))
            cv2.line(canvas, p0, p1, color, thickness)


def label(canvas, text, x, y, color, scale=0.4, thickness=1):
    y = max(y, 12)
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thickness,     cv2.LINE_AA)


def draw_gt_panel(image_rgb, gt_persons, gt_ppe):
    canvas = image_rgb.copy()

    for g in gt_persons:
        x1, y1, x2, y2 = g["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_GT_PERSON, 2)
        label(canvas, "person", x1, y1 - 4, C_GT_PERSON)

    for g in gt_ppe:
        x1, y1, x2, y2 = g["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_GT_PPE, 2)
        label(canvas, g["class_name"], x1, y1 - 4, C_GT_PPE)

    _add_panel_title(canvas, "GROUND TRUTH")
    return canvas


def draw_pred_panel(image_rgb, frame_result, gt_persons, gt_ppe):
    canvas = draw_pact_result(image_rgb, frame_result)

    # ── person TP/FP markers ─────────────────────────────────────────────────
    pred_persons = [{"bbox": p.person_bbox} for p in frame_result.persons]
    gt_p_dicts   = [{"bbox": g["bbox"]} for g in gt_persons]
    _, matched_pred_p = match_boxes(gt_p_dicts, pred_persons)

    for j, p in enumerate(frame_result.persons):
        x1, y1, x2, y2 = p.person_bbox
        color = C_PERSON_TP if j in matched_pred_p else C_PERSON_FP
        tag   = "TP" if j in matched_pred_p else "FP"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label(canvas, tag, x2 - 22, y1 + 14, color, scale=0.38)

    # ── PPE TP / FP / FN markers ─────────────────────────────────────────────
    pred_ppe_all: List[Dict] = []
    seen: set = set()
    for p in frame_result.persons:
        for det in p.ppe_detections:
            key = det["bbox"]
            if key not in seen:
                pred_ppe_all.append({"bbox": det["bbox"],
                                     "class_name": det["class_name"]})
                seen.add(key)

    # group by class for per-class matching
    from collections import defaultdict
    gt_by_cls:   Dict[str, List] = defaultdict(list)
    pred_by_cls: Dict[str, List] = defaultdict(list)
    for g in gt_ppe:
        gt_by_cls[g["class_name"]].append(g)
    for d in pred_ppe_all:
        pred_by_cls[d["class_name"]].append(d)

    for cls in set(list(gt_by_cls.keys()) + list(pred_by_cls.keys())):
        matched_gt_idx, matched_pred_idx = match_boxes(gt_by_cls[cls], pred_by_cls[cls])

        # FP — predicted but unmatched
        for j, det in enumerate(pred_by_cls[cls]):
            if j not in matched_pred_idx:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), C_PRED_FP, 2)
                label(canvas, f"FP:{cls}", x1, y1 - 4, C_PRED_FP)

        # FN — GT present but missed
        for i, g in enumerate(gt_by_cls[cls]):
            if i not in matched_gt_idx:
                x1, y1, x2, y2 = g["bbox"]
                draw_dashed_rect(canvas, x1, y1, x2, y2, C_GT_FN, thickness=2)
                label(canvas, f"FN:{cls}", x1, y2 + 12, C_GT_FN)

    _add_panel_title(canvas, "PACT OUTPUT  (TP=green  FP=red  FN=dashed)")
    return canvas


def _add_panel_title(canvas, text):
    h, w = canvas.shape[:2]
    bar = np.zeros((22, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    canvas[:22] = bar


def make_sample(image_rgb, frame_result, gt_persons, gt_ppe):
    left  = draw_gt_panel(image_rgb, gt_persons, gt_ppe)
    right = draw_pred_panel(image_rgb, frame_result, gt_persons, gt_ppe)
    return np.concatenate(
        [cv2.cvtColor(left,  cv2.COLOR_RGB2BGR),
         cv2.cvtColor(right, cv2.COLOR_RGB2BGR)],
        axis=1,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chv", choices=["chv", "cppe5", "sh17"])
    parser.add_argument("--split",   default="original", choices=["original", "8020"])
    parser.add_argument("--n",       default=8,  type=int, help="Number of sample images")
    parser.add_argument("--seed",    default=42, type=int, help="Random seed for image selection")
    args = parser.parse_args()

    cfg = DATASET_CFG[args.dataset]

    split_file = cfg["splits"][args.split]
    weights    = cfg["weights"][args.split]

    image_paths = [Path(p.strip()) for p in open(split_file) if Path(p.strip()).exists()]
    random.seed(args.seed)
    sample_paths = random.sample(image_paths, min(args.n, len(image_paths)))

    out_dir = RESULTS_DIR / args.dataset / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  PACT Visual Samples — {args.dataset.upper()} ({args.split})")
    print(f"  Saving {len(sample_paths)} samples → {out_dir}\n")

    pipeline = PACTPipeline(
        det_weights  = str(weights),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = args.dataset,
        rule         = cfg["rule"],
        device       = "0",
    )

    for img_path in sample_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        label_path = cfg["label_dir"] / (img_path.stem + ".txt")
        gt_persons, gt_ppe = load_gt(
            label_path, w, h,
            cfg["class_names"], cfg["person_class_ids"]
        )

        frame_result = pipeline.run(img_rgb, image_path=str(img_path))
        composite    = make_sample(img_rgb, frame_result, gt_persons, gt_ppe)

        out_path = out_dir / f"sample_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), composite)
        print(f"  Saved: {out_path.name}")

    print(f"\n  Done. Open {out_dir} to view samples.")


if __name__ == "__main__":
    main()
