"""
F1 ablation on CHV + SH17 test sets.
Sweeps iou_threshold x anchor_margin (delta) x visibility_threshold.
Model outputs cached once; param combos applied in Python.
Output: results/eval/ablation_f1_all.csv
"""
import csv
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline, _arm_hand_anchor
from landmarks.base import COCO_PPE_ANCHOR_GROUPS
from landmarks.core import build_compliance_map, compute_iou, get_ppe_anchors

IOU_VALS  = [0.01, 0.05, 0.10, 0.20]
MARG_VALS = [0.10, 0.15, 0.20, 0.25]
VIS_VALS  = [0.20, 0.30, 0.40, 0.50]
OUT       = ROOT / "results/eval/ablation_f1_all.csv"

DATASETS = {
    "chv": {
        "split":      ROOT / "dataset/CHV/yolo_original/test_images.txt",
        "labels":     ROOT / "dataset/CHV/raw/labels",
        "weights":    ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "rule":       "chv",
        "classes":    {0:"person", 1:"vest", 2:"helmet", 3:"helmet", 4:"helmet", 5:"helmet"},
        "person_ids": {0},
    },
    "sh17": {
        "split":      ROOT / "dataset/SH17/yolo_original/test_images.txt",
        "labels":     ROOT / "dataset/SH17/raw_640/labels",
        "weights":    ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "rule":       "sh17",
        "classes":    {0:"person", 1:"ear", 2:"earmuffs", 3:"face", 4:"face-guard",
                       5:"face-mask-medical", 6:"foot", 7:"tools", 8:"glasses",
                       9:"gloves", 10:"helmet", 11:"hands", 12:"head", 13:"medical-suit",
                       14:"shoes", 15:"safety-suit", 16:"safety-vest"},
        "person_ids": {0},
    },
}


def yolo_to_bbox(cx, cy, w, h, iw, ih):
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    return x1, y1, x2, y2


def load_gt(label_path, iw, ih, classes, person_ids):
    persons, ppe = [], []
    if not label_path.exists():
        return persons, ppe
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        bbox = yolo_to_bbox(*map(float, parts[1:5]), iw, ih)
        name = classes.get(cid, f"cls_{cid}")
        (persons if cid in person_ids else ppe).append({"class_name": name, "bbox": bbox})
    return persons, ppe


def find_owner(ppe_bbox, person_bboxes):
    x1, y1, x2, y2 = ppe_bbox
    best_idx, best_r = -1, 0.0
    for i, (px1, py1, px2, py2) in enumerate(person_bboxes):
        inter = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
        area = max(1, (x2 - x1) * (y2 - y1))
        r = inter / area
        if r > best_r:
            best_r, best_idx = r, i
    return best_idx if best_r > 0.05 else None


def build_anchors(lm_result, person_bbox, delta, tau_v):
    anchors = get_ppe_anchors(
        lm_result,
        anchor_groups=COCO_PPE_ANCHOR_GROUPS,
        margin=delta,
        visibility_threshold=tau_v,
    )
    px1, py1, px2, py2 = person_bbox
    iw, ih = lm_result.image_width, lm_result.image_height

    if anchors.get("helmet_region") is not None:
        ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
        anchors["helmet_region"]["bbox"] = (ax1, py1, ax2, ay2)
    else:
        bb = (px1, py1, px2, py1 + int((py2 - py1) * 0.35))
        anchors["helmet_region"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

    for side, key in [("left", "left_hand"), ("right", "right_hand")]:
        if anchors.get(key) is None:
            bb = _arm_hand_anchor(lm_result, person_bbox, side, tau_v)
            anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

    lms = lm_result.landmarks
    for knee_i, key in [(13, "left_foot"), (14, "right_foot")]:
        if anchors.get(key) is None:
            if knee_i < len(lms) and lms[knee_i].visibility >= tau_v:
                ky = int(lms[knee_i].y * ih)
                bb = (px1, ky, px2, py2)
            else:
                bb = (px1, py1 + int((py2 - py1) * 0.6), px2, py2)
            anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

    return anchors


def cache_dataset(name, cfg):
    pipeline = PACTPipeline(
        det_weights  = str(cfg["weights"]),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = name,
        rule         = cfg["rule"],
        device       = "0",
    )
    image_paths = [Path(p.strip()) for p in open(cfg["split"])]
    cache = []

    for img_path in tqdm(image_paths, desc=name, ncols=80):
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        label_path = cfg["labels"] / (img_path.stem + ".txt")
        gt_persons, gt_ppe = load_gt(label_path, w, h, cfg["classes"], cfg["person_ids"])

        ppe_dets = pipeline._detect_ppe(img_rgb)
        persons  = pipeline._detect_persons(img_rgb)

        cache.append({
            "img_w": w, "img_h": h,
            "gt_persons": gt_persons,
            "gt_ppe":     gt_ppe,
            "ppe_dets":   ppe_dets,
            "persons":    persons,
        })

    return cache


def compute_f1(cache, iou_t, delta, tau_v):
    tp_tot = fp_tot = fn_tot = 0

    for entry in cache:
        img_w, img_h = entry["img_w"], entry["img_h"]
        persons   = entry["persons"]
        ppe_dets  = entry["ppe_dets"]
        gt_persons = entry["gt_persons"]
        gt_ppe     = entry["gt_ppe"]

        all_bboxes = [p[0] for p in persons]
        gt_p_bboxes = [g["bbox"] for g in gt_persons]

        for pidx, (person_bbox, lm_result) in enumerate(persons):
            scores = [compute_iou(ppe["bbox"], pb) for pb in all_bboxes] if all_bboxes else []
            assigned = []
            for ppe in ppe_dets:
                s = [compute_iou(ppe["bbox"], pb) for pb in all_bboxes]
                if s and max(s) > 0 and all_bboxes[int(np.argmax(s))] == person_bbox:
                    assigned.append(ppe)

            anchors = build_anchors(lm_result, person_bbox, delta, tau_v)
            cmap = build_compliance_map(
                anchors       = anchors,
                detections    = assigned,
                iou_threshold = iou_t,
                anchor_margin = 0.0,
                image_width   = img_w,
                image_height  = img_h,
            )
            worn_pred = {cls for cls, v in cmap.items() if v is not None}

            worn_gt = set()
            for g in gt_ppe:
                owner = find_owner(g["bbox"], gt_p_bboxes)
                if owner is None:
                    continue
                if gt_p_bboxes and compute_iou(gt_p_bboxes[owner], person_bbox) > 0.3:
                    worn_gt.add(g["class_name"])

            tp_tot += len(worn_pred & worn_gt)
            fp_tot += len(worn_pred - worn_gt)
            fn_tot += len(worn_gt  - worn_pred)

    p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4), tp_tot, fp_tot, fn_tot


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_cache = []

    for ds_name, cfg in DATASETS.items():
        print(f"\nCaching {ds_name}...")
        cache = cache_dataset(ds_name, cfg)
        all_cache.extend(cache)
        print(f"  {len(cache)} images")

    print(f"\nTotal: {len(all_cache)} images")

    combos = list(product(IOU_VALS, MARG_VALS, VIS_VALS))
    rows = []

    for iou_t, delta, tau_v in tqdm(combos, desc="sweep", ncols=80):
        p, r, f, tp, fp, fn = compute_f1(all_cache, iou_t, delta, tau_v)
        rows.append({
            "iou_threshold":       iou_t,
            "delta":               delta,
            "visibility_threshold": tau_v,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f,
        })

    rows.sort(key=lambda x: -x["f1"])
    fields = ["iou_threshold", "delta", "visibility_threshold",
              "tp", "fp", "fn", "precision", "recall", "f1"]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    print(f"\nBest: iou={best['iou_threshold']}, delta={best['delta']}, "
          f"vis={best['visibility_threshold']}  F1={best['f1']}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
