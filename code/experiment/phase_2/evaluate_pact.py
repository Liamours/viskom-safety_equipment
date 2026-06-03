import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compliance.pipeline import PACTPipeline
from landmarks.core import compute_iou

ROOT        = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
RESULTS_DIR = ROOT / "results/phase_2"


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

IOU_THRESH = 0.5
DEFAULT_ASSIGN_IOU = 0.10
SENSITIVITY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]

PAIRING_PROTOCOL = {
    "chv": (
        "GT person-PPE pairs are derived, not manually annotated: each GT PPE box "
        "is paired with the GT person box that covers the largest fraction of the "
        "PPE box; items with <=0.10 coverage are ignored."
    ),
    "sh17": (
        "GT person-PPE pairs are derived, not manually annotated: each GT PPE box "
        "is paired with the GT person box that covers the largest fraction of the "
        "PPE box; items with <=0.10 coverage are ignored."
    ),
    "cppe5": (
        "Unsupported for Stage 2 in this evaluator: CPPE-5 has no person-class "
        "ground-truth boxes, so person-PPE assignment accuracy is not reproducible "
        "without added person annotations or a clearly caveated derived-person protocol."
    ),
}



def yolo_to_pixel(cx, cy, w, h, img_w, img_h) -> Tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def load_gt(label_path: Path, img_w: int, img_h: int,
            class_names: Dict[int, str],
            person_class_ids) -> Tuple[List[Dict], List[Dict]]:
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


def match_detections(gt_boxes: List[Tuple], pred_boxes: List[Tuple],
                     iou_thresh: float = IOU_THRESH):
    matched_pred = set()
    tp = 0
    for gt in gt_boxes:
        best_iou, best_j = 0.0, -1
        for j, pred in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched_pred.add(best_j)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def find_owner(ppe_bbox, person_bboxes: List[Tuple]) -> Optional[int]:
    best_idx, best_ratio = -1, 0.0
    x1, y1, x2, y2 = ppe_bbox
    for i, pb in enumerate(person_bboxes):
        px1, py1, px2, py2 = pb
        ix1, iy1 = max(x1, px1), max(y1, py1)
        ix2, iy2 = min(x2, px2), min(y2, py2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (x2 - x1) * (y2 - y1))
        ratio = inter / area
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    return best_idx if best_ratio > 0.1 else None


def det_accepted_by_anchor(person, det: Dict) -> bool:
    status = person.compliance_map.get(det["class_name"])
    if status is None:
        return False
    return tuple(status.get("bbox", ())) == tuple(det["bbox"])



def evaluate(dataset: str, split: str, n: int, assignment_iou: float = DEFAULT_ASSIGN_IOU):
    cfg = DATASET_CFG[dataset]

    split_file = cfg["splits"].get(split)
    if split_file is None or not Path(split_file).exists():
        print(f"  [!] Split file not found: {split_file}")
        return

    weights = cfg["weights"].get(split)
    if not Path(weights).exists():
        print(f"  [!] Weights not found: {weights}")
        return

    image_paths = [Path(p.strip()) for p in open(split_file)]
    if n > 0:
        image_paths = image_paths[:n]

    pipeline = PACTPipeline(
        det_weights  = str(weights),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = dataset,
        rule         = cfg["rule"],
        device       = "0",
        iou_threshold = assignment_iou,
    )

    person_tp, person_fp, person_fn = 0, 0, 0
    ppe_tp_map:  Dict[str, int] = defaultdict(int)
    ppe_fp_map:  Dict[str, int] = defaultdict(int)
    ppe_fn_map:  Dict[str, int] = defaultdict(int)

    assign_total, assign_correct = 0, 0
    anchor_total, anchor_correct, anchor_rejected = 0, 0, 0

    for img_path in tqdm(image_paths, desc=f"{dataset}:{split}", unit="img"):
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w   = img_rgb.shape[:2]

        label_path = cfg["label_dir"] / (img_path.stem + ".txt")
        gt_persons, gt_ppe = load_gt(
            label_path, w, h,
            cfg["class_names"], cfg["person_class_ids"]
        )

        frame_result = pipeline.run(img_rgb, image_path=str(img_path))

        pred_person_bboxes = [p.person_bbox for p in frame_result.persons]
        gt_person_bboxes   = [g["bbox"] for g in gt_persons]

        tp, fp, fn = match_detections(gt_person_bboxes, pred_person_bboxes)
        person_tp += tp
        person_fp += fp
        person_fn += fn

        pred_ppe_all: List[Dict] = []
        seen_bboxes = set()
        for p in frame_result.persons:
            for det in p.ppe_detections:
                key = det["bbox"]
                if key not in seen_bboxes:
                    pred_ppe_all.append(det)
                    seen_bboxes.add(key)

        gt_by_class:   Dict[str, List] = defaultdict(list)
        pred_by_class: Dict[str, List] = defaultdict(list)

        for g in gt_ppe:
            gt_by_class[g["class_name"]].append(g["bbox"])
        for d in pred_ppe_all:
            pred_by_class[d["class_name"]].append(d["bbox"])

        all_ppe_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
        for cls in all_ppe_classes:
            tp, fp, fn = match_detections(gt_by_class[cls], pred_by_class[cls])
            ppe_tp_map[cls] += tp
            ppe_fp_map[cls] += fp
            ppe_fn_map[cls] += fn

        if len(gt_persons) < 2 or len(frame_result.persons) < 2:
            continue

        gt_p_bboxes = [g["bbox"] for g in gt_persons]

        for gt_ppe_item in gt_ppe:
            gt_owner_idx = find_owner(gt_ppe_item["bbox"], gt_p_bboxes)
            if gt_owner_idx is None:
                continue

            best_pred_det, best_iou = None, 0.0
            for p in frame_result.persons:
                for det in p.ppe_detections:
                    if det["class_name"] != gt_ppe_item["class_name"]:
                        continue
                    iou = compute_iou(det["bbox"], gt_ppe_item["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_det = (p.person_id, det)

            if best_iou < IOU_THRESH or best_pred_det is None:
                continue

            pred_person_id = best_pred_det[0]
            pred_p_bboxes = [p.person_bbox for p in frame_result.persons]
            best_p_iou, best_p_idx = 0.0, -1
            for j, pb in enumerate(pred_p_bboxes):
                iou = compute_iou(gt_p_bboxes[gt_owner_idx], pb)
                if iou > best_p_iou:
                    best_p_iou, best_p_idx = iou, j

            assign_total += 1
            if best_p_idx == pred_person_id:
                assign_correct += 1

            anchor_total += 1
            pred_person = frame_result.persons[pred_person_id]
            if det_accepted_by_anchor(pred_person, best_pred_det[1]):
                if best_p_idx == pred_person_id:
                    anchor_correct += 1
            else:
                anchor_rejected += 1

    p1, r1, f1_person = prf(person_tp, person_fp, person_fn)

    ppe_results = {}
    all_classes = sorted(set(ppe_tp_map) | set(ppe_fp_map) | set(ppe_fn_map))
    for cls in all_classes:
        p, r, f = prf(ppe_tp_map[cls], ppe_fp_map[cls], ppe_fn_map[cls])
        ppe_results[cls] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "tp": ppe_tp_map[cls], "fp": ppe_fp_map[cls], "fn": ppe_fn_map[cls],
        }

    if ppe_results:
        macro_p = np.mean([v["precision"] for v in ppe_results.values()])
        macro_r = np.mean([v["recall"]    for v in ppe_results.values()])
        macro_f = np.mean([v["f1"]        for v in ppe_results.values()])
    else:
        macro_p = macro_r = macro_f = 0.0

    assign_acc = assign_correct / assign_total if assign_total > 0 else None
    anchor_acc = anchor_correct / anchor_total if anchor_total > 0 else None

    results = {
        "dataset": dataset,
        "split":   split,
        "n_images": len(image_paths),
        "assignment_iou_threshold": assignment_iou,
        "pairing_protocol": PAIRING_PROTOCOL.get(dataset, ""),
        "level_1a_person_detection": {
            "precision": round(p1, 4),
            "recall":    round(r1, 4),
            "f1":        round(f1_person, 4),
            "tp": person_tp, "fp": person_fp, "fn": person_fn,
        },
        "level_1b_ppe_detection": {
            "per_class":     ppe_results,
            "macro_precision": round(macro_p, 4),
            "macro_recall":    round(macro_r, 4),
            "macro_f1":        round(macro_f, 4),
        },
        "level_2_assignment": {
            "total_gt_pairs": assign_total,
            "correct":        assign_correct,
            "accuracy":       round(assign_acc, 4) if assign_acc is not None else "N/A",
            "note": (
                "CHV/SH17 GT pairs are derived by max PPE-box coverage inside GT person boxes. "
                "CPPE-5 should be N/A unless derived person boxes are explicitly caveated."
            ),
        },
        "level_2_anchor_threshold": {
            "threshold": assignment_iou,
            "total_gt_pairs": anchor_total,
            "correct_after_anchor_acceptance": anchor_correct,
            "rejected_by_anchor_threshold": anchor_rejected,
            "accuracy": round(anchor_acc, 4) if anchor_acc is not None else "N/A",
        },
    }

    return results


def print_results(res: dict):
    l1a = res["level_1a_person_detection"]
    l1b = res["level_1b_ppe_detection"]
    l2 = res["level_2_assignment"]
    l2a = res["level_2_anchor_threshold"]
    print(
        f"{res['dataset']}:{res['split']} "
        f"images={res['n_images']} "
        f"person_f1={l1a['f1']} "
        f"ppe_macro_f1={l1b['macro_f1']} "
        f"assign={l2['accuracy']} "
        f"anchor@{l2a['threshold']:.2f}={l2a['accuracy']}"
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chv", choices=["chv", "cppe5", "sh17"])
    parser.add_argument("--split",   default="original", choices=["original", "8020"])
    parser.add_argument("--n",       default=0, type=int,
                        help="Max images to evaluate (0 = all)")
    parser.add_argument("--assignment-iou", default=DEFAULT_ASSIGN_IOU, type=float,
                        help="Anchor IoU/overlap threshold used by PACT assignment")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run anchor-threshold sensitivity: 0.05, 0.10, 0.15, 0.20, 0.30")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sensitivity:
        sensitivity = []
        for thr in SENSITIVITY_THRESHOLDS:
            res = evaluate(args.dataset, args.split, args.n, assignment_iou=thr)
            if res is None:
                return
            sensitivity.append(res["level_2_anchor_threshold"])

        out_json = out_dir / f"pact_assignment_sensitivity_{args.split}.json"
        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "pairing_protocol": PAIRING_PROTOCOL.get(args.dataset, ""),
            "sensitivity": sensitivity,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(out_json)
        return

    results = evaluate(args.dataset, args.split, args.n, assignment_iou=args.assignment_iou)
    if results is None:
        return

    print_results(results)

    out_json = out_dir / f"pact_eval_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(out_json)


if __name__ == "__main__":
    main()
