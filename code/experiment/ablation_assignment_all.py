"""
Assignment accuracy ablation on all frames (no minimum-person filter).
Sweeps tau_a x delta on CHV and SH17, all test images.
tau_v fixed at 0.30.
Output: results/eval/ablation_assignment_all.csv
"""
import csv
import sys
from itertools import product
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline, _arm_hand_anchor
from landmarks.base import COCO_PPE_ANCHOR_GROUPS
from landmarks.core import PPE_CLASS_TO_ANCHOR, compute_iou, compute_overlap_ratio, get_ppe_anchors

TAU_A_VALS = [0.05, 0.10, 0.15, 0.20]
DELTA_VALS  = [0.05, 0.10, 0.15, 0.20, 0.25]
TAU_V       = 0.30
IOU_MATCH   = 0.50
OUT         = ROOT / "results/eval/ablation_assignment_all.csv"

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
    return best_idx if best_r > 0.1 else None


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


def anchor_score(ppe_bbox, anchors, ppe_cls, tau_a):
    groups = PPE_CLASS_TO_ANCHOR.get(ppe_cls, [])
    best = 0.0
    for g in groups:
        a = anchors.get(g)
        if a is None:
            continue
        ab = a.get("bbox")
        if ab is None:
            continue
        s = compute_iou(ppe_bbox, ab)
        if s < tau_a:
            s = compute_overlap_ratio(ppe_bbox, ab)
        best = max(best, s)
    return best


def assign_all(ppe_dets, all_anchors, tau_a):
    result = {}
    for pi, ppe in enumerate(ppe_dets):
        best_s, best_p = 0.0, -1
        for idx, anchors in enumerate(all_anchors):
            s = anchor_score(ppe["bbox"], anchors, ppe["class_name"], tau_a)
            if s > best_s:
                best_s, best_p = s, idx
        result[pi] = best_p if best_s >= tau_a else -1
    return result


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


def compute_accuracy(cache, tau_a, delta):
    correct = total = 0
    for entry in cache:
        gt_persons = entry["gt_persons"]
        gt_ppe     = entry["gt_ppe"]
        persons    = entry["persons"]
        ppe_dets   = entry["ppe_dets"]

        if len(gt_persons) < 1 or len(persons) < 1:
            continue

        all_anchors = [
            build_anchors(lm, pbbox, delta, TAU_V)
            for pbbox, lm in persons
        ]
        assignment = assign_all(ppe_dets, all_anchors, tau_a)

        gt_p_bboxes   = [g["bbox"] for g in gt_persons]
        pred_p_bboxes = [pb for pb, _ in persons]

        for gt_item in gt_ppe:
            gt_owner = find_owner(gt_item["bbox"], gt_p_bboxes)
            if gt_owner is None:
                continue

            best_iou, best_pi = 0.0, -1
            for pi, ppe in enumerate(ppe_dets):
                if ppe["class_name"] != gt_item["class_name"]:
                    continue
                s = compute_iou(ppe["bbox"], gt_item["bbox"])
                if s > best_iou:
                    best_iou, best_pi = s, pi

            if best_iou < IOU_MATCH or best_pi < 0:
                continue

            pred_person = assignment.get(best_pi, -1)
            if pred_person < 0:
                continue

            best_p_iou, best_p_idx = 0.0, -1
            for j, pb in enumerate(pred_p_bboxes):
                s = compute_iou(gt_p_bboxes[gt_owner], pb)
                if s > best_p_iou:
                    best_p_iou, best_p_idx = s, j

            total += 1
            if best_p_idx == pred_person:
                correct += 1

    acc = round(correct / total, 4) if total > 0 else None
    return correct, total, acc


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_name, cfg in DATASETS.items():
        print(f"\nCaching {ds_name}...")
        cache = cache_dataset(ds_name, cfg)
        print(f"  {len(cache)} images")

        combos = list(product(TAU_A_VALS, DELTA_VALS))
        for tau_a, delta in tqdm(combos, desc="sweep", ncols=80):
            correct, total, acc = compute_accuracy(cache, tau_a, delta)
            rows.append({
                "dataset":  ds_name,
                "tau_a":    tau_a,
                "delta":    delta,
                "correct":  correct,
                "total":    total,
                "accuracy": acc if acc is not None else "",
            })

    fields = ["dataset", "tau_a", "delta", "correct", "total", "accuracy"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
