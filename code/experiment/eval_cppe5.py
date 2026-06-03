import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline

OUT_DIR  = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_PATH  = ROOT / "dataset/CPPE-5/raw/annotations/test.json"
IMG_DIR   = ROOT / "dataset/CPPE-5/raw/images"
WEIGHTS   = ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt"
POSE_W    = ROOT / "models/yolov8x-pose.pt"

PERSON_IOU_THRESH   = 0.50
CLUSTER_EXPAND      = 0.10
PPE_MATCH_IOU       = 0.50
PPE_ASSIGN_OVERLAP  = 0.20

CAT_MAP = {1: "coverall", 2: "face_shield", 3: "gloves", 4: "goggles", 5: "mask"}



def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(det, anchor):
    ax1, ay1, ax2, ay2 = det
    bx1, by1, bx2, by2 = anchor
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    det_area = max(1, (ax2-ax1)*(ay2-ay1))
    return inter / det_area


def expand(bbox, margin, w, h):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    return (max(0, x1-dx), max(0, y1-dy), min(w, x2+dx), min(h, y2+dy))


def enclosing(boxes):
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )



def cluster_gt_persons(anns, img_w, img_h):
    boxes = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        boxes.append((x, y, x + bw, y + bh))

    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    expanded = [expand(b, CLUSTER_EXPAND, img_w, img_h) for b in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            if iou(expanded[i], expanded[j]) > 0:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    persons = []
    for idxs in clusters.values():
        ann_group = [anns[i] for i in idxs]
        bbox = enclosing([boxes[i] for i in idxs])
        persons.append({"bbox": bbox, "anns": ann_group})
    return persons



def person_detection_metrics(gt_persons, pred_persons):
    matched_gt  = set()
    matched_pred = set()
    tp = 0
    for pi, pred in enumerate(pred_persons):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_persons):
            s = iou(pred["bbox"], gt["bbox"])
            if s > best_iou:
                best_iou, best_gi = s, gi
        if best_iou >= PERSON_IOU_THRESH and best_gi not in matched_gt:
            tp += 1
            matched_gt.add(best_gi)
            matched_pred.add(pi)
    fp = len(pred_persons) - len(matched_pred)
    fn = len(gt_persons) - len(matched_gt)
    return tp, fp, fn, matched_gt, matched_pred


def ppe_assignment_metrics(gt_persons, pred_frame):
    total_assigned = 0
    correct        = 0
    wrong_person   = 0
    no_gt_match    = 0

    for pred_p in pred_frame.persons:
        person_bbox = pred_p.person_bbox
        for ppe in pred_p.ppe_detections:
            total_assigned += 1
            ppe_cls  = ppe["class_name"]
            ppe_bbox = ppe["bbox"]

            best_iou_score, best_gt_ann = 0.0, None
            for gt_p in gt_persons:
                for ann in gt_p["anns"]:
                    if CAT_MAP.get(ann["category_id"]) != ppe_cls:
                        continue
                    x, y, bw, bh = ann["bbox"]
                    gt_bbox = (x, y, x+bw, y+bh)
                    s = iou(ppe_bbox, gt_bbox)
                    if s > best_iou_score:
                        best_iou_score = s
                        best_gt_ann = (gt_bbox, gt_p)

            if best_gt_ann is None or best_iou_score < PPE_MATCH_IOU:
                no_gt_match += 1
                continue

            gt_bbox, gt_p = best_gt_ann
            gt_person_bbox = gt_p["bbox"]
            if overlap_ratio(gt_bbox, person_bbox) >= PPE_ASSIGN_OVERLAP:
                correct += 1
            else:
                wrong_person += 1

    return total_assigned, correct, wrong_person, no_gt_match



def main():
    coco = json.load(open(ANN_PATH))
    id_to_img  = {i["id"]: i for i in coco["images"]}
    id_to_anns = defaultdict(list)
    for a in coco["annotations"]:
        id_to_anns[a["image_id"]].append(a)

    pipeline = PACTPipeline(
        det_weights  = str(WEIGHTS),
        pose_weights = str(POSE_W),
        dataset      = "cppe5",
        rule         = "cppe5",
        device       = "0",
    )

    person_rows = []
    assign_rows = []

    agg_person = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
    agg_assign = {"total": 0, "correct": 0, "wrong_person": 0, "no_gt_match": 0}

    for img_meta in tqdm(coco["images"], desc="cppe5-eval", unit="img", ncols=80):
        img_id   = img_meta["id"]
        img_path = IMG_DIR / img_meta["file_name"]
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_w, img_h = img_meta["width"], img_meta["height"]
        anns         = id_to_anns[img_id]
        gt_persons   = cluster_gt_persons(anns, img_w, img_h)

        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame      = pipeline.run(img_rgb, image_path=str(img_path))
        pred_persons_raw = [{"bbox": p.person_bbox} for p in frame.persons]

        tp, fp, fn, matched_gt, matched_pred = person_detection_metrics(gt_persons, pred_persons_raw)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

        person_rows.append({
            "image":      img_meta["file_name"],
            "gt_persons": len(gt_persons),
            "pred_persons": len(pred_persons_raw),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        })

        agg_person["tp"]   += tp
        agg_person["fp"]   += fp
        agg_person["fn"]   += fn
        agg_person["gt"]   += len(gt_persons)
        agg_person["pred"] += len(pred_persons_raw)

        total, correct, wrong, no_gt = ppe_assignment_metrics(gt_persons, frame)
        acc = correct / total if total > 0 else None

        assign_rows.append({
            "image":          img_meta["file_name"],
            "total_assigned": total,
            "correct":        correct,
            "wrong_person":   wrong,
            "no_gt_match":    no_gt,
            "accuracy":       round(acc, 4) if acc is not None else "N/A",
        })

        agg_assign["total"]        += total
        agg_assign["correct"]      += correct
        agg_assign["wrong_person"] += wrong
        agg_assign["no_gt_match"]  += no_gt

    import csv
    person_fields = ["image", "gt_persons", "pred_persons", "tp", "fp", "fn", "precision", "recall", "f1"]
    assign_fields = ["image", "total_assigned", "correct", "wrong_person", "no_gt_match", "accuracy"]

    with open(OUT_DIR / "cppe5_person_detection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=person_fields)
        w.writeheader(); w.writerows(person_rows)

    with open(OUT_DIR / "cppe5_ppe_assignment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=assign_fields)
        w.writeheader(); w.writerows(assign_rows)

    tp, fp, fn = agg_person["tp"], agg_person["fp"], agg_person["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    t = agg_assign["total"]
    assign_acc = agg_assign["correct"] / t if t > 0 else 0.0

    summary = {
        "derived_gt_warning": (
            "CPPE-5 has no person-class ground truth. Person boxes are derived "
            "by clustering PPE boxes, so assignment accuracy is an internal proxy "
            "and should not be presented as directly peer-validated."
        ),
        "person_detection": {
            "gt_persons":   agg_person["gt"],
            "pred_persons": agg_person["pred"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        },
        "ppe_assignment": {
            "total_assigned": t,
            "correct":        agg_assign["correct"],
            "wrong_person":   agg_assign["wrong_person"],
            "no_gt_match":    agg_assign["no_gt_match"],
            "accuracy":       round(assign_acc, 4),
        },
        "thresholds": {
            "person_iou":        PERSON_IOU_THRESH,
            "cluster_expand":    CLUSTER_EXPAND,
            "ppe_match_iou":     PPE_MATCH_IOU,
            "ppe_assign_overlap": PPE_ASSIGN_OVERLAP,
        },
    }

    (OUT_DIR / "cppe5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"cppe5 person_f1={f1:.4f} "
        f"assignment_proxy={assign_acc:.4f} "
        f"out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
