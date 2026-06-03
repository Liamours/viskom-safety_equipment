"""
Baseline assignment comparison on CPPE-5 test set.
Strategies (all share same model detections):
  1. iou_no_anchor   — best IoU person assignment, all assigned PPE = worn
  2. centroid        — nearest centroid, all assigned PPE = worn
  3. hungarian       — optimal IoU assignment, all assigned PPE = worn
  4. pact            — best IoU + anatomical anchor validation (full pipeline)

Output: results/eval/baselines_per_image.csv
        results/eval/baselines_summary.csv
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from experiment.eval_utils import (
    CAT_MAP, PPE_ASSIGN_OVR, PPE_MATCH_IOU, PERSON_IOU_THRESH,
    cluster_gt_persons, iou, overlap_ratio, prec_rec_f1,
)

ANN_PATH = ROOT / "dataset/CPPE-5/raw/annotations/test.json"
IMG_DIR  = ROOT / "dataset/CPPE-5/raw/images"
WEIGHTS  = ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt"
POSE_W   = ROOT / "models/yolov8x-pose.pt"
OUT_DIR  = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["iou_no_anchor", "centroid", "hungarian", "pact"]


def centroid(bbox):
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


def assign_iou(ppe_list, person_bboxes):
    result = defaultdict(list)
    for ppe in ppe_list:
        scores = [iou(ppe["bbox"], pb) for pb in person_bboxes]
        if not scores: continue
        best = int(np.argmax(scores))
        if scores[best] > 0:
            result[best].append(ppe)
    return result


def assign_centroid(ppe_list, person_bboxes):
    result = defaultdict(list)
    cents = [centroid(pb) for pb in person_bboxes]
    for ppe in ppe_list:
        pc = centroid(ppe["bbox"])
        dists = [((pc[0]-c[0])**2+(pc[1]-c[1])**2)**0.5 for c in cents]
        if dists:
            result[int(np.argmin(dists))].append(ppe)
    return result


def assign_hungarian(ppe_list, person_bboxes):
    result = defaultdict(list)
    if not ppe_list or not person_bboxes: return result
    cost = np.array([[1.0 - iou(p["bbox"], pb) for pb in person_bboxes] for p in ppe_list])
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] < 1.0:
            result[c].append(ppe_list[r])
    return result


def eval_assignment(assigned_map, person_bboxes, gt_persons, ppe_list_all):
    tp = fp = fn = 0
    for pi, pb in enumerate(person_bboxes):
        worn_pred = set(p["class_name"] for p in assigned_map.get(pi, []))
        worn_gt   = set()
        for gp in gt_persons:
            if overlap_ratio(gp["bbox"], pb) >= PPE_ASSIGN_OVR:
                for a in gp["anns"]:
                    worn_gt.add(CAT_MAP.get(a["category_id"], ""))
        tp += len(worn_pred & worn_gt)
        fp += len(worn_pred - worn_gt)
        fn += len(worn_gt  - worn_pred)
    return tp, fp, fn


def main():
    coco = json.load(open(ANN_PATH))
    id_to_anns = defaultdict(list)
    for a in coco["annotations"]: id_to_anns[a["image_id"]].append(a)

    pipeline = PACTPipeline(
        det_weights=str(WEIGHTS), pose_weights=str(POSE_W),
        dataset="cppe5", rule="cppe5", device="0",
    )

    agg = {s: {"tp":0,"fp":0,"fn":0} for s in STRATEGIES}
    per_image_rows = []

    for img_meta in tqdm(coco["images"], desc="baselines", ncols=80):
        img_id   = img_meta["id"]
        img_path = IMG_DIR / img_meta["file_name"]
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None: continue

        img_w, img_h = img_meta["width"], img_meta["height"]
        anns         = id_to_anns[img_id]
        gt_persons   = cluster_gt_persons(anns, img_w, img_h)

        img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ppe_dets    = pipeline._detect_ppe(img_rgb)
        persons_raw = pipeline._detect_persons(img_rgb)
        person_bboxes = [p[0] for p in persons_raw]

        if not person_bboxes:
            continue

        row = {"image": img_meta["file_name"]}

        for strategy in STRATEGIES:
            if strategy == "pact":
                frame = pipeline.run(img_rgb, image_path=str(img_path))
                assigned_map = defaultdict(list)
                for i, p in enumerate(frame.persons):
                    for ppe in p.ppe_detections:
                        if p.compliance_map.get(ppe["class_name"]) is not None:
                            assigned_map[i].append(ppe)
                pbboxes = [p.person_bbox for p in frame.persons]
            else:
                pbboxes = person_bboxes
                if strategy == "iou_no_anchor":
                    assigned_map = assign_iou(ppe_dets, pbboxes)
                elif strategy == "centroid":
                    assigned_map = assign_centroid(ppe_dets, pbboxes)
                elif strategy == "hungarian":
                    assigned_map = assign_hungarian(ppe_dets, pbboxes)

            tp, fp, fn = eval_assignment(assigned_map, pbboxes, gt_persons, ppe_dets)
            p_, r_, f_ = prec_rec_f1(tp, fp, fn)
            agg[strategy]["tp"] += tp
            agg[strategy]["fp"] += fp
            agg[strategy]["fn"] += fn
            row[f"{strategy}_p"]  = p_
            row[f"{strategy}_r"]  = r_
            row[f"{strategy}_f1"] = f_

        per_image_rows.append(row)

    per_fields = ["image"] + [f"{s}_{m}" for s in STRATEGIES for m in ["p","r","f1"]]
    with open(OUT_DIR / "baselines_per_image.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_fields)
        w.writeheader(); w.writerows(per_image_rows)

    summary_rows = []
    print("\n── Baseline Comparison (CPPE-5 test) ──")
    print(f"{'Strategy':<22} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for s in STRATEGIES:
        tp,fp,fn = agg[s]["tp"],agg[s]["fp"],agg[s]["fn"]
        p_,r_,f_ = prec_rec_f1(tp,fp,fn)
        summary_rows.append({"strategy":s,"tp":tp,"fp":fp,"fn":fn,
                              "precision":p_,"recall":r_,"f1":f_})
        print(f"  {s:<20} {p_*100:>9.1f}% {r_*100:>7.1f}% {f_*100:>7.1f}%")

    with open(OUT_DIR / "baselines_summary.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strategy","tp","fp","fn","precision","recall","f1"])
        w.writeheader(); w.writerows(summary_rows)

    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    main()
