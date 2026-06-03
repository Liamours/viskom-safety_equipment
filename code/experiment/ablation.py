"""
Parameter ablation on CPPE-5 test set.
Model inference runs ONCE per image (cached). Param combos applied in Python.

Sweep:
  iou_threshold      : [0.01, 0.05, 0.10, 0.20]
  anchor_margin      : [0.10, 0.15, 0.20, 0.25]
  visibility_threshold: [0.20, 0.30, 0.40, 0.50]

Output: results/eval/ablation.csv
"""
import csv, json, sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline, _arm_hand_anchor
from experiment.eval_utils import (
    CAT_MAP, PPE_ASSIGN_OVR, cluster_gt_persons, iou, overlap_ratio, prec_rec_f1,
)
from landmarks.base import COCO_PPE_ANCHOR_GROUPS
from landmarks.core import build_compliance_map, get_ppe_anchors

ANN_PATH = ROOT / "dataset/CPPE-5/raw/annotations/test.json"
IMG_DIR  = ROOT / "dataset/CPPE-5/raw/images"
WEIGHTS  = ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt"
POSE_W   = ROOT / "models/yolov8x-pose.pt"
OUT_DIR  = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_VALS  = [0.01, 0.05, 0.10, 0.20]
MARG_VALS = [0.10, 0.15, 0.20, 0.25]
VIS_VALS  = [0.20, 0.30, 0.40, 0.50]


def build_anchors_with_fallbacks(lm_result, person_bbox, anchor_margin, vis_thresh):
    anchors = get_ppe_anchors(
        lm_result,
        anchor_groups=COCO_PPE_ANCHOR_GROUPS,
        margin=anchor_margin,
        visibility_threshold=vis_thresh,
    )
    px1, py1, px2, py2 = person_bbox
    if anchors.get("helmet_region") is not None:
        ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
        anchors["helmet_region"]["bbox"] = (ax1, py1, ax2, ay2)
    else:
        bb = (px1, py1, px2, py1 + int((py2-py1)*0.35))
        anchors["helmet_region"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}
    for side, key in [("left","left_hand"),("right","right_hand")]:
        if anchors.get(key) is None:
            bb = _arm_hand_anchor(lm_result, person_bbox, side, vis_thresh)
            anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}
    for knee_i, key in [(13,"left_foot"),(14,"right_foot")]:
        if anchors.get(key) is None:
            lms = lm_result.landmarks
            iw, ih = lm_result.image_width, lm_result.image_height
            if knee_i < len(lms) and lms[knee_i].visibility >= vis_thresh:
                ky = int(lms[knee_i].y * ih)
                bb = (px1, ky, px2, py2)
            else:
                bb = (px1, py1+int((py2-py1)*0.6), px2, py2)
            anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}
    return anchors


def eval_compliance(assigned_ppe, anchors, gt_persons, person_bbox, iou_thresh, img_w, img_h):
    cmap = build_compliance_map(
        anchors=anchors, detections=assigned_ppe,
        iou_threshold=iou_thresh, anchor_margin=0.0,
        image_width=img_w, image_height=img_h,
    )
    worn_pred = {cls for cls, v in cmap.items() if v is not None}
    worn_gt = set()
    for gp in gt_persons:
        if overlap_ratio(gp["bbox"], person_bbox) >= PPE_ASSIGN_OVR:
            for a in gp["anns"]:
                worn_gt.add(CAT_MAP.get(a["category_id"], ""))
    tp = len(worn_pred & worn_gt)
    fp = len(worn_pred - worn_gt)
    fn = len(worn_gt  - worn_pred)
    return tp, fp, fn


def main():
    coco = json.load(open(ANN_PATH))
    id_to_anns = defaultdict(list)
    for a in coco["annotations"]: id_to_anns[a["image_id"]].append(a)

    pipeline = PACTPipeline(
        det_weights=str(WEIGHTS), pose_weights=str(POSE_W),
        dataset="cppe5", rule="cppe5", device="0",
    )

    cache = []
    print("Caching model outputs...")
    for img_meta in tqdm(coco["images"], ncols=80):
        img_id   = img_meta["id"]
        img_path = IMG_DIR / img_meta["file_name"]
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_w, img_h = img_meta["width"], img_meta["height"]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ppe_dets = pipeline._detect_ppe(img_rgb)
        persons  = pipeline._detect_persons(img_rgb)
        anns     = id_to_anns[img_id]
        gt_persons = cluster_gt_persons(anns, img_w, img_h)
        cache.append({
            "img_w": img_w, "img_h": img_h,
            "ppe_dets": ppe_dets, "persons": persons,
            "gt_persons": gt_persons,
        })

    combos = list(product(IOU_VALS, MARG_VALS, VIS_VALS))
    rows = []
    best_f1, best_cfg = 0.0, None

    for iou_t, marg, vis in tqdm(combos, desc="ablation", ncols=80):
        tp_tot = fp_tot = fn_tot = 0
        for entry in cache:
            img_w, img_h = entry["img_w"], entry["img_h"]
            all_bboxes = [p[0] for p in entry["persons"]]
            for person_bbox, lm_result in entry["persons"]:
                assigned = [
                    ppe for ppe in entry["ppe_dets"]
                    if max(iou(ppe["bbox"], pb) for pb in all_bboxes) > 0
                    and all_bboxes[int(np.argmax([iou(ppe["bbox"], pb) for pb in all_bboxes]))] == person_bbox
                ]
                anchors = build_anchors_with_fallbacks(lm_result, person_bbox, marg, vis)
                tp, fp, fn = eval_compliance(
                    assigned, anchors, entry["gt_persons"],
                    person_bbox, iou_t, img_w, img_h,
                )
                tp_tot += tp; fp_tot += fp; fn_tot += fn

        p_, r_, f_ = prec_rec_f1(tp_tot, fp_tot, fn_tot)
        rows.append({
            "iou_threshold": iou_t, "anchor_margin": marg,
            "visibility_threshold": vis,
            "tp": tp_tot, "fp": fp_tot, "fn": fn_tot,
            "precision": p_, "recall": r_, "f1": f_,
        })
        if f_ > best_f1:
            best_f1 = f_
            best_cfg = (iou_t, marg, vis)

    rows.sort(key=lambda r: -r["f1"])
    fields = ["iou_threshold","anchor_margin","visibility_threshold",
              "tp","fp","fn","precision","recall","f1"]
    with open(OUT_DIR / "ablation.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print(f"\nBest config: iou={best_cfg[0]}, margin={best_cfg[1]}, vis={best_cfg[2]}  F1={best_f1*100:.1f}%")
    print(f"→ {OUT_DIR / 'ablation.csv'}")


if __name__ == "__main__":
    main()
