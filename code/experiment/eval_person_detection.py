"""
Person detection evaluation (YOLOv8x-pose) across all 3 datasets.

GT person sources:
  CHV    — YOLO labels, class index 0 = person
  SH17   — YOLO labels, class index 0 = person
  CPPE-5 — no person class; GT persons derived by clustering PPE annotations

Output: results/eval/person_detection_{chv|cppe5|sh17}.csv
        results/eval/person_detection_summary.json
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from experiment.eval_utils import (
    PERSON_IOU_THRESH, cluster_gt_persons, iou,
    load_cppe5_gt, load_yolo_person_bboxes, prec_rec_f1,
)

OUT_DIR  = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSE_W = ROOT / "models/yolov8x-pose.pt"

CONFIGS = {
    "chv": {
        "test_txt":  ROOT / "dataset/CHV/yolo_original/test_images.txt",
        "label_dir": ROOT / "dataset/CHV/raw/labels",
        "gt_mode":   "yolo",
        "person_class": 0,
        "weights":   ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "rule": "chv", "dataset": "chv",
    },
    "sh17": {
        "test_txt":  ROOT / "dataset/SH17/yolo_original/test_images.txt",
        "label_dir": ROOT / "dataset/SH17/raw/labels",
        "gt_mode":   "yolo",
        "person_class": 0,
        "weights":   ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "rule": "sh17", "dataset": "sh17",
    },
    "cppe5": {
        "ann_path":  ROOT / "dataset/CPPE-5/raw/annotations/test.json",
        "img_dir":   ROOT / "dataset/CPPE-5/raw/images",
        "gt_mode":   "coco_cluster",
        "weights":   ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "rule": "cppe5", "dataset": "cppe5",
    },
}

FIELDS = ["image","gt_persons","pred_persons","tp","fp","fn","precision","recall","f1"]


def match_persons(gt_bboxes, pred_bboxes):
    matched_gt = set()
    tp = 0
    for pb in pred_bboxes:
        best_s, best_i = 0.0, -1
        for gi, gb in enumerate(gt_bboxes):
            s = iou(pb, gb)
            if s > best_s: best_s, best_i = s, gi
        if best_s >= PERSON_IOU_THRESH and best_i not in matched_gt:
            tp += 1
            matched_gt.add(best_i)
    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes)   - len(matched_gt)
    return tp, fp, fn


def run_yolo_dataset(name, cfg, pipeline):
    images = [Path(p.strip()) for p in
              cfg["test_txt"].read_text().splitlines() if p.strip()]
    rows = []
    agg = {"tp":0,"fp":0,"fn":0,"gt":0,"pred":0}

    for img_path in tqdm(images, desc=name, ncols=80):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_h, img_w = img_bgr.shape[:2]

        label_path = cfg["label_dir"] / f"{img_path.stem}.txt"
        gt_bboxes  = load_yolo_person_bboxes(label_path, img_w, img_h, cfg["person_class"])
        if not gt_bboxes: continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        persons  = pipeline._detect_persons(img_rgb)
        pred_bboxes = [p[0] for p in persons]

        tp, fp, fn = match_persons(gt_bboxes, pred_bboxes)
        p_, r_, f_ = prec_rec_f1(tp, fp, fn)
        agg["tp"]+=tp; agg["fp"]+=fp; agg["fn"]+=fn
        agg["gt"]+=len(gt_bboxes); agg["pred"]+=len(pred_bboxes)
        rows.append({
            "image": img_path.name, "gt_persons": len(gt_bboxes),
            "pred_persons": len(pred_bboxes),
            "tp":tp,"fp":fp,"fn":fn,"precision":p_,"recall":r_,"f1":f_,
        })

    return rows, agg


def run_cppe5_dataset(name, cfg, pipeline):
    images, id_to_img, id_to_anns = load_cppe5_gt(cfg["ann_path"])
    rows = []
    agg = {"tp":0,"fp":0,"fn":0,"gt":0,"pred":0}

    for img_meta in tqdm(images, desc=name, ncols=80):
        img_id   = img_meta["id"]
        img_path = cfg["img_dir"] / img_meta["file_name"]
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_w, img_h = img_meta["width"], img_meta["height"]

        anns       = id_to_anns[img_id]
        gt_persons = cluster_gt_persons(anns, img_w, img_h)
        gt_bboxes  = [gp["bbox"] for gp in gt_persons]

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        persons  = pipeline._detect_persons(img_rgb)
        pred_bboxes = [p[0] for p in persons]

        tp, fp, fn = match_persons(gt_bboxes, pred_bboxes)
        p_, r_, f_ = prec_rec_f1(tp, fp, fn)
        agg["tp"]+=tp; agg["fp"]+=fp; agg["fn"]+=fn
        agg["gt"]+=len(gt_bboxes); agg["pred"]+=len(pred_bboxes)
        rows.append({
            "image": img_meta["file_name"], "gt_persons": len(gt_bboxes),
            "pred_persons": len(pred_bboxes),
            "tp":tp,"fp":fp,"fn":fn,"precision":p_,"recall":r_,"f1":f_,
        })

    return rows, agg


def main():
    summary = {}
    print("\n── Person Detection: YOLOv8x-pose ──")
    print(f"{'Dataset':<10} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")

    for name, cfg in CONFIGS.items():
        pipeline = PACTPipeline(
            det_weights  = str(cfg["weights"]),
            pose_weights = str(POSE_W),
            dataset      = cfg["dataset"],
            rule         = cfg["rule"],
            device       = "0",
        )

        if cfg["gt_mode"] == "yolo":
            rows, agg = run_yolo_dataset(name, cfg, pipeline)
        else:
            rows, agg = run_cppe5_dataset(name, cfg, pipeline)

        with open(OUT_DIR / f"person_detection_{name}.csv","w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)

        tp,fp,fn = agg["tp"],agg["fp"],agg["fn"]
        p_,r_,f_ = prec_rec_f1(tp,fp,fn)
        summary[name] = {
            "gt_persons": agg["gt"], "pred_persons": agg["pred"],
            "tp":tp,"fp":fp,"fn":fn,
            "precision":p_,"recall":r_,"f1":f_,
            "gt_source": cfg["gt_mode"],
        }
        print(f"  {name:<8} {agg['gt']:>6} {agg['pred']:>6} {tp:>6} {fp:>6} {fn:>6} "
              f"{p_*100:>6.1f}% {r_*100:>6.1f}% {f_*100:>6.1f}%")

    (OUT_DIR / "person_detection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    main()
