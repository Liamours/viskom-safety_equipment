"""
Inference: image → PACT pipeline → JSON report + correctness metrics + viz.
Uses original-split weights.

Usage:
    python code/experiment/run_inference.py                        # test split
    python code/experiment/run_inference.py --split train          # 10% of train
    python code/experiment/run_inference.py --split train --frac 0.05
    python code/experiment/run_inference.py --dataset cppe5
    python code/experiment/run_inference.py --limit 20
    python code/experiment/run_inference.py --no-viz
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import DATASET_CLASS_MAPS, PACTPipeline
from landmarks.base import COCO_CONNECTIONS
from landmarks.core import draw_pose_landmarks
from reporting.generator import generate_report_data

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GT_CLASS_NAMES = {
    "chv":   ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"],
    "cppe5": ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"],
    "sh17":  ["person", "ear", "earmuffs", "face", "face-guard", "face-mask-medical",
               "foot", "tools", "glasses", "gloves", "helmet", "hands", "head",
               "medical-suit", "shoes", "safety-suit", "safety-vest"],
}

CONFIGS = {
    "chv": {
        "test_txt":  ROOT / "dataset/CHV/yolo_original/test_images.txt",
        "train_txt": ROOT / "dataset/CHV/yolo_original/train_images.txt",
        "label_dir": ROOT / "dataset/CHV/raw/labels",
        "weights":   ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "rule":      "chv",
        "dataset":   "chv",
    },
    "cppe5": {
        "test_txt":  ROOT / "dataset/CPPE-5/yolo_original/test_images.txt",
        "train_txt": ROOT / "dataset/CPPE-5/yolo_original/train_images.txt",
        "label_dir": ROOT / "dataset/CPPE-5/raw/labels",
        "weights":   ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "rule":      "cppe5",
        "dataset":   "cppe5",
    },
    "sh17": {
        "test_txt":  ROOT / "dataset/SH17/yolo_original/test_images.txt",
        "train_txt": ROOT / "dataset/SH17/yolo_original/train_images.txt",
        "label_dir": ROOT / "dataset/SH17/raw/labels",
        "weights":   ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "rule":      "sh17",
        "dataset":   "sh17",
    },
}

PER_IMAGE_FIELDS = [
    "image", "num_persons", "num_compliant", "compliance_rate",
    "gt_classes", "pred_classes", "tp", "fp", "fn",
    "precision", "recall", "f1",
]

SUMMARY_FIELDS = [
    "class", "gt_count", "tp", "fp", "fn", "precision", "recall", "f1",
]

COLOR_LANDMARK   = (0,   255, 128)
COLOR_CONNECTION = (255, 255, 255)
COLOR_PPE_BOX    = (255, 140,   0)
COLOR_PERSON_BOX = (200, 200, 200)
COLOR_LABEL_BG   = (30,  30,  30)
COLOR_LABEL_FG   = (255, 255, 255)


def _put_label(img_bgr: np.ndarray, text: str, x: int, y: int) -> None:
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img_bgr, (x, y - th - bl - 2), (x + tw + 4, y + 2), COLOR_LABEL_BG, -1)
    cv2.putText(img_bgr, text, (x + 2, y - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL_FG, 1, cv2.LINE_AA)


def viz_landmarks(img_bgr: np.ndarray, frame) -> np.ndarray:
    canvas = img_bgr.copy()
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    for person in frame.persons:
        if person.landmark_result is not None:
            canvas_rgb = draw_pose_landmarks(
                canvas_rgb, person.landmark_result,
                landmark_color=COLOR_LANDMARK,
                connection_color=COLOR_CONNECTION,
                landmark_radius=4,
                connection_thickness=2,
                connections=COCO_CONNECTIONS,
            )
    return cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)


VIZ_CONF_THRESHOLD = 0.5


def viz_detections(img_bgr: np.ndarray, frame) -> np.ndarray:
    canvas = img_bgr.copy()
    for i, person in enumerate(frame.persons):
        x1, y1, x2, y2 = person.person_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PERSON_BOX, 2)
        _put_label(canvas, f"W{i + 1}", x1, y1)
        for ppe in person.ppe_detections:
            if ppe["confidence"] < VIZ_CONF_THRESHOLD:
                continue
            bx1, by1, bx2, by2 = ppe["bbox"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COLOR_PPE_BOX, 2)
            _put_label(canvas, f"{ppe['class_name']} {ppe['confidence']:.2f}", bx1, by1)
    return canvas


def viz_combined(img_bgr: np.ndarray, frame) -> np.ndarray:
    canvas = viz_landmarks(img_bgr, frame)
    for i, person in enumerate(frame.persons):
        x1, y1, x2, y2 = person.person_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PERSON_BOX, 2)
        _put_label(canvas, f"W{i + 1}", x1, y1)
        for ppe in person.ppe_detections:
            if ppe["confidence"] < VIZ_CONF_THRESHOLD:
                continue
            bx1, by1, bx2, by2 = ppe["bbox"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COLOR_PPE_BOX, 2)
            _put_label(canvas, f"{ppe['class_name']} {ppe['confidence']:.2f}", bx1, by1)
    return canvas


def save_viz(img_bgr: np.ndarray, frame, out_dir: Path, stem: str) -> None:
    cv2.imwrite(str(out_dir / "viz_landmarks"  / f"{stem}.jpg"), viz_landmarks(img_bgr, frame))
    cv2.imwrite(str(out_dir / "viz_detections" / f"{stem}.jpg"), viz_detections(img_bgr, frame))
    cv2.imwrite(str(out_dir / "viz_combined"   / f"{stem}.jpg"), viz_combined(img_bgr, frame))


def load_gt_classes(label_path: Path, dataset: str) -> set:
    if not label_path.exists():
        return set()
    idx_to_name = GT_CLASS_NAMES[dataset]
    cmap        = DATASET_CLASS_MAPS.get(dataset, {})
    classes     = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        idx  = int(parts[0])
        name = idx_to_name[idx] if idx < len(idx_to_name) else str(idx)
        norm = cmap.get(name, name.lower())
        if norm != "person":
            classes.add(norm)
    return classes


def pred_classes_from_frame(frame_dict: dict) -> set:
    classes = set()
    for p in frame_dict.get("persons", []):
        classes.update(p.get("worn", []))
    return classes


def class_metrics(tp: int, fp: int, fn: int) -> tuple:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def build_pipeline(cfg: dict) -> PACTPipeline:
    return PACTPipeline(
        det_weights  = str(cfg["weights"]),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = cfg["dataset"],
        rule         = cfg["rule"],
        device       = "0",
    )


def collect_images(txt: Path, limit: int, frac: float = 1.0, seed: int = 42) -> list:
    imgs = [Path(p.strip()) for p in txt.read_text().splitlines() if p.strip()]
    if frac < 1.0:
        random.seed(seed)
        imgs = random.sample(imgs, max(1, int(len(imgs) * frac)))
        imgs.sort()
    return imgs[:limit] if limit > 0 else imgs


def run_dataset(name: str, cfg: dict, split: str, frac: float, limit: int, out_root: Path, do_viz: bool) -> None:
    out_dir = out_root / split / name
    for sub in ["json", "viz_landmarks", "viz_detections", "viz_combined"]:
        if sub == "json" or do_viz:
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    txt_key  = "train_txt" if split == "train" else "test_txt"
    images   = collect_images(cfg[txt_key], limit, frac=frac)
    pipeline = build_pipeline(cfg)

    per_image_rows = []
    class_accum    = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
    t0             = time.time()

    for img_path in tqdm(images, desc=name, unit="img", ncols=80):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame   = pipeline.run(img_rgb, image_path=str(img_path))
        fd      = frame.to_dict()
        report  = generate_report_data(fd, rule=cfg["rule"], dataset=name, image_path=str(img_path))

        (out_dir / "json" / f"{img_path.stem}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        if do_viz:
            save_viz(img_bgr, frame, out_dir, img_path.stem)

        label_path = cfg["label_dir"] / f"{img_path.stem}.txt"
        gt   = load_gt_classes(label_path, name)
        pred = pred_classes_from_frame(fd)

        tp_set = gt & pred
        fp_set = pred - gt
        fn_set = gt - pred
        tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
        prec, rec, f1 = class_metrics(tp, fp, fn)

        for cls in gt:
            class_accum[cls]["gt"] += 1
            if cls in pred:
                class_accum[cls]["tp"] += 1
            else:
                class_accum[cls]["fn"] += 1
        for cls in fp_set:
            class_accum[cls]["fp"] += 1

        per_image_rows.append({
            "image":           img_path.name,
            "num_persons":     fd["num_persons"],
            "num_compliant":   fd["num_compliant"],
            "compliance_rate": round(fd.get("compliance_rate", 0.0) or 0.0, 4),
            "gt_classes":      "|".join(sorted(gt)),
            "pred_classes":    "|".join(sorted(pred)),
            "tp":  tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

    with open(out_dir / "correctness_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        w.writeheader(); w.writerows(per_image_rows)

    summary_rows = []
    for cls, counts in sorted(class_accum.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec, rec, f1 = class_metrics(tp, fp, fn)
        summary_rows.append({
            "class": cls, "gt_count": counts["gt"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

    with open(out_dir / "correctness_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader(); w.writerows(summary_rows)

    valid     = per_image_rows
    mean_f1   = sum(r["f1"] for r in valid) / len(valid) if valid else 0.0
    mean_rate = sum(r["compliance_rate"] for r in valid) / len(valid) if valid else 0.0

    tqdm.write(
        f"[{name}] {len(valid)} images | mean F1 {mean_f1*100:.1f}% | "
        f"mean compliance {mean_rate*100:.1f}% | {time.time()-t0:.1f}s"
    )
    tqdm.write(f"  → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "chv", "cppe5", "sh17"])
    parser.add_argument("--split",   default="test", choices=["test", "train"])
    parser.add_argument("--frac",    type=float, default=0.10, help="fraction of train split to sample (default 0.10)")
    parser.add_argument("--limit",   type=int,   default=0,    help="hard cap on images per dataset (0 = no cap)")
    parser.add_argument("--no-viz",  action="store_true", help="skip saving visualization images")
    args = parser.parse_args()

    frac     = args.frac if args.split == "train" else 1.0
    out_root = ROOT / "results/reports"
    targets  = list(CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        run_dataset(name, CONFIGS[name], args.split, frac, args.limit, out_root, do_viz=not args.no_viz)


if __name__ == "__main__":
    main()
