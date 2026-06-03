"""
Benchmark all three datasets. Fixed timing: each model runs once per image.
Stages: PPE detect | Pose detect | Anchor+compliance | Report
Output: results/eval/benchmark_all.csv, benchmark_all_summary.json
"""
import csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline, _arm_hand_anchor
from compliance.rules import COMPLIANCE_RULES
from landmarks.base import COCO_PPE_ANCHOR_GROUPS
from landmarks.core import build_compliance_map, get_ppe_anchors
from compliance.pipeline import PersonResult, FrameResult
from reporting.generator import generate_report_data

WARMUP = 3
OUT_DIR = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "cppe5": {
        "images_src": "coco_json",
        "ann":        ROOT / "dataset/CPPE-5/raw/annotations/test.json",
        "img_dir":    ROOT / "dataset/CPPE-5/raw/images",
        "weights":    ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "dataset":    "cppe5",
        "rule":       "cppe5",
    },
    "chv": {
        "images_src": "txt",
        "split":      ROOT / "dataset/CHV/yolo_original/test_images.txt",
        "weights":    ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "dataset":    "chv",
        "rule":       "chv",
    },
    "sh17": {
        "images_src": "txt",
        "split":      ROOT / "dataset/SH17/yolo_original/test_images.txt",
        "weights":    ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "dataset":    "sh17",
        "rule":       "sh17",
    },
}


def ms(t):
    return round(t * 1000, 2)


def load_images(cfg):
    if cfg["images_src"] == "coco_json":
        coco = json.load(open(cfg["ann"]))
        imgs = []
        for m in coco["images"]:
            p = cfg["img_dir"] / m["file_name"]
            bgr = cv2.imread(str(p))
            if bgr is not None:
                imgs.append(bgr)
        return imgs
    else:
        paths = [Path(l.strip()) for l in open(cfg["split"])]
        imgs = []
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is not None:
                imgs.append(bgr)
        return imgs


def run_anchor_compliance(pipeline, ppe_dets, persons, img_rgb):
    h, w = img_rgb.shape[:2]
    all_bboxes = [p[0] for p in persons]
    person_results = []

    for idx, (person_bbox, lm_result) in enumerate(persons):
        assigned = pipeline._assign_ppe(ppe_dets, person_bbox, all_bboxes)

        anchors = get_ppe_anchors(
            lm_result,
            anchor_groups=COCO_PPE_ANCHOR_GROUPS,
            margin=pipeline.anchor_margin,
            visibility_threshold=0.3,
        )

        px1, py1, px2, py2 = person_bbox
        if anchors.get("helmet_region") is not None:
            ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
            anchors["helmet_region"]["bbox"] = (ax1, py1, ax2, ay2)
        else:
            bb = (px1, py1, px2, py1 + int((py2 - py1) * 0.35))
            anchors["helmet_region"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

        for side, key in [("left", "left_hand"), ("right", "right_hand")]:
            if anchors.get(key) is None:
                bb = _arm_hand_anchor(lm_result, person_bbox, side, 0.3)
                anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

        def _foot(knee_i):
            lms = lm_result.landmarks
            iw2, ih2 = lm_result.image_width, lm_result.image_height
            if knee_i < len(lms) and lms[knee_i].visibility >= 0.3:
                ky = int(lms[knee_i].y * ih2)
                return (px1, ky, px2, py2)
            return (px1, py1 + int((py2 - py1) * 0.6), px2, py2)

        for knee_i, key in [(13, "left_foot"), (14, "right_foot")]:
            if anchors.get(key) is None:
                bb = _foot(knee_i)
                anchors[key] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

        compliance_map = build_compliance_map(
            anchors=anchors, detections=assigned,
            iou_threshold=pipeline.iou_threshold,
            anchor_margin=0.0, image_width=w, image_height=h,
        )
        person_results.append(PersonResult(
            person_id=idx, person_bbox=person_bbox,
            landmark_result=lm_result, ppe_detections=assigned,
            compliance_map=compliance_map,
            required_ppe=pipeline.required_ppe,
        ))

    return FrameResult(image_path="", persons=person_results)


def benchmark_dataset(name, cfg):
    print(f"\nLoading {name} images...")
    images = load_images(cfg)
    print(f"  {len(images)} images")

    pipeline = PACTPipeline(
        det_weights=str(cfg["weights"]),
        pose_weights=str(ROOT / "models/yolov8x-pose.pt"),
        dataset=cfg["dataset"], rule=cfg["rule"], device="0",
    )

    print(f"  Warming up ({WARMUP})...")
    for bgr in images[:WARMUP]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pipeline._detect_ppe(rgb)
        pipeline._detect_persons(rgb)

    rows = []
    for bgr in tqdm(images, desc=name, ncols=80):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        ppe_dets = pipeline._detect_ppe(rgb)
        t1 = time.perf_counter()
        persons  = pipeline._detect_persons(rgb)
        t2 = time.perf_counter()
        frame    = run_anchor_compliance(pipeline, ppe_dets, persons, rgb)
        t3 = time.perf_counter()
        fd       = frame.to_dict()
        report   = generate_report_data(fd, rule=cfg["rule"], dataset=name)
        t4 = time.perf_counter()

        rows.append({
            "dataset":    name,
            "n_persons":  len(persons),
            "t_ppe_ms":   ms(t1 - t0),
            "t_pose_ms":  ms(t2 - t1),
            "t_anchor_ms":ms(t3 - t2),
            "t_report_ms":ms(t4 - t3),
            "t_total_ms": ms(t4 - t0),
        })

    return rows


def summarize(rows, label):
    def avg(k): return round(sum(r[k] for r in rows) / len(rows), 2)
    total = avg("t_total_ms")
    return {
        "label":        label,
        "n_images":     len(rows),
        "mean_fps":     round(1000 / total, 2),
        "mean_total_ms":total,
        "mean_ppe_ms":  avg("t_ppe_ms"),
        "mean_pose_ms": avg("t_pose_ms"),
        "mean_anchor_ms": avg("t_anchor_ms"),
        "mean_report_ms": avg("t_report_ms"),
    }


def main():
    all_rows = []
    summaries = []

    for name, cfg in DATASETS.items():
        rows = benchmark_dataset(name, cfg)
        all_rows.extend(rows)
        s = summarize(rows, name)
        summaries.append(s)
        print(f"  {name}: {s['mean_total_ms']} ms  ({s['mean_fps']} FPS)")
        print(f"    PPE={s['mean_ppe_ms']}ms  Pose={s['mean_pose_ms']}ms  "
              f"Anchor={s['mean_anchor_ms']}ms  Report={s['mean_report_ms']}ms")

    overall = summarize(all_rows, "overall")
    summaries.append(overall)
    print(f"\nOverall ({overall['n_images']} images): "
          f"{overall['mean_total_ms']} ms  ({overall['mean_fps']} FPS)")

    fields = ["dataset", "n_persons", "t_ppe_ms", "t_pose_ms",
              "t_anchor_ms", "t_report_ms", "t_total_ms"]
    with open(OUT_DIR / "benchmark_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    with open(OUT_DIR / "benchmark_all_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n-> {OUT_DIR}/benchmark_all_summary.json")


if __name__ == "__main__":
    main()
