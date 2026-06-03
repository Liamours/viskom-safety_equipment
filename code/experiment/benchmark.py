"""
End-to-end runtime benchmark on CPPE-5 test set.
Times each pipeline stage and shows scaling with number of persons.

Output: results/eval/benchmark.csv
        results/eval/benchmark_summary.json
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from reporting.generator import generate_report_data

ANN_PATH = ROOT / "dataset/CPPE-5/raw/annotations/test.json"
IMG_DIR  = ROOT / "dataset/CPPE-5/raw/images"
WEIGHTS  = ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt"
POSE_W   = ROOT / "models/yolov8x-pose.pt"
OUT_DIR  = ROOT / "results/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 3


def ms(t): return round(t * 1000, 2)


def main():
    coco = json.load(open(ANN_PATH))
    pipeline = PACTPipeline(
        det_weights=str(WEIGHTS), pose_weights=str(POSE_W),
        dataset="cppe5", rule="cppe5", device="0",
    )

    images = []
    for img_meta in coco["images"]:
        img_bgr = cv2.imread(str(IMG_DIR / img_meta["file_name"]))
        if img_bgr is not None:
            images.append((img_meta["file_name"], img_bgr))

    print(f"Warming up ({WARMUP} images)...")
    for _, img_bgr in images[:WARMUP]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pipeline.run(img_rgb)

    rows = []
    for fname, img_bgr in tqdm(images, desc="benchmark", ncols=80):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        ppe_dets = pipeline._detect_ppe(img_rgb)
        t1 = time.perf_counter()
        persons  = pipeline._detect_persons(img_rgb)
        t2 = time.perf_counter()
        frame    = pipeline.run(img_rgb)
        t3 = time.perf_counter()
        fd       = frame.to_dict()
        report   = generate_report_data(fd, rule="cppe5", dataset="cppe5")
        t4 = time.perf_counter()

        n_persons = len(persons)
        rows.append({
            "image":        fname,
            "n_persons":    n_persons,
            "t_ppe_ms":     ms(t1-t0),
            "t_pose_ms":    ms(t2-t1),
            "t_pipeline_ms":ms(t3-t0),
            "t_report_ms":  ms(t4-t3),
            "t_total_ms":   ms(t4-t0),
        })

    fields = ["image","n_persons","t_ppe_ms","t_pose_ms","t_pipeline_ms","t_report_ms","t_total_ms"]
    with open(OUT_DIR / "benchmark.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    by_n = defaultdict(list)
    for r in rows: by_n[r["n_persons"]].append(r["t_total_ms"])

    def avg(lst): return round(sum(lst)/len(lst), 2)

    scaling = {str(n): {"count": len(ts), "mean_total_ms": avg(ts)}
               for n, ts in sorted(by_n.items())}

    all_total = [r["t_total_ms"] for r in rows]
    all_ppe   = [r["t_ppe_ms"]   for r in rows]
    all_pose  = [r["t_pose_ms"]  for r in rows]
    summary   = {
        "n_images":        len(rows),
        "mean_total_ms":   avg(all_total),
        "mean_fps":        round(1000/avg(all_total), 2),
        "mean_ppe_ms":     avg(all_ppe),
        "mean_pose_ms":    avg(all_pose),
        "mean_report_ms":  avg([r["t_report_ms"] for r in rows]),
        "scaling_by_n_persons": scaling,
    }

    (OUT_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n── Runtime Summary ──")
    print(f"  Images tested : {summary['n_images']}")
    print(f"  Mean total    : {summary['mean_total_ms']} ms  ({summary['mean_fps']} FPS)")
    print(f"  PPE detect    : {summary['mean_ppe_ms']} ms")
    print(f"  Pose detect   : {summary['mean_pose_ms']} ms")
    print(f"  Report gen    : {summary['mean_report_ms']} ms")
    print(f"\n  Scaling (mean total ms by #persons):")
    for n, d in scaling.items():
        print(f"    {n} person(s): {d['mean_total_ms']} ms  (n={d['count']})")
    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    main()
