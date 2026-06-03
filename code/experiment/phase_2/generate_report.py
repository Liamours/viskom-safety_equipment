"""
PPE Compliance Report Generator
================================
Runs the PACT pipeline on one or more images and produces
a self-contained HTML report per image.

Usage
-----
# Single image
python generate_report.py --image path/to/image.jpg --dataset sh17

# Sample N images from a dataset split
python generate_report.py --dataset sh17 --split original --n 5

# All test images
python generate_report.py --dataset chv --split original --n 0

Output
------
results/phase_2/<dataset>/reports/report_<stem>.html
"""
import sys
import json
import argparse
import random
from pathlib import Path

import cv2

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from reporting.generator import generate_report_data
from reporting.renderer  import render_html_report

RESULTS_DIR = ROOT / "results/phase_2"

DATASET_CFG = {
    "chv": {
        "splits": {
            "original": ROOT / "dataset/CHV/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/CHV/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_chv_8020/weights/best.pt",
        },
        "rule": "chv",
    },
    "cppe5": {
        "splits": {
            "original": ROOT / "dataset/CPPE-5/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/CPPE-5/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_cppe_5_8020/weights/best.pt",
        },
        "rule": "cppe5",
    },
    "sh17": {
        "splits": {
            "original": ROOT / "dataset/SH17/yolo_original/test_images.txt",
            "8020":     ROOT / "dataset/SH17/yolo_8020/test_images.txt",
        },
        "weights": {
            "original": ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
            "8020":     ROOT / "runs/detect/phase1_sh17_8020/weights/best.pt",
        },
        "rule": "sh17",
    },
}


def process_image(
    img_path: Path,
    pipeline: PACTPipeline,
    rule: str,
    dataset: str,
    out_dir: Path,
    save_json: bool = True,
) -> Path:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [!] Cannot read: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # PACT → FrameResult
    frame_result = pipeline.run(img_rgb, image_path=str(img_path))
    frame_dict   = frame_result.to_dict()

    # JSON sidecar (raw PACT output)
    if save_json:
        json_path = out_dir / f"pact_{img_path.stem}.json"
        json_path.write_text(json.dumps(frame_dict, indent=2), encoding="utf-8")

    # report data (rule-based text)
    report_data = generate_report_data(
        frame_dict   = frame_dict,
        rule         = rule,
        dataset      = dataset,
        image_path   = str(img_path),
        inspector    = "PACT Automated Vision System",
    )

    # HTML report
    html_path = out_dir / f"report_{img_path.stem}.html"
    render_html_report(report_data, image=img_bgr, output_path=html_path)

    n   = frame_dict["num_persons"]
    ok  = frame_dict["num_compliant"]
    tag = "ALL OK" if ok == n else f"{n - ok} VIOLATION(S)"
    print(f"  {img_path.name:<40} persons={n}  compliant={ok}  [{tag}]")
    print(f"    → {html_path}")

    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate PPE compliance HTML reports.")
    parser.add_argument("--dataset", default="sh17", choices=list(DATASET_CFG.keys()))
    parser.add_argument("--split",   default="original", choices=["original", "8020"])
    parser.add_argument("--n",       default=5,  type=int, help="Images to process (0=all)")
    parser.add_argument("--seed",    default=42, type=int)
    parser.add_argument("--image",   default=None, help="Single image path (overrides dataset)")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON sidecar output")
    args = parser.parse_args()

    if args.image:
        # single-image mode — infer dataset from arg
        img_path = Path(args.image)
        cfg      = DATASET_CFG[args.dataset]
        weights  = cfg["weights"][args.split]
        rule     = cfg["rule"]
        out_dir  = RESULTS_DIR / args.dataset / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = PACTPipeline(
            det_weights  = str(weights),
            pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
            dataset      = args.dataset,
            rule         = rule,
            device       = "0",
        )
        process_image(img_path, pipeline, rule, args.dataset, out_dir,
                      save_json=not args.no_json)
        return

    # batch mode
    cfg        = DATASET_CFG[args.dataset]
    split_file = cfg["splits"][args.split]
    weights    = cfg["weights"][args.split]
    rule       = cfg["rule"]

    if not Path(split_file).exists():
        print(f"[!] Split file not found: {split_file}")
        return
    if not Path(weights).exists():
        print(f"[!] Weights not found: {weights}")
        return

    image_paths = [Path(p.strip()) for p in open(split_file) if Path(p.strip()).exists()]
    if args.n > 0:
        random.seed(args.seed)
        image_paths = random.sample(image_paths, min(args.n, len(image_paths)))

    out_dir = RESULTS_DIR / args.dataset / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  PPE Compliance Report Generator")
    print(f"  Dataset: {args.dataset.upper()}  Split: {args.split}  Images: {len(image_paths)}")
    print(f"  Output: {out_dir}\n")

    pipeline = PACTPipeline(
        det_weights  = str(weights),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = args.dataset,
        rule         = rule,
        device       = "0",
    )

    for img_path in image_paths:
        process_image(img_path, pipeline, rule, args.dataset, out_dir,
                      save_json=not args.no_json)

    print(f"\n  Done. {len(image_paths)} report(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
