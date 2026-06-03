import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compliance.pipeline import PACTPipeline
from compliance.visualizer import draw_pact_result

ROOT         = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
RESULTS_DIR  = ROOT / "results/phase_2"

WEIGHTS = {
    "chv":    ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
    "cppe5":  ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
    "sh17":   ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
}

SAMPLE_DIRS = {
    "chv":   ROOT / "dataset/CHV/raw/images",
    "cppe5": ROOT / "dataset/CPPE-5/raw/images",
    "sh17":  ROOT / "dataset/SH17/raw_640/images",
}


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="PACT — Pose-Anchored Compliance Tracker test")
    parser.add_argument("--dataset", default="chv",   choices=["chv", "cppe5", "sh17"])
    parser.add_argument("--rule",    default="chv",   help="Compliance rule set (see rules.py)")
    parser.add_argument("--image",   default=None,    help="Path to a specific image (optional)")
    parser.add_argument("--n",       default=5, type=int, help="Number of sample images to test")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PACT — Pose-Anchored Compliance Tracker")
    print(f"  Dataset : {args.dataset.upper()}")
    print(f"  Rule    : {args.rule}")
    print("=" * 60)

    pipeline = PACTPipeline(
        det_weights  = str(WEIGHTS[args.dataset]),
        pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
        dataset      = args.dataset,
        rule         = args.rule,
        device       = "0",
    )

    if args.image:
        image_paths = [Path(args.image)]
    else:
        sample_dir  = SAMPLE_DIRS[args.dataset]
        image_paths = sorted(sample_dir.glob("*.jpg"))[:args.n] + \
                      sorted(sample_dir.glob("*.png"))[:args.n]
        image_paths = image_paths[:args.n]

    all_results = []

    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}")

        image_rgb  = load_image(img_path)
        result     = pipeline.run(image_rgb, image_path=str(img_path))
        annotated  = draw_pact_result(image_rgb, result)

        out_img = out_dir / f"pact_{img_path.stem}.jpg"
        cv2.imwrite(str(out_img), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        print(f"    Persons   : {result.num_persons}")
        print(f"    Compliant : {result.num_compliant}/{result.num_persons}")

        for p in result.persons:
            status = "COMPLIANT" if p.is_compliant else f"MISSING: {', '.join(p.missing)}"
            print(f"    Person {p.person_id} — {status}")
            if p.worn:
                print(f"      Worn: {', '.join(p.worn)}")

        all_results.append(result.to_dict())

    json_out = out_dir / "pact_results.json"
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Annotated images → {out_dir}")
    print(f"  JSON results     → {json_out}")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
