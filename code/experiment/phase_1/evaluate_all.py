import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detection.dataset import DATASETS, prepare_dataset
from detection.evaluator import evaluate, print_results, save_results
from detection.predictor import Predictor

RUNS_DIR    = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/runs/detect")
RESULTS_DIR = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/results/phase_1")
DEVICE      = "0"

EXPERIMENTS = [
    ("CHV",    "original", RUNS_DIR / "phase1_chv_original/weights/best.pt"),
    ("CHV",    "8020",     RUNS_DIR / "phase1_chv_8020/weights/best.pt"),
    ("CPPE-5", "original", RUNS_DIR / "phase1_cppe_5_original/weights/best.pt"),
    ("CPPE-5", "8020",     RUNS_DIR / "phase1_cppe_5_8020/weights/best.pt"),
    ("SH17",   "original", RUNS_DIR / "phase1_sh17_original/weights/best.pt"),
    ("SH17",   "8020",     RUNS_DIR / "phase1_sh17_8020/weights/best.pt"),
]

RUN_PREDICT = False


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 1 — EVALUATION")
    print("=" * 60)

    for name, split, weights_path in EXPERIMENTS:
        if not weights_path.exists():
            print(f"\n  [{name}/{split}] weights not found → {weights_path}")
            continue

        cfg       = DATASETS[name]
        data_yaml = prepare_dataset(cfg, split_variant=split)
        results   = evaluate(weights_path, data_yaml, split="test", device=DEVICE)
        print_results(f"{name} ({split})", results)

        out_name = f"{name.lower().replace('-', '_')}_{split}_results.json"
        save_results(results, RESULTS_DIR / out_name)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
