import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detection.dataset import DATASETS, prepare_dataset
from detection.trainer import Trainer

DATASET   = "CPPE-5"
MODEL     = "models/yolo26l.pt"
EPOCHS    = 150
BATCH     = 8
IMGSZ     = 640
PATIENCE  = 20
DEVICE    = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="original", choices=["original", "8020"])
    args = parser.parse_args()

    run_name  = f"phase1_{DATASET.lower().replace('-', '_')}_{args.split}"
    cfg       = DATASETS[DATASET]
    data_yaml = prepare_dataset(cfg, split_variant=args.split)

    trainer = Trainer(model_name=MODEL)
    best    = trainer.train(
        data_yaml = data_yaml,
        run_name  = run_name,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        patience  = PATIENCE,
        device    = DEVICE,
        exist_ok  = True,
    )

    print(f"\n[{DATASET}] Training complete.")
    print(f"  Best weights : {best}")


if __name__ == "__main__":
    main()
