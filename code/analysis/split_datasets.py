import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE       = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")
TEST_RATIO = 0.20
SEED       = 42

DATASETS: Dict[str, Dict] = {
    "CHV": {
        "format":    "yolo",
        "split_dir": BASE / "CHV/split",
        "original": {
            "train_txts": [
                BASE / "CHV/raw/data split/train.txt",
                BASE / "CHV/raw/data split/valid.txt",
            ],
            "test_txts": [BASE / "CHV/raw/data split/test.txt"],
        },
        "all_sources": {
            "txts": [
                BASE / "CHV/raw/data split/train.txt",
                BASE / "CHV/raw/data split/valid.txt",
                BASE / "CHV/raw/data split/test.txt",
            ],
        },
    },
    "CPPE-5": {
        "format":    "coco",
        "split_dir": BASE / "CPPE-5/split",
        "original": {
            "train_jsons": [BASE / "CPPE-5/raw/annotations/train.json"],
            "test_jsons":  [BASE / "CPPE-5/raw/annotations/test.json"],
        },
        "all_sources": {
            "jsons": [
                BASE / "CPPE-5/raw/annotations/train.json",
                BASE / "CPPE-5/raw/annotations/test.json",
            ],
        },
    },
    "SH17": {
        "format":    "yolo",
        "split_dir": BASE / "SH17/split",
        "original": {
            "train_txts": [BASE / "SH17/raw/train_files.txt"],
            "test_txts":  [BASE / "SH17/raw/val_files.txt"],
        },
        "all_sources": {
            "txts": [
                BASE / "SH17/raw/train_files.txt",
                BASE / "SH17/raw/val_files.txt",
            ],
        },
    },
}


def _read_yolo_txt(path: Path) -> List[str]:
    return [Path(line.strip()).name for line in path.read_text().splitlines() if line.strip()]


def _read_coco_json(path: Path) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    return [Path(img["file_name"]).name for img in data["images"]]


def _dedup(filenames: List[str]) -> List[str]:
    seen = set()
    out  = []
    for f in filenames:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _collect_txts(paths: List[Path]) -> List[str]:
    out = []
    for p in paths:
        out.extend(_read_yolo_txt(p))
    return _dedup(out)


def _collect_jsons(paths: List[Path]) -> List[str]:
    out = []
    for p in paths:
        out.extend(_read_coco_json(p))
    return _dedup(out)


def split_original(name: str, cfg: dict) -> Tuple[List[str], List[str]]:
    fmt    = cfg["format"]
    orig   = cfg["original"]
    if fmt == "yolo":
        trainval = _collect_txts(orig["train_txts"])
        test     = _collect_txts(orig["test_txts"])
    else:
        trainval = _collect_jsons(orig["train_jsons"])
        test     = _collect_jsons(orig["test_jsons"])
    return trainval, test


def split_8020(name: str, cfg: dict) -> Tuple[List[str], List[str]]:
    fmt = cfg["format"]
    src = cfg["all_sources"]
    if fmt == "yolo":
        all_files = _collect_txts(src["txts"])
    else:
        all_files = _collect_jsons(src["jsons"])

    rng = random.Random(SEED)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * TEST_RATIO))
    return shuffled[:-n_test], shuffled[-n_test:]


def save_split(split_dir: Path, variant: str, trainval: List[str], test: List[str]):
    out = split_dir / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "trainval.txt").write_text("\n".join(trainval))
    (out / "test.txt").write_text("\n".join(test))


def print_summary(name: str, variant: str, trainval: List[str], test: List[str]):
    total = len(trainval) + len(test)
    print(f"    [{variant}]  total={total}  trainval={len(trainval)} ({100*len(trainval)/total:.0f}%)  test={len(test)} ({100*len(test)/total:.0f}%)")


def main():
    print("=" * 60)
    print("  DATASET SPLIT GENERATION")
    print("=" * 60)

    for name, cfg in DATASETS.items():
        print(f"\n  {name}")

        tv_orig, t_orig  = split_original(name, cfg)
        save_split(cfg["split_dir"], "original", tv_orig, t_orig)
        print_summary(name, "original", tv_orig, t_orig)

        tv_8020, t_8020  = split_8020(name, cfg)
        save_split(cfg["split_dir"], "8020", tv_8020, t_8020)
        print_summary(name, "8020", tv_8020, t_8020)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
