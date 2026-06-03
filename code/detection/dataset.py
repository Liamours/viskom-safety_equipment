import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")

VAL_RATIO = 0.15
SEED      = 42

SH17_NAMES: Dict[int, str] = {
    0:  "person",           1:  "ear",              2:  "earmuffs",
    3:  "face",             4:  "face-guard",        5:  "face-mask-medical",
    6:  "foot",             7:  "tools",             8:  "glasses",
    9:  "gloves",           10: "helmet",            11: "hands",
    12: "head",             13: "medical-suit",      14: "shoes",
    15: "safety-suit",      16: "safety-vest",
}


@dataclass
class DatasetConfig:
    name:       str
    images_dir: Path
    labels_dir: Path
    split_dir:  Path
    fmt:        str
    nc:         int
    names:      Optional[Dict[int, str]]
    coco_jsons: Optional[Dict[str, Path]] = field(default=None)

    @property
    def yolo_dir(self) -> Path:
        return BASE / self.name / "yolo"

    @property
    def bb_predict_dir(self) -> Path:
        return BASE / self.name / "bb-predict"


DATASETS: Dict[str, DatasetConfig] = {
    "CHV": DatasetConfig(
        name="CHV",
        images_dir=BASE / "CHV/raw/images",
        labels_dir=BASE / "CHV/raw/annotations",
        split_dir =BASE / "CHV/split",
        fmt="yolo",
        nc=6,
        names=None,
    ),
    "CPPE-5": DatasetConfig(
        name="CPPE-5",
        images_dir=BASE / "CPPE-5/raw/images",
        labels_dir=BASE / "CPPE-5/raw/annotations",
        split_dir =BASE / "CPPE-5/split",
        fmt="coco",
        nc=5,
        names=None,
        coco_jsons={
            "train": BASE / "CPPE-5/raw/annotations/train.json",
            "test":  BASE / "CPPE-5/raw/annotations/test.json",
        },
    ),
    "SH17": DatasetConfig(
        name="SH17",
        images_dir=BASE / "SH17/raw_640/images",
        labels_dir=BASE / "SH17/raw_640/labels",
        split_dir =BASE / "SH17/split",
        fmt="yolo",
        nc=17,
        names=SH17_NAMES,
    ),
}


def _load_split(split_dir: Path, split_name: str) -> List[str]:
    p = split_dir / f"{split_name}.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _split_list(
    items: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def _try_load_yolo_names(labels_dir: Path) -> Optional[Dict[int, str]]:
    for fname in ["classes.txt", "_classes.txt"]:
        p = labels_dir / fname
        if p.exists():
            lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            return {i: n for i, n in enumerate(lines)}
    for yp in list(labels_dir.parent.glob("*.yaml")) + list(labels_dir.parent.glob("*.yml")):
        try:
            with open(yp) as f:
                data = yaml.safe_load(f)
            names = data.get("names", None)
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
        except Exception:
            pass
    return None


def _load_coco_names(json_path: Path) -> Dict[int, str]:
    with open(json_path) as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    return {i: c["name"] for i, c in enumerate(categories)}


def _ensure_yolo_labels_dir(images_dir: Path, src_labels_dir: Path):
    expected = images_dir.parent / "labels"
    if expected == src_labels_dir or expected.exists():
        return
    expected.mkdir(parents=True, exist_ok=True)
    for lf in src_labels_dir.glob("*.txt"):
        target = expected / lf.name
        if target.exists():
            continue
        try:
            os.link(str(lf), str(target))
        except OSError:
            shutil.copy2(str(lf), str(target))


def _write_image_list(filenames: List[str], images_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(images_dir / f) for f in filenames))


def _convert_coco_to_yolo_labels(
    json_path: Path,
    labels_out_dir: Path,
    category_map: Dict[int, int],
) -> List[str]:
    with open(json_path) as f:
        data = json.load(f)

    id_to_meta = {
        img["id"]: {
            "fname": Path(img["file_name"]).name,
            "w":     img["width"],
            "h":     img["height"],
        }
        for img in data["images"]
    }

    anns_by_image: Dict[int, List] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    labels_out_dir.mkdir(parents=True, exist_ok=True)
    filenames: List[str] = []

    for img_id, meta in id_to_meta.items():
        fname = meta["fname"]
        iw, ih = meta["w"], meta["h"]
        lines: List[str] = []

        for ann in anns_by_image.get(img_id, []):
            cls_idx = category_map.get(ann["category_id"])
            if cls_idx is None:
                continue
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            bw = w / iw
            bh = h / ih
            cx  = max(0.0, min(1.0, cx))
            cy  = max(0.0, min(1.0, cy))
            bw  = max(0.0, min(1.0, bw))
            bh  = max(0.0, min(1.0, bh))
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        stem = Path(fname).stem
        (labels_out_dir / f"{stem}.txt").write_text("\n".join(lines))
        filenames.append(fname)

    return filenames


def _write_data_yaml(
    out_path: Path,
    nc: int,
    names: Dict[int, str],
    train: str,
    val: str,
    test: str,
    path: str = "",
):
    data = {
        "path":  path,
        "train": train,
        "val":   val,
        "test":  test,
        "nc":    nc,
        "names": [names[i] for i in range(nc)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


SPLIT_VARIANTS = ("original", "8020")


def prepare_dataset(
    config:        DatasetConfig,
    split_variant: str   = "original",
    val_ratio:     float = VAL_RATIO,
    seed:          int   = SEED,
) -> Path:
    if split_variant not in SPLIT_VARIANTS:
        raise ValueError(f"split_variant must be one of {SPLIT_VARIANTS}, got '{split_variant}'")

    split_dir = config.split_dir / split_variant
    yolo_dir  = BASE / config.name / f"yolo_{split_variant}"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    trainval   = _load_split(split_dir, "trainval")
    test_files = _load_split(split_dir, "test")

    if not trainval:
        raise FileNotFoundError(
            f"Split files not found at {split_dir}. "
            f"Run split_datasets.py first."
        )

    train_files, val_files = _split_list(trainval, val_ratio, seed)

    names     = config.names
    data_yaml = yolo_dir / "data.yaml"

    if config.fmt == "yolo":
        _ensure_yolo_labels_dir(config.images_dir, config.labels_dir)

        if names is None:
            names = _try_load_yolo_names(config.labels_dir) or {i: f"class_{i}" for i in range(config.nc)}

        train_txt = yolo_dir / "train_images.txt"
        val_txt   = yolo_dir / "val_images.txt"
        test_txt  = yolo_dir / "test_images.txt"

        _write_image_list(train_files, config.images_dir, train_txt)
        _write_image_list(val_files,   config.images_dir, val_txt)
        _write_image_list(test_files,  config.images_dir, test_txt)

        _write_data_yaml(
            data_yaml,
            nc=config.nc,
            names=names,
            train=str(train_txt),
            val=str(val_txt),
            test=str(test_txt),
        )

    elif config.fmt == "coco":
        if names is None:
            names = _load_coco_names(config.coco_jsons["train"])

        categories   = sorted(
            json.loads(config.coco_jsons["train"].read_text()).get("categories", []),
            key=lambda c: c["id"],
        )
        category_map = {c["id"]: i for i, c in enumerate(categories)}

        labels_dir = config.images_dir.parent / "labels"
        _convert_coco_to_yolo_labels(config.coco_jsons["train"], labels_dir, category_map)
        _convert_coco_to_yolo_labels(config.coco_jsons["test"],  labels_dir, category_map)

        train_txt = yolo_dir / "train_images.txt"
        val_txt   = yolo_dir / "val_images.txt"
        test_txt  = yolo_dir / "test_images.txt"

        _write_image_list(train_files, config.images_dir, train_txt)
        _write_image_list(val_files,   config.images_dir, val_txt)
        _write_image_list(test_files,  config.images_dir, test_txt)

        _write_data_yaml(
            data_yaml,
            nc=config.nc,
            names=names,
            train=str(train_txt),
            val=str(val_txt),
            test=str(test_txt),
        )

    print(f"  [{config.name}] split={split_variant}  data.yaml → {data_yaml}")
    print(f"           train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

    return data_yaml
