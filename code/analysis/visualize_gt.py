import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.image import load_as_rgb, to_numpy_rgb

BASE = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")

SH17_CLASSES: Dict[int, str] = {
    0:  "person",           1:  "ear",          2:  "earmuffs",
    3:  "face",             4:  "face-guard",   5:  "face-mask-medical",
    6:  "foot",             7:  "tools",         8:  "glasses",
    9:  "gloves",           10: "helmet",        11: "hands",
    12: "head",             13: "medical-suit",  14: "shoes",
    15: "safety-suit",      16: "safety-vest",
}

DATASETS: Dict[str, Dict] = {
    "CHV": {
        "images":     BASE / "CHV/raw/images",
        "labels":     BASE / "CHV/raw/annotations",
        "split_dir":  BASE / "CHV/split",
        "format":     "yolo",
        "classes":    {},
        "output":     BASE / "CHV/bb-gt",
        "split_mode": "subdir",
    },
    "CPPE-5": {
        "images":     BASE / "CPPE-5/raw/images",
        "labels":     BASE / "CPPE-5/raw/annotations",
        "split_dir":  BASE / "CPPE-5/split",
        "format":     "coco",
        "classes":    {},
        "output":     BASE / "CPPE-5/bb-gt",
        "split_mode": "subdir",
    },
    "SH17": {
        "images":     BASE / "SH17/raw/images",
        "labels":     BASE / "SH17/raw/labels",
        "split_dir":  BASE / "SH17/split",
        "format":     "yolo",
        "classes":    SH17_CLASSES,
        "output":     BASE / "SH17/bb-gt",
        "split_mode": "subdir",
    },
}

BOX_THICKNESS   = 2
LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE     = 0.45
LABEL_THICKNESS = 1
OUTPUT_EXT      = ".jpg"
JPEG_QUALITY    = 92


def _make_palette(n: int) -> List[Tuple[int, int, int]]:
    palette = []
    for i in range(max(n, 1)):
        hue = int(180 * i / max(n, 1))
        hsv = np.array([[[hue, 200, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


def _load_split(split_dir: Path, split_name: str) -> List[str]:
    path = split_dir / f"{split_name}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_yolo_classes(label_dir: Path) -> Dict[int, str]:
    for name in ["classes.txt", "_classes.txt"]:
        p = label_dir / name
        if p.exists():
            lines = p.read_text().splitlines()
            return {i: line.strip() for i, line in enumerate(lines) if line.strip()}
    for yp in list(label_dir.parent.glob("*.yaml")) + list(label_dir.parent.glob("*.yml")):
        try:
            import yaml
            with open(yp) as f:
                data = yaml.safe_load(f)
            names = data.get("names", {})
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
        except Exception:
            pass
    return {}


def _load_coco_lookup(ann_dir: Path) -> Tuple[Dict[str, List], Dict[int, str]]:
    image_anns: Dict[str, List] = {}
    categories: Dict[int, str] = {}
    for json_path in sorted(ann_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            continue
        if "images" not in data or "annotations" not in data:
            continue
        categories.update({c["id"]: c["name"] for c in data.get("categories", [])})
        id_to_name = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
        for ann in data["annotations"]:
            fname = id_to_name.get(ann["image_id"])
            if fname is None:
                continue
            image_anns.setdefault(fname, []).append({
                "bbox":        ann["bbox"],
                "category_id": ann["category_id"],
            })
    return image_anns, categories


def _draw_label(canvas: np.ndarray, text: str, x1: int, y1: int, color: Tuple) -> np.ndarray:
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    ly = max(y1 - 4, th + 4)
    cv2.rectangle(canvas, (x1, ly - th - 3), (x1 + tw + 4, ly + 2), color, -1)
    cv2.putText(canvas, text, (x1 + 2, ly - 1), LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return canvas


def _draw_yolo_boxes(
    canvas: np.ndarray,
    label_path: Path,
    img_w: int,
    img_h: int,
    class_names: Dict[int, str],
    palette: List[Tuple],
) -> np.ndarray:
    if not label_path.exists():
        return canvas
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        color = palette[cls_id % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        _draw_label(canvas, class_names.get(cls_id, str(cls_id)), x1, y1, color)
    return canvas


def _draw_coco_boxes(
    canvas: np.ndarray,
    anns: List[Dict],
    categories: Dict[int, str],
    palette: List[Tuple],
) -> np.ndarray:
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cat_id = ann["category_id"]
        color = palette[cat_id % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        _draw_label(canvas, categories.get(cat_id, str(cat_id)), x1, y1, color)
    return canvas


def _save_image(canvas: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _process_split(
    split_name: str,
    filenames: List[str],
    images_dir: Path,
    output_dir: Path,
    fmt: str,
    labels_dir: Optional[Path],
    class_names: Dict[int, str],
    palette: List[Tuple],
    image_anns: Optional[Dict],
    categories: Optional[Dict],
    skip_existing: bool = True,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / f"{split_name}.txt").write_text("\n".join(filenames))
    stats = {"total": len(filenames), "processed": 0, "skipped": 0, "errors": 0}

    for fname in tqdm(filenames, desc=f"    {split_name}", unit="img", leave=False):
        out_path = output_dir / (Path(fname).stem + OUTPUT_EXT)
        if skip_existing and out_path.exists():
            stats["skipped"] += 1
            continue
        try:
            canvas = cv2.cvtColor(to_numpy_rgb(load_as_rgb(images_dir / fname)), cv2.COLOR_RGB2BGR)
            h, w = canvas.shape[:2]
            if fmt == "yolo":
                label_path = labels_dir / (Path(fname).stem + ".txt")
                canvas = _draw_yolo_boxes(canvas, label_path, w, h, class_names, palette)
            else:
                canvas = _draw_coco_boxes(canvas, image_anns.get(fname, []), categories, palette)
            _save_image(canvas, out_path)
            stats["processed"] += 1
        except Exception as e:
            tqdm.write(f"    ERROR [{fname}]: {e}")
            stats["errors"] += 1
    return stats


def process_dataset(name: str, cfg: dict):
    fmt        = cfg["format"]
    images_dir = cfg["images"]
    labels_dir = cfg["labels"]
    split_dir  = cfg["split_dir"]
    output_dir = cfg["output"]

    trainval = _load_split(split_dir, "trainval")
    test     = _load_split(split_dir, "test")

    if not trainval and not test:
        print(f"\n  [{name}]  ERROR: split files not found in {split_dir}. Run split_datasets.py first.")
        return

    print(f"\n  [{name}]  trainval={len(trainval)}  test={len(test)}")

    image_anns = categories = None
    if fmt == "yolo":
        class_names = cfg["classes"] or _load_yolo_classes(labels_dir)
        n_classes   = (max(class_names.keys()) + 1) if class_names else 20
    else:
        image_anns, categories = _load_coco_lookup(labels_dir)
        class_names = {}
        n_classes   = (max(categories.keys()) + 1) if categories else 10

    palette = _make_palette(n_classes)

    for split_name, filenames in [("trainval", trainval), ("test", test)]:
        sub_dir = output_dir / split_name
        stats = _process_split(split_name, filenames, images_dir, sub_dir, fmt,
                                labels_dir, class_names, palette, image_anns, categories)
        print(f"    [{split_name}]  processed={stats['processed']}  skipped={stats['skipped']}  errors={stats['errors']}")


def main():
    print("=" * 60)
    print("  GT BOUNDING BOX VISUALIZATION")
    print("=" * 60)

    for name, cfg in DATASETS.items():
        process_dataset(name, cfg)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
