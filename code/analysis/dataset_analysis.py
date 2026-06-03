import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

BASE = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")

PATHS = {
    "CHV": {
        "images": BASE / "CHV/raw/images",
        "labels": BASE / "CHV/raw/annotations",
        "splits": BASE / "CHV/raw/data split",
    },
    "CPPE-5": {
        "images": BASE / "CPPE-5/raw/images",
        "annotations": BASE / "CPPE-5/raw/annotations",
    },
    "SH17": {
        "images": BASE / "SH17/raw_640/images",
        "labels": BASE / "SH17/raw_640/labels",
        "split_train": BASE / "SH17/raw/train_files.txt",
        "split_val": BASE / "SH17/raw/val_files.txt",
    },
}

CHV_CLASSES = {
    0: "person",
    1: "vest",
    2: "blue_helmet",
    3: "red_helmet",
    4: "white_helmet",
    5: "yellow_helmet",
}

SH17_CLASSES = {
    0: "person",
    1: "ear",
    2: "earmuffs",
    3: "face",
    4: "face-guard",
    5: "face-mask-medical",
    6: "foot",
    7: "tools",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def scan_images(image_dir: Path, dataset_name: str) -> dict:
    paths = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    heights, widths, channels = [], [], []
    modalities = Counter()

    for p in tqdm(paths, desc=f"  Scanning images", unit="img"):
        try:
            img = Image.open(p)
            w, h = img.size
            modalities[img.mode] += 1
            heights.append(h)
            widths.append(w)
            channels.append(len(img.getbands()))
        except Exception:
            continue

    h_arr = np.array(heights)
    w_arr = np.array(widths)
    c_arr = np.array(channels)

    return {
        "count": len(heights),
        "modalities": dict(modalities),
        "height": {
            "min": int(h_arr.min()),
            "max": int(h_arr.max()),
            "mean": float(h_arr.mean()),
            "std": float(h_arr.std()),
        },
        "width": {
            "min": int(w_arr.min()),
            "max": int(w_arr.max()),
            "mean": float(w_arr.mean()),
            "std": float(w_arr.std()),
        },
        "channels": {
            "min": int(c_arr.min()),
            "max": int(c_arr.max()),
            "mean": float(c_arr.mean()),
        },
    }


def scan_yolo_labels(label_dir: Path, class_map: dict) -> dict:
    paths = list(label_dir.glob("*.txt"))
    instance_counts = Counter()
    annotated = 0

    for p in tqdm(paths, desc=f"  Scanning labels", unit="file"):
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if lines:
            annotated += 1
        for line in lines:
            parts = line.split()
            if parts:
                cls_id = int(parts[0])
                instance_counts[class_map.get(cls_id, f"class_{cls_id}")] += 1

    return {
        "format": "YOLO TXT",
        "annotated_images": annotated,
        "total_instances": sum(instance_counts.values()),
        "class_distribution": dict(instance_counts.most_common()),
    }


def scan_coco_json(annotation_dir: Path) -> dict:
    splits = {}

    for json_path in sorted(annotation_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)

        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        instance_counts = Counter()
        annotated_ids = set()

        for ann in tqdm(data.get("annotations", []), desc=f"  Scanning {json_path.stem}", unit="ann"):
            cls_name = categories.get(ann["category_id"], f"class_{ann['category_id']}")
            instance_counts[cls_name] += 1
            annotated_ids.add(ann["image_id"])

        splits[json_path.stem] = {
            "total_images": len(data.get("images", [])),
            "annotated_images": len(annotated_ids),
            "total_instances": sum(instance_counts.values()),
            "class_distribution": dict(instance_counts.most_common()),
        }

    return {"format": "COCO JSON", "splits": splits}


def scan_splits_yolo(split_dir: Path) -> dict:
    result = {}
    for f in sorted(split_dir.glob("*.txt")):
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        result[f.stem] = len(lines)
    return result


def scan_splits_txt(train_path: Path, val_path: Path) -> dict:
    result = {}
    for label, path in [("train", train_path), ("val", val_path)]:
        if path.exists():
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            result[label] = len(lines)
    return result


def divider(char="=", width=65):
    print(char * width)


def print_dim_row(label, stat):
    print(f"    {label:<10} min={stat['min']:<6} max={stat['max']:<6} mean={stat['mean']:.1f}  std={stat['std']:.1f}")


def print_report(name: str, image_stats: dict, annotation_stats: dict, split_info: dict):
    divider()
    print(f"  DATASET : {name}")
    divider()

    print("\n  [MODALITY & DIMENSIONS]")
    print(f"    Image count : {image_stats['count']}")
    print(f"    Modalities  : {image_stats['modalities']}")
    print_dim_row("Height", image_stats["height"])
    print_dim_row("Width", image_stats["width"])
    print(f"    Channels    : min={image_stats['channels']['min']}  max={image_stats['channels']['max']}  mean={image_stats['channels']['mean']:.1f}")

    print("\n  [ANNOTATIONS]")
    fmt = annotation_stats.get("format")
    print(f"    Format : {fmt}")

    if fmt == "YOLO TXT":
        print(f"    Annotated images : {annotation_stats['annotated_images']}")
        print(f"    Total instances  : {annotation_stats['total_instances']}")
        print(f"\n    {'Class':<25} {'Count':>7}")
        print(f"    {'-'*25} {'-'*7}")
        for cls, cnt in annotation_stats["class_distribution"].items():
            print(f"    {cls:<25} {cnt:>7}")

    elif fmt == "COCO JSON":
        for split_name, info in annotation_stats["splits"].items():
            print(f"\n    Split: {split_name}")
            print(f"      Total images     : {info['total_images']}")
            print(f"      Annotated images : {info['annotated_images']}")
            print(f"      Total instances  : {info['total_instances']}")
            print(f"\n      {'Class':<25} {'Count':>7}")
            print(f"      {'-'*25} {'-'*7}")
            for cls, cnt in info["class_distribution"].items():
                print(f"      {cls:<25} {cnt:>7}")

    if split_info:
        print(f"\n  [SPLITS]")
        for split_name, count in split_info.items():
            print(f"    {split_name:<10} : {count} images")

    print()


def main():
    divider("*")
    print("  PPE DATASET ANALYSIS")
    divider("*")

    print("\n[CHV]")
    chv_images = scan_images(PATHS["CHV"]["images"], "CHV")
    chv_labels = scan_yolo_labels(PATHS["CHV"]["labels"], CHV_CLASSES)
    chv_splits = scan_splits_yolo(PATHS["CHV"]["splits"])
    print_report("CHV", chv_images, chv_labels, chv_splits)

    print("\n[CPPE-5]")
    cppe_images = scan_images(PATHS["CPPE-5"]["images"], "CPPE-5")
    cppe_annotations = scan_coco_json(PATHS["CPPE-5"]["annotations"])
    print_report("CPPE-5", cppe_images, cppe_annotations, {})

    print("\n[SH17]")
    sh17_images = scan_images(PATHS["SH17"]["images"], "SH17")
    sh17_labels = scan_yolo_labels(PATHS["SH17"]["labels"], SH17_CLASSES)
    sh17_splits = scan_splits_txt(PATHS["SH17"]["split_train"], PATHS["SH17"]["split_val"])
    print_report("SH17", sh17_images, sh17_labels, sh17_splits)

    divider("*")
    print("  CROSS-DATASET SUMMARY")
    divider("*")
    print(f"\n  {'Dataset':<10} {'Images':>8} {'Instances':>10} {'Classes':>8} {'Format':<12}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")

    chv_cls = len(chv_labels["class_distribution"])
    cppe_cls = sum(len(s["class_distribution"]) for s in cppe_annotations["splits"].values())
    cppe_inst = sum(s["total_instances"] for s in cppe_annotations["splits"].values())
    sh17_cls = len(sh17_labels["class_distribution"])

    print(f"  {'CHV':<10} {chv_images['count']:>8} {chv_labels['total_instances']:>10} {chv_cls:>8} {'YOLO TXT':<12}")
    print(f"  {'CPPE-5':<10} {cppe_images['count']:>8} {cppe_inst:>10} {cppe_cls:>8} {'COCO JSON':<12}")
    print(f"  {'SH17':<10} {sh17_images['count']:>8} {sh17_labels['total_instances']:>10} {sh17_cls:>8} {'YOLO TXT':<12}")
    print()


if __name__ == "__main__":
    main()
