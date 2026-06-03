from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

from .image import NON_RGB_MODES, YOLO_SIZE, MEDIAPIPE_MAX_SIDE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

BASE = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")

DATASET_IMAGE_DIRS: Dict[str, Path] = {
    "CHV":    BASE / "CHV/raw/images",
    "CPPE-5": BASE / "CPPE-5/raw/images",
    "SH17":   BASE / "SH17/raw/images",
}


def _collect_images(image_dir: Path) -> List[Path]:
    return [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def check_modalities(image_dir: Path, dataset_name: str) -> Dict:
    paths = _collect_images(image_dir)
    mode_counts = Counter()
    non_rgb: List[Tuple[str, str]] = []
    errors: List[Tuple[str, str]] = []

    for p in tqdm(paths, desc=f"  [{dataset_name}] Modality check", unit="img"):
        try:
            img = Image.open(p)
            mode_counts[img.mode] += 1
            if img.mode != "RGB":
                non_rgb.append((p.name, img.mode))
        except Exception as e:
            errors.append((p.name, str(e)))

    return {
        "total": len(paths),
        "mode_counts": dict(mode_counts),
        "requires_conversion": len(non_rgb),
        "non_rgb_files": non_rgb,
        "load_errors": errors,
        "rgb_compatible": len(non_rgb) == 0 and len(errors) == 0,
    }


def check_dimensions(
    image_dir: Path,
    dataset_name: str,
    yolo_size: int = YOLO_SIZE,
    mediapipe_max: int = MEDIAPIPE_MAX_SIDE,
) -> Dict:
    paths = _collect_images(image_dir)
    needs_yolo_resize: List[Tuple[str, int, int]] = []
    needs_mp_resize: List[Tuple[str, int, int]] = []
    widths: List[int] = []
    heights: List[int] = []

    for p in tqdm(paths, desc=f"  [{dataset_name}] Dimension check", unit="img"):
        try:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            if w != yolo_size or h != yolo_size:
                needs_yolo_resize.append((p.name, w, h))
            if max(w, h) > mediapipe_max:
                needs_mp_resize.append((p.name, w, h))
        except Exception:
            continue

    return {
        "total": len(widths),
        "yolo_target": yolo_size,
        "mediapipe_max": mediapipe_max,
        "needs_yolo_resize": len(needs_yolo_resize),
        "needs_mp_resize": len(needs_mp_resize),
        "size_range": {
            "width":  (min(widths), max(widths)),
            "height": (min(heights), max(heights)),
        },
        "yolo_compatible":       len(needs_yolo_resize) == 0,
        "mediapipe_compatible":  len(needs_mp_resize) == 0,
        "flagged_yolo":          needs_yolo_resize,
        "flagged_mediapipe":     needs_mp_resize,
    }


def run_checks(
    datasets: Dict[str, Path] = None,
    yolo_size: int = YOLO_SIZE,
    mediapipe_max: int = MEDIAPIPE_MAX_SIDE,
) -> Dict:
    if datasets is None:
        datasets = DATASET_IMAGE_DIRS

    results = {}
    for name, image_dir in datasets.items():
        print(f"\n[{name}]")
        results[name] = {
            "modality":   check_modalities(image_dir, name),
            "dimensions": check_dimensions(image_dir, name, yolo_size, mediapipe_max),
        }
    return results


def print_check_report(results: Dict):
    divider = "=" * 65

    for name, data in results.items():
        print(f"\n{divider}")
        print(f"  DATASET : {name}")
        print(divider)

        m = data["modality"]
        print(f"\n  [MODALITY]")
        print(f"    Total images       : {m['total']}")
        print(f"    Mode distribution  : {m['mode_counts']}")
        print(f"    Needs conversion   : {m['requires_conversion']}")
        print(f"    Load errors        : {len(m['load_errors'])}")
        print(f"    RGB compatible     : {'YES' if m['rgb_compatible'] else 'NO  <-- action required'}")

        if m["non_rgb_files"]:
            print(f"\n    Non-RGB files (first 10):")
            for fname, mode in m["non_rgb_files"][:10]:
                print(f"      {fname:<55} [{mode}]")
            if len(m["non_rgb_files"]) > 10:
                print(f"      ... and {len(m['non_rgb_files']) - 10} more")

        if m["load_errors"]:
            print(f"\n    Load errors (first 5):")
            for fname, err in m["load_errors"][:5]:
                print(f"      {fname:<55} {err}")

        d = data["dimensions"]
        print(f"\n  [DIMENSIONS]")
        print(f"    Total images       : {d['total']}")
        print(f"    Width range        : {d['size_range']['width'][0]} – {d['size_range']['width'][1]} px")
        print(f"    Height range       : {d['size_range']['height'][0]} – {d['size_range']['height'][1]} px")
        print(f"    Needs YOLO resize  : {d['needs_yolo_resize']} / {d['total']}  (target: {d['yolo_target']}×{d['yolo_target']}px, letterbox)")
        print(f"    Needs MP resize    : {d['needs_mp_resize']} / {d['total']}  (max side: {d['mediapipe_max']}px)")
        print(f"    YOLO compatible    : {'YES' if d['yolo_compatible'] else 'NO  <-- action required'}")
        print(f"    MediaPipe compat.  : {'YES' if d['mediapipe_compatible'] else 'NO  <-- action required'}")

    print(f"\n{divider}")
    print(f"  COMPATIBILITY SUMMARY")
    print(divider)
    print(f"\n  {'Dataset':<10} {'RGB':>8} {'YOLO-ready':>12} {'MP-ready':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
    for name, data in results.items():
        rgb_ok = "YES" if data["modality"]["rgb_compatible"] else "NO"
        yolo_ok = "YES" if data["dimensions"]["yolo_compatible"] else "NO"
        mp_ok = "YES" if data["dimensions"]["mediapipe_compatible"] else "NO"
        print(f"  {name:<10} {rgb_ok:>8} {yolo_ok:>12} {mp_ok:>10}")
    print()


def main():
    results = run_checks()
    print_check_report(results)


if __name__ == "__main__":
    main()
