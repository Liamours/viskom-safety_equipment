"""
Resize SH17 images so the longest side = TARGET_SIZE, preserving aspect ratio.
Labels are copied unchanged (YOLO format uses normalized coords — scale-invariant).

Output:
    dataset/SH17/raw_640/images/  <- resized images
    dataset/SH17/raw_640/labels/  <- copied labels (identical to raw/labels)
"""

import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 640
SRC_IMAGES  = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset/SH17/raw/images")
SRC_LABELS  = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset/SH17/raw/labels")
DST_IMAGES  = SRC_IMAGES.parent.parent / "raw_640" / "images"
DST_LABELS  = SRC_IMAGES.parent.parent / "raw_640" / "labels"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_image(src: Path, dst: Path) -> tuple[int, int]:
    with Image.open(src) as img:
        w, h     = img.size
        scale    = TARGET_SIZE / max(w, h)
        new_w    = int(w * scale)
        new_h    = int(h * scale)
        resized  = img.resize((new_w, new_h), Image.LANCZOS)
        resized.save(dst)
    return (w, h)


def main():
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    images = [p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS]
    total  = len(images)
    print(f"Found {total} images in {SRC_IMAGES}")
    print(f"Resizing longest side → {TARGET_SIZE}px  (aspect ratio preserved)")
    print(f"Output: {DST_IMAGES}\n")

    skipped = 0
    for src_img in tqdm(sorted(images), desc="Resizing SH17", unit="img"):
        dst_img = DST_IMAGES / src_img.name
        resize_image(src_img, dst_img)

        label_src = SRC_LABELS / (src_img.stem + ".txt")
        if label_src.exists():
            shutil.copy2(label_src, DST_LABELS / label_src.name)
        else:
            skipped += 1

    print(f"\nDone. {total} images resized, {skipped} missing labels.")
    print(f"Images : {DST_IMAGES}")
    print(f"Labels : {DST_LABELS}")


if __name__ == "__main__":
    main()
