from pathlib import Path

import numpy as np
from PIL import Image

YOLO_SIZE = 640
MEDIAPIPE_MAX_SIDE = 1280
LETTERBOX_FILL = (114, 114, 114)
NON_RGB_MODES = {"CMYK", "RGBA", "P", "L", "LA", "PA", "1", "I", "F"}


def is_rgb_compatible(img: Image.Image) -> bool:
    return img.mode == "RGB"


def load_as_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def letterbox(
    img: Image.Image,
    target: int = YOLO_SIZE,
    fill: tuple = LETTERBOX_FILL,
) -> Image.Image:
    w, h = img.size
    scale = target / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), fill)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas.paste(img, (pad_x, pad_y))
    return canvas, (pad_x, pad_y, scale)


def resize_fit(
    img: Image.Image,
    target: int = YOLO_SIZE,
) -> Image.Image:
    return img.resize((target, target), Image.BILINEAR)


def resize_longest_side(
    img: Image.Image,
    max_side: int = MEDIAPIPE_MAX_SIDE,
) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def to_numpy_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def prepare_for_yolo(path: Path, target: int = YOLO_SIZE) -> tuple:
    img = load_as_rgb(path)
    img_letterboxed, padding_meta = letterbox(img, target)
    return to_numpy_rgb(img_letterboxed), padding_meta


def prepare_for_mediapipe(path: Path, max_side: int = MEDIAPIPE_MAX_SIDE) -> np.ndarray:
    img = load_as_rgb(path)
    img = resize_longest_side(img, max_side)
    return to_numpy_rgb(img)
