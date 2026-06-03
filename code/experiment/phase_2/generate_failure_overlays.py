import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
OUT_DIR = ROOT / "conference/draft/figures/failure_cases"
FONT_PATH = Path("C:/Windows/Fonts/times.ttf")
FONT_LABEL = ImageFont.truetype(str(FONT_PATH), 17)
FONT_TITLE = ImageFont.truetype(str(FONT_PATH), 28)

SAMPLES = [
    {
        "title": "SH17: missed vest",
        "out": "wrongcase_sh17.png",
        "image": ROOT / "dataset/SH17/raw_640/images/pexels-photo-15200454.jpeg",
        "label": ROOT / "dataset/SH17/raw_640/labels/pexels-photo-15200454.txt",
        "weights": ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "names": {
            0: "person", 1: "ear", 2: "earmuffs", 3: "face", 4: "face-guard",
            5: "face-mask-medical", 6: "foot", 7: "tools", 8: "glasses",
            9: "gloves", 10: "helmet", 11: "hands", 12: "head",
            13: "medical-suit", 14: "shoes", 15: "safety-suit", 16: "safety-vest",
        },
    },
    {
        "title": "CPPE-5: false coverall/mask",
        "out": "wrongcase_cppe5.png",
        "image": ROOT / "dataset/CPPE-5/raw/images/1017.png",
        "label": ROOT / "dataset/CPPE-5/raw/labels/1017.txt",
        "weights": ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "names": {0: "coverall", 1: "face_shield", 2: "gloves", 3: "goggles", 4: "mask"},
    },
    {
        "title": "CHV: missed vest",
        "out": "wrongcase_chv.png",
        "image": ROOT / "dataset/CHV/raw/images/ppe_1044.jpg",
        "label": ROOT / "dataset/CHV/raw/labels/ppe_1044.txt",
        "weights": ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "names": {0: "person", 1: "vest", 2: "helmet", 3: "helmet", 4: "helmet", 5: "helmet"},
    },
]


def yolo_box(row, width, height):
    cls, cx, cy, bw, bh = row
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return int(cls), (x1, y1, x2, y2)


def read_gt(path, width, height):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        boxes.append(yolo_box([float(v) for v in parts[:5]], width, height))
    return boxes


def predict(model, image):
    result = model(image, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
        boxes.append((cls, xyxy, conf))
    return boxes


def draw_text(image, text, xy, color, font):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=color, font=font)
    image[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_label(image, text, x, y, color):
    x = max(0, x)
    y = max(18, y)
    pad = 3
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=FONT_LABEL)
    width = box[2] - box[0] + pad * 2
    height = box[3] - box[1] + pad * 2
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - height), (x + width, y), color, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    draw_text(image, text, (x + pad, y - height + 1), (255, 255, 255), FONT_LABEL)


def draw_boxes(image, boxes, names, color, prefix, with_conf=False):
    for item in boxes:
        cls, bbox = item[:2]
        conf = item[2] if len(item) > 2 else None
        x1, y1, x2, y2 = bbox
        name = names.get(cls, str(cls))
        text = f"{prefix} {name}" if not with_conf else f"{prefix} {name} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        draw_label(image, text, x1, max(20, y1), color)


def fit_panel(image, size=(700, 520)):
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    panel = np.full((target_h + 48, target_w, 3), 255, dtype=np.uint8)
    x = (target_w - nw) // 2
    y = 48 + (target_h - nh) // 2
    panel[y:y + nh, x:x + nw] = resized
    return panel


def render_sample(sample, model):
    image = cv2.imread(str(sample["image"]))
    if image is None:
        raise FileNotFoundError(sample["image"])
    height, width = image.shape[:2]
    gt_boxes = read_gt(sample["label"], width, height)
    pred_boxes = predict(model, image)
    draw_boxes(image, gt_boxes, sample["names"], (50, 135, 50), "GT")
    draw_boxes(image, pred_boxes, sample["names"], (80, 110, 190), "PRED", True)
    panel = fit_panel(image)
    draw_text(panel, sample["title"], (16, 12), (20, 20, 20), FONT_TITLE)
    return panel


def main():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        sys.exit("ultralytics is required")
    out_paths = []
    models = {}
    for sample in tqdm(SAMPLES, desc="failure-overlays", unit="img"):
        weights = sample["weights"]
        if weights not in models:
            models[weights] = YOLO(str(weights))
        panel = render_sample(sample, models[weights])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / sample["out"]
        cv2.imwrite(str(out_path), panel)
        out_paths.append(out_path)
    for path in out_paths:
        print(path)


if __name__ == "__main__":
    main()
