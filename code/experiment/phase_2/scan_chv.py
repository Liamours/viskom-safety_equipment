import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
det = YOLO(str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"))
pose = YOLO(str(ROOT / "models/yolov8x-pose.pt"))

# class_1=vest, class_2/3/4/5=helmet
img_dir = ROOT / "landmarks/CHV/detected-mediapipe"
imgs = sorted(img_dir.glob("*.jpg"))

for img_path in imgs[:50]:
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    res = det(img, conf=0.25, iou=0.45, imgsz=640, device="0", verbose=False)[0]
    names = [res.names[int(b.cls[0])] for b in res.boxes]
    has_vest   = any(n == "class_1" for n in names)
    has_helmet = any(n in ("class_2","class_3","class_4","class_5") for n in names)
    n_persons  = sum(1 for n in names if n == "class_0")
    # want: vest + helmet + multiple non-person detections
    if has_vest and has_helmet and n_persons >= 2:
        print(f"GOOD: {img_path.name}  persons={n_persons}  {names}")
    elif has_vest:
        print(f"vest: {img_path.name}  {names}")
