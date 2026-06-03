import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
det  = YOLO(str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"))
pose = YOLO(str(ROOT / "models/yolov8x-pose.pt"))

img_dir = ROOT / "landmarks/CHV/detected-mediapipe"
imgs = sorted(img_dir.glob("*.jpg"))

for img_path in imgs:
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    res = det(img, conf=0.25, iou=0.45, imgsz=640, device="0", verbose=False)[0]
    names = [res.names[int(b.cls[0])] for b in res.boxes]
    n_persons  = sum(1 for n in names if n == "class_0")
    n_vest     = sum(1 for n in names if n == "class_1")
    n_helmet   = sum(1 for n in names if n in ("class_2","class_3","class_4","class_5"))
    # want variety: some vest, some helmet, some person without
    if n_persons >= 2 and n_vest >= 1 and n_helmet >= 1 and n_helmet < n_persons:
        print(f"VARIED: {img_path.name}  persons={n_persons} vests={n_vest} helmets={n_helmet}")
    elif n_persons >= 3 and n_vest >= 1 and n_helmet >= 1:
        print(f"MULTI:  {img_path.name}  persons={n_persons} vests={n_vest} helmets={n_helmet}")
