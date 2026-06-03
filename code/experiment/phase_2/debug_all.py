import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO
import numpy as np

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")

configs = [
    ("CHV",   "dataset/CHV/ppe_0021.jpg",                "runs/detect/phase1_chv_original/weights/best.pt"),
    ("CPPE5", "dataset/CPPE-5/130.png",                  "runs/detect/phase1_cppe_5_original/weights/best.pt"),
    ("SH17",  "dataset/SH17/pexels-photo-18110372.jpeg", "runs/detect/phase1_sh17_original/weights/best.pt"),
]

pose_model = YOLO(str(ROOT / "models/yolov8x-pose.pt"))

for tag, img_rel, det_rel in configs:
    print(f"\n{'='*55}")
    print(f"[{tag}]")
    img_bgr = cv2.imread(str(ROOT / img_rel))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"  Image: {w}x{h}")

    det_model = YOLO(str(ROOT / det_rel))
    for conf in [0.25, 0.15, 0.10]:
        res = det_model(img_rgb, conf=conf, iou=0.45, imgsz=640, device="0", verbose=False)[0]
        items = [(res.names[int(b.cls[0])], f"{float(b.conf[0]):.2f}", list(map(int, b.xyxy[0].tolist()))) for b in res.boxes]
        print(f"  conf>={conf}: {[(n,c) for n,c,_ in items]}")
        if conf == 0.10:
            for name, conf_v, bbox in items:
                print(f"    {name:20s} {conf_v}  {bbox}")

    pose_res = pose_model(img_rgb, conf=0.25, iou=0.45, imgsz=640, device="0", verbose=False)[0]
    print(f"  Persons (pose): {len(pose_res.boxes)}")
    for i, box in enumerate(pose_res.boxes):
        bbox = list(map(int, box.xyxy[0].tolist()))
        print(f"    P{i}: {bbox}  conf={float(box.conf[0]):.2f}")
