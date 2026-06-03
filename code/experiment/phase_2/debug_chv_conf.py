import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
det = YOLO(str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"))
img = cv2.cvtColor(cv2.imread(str(ROOT / "dataset/CHV/ppe_0021.jpg")), cv2.COLOR_BGR2RGB)

res = det(img, conf=0.01, iou=0.45, imgsz=640, device="0", verbose=False)[0]
print("ALL detections conf>=0.01:")
for box in res.boxes:
    name = res.names[int(box.cls[0])]
    print(f"  {name}  conf={float(box.conf[0]):.3f}  bbox={list(map(int,box.xyxy[0].tolist()))}")
