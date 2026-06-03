import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compliance.pipeline import PACTPipeline

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
img_bgr = cv2.imread(str(ROOT / "dataset/CHV/ppe_0021.jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

pipe = PACTPipeline(
    det_weights  = str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"),
    pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
    dataset="chv", rule="chv", device="0",
)

# check vest at lower conf
from ultralytics import YOLO
det = YOLO(str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"))
res = det(img_rgb, conf=0.10, iou=0.45, imgsz=640, device="0", verbose=False)[0]
print("=== ALL DETECTIONS conf>=0.10 ===")
for box in res.boxes:
    name = res.names[int(box.cls[0])]
    print(f"  {name:15s} conf={float(box.conf[0]):.3f}  bbox={list(map(int,box.xyxy[0].tolist()))}")

print("\n=== PACT RESULT AFTER FIX ===")
frame = pipe.run(img_rgb)
for p in frame.persons:
    print(f"Person {p.person_id}: compliant={p.is_compliant}  worn={p.worn}  missing={p.missing}")
