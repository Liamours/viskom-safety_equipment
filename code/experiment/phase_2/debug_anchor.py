import cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compliance.pipeline import PACTPipeline, DATASET_CLASS_MAPS
from landmarks.base import COCO_PPE_ANCHOR_GROUPS
from landmarks.core import get_ppe_anchors, compute_iou, build_compliance_map
from ultralytics import YOLO
import numpy as np

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")

# ---- SH17 ----
print("=== SH17 anchor debug ===")
img_bgr = cv2.imread(str(ROOT / "dataset/SH17/pexels-photo-18110372.jpeg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

det_model  = YOLO(str(ROOT / "runs/detect/phase1_sh17_original/weights/best.pt"))
pose_model = YOLO(str(ROOT / "models/yolov8x-pose.pt"))

det_res  = det_model(img_rgb,  conf=0.25, iou=0.45, imgsz=640, device="0", verbose=False)[0]
pose_res = pose_model(img_rgb, conf=0.25, iou=0.45, imgsz=640, device="0", verbose=False)[0]

kps_xy   = pose_res.keypoints.xy.cpu().numpy()
kps_conf = pose_res.keypoints.conf.cpu().numpy()

class_map = DATASET_CLASS_MAPS["sh17"]

ppe_list = []
for box in det_res.boxes:
    name = det_res.names[int(box.cls[0])]
    norm = class_map.get(name, name.lower())
    if norm == "person": continue
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    ppe_list.append({"class_name": norm, "confidence": float(box.conf[0]), "bbox": (x1,y1,x2,y2)})

persons = []
for idx, box in enumerate(pose_res.boxes):
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    from landmarks.base import LandmarkResult, Landmark
    landmarks = []
    for i, (px, py) in enumerate(kps_xy[idx]):
        vis = float(kps_conf[idx][i])
        landmarks.append(Landmark(x=float(px)/w, y=float(py)/h, z=0.0, visibility=vis))
    lm = LandmarkResult(landmarks=landmarks, image_width=w, image_height=h)
    persons.append(((x1,y1,x2,y2), lm))

all_bboxes = [p[0] for p in persons]

for pidx, (person_bbox, lm_result) in enumerate(persons):
    print(f"\nPerson {pidx}: bbox={person_bbox}")

    # keypoint visibility for head kps (0-4: nose,l_eye,r_eye,l_ear,r_ear)
    for ki in range(5):
        lm = lm_result.landmarks[ki]
        print(f"  kp{ki}: x={lm.x*w:.0f} y={lm.y*h:.0f} vis={lm.visibility:.2f}")

    anchors = get_ppe_anchors(lm_result, anchor_groups=COCO_PPE_ANCHOR_GROUPS, margin=0.2)

    if anchors.get("helmet_region") is not None:
        ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
        print(f"  helmet_region BEFORE fix: ({ax1},{ay1},{ax2},{ay2})")
        anchors["helmet_region"]["bbox"] = (ax1, person_bbox[1], ax2, ay2)
        ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
        print(f"  helmet_region AFTER  fix: ({ax1},{ay1},{ax2},{ay2})")
    else:
        print("  helmet_region: None")

    # assigned PPE
    assigned = []
    for ppe in ppe_list:
        scores = [compute_iou(ppe["bbox"], pb) for pb in all_bboxes]
        best_idx = int(np.argmax(scores))
        if all_bboxes[best_idx] == person_bbox or scores[best_idx] == 0.0:
            assigned.append(ppe)
            print(f"  PPE assigned: {ppe['class_name']} {ppe['bbox']} iou_scores={[f'{s:.3f}' for s in scores]}")

    cmap = build_compliance_map(anchors=anchors, detections=assigned,
                                iou_threshold=0.05, anchor_margin=0.2,
                                image_width=w, image_height=h)
    print(f"  compliance_map: {cmap}")
