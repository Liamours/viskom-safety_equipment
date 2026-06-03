import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmarks.base import COCO_PPE_ANCHOR_GROUPS, LandmarkResult, Landmark
from landmarks.core import build_compliance_map, get_ppe_anchors, compute_iou
from compliance.rules import COMPLIANCE_RULES, DEFAULT_RULE

Bbox  = Tuple[int, int, int, int]
CONF  = 0.25
IOU   = 0.45

# COCO 17-keypoint indices used for arm-based hand anchor estimation
_COCO_ARM = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}


def _arm_hand_anchor(
    lm: LandmarkResult,
    person_bbox: Bbox,
    side: str,
    vis_thresh: float = 0.3,
) -> Bbox:
    idx   = _COCO_ARM[side]
    lms   = lm.landmarks
    iw, ih = lm.image_width, lm.image_height
    px1, py1, px2, py2 = person_bbox

    def kp(i) -> Optional[Tuple[int, int]]:
        if i >= len(lms) or lms[i].visibility < vis_thresh:
            return None
        return int(lms[i].x * iw), int(lms[i].y * ih)

    shoulder = kp(idx["shoulder"])
    elbow    = kp(idx["elbow"])

    if shoulder and elbow:
        # extrapolate: shoulder → elbow → estimated wrist
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        cx = elbow[0] + dx
        cy = elbow[1] + dy
        r  = max(int((dx**2 + dy**2) ** 0.5 * 0.6), 20)
    elif elbow:
        cx, cy = elbow
        r = int((py2 - py1) * 0.15)
    elif shoulder:
        # no elbow: hand region = lower half of person from shoulder level
        return (px1, shoulder[1], px2, py2)
    else:
        # nothing visible: full person bbox
        return (px1, py1, px2, py2)

    return (
        max(px1, cx - r),
        max(py1, cy - r),
        min(px2, cx + r),
        min(py2, cy + r),
    )
IMGSZ = 640

# Per-dataset class name maps: model output name -> semantic name used in PPE_CLASS_TO_ANCHOR
DATASET_CLASS_MAPS: Dict[str, Dict[str, str]] = {
    "chv": {
        "class_0": "person",
        "class_1": "vest",
        "class_2": "helmet",
        "class_3": "helmet",
        "class_4": "helmet",
        "class_5": "helmet",
    },
    "cppe5": {
        "Coverall":   "coverall",
        "Face_Shield": "face_shield",
        "Gloves":     "gloves",
        "Goggles":    "goggles",
        "Mask":       "mask",
    },
    "sh17": {},  # names already match
}


class PersonResult:
    def __init__(
        self,
        person_id:       int,
        person_bbox:     Bbox,
        landmark_result: Optional[LandmarkResult],
        ppe_detections:  List[Dict],
        compliance_map:  Dict,
        required_ppe:    List[str],
    ):
        self.person_id       = person_id
        self.person_bbox     = person_bbox
        self.landmark_result = landmark_result
        self.ppe_detections  = ppe_detections
        self.compliance_map  = compliance_map
        self.required_ppe    = required_ppe

    @property
    def worn(self) -> List[str]:
        return [cls for cls, status in self.compliance_map.items() if status is not None]

    @property
    def missing(self) -> List[str]:
        return [cls for cls in self.required_ppe if cls not in self.worn]

    @property
    def is_compliant(self) -> bool:
        return len(self.missing) == 0

    @property
    def compliance_score(self) -> float:
        if not self.required_ppe:
            return 1.0
        worn_required = [p for p in self.required_ppe if p in self.worn]
        return round(len(worn_required) / len(self.required_ppe), 4)


class FrameResult:
    def __init__(self, image_path: str, persons: List[PersonResult]):
        self.image_path = image_path
        self.persons    = persons

    @property
    def num_persons(self) -> int:
        return len(self.persons)

    @property
    def num_compliant(self) -> int:
        return sum(1 for p in self.persons if p.is_compliant)

    @property
    def compliance_rate(self) -> float:
        if not self.persons:
            return 0.0
        return round(sum(p.compliance_score for p in self.persons) / len(self.persons), 4)

    def to_dict(self) -> dict:
        return {
            "image":           self.image_path,
            "num_persons":     self.num_persons,
            "num_compliant":   self.num_compliant,
            "compliance_rate": self.compliance_rate,
            "persons": [
                {
                    "person_id":        p.person_id,
                    "compliant":        p.is_compliant,
                    "compliance_score": p.compliance_score,
                    "worn":             p.worn,
                    "missing":          p.missing,
                    "person_bbox":      list(p.person_bbox),
                }
                for p in self.persons
            ],
        }


class PACTPipeline:
    """
    PACT — Pose-Anchored Compliance Tracker

    Uses a pose model to detect persons and extract body keypoints,
    and a detection model to find PPE items. PPE is assigned to the
    person it overlaps most, then validated against anatomical anchor
    regions derived from keypoints.
    """

    def __init__(
        self,
        det_weights:   str,
        pose_weights:  str   = "models/yolov8x-pose.pt",
        dataset:       str   = "chv",
        rule:          str   = DEFAULT_RULE,
        conf:          float = CONF,
        iou:           float = IOU,
        imgsz:         int   = IMGSZ,
        device:        str   = "0",
        anchor_margin: float = 0.2,
        iou_threshold: float = 0.05,
    ):
        from ultralytics import YOLO

        self.det_model     = YOLO(det_weights)
        self.pose_model    = YOLO(pose_weights)
        self.class_map     = DATASET_CLASS_MAPS.get(dataset, {})
        self.required_ppe  = COMPLIANCE_RULES.get(rule, COMPLIANCE_RULES[DEFAULT_RULE])
        self.conf          = conf
        self.iou           = iou
        self.imgsz         = imgsz
        self.device        = device
        self.anchor_margin = anchor_margin
        self.iou_threshold = iou_threshold

    def _normalize_class(self, name: str) -> str:
        return self.class_map.get(name, name.lower())

    def _detect_ppe(self, image: np.ndarray) -> List[Dict]:
        result = self.det_model(
            image, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]

        ppe = []
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            norm     = self._normalize_class(cls_name)
            if norm == "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            ppe.append({
                "class_name": norm,
                "confidence": float(box.conf[0]),
                "bbox":       (x1, y1, x2, y2),
            })
        return ppe

    def _detect_persons(self, image: np.ndarray) -> List[Tuple[Bbox, LandmarkResult]]:
        h, w = image.shape[:2]
        result = self.pose_model(
            image, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]

        persons = []

        if result.boxes is None or result.keypoints is None:
            return persons

        kps_xy   = result.keypoints.xy.cpu().numpy()
        kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        for idx, box in enumerate(result.boxes):
            # Skip low-confidence person detections (spurious / background)
            if float(box.conf[0]) < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_bbox = (x1, y1, x2, y2)

            landmarks = []
            for i, (px, py) in enumerate(kps_xy[idx]):
                vis = float(kps_conf[idx][i]) if kps_conf is not None else 1.0
                landmarks.append(Landmark(
                    x=float(px) / w,
                    y=float(py) / h,
                    z=0.0,
                    visibility=vis,
                ))

            lm_result = LandmarkResult(landmarks=landmarks, image_width=w, image_height=h)
            persons.append((person_bbox, lm_result))

        return persons

    def _assign_ppe(self, ppe_list: List[Dict], person_bbox: Bbox, all_bboxes: List[Bbox]) -> List[Dict]:
        # Fix #1: removed `or scores[best_idx] == 0.0` — floating PPE (no IoU with
        # any person) was being assigned to EVERY person. Now only assign if this
        # person is the best IoU match AND that score is > 0.
        assigned = []
        for ppe in ppe_list:
            scores = [compute_iou(ppe["bbox"], pb) for pb in all_bboxes]
            if not scores:
                continue
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 0 and all_bboxes[best_idx] == person_bbox:
                assigned.append(ppe)
        return assigned

    def run(self, image: np.ndarray, image_path: str = "") -> FrameResult:
        h, w = image.shape[:2]

        ppe_detections = self._detect_ppe(image)
        persons        = self._detect_persons(image)
        all_bboxes     = [p[0] for p in persons]

        person_results = []

        for idx, (person_bbox, lm_result) in enumerate(persons):
            assigned_ppe = self._assign_ppe(ppe_detections, person_bbox, all_bboxes)

            # Fix #4: lower visibility_threshold to 0.3 so partially-visible
            # keypoints (e.g. wrists under gloves, face under mask) still produce
            # anchors. Default 0.5 was silently dropping valid CPPE-5 wrist kps.
            anchors = get_ppe_anchors(
                lm_result,
                anchor_groups=COCO_PPE_ANCHOR_GROUPS,
                margin=self.anchor_margin,
                visibility_threshold=0.3,
            )

            # Helmets sit above face keypoints — extend helmet_region upward
            # to the person bbox top so the anchor covers the full head+helmet.
            if anchors.get("helmet_region") is not None:
                ax1, ay1, ax2, ay2 = anchors["helmet_region"]["bbox"]
                anchors["helmet_region"]["bbox"] = (ax1, person_bbox[1], ax2, ay2)
            else:
                # Fallback: head keypoints undetected (e.g. worker bent over / facing away).
                # Use top 35% of person bbox as the helmet anchor region.
                px1, py1, px2, py2 = person_bbox
                anchors["helmet_region"] = {
                    "bbox":     (px1, py1, px2, py1 + int((py2 - py1) * 0.35)),
                    "raw_bbox": (px1, py1, px2, py1 + int((py2 - py1) * 0.35)),
                    "keypoints": [],
                }

            # Hand fallback: wrist keypoint invisible → estimate hand region
            # from shoulder→elbow direction (COCO kps 5/6=shoulder, 7/8=elbow).
            px1, py1, px2, py2 = person_bbox
            if anchors.get("left_hand") is None:
                bb = _arm_hand_anchor(lm_result, person_bbox, "left")
                anchors["left_hand"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}
            if anchors.get("right_hand") is None:
                bb = _arm_hand_anchor(lm_result, person_bbox, "right")
                anchors["right_hand"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

            # Foot fallback: ankle/heel invisible → use lower person bbox below knee.
            # COCO: left_knee=13, right_knee=14, left_ankle=15, right_ankle=16.
            def _foot_anchor(knee_i: int) -> Bbox:
                lms = lm_result.landmarks
                iw, ih = lm_result.image_width, lm_result.image_height
                if knee_i < len(lms) and lms[knee_i].visibility >= 0.3:
                    knee_y = int(lms[knee_i].y * ih)
                    return (px1, knee_y, px2, py2)
                return (px1, py1 + int((py2 - py1) * 0.6), px2, py2)

            if anchors.get("left_foot") is None:
                bb = _foot_anchor(13)
                anchors["left_foot"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}
            if anchors.get("right_foot") is None:
                bb = _foot_anchor(14)
                anchors["right_foot"] = {"bbox": bb, "raw_bbox": bb, "keypoints": []}

            # Fix #8/#13: anchors are already pre-expanded by get_ppe_anchors.
            # Pass anchor_margin=0.0 so associate_to_anchor reads raw_bbox and
            # expands once instead of double-expanding.
            compliance_map = build_compliance_map(
                anchors       = anchors,
                detections    = assigned_ppe,
                iou_threshold = self.iou_threshold,
                anchor_margin = 0.0,
                image_width   = w,
                image_height  = h,
            )

            person_results.append(PersonResult(
                person_id       = idx,
                person_bbox     = person_bbox,
                landmark_result = lm_result,
                ppe_detections  = assigned_ppe,
                compliance_map  = compliance_map,
                required_ppe    = self.required_ppe,
            ))

        return FrameResult(image_path=image_path, persons=person_results)
