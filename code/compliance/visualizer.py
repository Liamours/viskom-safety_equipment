import cv2
import numpy as np

from landmarks.core import draw_pose_landmarks
from landmarks.base import COCO_CONNECTIONS
from compliance.pipeline import FrameResult

COMPLIANT_COLOR     = (0,   200,   0)
NON_COMPLIANT_COLOR = (0,     0, 200)
PPE_COLOR           = (255, 140,   0)


def draw_pact_result(
    image_rgb:    np.ndarray,
    frame_result: FrameResult,
) -> np.ndarray:
    canvas = image_rgb.copy()

    for person in frame_result.persons:
        color = COMPLIANT_COLOR if person.is_compliant else NON_COMPLIANT_COLOR

        # pose landmarks only
        if person.landmark_result is not None:
            canvas = draw_pose_landmarks(canvas, person.landmark_result,
                                         connections=COCO_CONNECTIONS)

        # PPE bounding boxes only
        for ppe in person.ppe_detections:
            bx1, by1, bx2, by2 = ppe["bbox"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), PPE_COLOR, 2)
            cv2.putText(
                canvas, f"{ppe['class_name']} {ppe['confidence']:.2f}",
                (bx1, max(by1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PPE_COLOR, 1, cv2.LINE_AA,
            )

        # compliance label above person box (no person box drawn)
        x1, y1 = person.person_bbox[0], person.person_bbox[1]
        label = "COMPLIANT" if person.is_compliant else f"MISSING: {', '.join(person.missing)}"
        cv2.putText(
            canvas, f"P{person.person_id}: {label}",
            (x1, max(y1 - 10, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    # summary top-left
    total     = frame_result.num_persons
    compliant = frame_result.num_compliant
    cv2.putText(
        canvas, f"Compliant: {compliant}/{total}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        COMPLIANT_COLOR if compliant == total else NON_COMPLIANT_COLOR,
        2, cv2.LINE_AA,
    )

    return canvas
