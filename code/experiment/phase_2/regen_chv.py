import cv2, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compliance.pipeline import PACTPipeline
from landmarks.core import draw_pose_landmarks
from landmarks.base import COCO_CONNECTIONS
from reporting.generator import generate_report_data

ROOT    = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
OUT_DIR = ROOT / "results/phase_2/components"
PPE_COLOR = (255, 140, 0)

img_bgr = cv2.imread(str(ROOT / "dataset/CHV/ppe_0387.jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

pipe = PACTPipeline(
    det_weights  = str(ROOT / "runs/detect/phase1_chv_original/weights/best.pt"),
    pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
    dataset="chv", rule="chv", device="0",
)
frame_result = pipe.run(img_rgb, image_path=str(ROOT / "dataset/CHV/ppe_0387.jpg"))
frame_dict   = frame_result.to_dict()
report_data  = generate_report_data(frame_dict, rule="chv", dataset="chv",
                                    image_path=str(ROOT / "dataset/CHV/ppe_0387.jpg"))

# landmarks panel
canvas = img_rgb.copy()
for person in frame_result.persons:
    if person.landmark_result:
        canvas = draw_pose_landmarks(canvas, person.landmark_result,
                                     landmark_color=(0, 255, 128),
                                     connection_color=(255, 255, 255),
                                     landmark_radius=5, connection_thickness=2,
                                     connections=COCO_CONNECTIONS)
cv2.imwrite(str(OUT_DIR / "chv_02_landmarks.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print("saved: chv_02_landmarks.png")

# pact panel
canvas = img_rgb.copy()
for person in frame_result.persons:
    if person.landmark_result:
        canvas = draw_pose_landmarks(canvas, person.landmark_result,
                                     landmark_color=(0, 255, 128),
                                     connection_color=(255, 255, 255),
                                     landmark_radius=5, connection_thickness=2,
                                     connections=COCO_CONNECTIONS)
    for ppe in person.ppe_detections:
        x1, y1, x2, y2 = ppe["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PPE_COLOR, 3)
cv2.imwrite(str(OUT_DIR / "chv_03_pact.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print("saved: chv_03_pact.png")

# json
(OUT_DIR / "chv_05_json.json").write_text(json.dumps(frame_dict, indent=2), encoding="utf-8")
print("saved: chv_05_json.json")

# report txt
s = report_data["summary"]
rate_str = f"{s['compliance_rate']*100:.0f}%" if s["compliance_rate"] is not None else "N/A"
lines = [
    f"Context: {report_data['meta']['context']}",
    f"Rule: {report_data['meta']['rule']}",
    f"Timestamp: {report_data['meta']['timestamp']}",
    "",
    f"Workers detected: {s['num_persons']}",
    f"Compliant: {s['num_compliant']}",
    f"Violations: {s['num_violations']}",
    f"Compliance rate: {rate_str}",
    "",
    "Overall assessment:",
    report_data["assessment"],
    "",
    "Workers:",
]
for pd in report_data["persons"]:
    pid    = pd["person_id"] + 1
    status = "COMPLIANT" if pd["compliant"] else "NON-COMPLIANT"
    score  = pd.get("compliance_score")
    score_str = f" ({score*100:.0f}%)" if score is not None else ""
    lines.append(f"  Worker {pid}: {status}{score_str}")
    if pd["worn"]:    lines.append(f"    Worn: {', '.join(pd['worn'])}")
    if pd["missing"]: lines.append(f"    Missing: {', '.join(pd['missing'])}")
    lines.append(f"    {pd['narrative']}")
if report_data["action_items"]:
    lines += ["", "Action items:"]
    for item in report_data["action_items"]:
        lines.append(f"  {item}")
(OUT_DIR / "chv_06_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("saved: chv_06_report.txt")
print("\nFinal result:")
for p in frame_result.persons:
    print(f"  Person {p.person_id}: worn={p.worn}  missing={p.missing}")
