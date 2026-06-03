import shutil
from pathlib import Path

DATASET_BASE  = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/dataset")
LANDMARK_BASE = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/landmarks")

MOVES = {
    "landmark-mediapipe":   "mediapipe",
    "landmark-yolov8":      "yolov8",
    "detected-mediapipe":   "detected-mediapipe",
    "detected-yolov8":      "detected-yolov8",
    "undetected-mediapipe": "undetected-mediapipe",
    "undetected-yolov8":    "undetected-yolov8",
}

DATASETS = ["CHV", "CPPE-5", "SH17"]

for ds in DATASETS:
    for src_name, dst_name in MOVES.items():
        src = DATASET_BASE / ds / src_name
        dst = LANDMARK_BASE / ds / dst_name
        if not src.exists():
            print(f"  SKIP   {src} (not found)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  MOVED  {src} -> {dst}")

print("\nDone.")
