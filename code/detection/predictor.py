import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMGSZ          = 640
JPEG_QUALITY   = 92


class Predictor:
    def __init__(
        self,
        weights_path: Union[str, Path],
        conf:         float = CONF_THRESHOLD,
        iou:          float = IOU_THRESHOLD,
        imgsz:        int   = IMGSZ,
        device:       str   = "0",
    ):
        from ultralytics import YOLO
        self.model  = YOLO(str(weights_path))
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device

    def predict_split(
        self,
        image_paths:     List[str],
        img_output_dir:  Path,
        lbl_output_dir:  Path,
        skip_existing:   bool = True,
    ) -> Dict[str, int]:
        img_output_dir.mkdir(parents=True, exist_ok=True)
        lbl_output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"total": len(image_paths), "processed": 0, "skipped": 0, "errors": 0}

        pbar = tqdm(image_paths, total=len(image_paths), unit="img", leave=False)

        for img_path_str in pbar:
            results = self.model.predict(
                source  = img_path_str,
                conf    = self.conf,
                iou     = self.iou,
                imgsz   = self.imgsz,
                device  = self.device,
                verbose = False,
            )
            result = results[0]
            stem     = Path(img_path_str).stem
            out_img  = img_output_dir / (stem + ".jpg")
            out_lbl  = lbl_output_dir / (stem + ".txt")

            if skip_existing and out_img.exists() and out_lbl.exists():
                stats["skipped"] += 1
                continue

            try:
                cv2.imwrite(
                    str(out_img),
                    result.plot(),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

                lines: List[str] = []
                if result.boxes is not None and len(result.boxes):
                    for box in result.boxes:
                        cls_id  = int(box.cls[0])
                        conf    = float(box.conf[0])
                        xywhn   = box.xywhn[0].tolist()
                        lines.append(
                            f"{cls_id} "
                            f"{xywhn[0]:.6f} {xywhn[1]:.6f} "
                            f"{xywhn[2]:.6f} {xywhn[3]:.6f} "
                            f"{conf:.4f}"
                        )
                out_lbl.write_text("\n".join(lines))
                stats["processed"] += 1

            except Exception as e:
                tqdm.write(f"ERROR [{Path(img_path_str).name}]: {e}")
                stats["errors"] += 1

        pbar.close()
        return stats

    def predict_dataset(
        self,
        config,
        splits:       Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Dict]:
        if splits is None:
            splits = ["trainval", "test"]

        all_stats: Dict[str, Dict] = {}

        for split_name in splits:
            split_txt = config.split_dir / f"{split_name}.txt"
            if not split_txt.exists():
                print(f"  [{config.name}/{split_name}] split file not found, skipping")
                continue

            filenames   = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
            image_paths = [str(config.images_dir / f) for f in filenames]

            img_out = config.bb_predict_dir / split_name
            lbl_out = config.bb_predict_dir / f"{split_name}_labels"

            print(f"\n  [{config.name}/{split_name}]  n={len(image_paths)}")
            stats = self.predict_split(image_paths, img_out, lbl_out, skip_existing)
            all_stats[split_name] = stats
            print(
                f"    processed={stats['processed']}  "
                f"skipped={stats['skipped']}  "
                f"errors={stats['errors']}"
            )

        return all_stats
