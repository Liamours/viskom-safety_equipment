import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONF_THRESHOLD = 0.001
IOU_THRESHOLD  = 0.6
IMGSZ          = 640


def evaluate(
    weights_path: Union[str, Path],
    data_yaml:    Union[str, Path],
    split:        str   = "test",
    imgsz:        int   = IMGSZ,
    conf:         float = CONF_THRESHOLD,
    iou:          float = IOU_THRESHOLD,
    device:       str   = "0",
    verbose:      bool  = False,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    model   = YOLO(str(weights_path))
    metrics = model.val(
        data    = str(data_yaml),
        split   = split,
        imgsz   = imgsz,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = verbose,
    )

    box         = metrics.box
    class_names = model.names

    per_class = {}
    if box.ap_class_index is not None:
        for idx, cls_idx in enumerate(box.ap_class_index):
            name = class_names.get(int(cls_idx), str(cls_idx))
            per_class[name] = {
                "ap50":   round(float(box.ap50[idx]), 4),
                "ap5095": round(float(box.ap[idx]),   4),
            }

    return {
        "map50":     round(float(box.map50),  4),
        "map5095":   round(float(box.map),    4),
        "precision": round(float(box.mp),     4),
        "recall":    round(float(box.mr),     4),
        "per_class": per_class,
        "split":     split,
        "weights":   str(weights_path),
    }


def save_results(results: Dict[str, Any], output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {output_path}")


def print_results(name: str, results: Dict[str, Any]):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {name}  [{results.get('split', 'test')}]")
    print(sep)
    print(f"  mAP@0.5      : {results['map50']:.4f}")
    print(f"  mAP@0.5:0.95 : {results['map5095']:.4f}")
    print(f"  Precision    : {results['precision']:.4f}")
    print(f"  Recall       : {results['recall']:.4f}")

    per_class = results.get("per_class", {})
    if per_class:
        print(f"\n  {'Class':<22}  {'AP@0.5':>8}  {'AP@0.5:0.95':>12}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*12}")
        for cls_name, ap in sorted(per_class.items(), key=lambda x: -x[1]["ap50"]):
            print(f"  {cls_name:<22}  {ap['ap50']:>8.4f}  {ap['ap5095']:>12.4f}")

    print(sep)
