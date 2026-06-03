import sys
from pathlib import Path
from typing import Any, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS_DIR      = Path("C:/Users/lulay/Desktop/viskom-safety_equipment/runs")
DEFAULT_MODEL = "models/yolo26l.pt"


class Trainer:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def train(
        self,
        data_yaml:   Union[str, Path],
        run_name:    str,
        epochs:      int   = 100,
        imgsz:       int   = 640,
        batch:       int   = 16,
        patience:    int   = 20,
        lr0:         float = 0.01,
        lrf:         float = 0.01,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3,
        device:      str   = "0",
        pretrained:  bool  = True,
        exist_ok:    bool  = False,
        **kwargs:    Any,
    ) -> Path:
        from ultralytics import YOLO

        model = YOLO(self.model_name)
        model.train(
            data          = str(data_yaml),
            epochs        = epochs,
            imgsz         = imgsz,
            batch         = batch,
            patience      = patience,
            lr0           = lr0,
            lrf           = lrf,
            weight_decay  = weight_decay,
            warmup_epochs = warmup_epochs,
            device        = device,
            project       = str(RUNS_DIR / "detect"),
            name          = run_name,
            pretrained    = pretrained,
            exist_ok      = exist_ok,
            **kwargs,
        )

        best = RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
        print(f"\n  Best weights → {best}")
        return best

    @staticmethod
    def resume(run_name: str, **kwargs: Any) -> Path:
        from ultralytics import YOLO

        last = RUNS_DIR / "detect" / run_name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"last.pt not found at {last}")

        model = YOLO(str(last))
        model.train(
            resume   = True,
            project  = str(RUNS_DIR / "detect"),
            name     = run_name,
            exist_ok = True,
            **kwargs,
        )

        best = RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
        return best
