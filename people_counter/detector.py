from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "top-n": ROOT / "models" / "Yolo_v11_n.pt",
    "top-x": ROOT / "models" / "Yolo_v11_x.pt",
    "custom": ROOT / "models" / "custom.pt",
    "front": ROOT / "models" / "yolo11n.pt",
}


@dataclass
class Detection:
    box: tuple
    conf: float
    track_id: int | None

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


def resolve_model(name):
    path = MODELS.get(name, Path(name))
    if name == "front" and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        YOLO("yolo11n.pt").save(str(path))
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run: python download_models.py")
    return path


class PersonDetector:
    def __init__(self, model="top-n", conf=0.35, iou=0.65, imgsz=640, device=None, tta=False, tracker="bytetrack.yaml"):
        self.path = resolve_model(model)
        self.model = YOLO(str(self.path))
        self.name = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.tta = tta
        self.tracker = tracker
        names = {i: n.lower() for i, n in self.model.names.items()}
        self.classes = [i for i, n in names.items() if n in ("person", "persona")] or list(names)

    def _args(self):
        return dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            augment=self.tta,
            classes=self.classes,
            verbose=False,
        )

    @staticmethod
    def _parse(result):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [Detection(tuple(b), float(c), None if i is None else int(i)) for b, c, i in zip(xyxy, confs, ids)]

    def detect(self, frame: np.ndarray):
        return self._parse(self.model.predict(frame, **self._args())[0])

    def track(self, frame: np.ndarray):
        return self._parse(self.model.track(frame, persist=True, tracker=self.tracker, **self._args())[0])

    def reset_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()
