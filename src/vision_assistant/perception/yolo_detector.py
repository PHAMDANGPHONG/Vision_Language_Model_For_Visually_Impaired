"""YOLOv8n adapter using the Ultralytics package.

Used in Layer 3 spatial grounding to provide bbox-based positions that override
VLM's often-wrong spatial claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class YoloDetection:
    class_name: str
    class_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class YoloDetector:
    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        img_size: int = 480,
        allowed_classes: Optional[list[str]] = None,
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.img_size = img_size
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics not installed") from e
        logger.info("Loading YOLO from {}", self.weights)
        self._model = YOLO(str(self.weights))

    def detect(self, frame_bgr: np.ndarray) -> list[YoloDetection]:
        if self._model is None:
            self.load()
        results = self._model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
        )
        out: list[YoloDetection] = []
        if not results:
            return out
        r = results[0]
        names = r.names  # dict[int, str]
        if r.boxes is None:
            return out
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if self.allowed_classes and cls_name not in self.allowed_classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append(
                YoloDetection(
                    class_name=cls_name,
                    class_id=cls_id,
                    conf=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
        return out
