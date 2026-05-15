"""Webcam capture helper using OpenCV."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger


class CameraStream:
    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        # On Windows prefer DirectShow for lower init latency.
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        logger.info("Camera {} opened @ {}x{}@{}", self.index, self.width, self.height, self.fps)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            self.open()
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @contextmanager
    def session(self) -> Iterator["CameraStream"]:
        try:
            self.open()
            yield self
        finally:
            self.close()
