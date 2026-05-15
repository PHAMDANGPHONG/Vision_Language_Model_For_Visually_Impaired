"""SSIM-based scene-change detector.

If the new frame is visually identical to the recent history, we skip VLM inference.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np


def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Lightweight SSIM implementation (avoids scikit-image dep).

    Uses 8x8 mean-only approximation good enough for change detection.
    """
    if img1.shape != img2.shape:
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu1 = cv2.boxFilter(g1, -1, (8, 8))
    mu2 = cv2.boxFilter(g2, -1, (8, 8))
    s1 = cv2.boxFilter(g1 * g1, -1, (8, 8)) - mu1 * mu1
    s2 = cv2.boxFilter(g2 * g2, -1, (8, 8)) - mu2 * mu2
    s12 = cv2.boxFilter(g1 * g2, -1, (8, 8)) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
    return float((num / np.maximum(den, 1e-8)).mean())


class SceneChangeDetector:
    def __init__(self, history_size: int = 3, ssim_unchanged_min: float = 0.95) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=history_size)
        self.threshold = ssim_unchanged_min

    def is_static(self, frame_bgr: np.ndarray) -> tuple[bool, Optional[float]]:
        if not self.history:
            self.history.append(frame_bgr.copy())
            return False, None
        sims = [_ssim(frame_bgr, h) for h in self.history]
        max_sim = max(sims)
        self.history.append(frame_bgr.copy())
        return max_sim >= self.threshold, max_sim

    def reset(self) -> None:
        self.history.clear()
