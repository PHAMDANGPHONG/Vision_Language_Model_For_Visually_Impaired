"""Variance-of-Laplacian based blur detector.

Reference: Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy:
a comparative study" (ICPR 2000) — the canonical Laplacian-variance method.
"""

from __future__ import annotations

import cv2
import numpy as np


class BlurDetector:
    """Compute the variance of the Laplacian of a grayscale frame.

    A sharper image has higher Laplacian variance. Threshold is calibrated
    per camera (see eval/notebooks/blur_calibration.ipynb).
    """

    def __init__(self, threshold: float = 100.0) -> None:
        self.threshold = float(threshold)

    @staticmethod
    def laplacian_variance(frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def is_blurry(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        score = self.laplacian_variance(frame_bgr)
        return score < self.threshold, score
