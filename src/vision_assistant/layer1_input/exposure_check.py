"""Histogram-based exposure quality check."""

from __future__ import annotations

import cv2
import numpy as np


class ExposureChecker:
    def __init__(
        self,
        luminance_min: float = 40.0,
        luminance_max: float = 220.0,
        saturation_max_ratio: float = 0.30,
    ) -> None:
        self.lum_min = luminance_min
        self.lum_max = luminance_max
        self.sat_max = saturation_max_ratio

    def assess(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        """Returns (status, mean_luminance). status ∈ {ok, dark, overexposed}."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())
        total = gray.size
        dark_ratio = float((gray < 10).sum()) / total
        bright_ratio = float((gray > 245).sum()) / total

        if mean_lum < self.lum_min or dark_ratio > self.sat_max:
            return "dark", mean_lum
        if mean_lum > self.lum_max or bright_ratio > self.sat_max:
            return "overexposed", mean_lum
        return "ok", mean_lum
