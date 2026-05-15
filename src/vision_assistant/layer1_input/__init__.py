"""Layer 1 — Adaptive Input Filter (see README §5.1).

Goal: cheaply reject low-quality frames and skip redundant inference.
"""

from .blur_detector import BlurDetector
from .exposure_check import ExposureChecker
from .frame_filter import FrameFilter
from .scene_change import SceneChangeDetector

__all__ = ["BlurDetector", "ExposureChecker", "FrameFilter", "SceneChangeDetector"]
