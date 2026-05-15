"""Composite frame filter — combines blur, exposure, and scene-change checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import FrameAssessment, FrameQuality
from .blur_detector import BlurDetector
from .exposure_check import ExposureChecker
from .scene_change import SceneChangeDetector


class FrameFilter:
    def __init__(
        self,
        blur: BlurDetector,
        exposure: ExposureChecker,
        scene: SceneChangeDetector,
        skip_on_static_scene: bool = True,
    ) -> None:
        self.blur = blur
        self.exposure = exposure
        self.scene = scene
        self.skip_on_static = skip_on_static_scene

    @classmethod
    def from_config(cls, thresholds: dict[str, Any], skip_on_static_scene: bool = True) -> "FrameFilter":
        t = thresholds["layer1_input_filter"]
        return cls(
            blur=BlurDetector(threshold=t["blur"]["laplacian_var_min"]),
            exposure=ExposureChecker(
                luminance_min=t["exposure"]["luminance_min"],
                luminance_max=t["exposure"]["luminance_max"],
                saturation_max_ratio=t["exposure"]["saturation_max_ratio"],
            ),
            scene=SceneChangeDetector(
                history_size=t["scene_change"]["history_size"],
                ssim_unchanged_min=t["scene_change"]["ssim_unchanged_min"],
            ),
            skip_on_static_scene=skip_on_static_scene,
        )

    def assess(self, frame_bgr: np.ndarray) -> FrameAssessment:
        is_blurry, blur_score = self.blur.is_blurry(frame_bgr)
        if is_blurry:
            return FrameAssessment(
                quality=FrameQuality.BLURRY,
                blur_score=blur_score,
                luminance_mean=0.0,
                ssim_to_prev=None,
                should_infer=False,
                reason=f"blur_var={blur_score:.1f} < threshold",
            )

        exp_status, lum_mean = self.exposure.assess(frame_bgr)
        if exp_status == "dark":
            return FrameAssessment(
                quality=FrameQuality.DARK,
                blur_score=blur_score,
                luminance_mean=lum_mean,
                ssim_to_prev=None,
                should_infer=False,
                reason=f"luminance={lum_mean:.1f}",
            )
        if exp_status == "overexposed":
            return FrameAssessment(
                quality=FrameQuality.OVEREXPOSED,
                blur_score=blur_score,
                luminance_mean=lum_mean,
                ssim_to_prev=None,
                should_infer=False,
                reason=f"luminance={lum_mean:.1f}",
            )

        is_static, ssim = self.scene.is_static(frame_bgr)
        if is_static and self.skip_on_static:
            return FrameAssessment(
                quality=FrameQuality.STATIC,
                blur_score=blur_score,
                luminance_mean=lum_mean,
                ssim_to_prev=ssim,
                should_infer=False,
                reason=f"static_scene ssim={ssim:.3f}",
            )

        return FrameAssessment(
            quality=FrameQuality.OK,
            blur_score=blur_score,
            luminance_mean=lum_mean,
            ssim_to_prev=ssim,
            should_infer=True,
        )
