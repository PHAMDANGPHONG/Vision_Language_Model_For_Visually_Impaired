"""Tests for Layer 1 — Input Filter."""

from __future__ import annotations

from vision_assistant.layer1_input.blur_detector import BlurDetector
from vision_assistant.layer1_input.exposure_check import ExposureChecker
from vision_assistant.layer1_input.scene_change import SceneChangeDetector
from vision_assistant.layer1_input.frame_filter import FrameFilter
from vision_assistant.schemas import FrameQuality


def test_blur_detector_marks_uniform_frame_as_blurry(blurry_frame):
    bd = BlurDetector(threshold=100.0)
    is_blurry, score = bd.is_blurry(blurry_frame)
    assert is_blurry
    assert score < 1.0


def test_blur_detector_passes_sharp_frame(sharp_frame):
    bd = BlurDetector(threshold=50.0)
    is_blurry, score = bd.is_blurry(sharp_frame)
    assert not is_blurry
    assert score > 50.0


def test_exposure_checker_flags_dark_frame(dark_frame):
    ec = ExposureChecker(luminance_min=40.0)
    status, lum = ec.assess(dark_frame)
    assert status == "dark"
    assert lum < 40.0


def test_scene_change_marks_identical_frames_as_static(sharp_frame):
    scd = SceneChangeDetector(ssim_unchanged_min=0.95)
    # First call always returns False (no history yet)
    assert scd.is_static(sharp_frame)[0] is False
    # Second identical call should be flagged as static
    is_static, _ = scd.is_static(sharp_frame)
    assert is_static is True


def test_frame_filter_blurry_short_circuits(blurry_frame):
    ff = FrameFilter(
        blur=BlurDetector(threshold=100.0),
        exposure=ExposureChecker(),
        scene=SceneChangeDetector(),
    )
    result = ff.assess(blurry_frame)
    assert result.quality == FrameQuality.BLURRY
    assert result.should_infer is False


def test_frame_filter_passes_clean_frame(sharp_frame):
    ff = FrameFilter(
        blur=BlurDetector(threshold=50.0),
        exposure=ExposureChecker(luminance_min=10.0, luminance_max=250.0),
        scene=SceneChangeDetector(),
    )
    result = ff.assess(sharp_frame)
    assert result.quality == FrameQuality.OK
    assert result.should_infer is True
