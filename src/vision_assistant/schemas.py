"""Shared data schemas used across pipeline layers.

Centralized to avoid circular imports and make the contract between layers explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------- Frame quality ----------


class FrameQuality(str, Enum):
    OK = "ok"
    BLURRY = "blurry"
    DARK = "dark"
    OVEREXPOSED = "overexposed"
    STATIC = "static"  # unchanged since last inference; skip


@dataclass
class FrameAssessment:
    """Output of Layer 1 — Input Filter."""

    quality: FrameQuality
    blur_score: float
    luminance_mean: float
    ssim_to_prev: Optional[float]
    should_infer: bool
    reason: str = ""


# ---------- VLM raw output ----------


@dataclass
class DetectedObject:
    name: str
    position: str = "unknown"  # left|right|center|front|behind|above|below
    distance: str = "unknown"  # near|mid|far


@dataclass
class VLMRawResponse:
    """Parsed JSON response from the VLM, before post-processing."""

    status: str  # "ok" | "unclear"
    scene: str = ""
    objects: list[DetectedObject] = field(default_factory=list)
    hazards: list[str] = field(default_factory=list)
    answer: str = ""
    raw_text: str = ""
    reason: str = ""
    suggested_action: str = ""


# ---------- Confidence ----------


@dataclass
class ConfidenceReport:
    """Output of Layer 2 — Confidence Estimator."""

    score: float                     # 0.0 – 1.0
    logprob_norm: float              # normalized log-probability
    self_consistency_sim: float      # avg BERTScore F1 across samples
    vague_phrase_rate: float
    is_confident: bool
    sampled_responses: list[str] = field(default_factory=list)


# ---------- Layer 3 final output ----------


@dataclass
class AssistantResponse:
    """Final structured output passed to TTS."""

    text_vi: str                              # Vietnamese speech text
    priority_segments: list[tuple[str, str]]  # list of (priority_tag, sentence)
    hazards: list[str] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)


# ---------- Pipeline I/O ----------


@dataclass
class PipelineInput:
    frame_bgr: np.ndarray
    user_query: str = ""
    timestamp: float = 0.0


@dataclass
class PipelineOutput:
    response: Optional[AssistantResponse]
    frame_assessment: FrameAssessment
    confidence: Optional[ConfidenceReport]
    error: Optional[str] = None
