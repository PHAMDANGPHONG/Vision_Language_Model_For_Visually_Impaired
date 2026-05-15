"""Unit tests for the Confidence Estimator (Layer 2)."""

from __future__ import annotations

import math

import numpy as np

from vision_assistant.layer2_vlm.confidence_estimator import ConfidenceEstimator
from vision_assistant.schemas import VLMRawResponse


def _make_resp(text: str, logprobs: list[float] | None = None) -> VLMRawResponse:
    r = VLMRawResponse(status="ok", answer=text, raw_text=text)
    r.__dict__["_logprobs"] = logprobs or []
    return r


def test_logprob_normalisation_high_when_close_to_zero():
    ce = ConfidenceEstimator()
    norm = ce._normalize_logprob([0.0, -0.1, -0.05])
    assert norm > 0.9


def test_logprob_normalisation_low_when_very_negative():
    ce = ConfidenceEstimator()
    norm = ce._normalize_logprob([-3.0, -2.5, -3.5])
    assert norm < 0.1


def test_vague_phrase_rate_increases_with_hits():
    ce = ConfidenceEstimator(vague_phrases=["i'm not sure", "possibly"])
    rate_low = ce._vague_phrase_rate("There is a cup on the table.")
    rate_high = ce._vague_phrase_rate("I'm not sure but possibly there is something.")
    assert rate_low == 0.0
    assert rate_high > 0.0


def test_self_consistency_high_with_identical_samples():
    ce = ConfidenceEstimator()
    sim = ce._self_consistency_similarity("a cup on the table", ["a cup on the table"] * 3)
    assert sim > 0.9


def test_self_consistency_low_with_unrelated_samples():
    ce = ConfidenceEstimator()
    sim = ce._self_consistency_similarity("cup on table", ["dog in park", "running shoes"])
    assert sim < 0.3


def test_evaluate_marks_high_confidence_when_signals_agree():
    ce = ConfidenceEstimator(min_confidence=0.45, vague_phrases=["unclear"])
    primary = _make_resp("A red cup on the table.", logprobs=[-0.1] * 10)

    def fake_sample(image_bgr, query, gen_params):
        return _make_resp("A red cup on the table.")

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    report = ce.evaluate(img, "what is this?", primary, fake_sample)
    assert report.is_confident
    assert report.score > 0.45


def test_evaluate_flags_low_confidence_with_vague_text():
    ce = ConfidenceEstimator(min_confidence=0.45, vague_phrases=["i'm not sure"])
    primary = _make_resp("I'm not sure I'm not sure I'm not sure.", logprobs=[-3.0] * 10)

    def fake_sample(image_bgr, query, gen_params):
        return _make_resp("totally different garbage output.")

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    report = ce.evaluate(img, "?", primary, fake_sample)
    assert not report.is_confident
