"""Confidence Estimator — main methodological contribution (README §5.2.2).

Combines 3 signals:
    (1) Normalized average token log-probability       (w1 = 0.4)
    (2) Self-consistency via BERTScore across samples  (w2 = 0.4)
    (3) Vague-phrase rate                              (w3 = 0.2, subtractive)

Returns a ConfidenceReport. Sub-threshold reports trigger Re-acquisition.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np
from loguru import logger

from ..schemas import ConfidenceReport, VLMRawResponse


class ConfidenceEstimator:
    def __init__(
        self,
        weight_logprob: float = 0.4,
        weight_self_consistency: float = 0.4,
        weight_vague_phrase: float = 0.2,
        min_confidence: float = 0.45,
        self_consistency_samples: int = 3,
        self_consistency_temperature: float = 0.5,
        vague_phrases: list[str] | None = None,
    ) -> None:
        self.w1 = weight_logprob
        self.w2 = weight_self_consistency
        self.w3 = weight_vague_phrase
        self.min_confidence = min_confidence
        self.n_samples = self_consistency_samples
        self.sample_temperature = self_consistency_temperature
        self.vague_phrases = [p.lower() for p in (vague_phrases or [])]

    # ---------- Public API ----------

    def evaluate(
        self,
        image_bgr,
        query: str,
        primary_response: VLMRawResponse,
        sample_fn: Callable[..., VLMRawResponse],
    ) -> ConfidenceReport:
        logprobs = primary_response.__dict__.get("_logprobs", [])
        logprob_norm = self._normalize_logprob(logprobs)

        samples = self._collect_samples(image_bgr, query, sample_fn)
        consistency = self._self_consistency_similarity(primary_response.raw_text, samples)
        vague_rate = self._vague_phrase_rate(primary_response.raw_text)

        score = (
            self.w1 * logprob_norm + self.w2 * consistency - self.w3 * vague_rate
        )
        score = float(max(0.0, min(1.0, score)))

        report = ConfidenceReport(
            score=score,
            logprob_norm=logprob_norm,
            self_consistency_sim=consistency,
            vague_phrase_rate=vague_rate,
            is_confident=score >= self.min_confidence,
            sampled_responses=samples,
        )
        logger.debug(
            "Confidence: total={:.2f} logprob={:.2f} consistency={:.2f} vague={:.2f}",
            score, logprob_norm, consistency, vague_rate,
        )
        return report

    # ---------- Signal 1: log-probability ----------

    @staticmethod
    def _normalize_logprob(logprobs: list[float]) -> float:
        """Map mean token logprob to [0, 1] via sigmoid-like rescaling.

        Reasonable token logprobs are roughly in [-3, 0]; map -3 → 0, 0 → 1.
        """
        if not logprobs:
            return 0.5  # neutral when unavailable
        mean_lp = sum(logprobs) / len(logprobs)
        prob = math.exp(mean_lp)              # in (0, 1]
        return float(max(0.0, min(1.0, prob)))

    # ---------- Signal 2: self-consistency ----------

    def _collect_samples(
        self, image_bgr, query: str, sample_fn: Callable[..., VLMRawResponse]
    ) -> list[str]:
        samples: list[str] = []
        from .vlm_engine import GenerationParams

        params = GenerationParams(
            max_new_tokens=128,
            temperature=self.sample_temperature,
            top_p=0.95,
            return_logprobs=False,
        )
        for i in range(self.n_samples):
            try:
                resp = sample_fn(image_bgr=image_bgr, query=query, gen_params=params)
                samples.append(resp.raw_text or resp.answer)
            except Exception as e:  # pragma: no cover
                logger.warning("Self-consistency sample {} failed: {}", i, e)
        return samples

    @staticmethod
    def _self_consistency_similarity(reference: str, samples: list[str]) -> float:
        """Average BERTScore F1 between reference and each sample.

        TODO Week 7-8: load multilingual `bert-base-multilingual-cased` lazily
        and cache the scorer. Until then, use Jaccard token overlap as cheap
        fallback to keep CI/tests fast.
        """
        if not samples:
            return 0.5

        try:  # Lazy heavy import
            from bert_score import BERTScorer  # type: ignore

            scorer = _get_bertscorer()
            _, _, f1 = scorer.score(samples, [reference] * len(samples))
            return float(f1.mean().item())
        except Exception:
            ref_tokens = set(re.findall(r"\w+", reference.lower()))
            jaccs = []
            for s in samples:
                tok = set(re.findall(r"\w+", s.lower()))
                inter = len(ref_tokens & tok)
                union = len(ref_tokens | tok) or 1
                jaccs.append(inter / union)
            return float(np.mean(jaccs))

    # ---------- Signal 3: vague-phrase rate ----------

    def _vague_phrase_rate(self, text: str) -> float:
        if not text:
            return 1.0
        text_l = text.lower()
        hits = sum(1 for p in self.vague_phrases if p in text_l)
        words = max(len(re.findall(r"\w+", text_l)), 1)
        # Rate ∈ [0, ~1] — clamp to 1.0 if many hits in short text
        return float(min(1.0, hits * 6.0 / words))


# ---------- BERTScorer singleton ----------

_BERTSCORER = None


def _get_bertscorer():  # pragma: no cover
    global _BERTSCORER
    if _BERTSCORER is None:
        from bert_score import BERTScorer

        _BERTSCORER = BERTScorer(
            model_type="bert-base-multilingual-cased",
            num_layers=9,
            lang="en",
            rescale_with_baseline=False,
        )
    return _BERTSCORER
