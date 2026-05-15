"""End-to-end orchestrator implementing the Triple-Layer Framework.

Flow (see README §5):
    [Frame] → Layer1.assess → if OK → Layer2.infer → if confident → Layer3.compose
                                                  → else        → re-acquisition feedback
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from .layer1_input.frame_filter import FrameFilter
from .layer2_vlm.confidence_estimator import ConfidenceEstimator
from .layer2_vlm.reacquisition import ReacquisitionAdvisor
from .layer2_vlm.vlm_engine import VLMEngine
from .layer3_output.priority_filter import PriorityFilter
from .layer3_output.spatial_grounding import SpatialGrounder
from .layer3_output.translator import OfflineTranslator
from .layer3_output.vi_formatter import VietnameseFormatter
from .perception.yolo_detector import YoloDetector
from .schemas import (
    AssistantResponse,
    FrameAssessment,
    FrameQuality,
    PipelineInput,
    PipelineOutput,
)


class AssistantPipeline:
    """High-level pipeline. Stateless across queries except for the frame history
    needed by SSIM-based scene-change detection."""

    def __init__(
        self,
        frame_filter: FrameFilter,
        vlm_engine: VLMEngine,
        confidence: ConfidenceEstimator,
        reacquire: ReacquisitionAdvisor,
        yolo: YoloDetector,
        grounder: SpatialGrounder,
        priority: PriorityFilter,
        translator: Optional[OfflineTranslator],
        formatter: VietnameseFormatter,
    ) -> None:
        self.frame_filter = frame_filter
        self.vlm = vlm_engine
        self.confidence = confidence
        self.reacquire = reacquire
        self.yolo = yolo
        self.grounder = grounder
        self.priority = priority
        self.translator = translator
        self.formatter = formatter

    def run(self, payload: PipelineInput) -> PipelineOutput:
        latency: dict[str, float] = {}

        # ---------- Layer 1 ----------
        t0 = time.perf_counter()
        assessment: FrameAssessment = self.frame_filter.assess(payload.frame_bgr)
        latency["layer1_ms"] = (time.perf_counter() - t0) * 1000

        if not assessment.should_infer:
            advice = self.formatter.format_unclear(assessment)
            return PipelineOutput(
                response=AssistantResponse(
                    text_vi=advice,
                    priority_segments=[("safety", advice)],
                    latency_ms=latency,
                ),
                frame_assessment=assessment,
                confidence=None,
            )

        # ---------- Layer 2 ----------
        t1 = time.perf_counter()
        vlm_raw = self.vlm.generate(image_bgr=payload.frame_bgr, query=payload.user_query)
        latency["vlm_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        confidence_report = self.confidence.evaluate(
            image_bgr=payload.frame_bgr,
            query=payload.user_query,
            primary_response=vlm_raw,
            sample_fn=self.vlm.generate,
        )
        latency["confidence_ms"] = (time.perf_counter() - t2) * 1000

        if not confidence_report.is_confident:
            hint = self.reacquire.suggest(vlm_raw)
            text = self.formatter.format_reacquisition(hint)
            return PipelineOutput(
                response=AssistantResponse(
                    text_vi=text,
                    priority_segments=[("safety", text)],
                    confidence=confidence_report.score,
                    latency_ms=latency,
                ),
                frame_assessment=assessment,
                confidence=confidence_report,
            )

        # ---------- Layer 3 ----------
        t3 = time.perf_counter()
        detections = self.yolo.detect(payload.frame_bgr)
        latency["yolo_ms"] = (time.perf_counter() - t3) * 1000

        t4 = time.perf_counter()
        grounded = self.grounder.merge(vlm_raw, detections, payload.frame_bgr.shape)
        prioritized = self.priority.filter(grounded)

        if self.translator and self.translator.is_needed():
            prioritized = self.translator.translate_response(prioritized)

        text_vi = self.formatter.compose(prioritized)
        latency["layer3_ms"] = (time.perf_counter() - t4) * 1000

        response = AssistantResponse(
            text_vi=text_vi,
            priority_segments=self.priority.last_segments,
            hazards=grounded.hazards,
            objects=grounded.objects,
            confidence=confidence_report.score,
            latency_ms=latency,
        )
        logger.info(
            "Pipeline OK | conf={:.2f} | total={:.0f}ms",
            confidence_report.score,
            sum(latency.values()),
        )
        return PipelineOutput(
            response=response, frame_assessment=assessment, confidence=confidence_report
        )
