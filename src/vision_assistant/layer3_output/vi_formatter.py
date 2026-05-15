"""Vietnamese natural-language formatter.

Template-based to keep latency low; no extra LLM call. Templates loaded from
configs/prompts/vi_formatter_templates.yaml.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from ..layer2_vlm.reacquisition import ReacquisitionHint
from ..schemas import FrameAssessment, FrameQuality
from .spatial_grounding import GroundedScene


class VietnameseFormatter:
    def __init__(self, templates: dict[str, Any], max_sentences: int = 2) -> None:
        self.t = templates
        self.max_sentences = max_sentences

    @classmethod
    def from_yaml(cls, path: str | Path, max_sentences: int = 2) -> "VietnameseFormatter":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f), max_sentences=max_sentences)

    # ---------- Public API ----------

    def compose(self, scene: GroundedScene) -> str:
        sentences: list[str] = []

        if scene.hazards:
            primary_hazard = scene.hazards[0]
            position = self._format_position_for_object(scene, primary_hazard)
            sentences.append(self._pick("hazard_alert").format(hazard=primary_hazard, position=position))

        if scene.raw_answer:
            sentences.append(scene.raw_answer.strip().rstrip("."))

        if not sentences and scene.objects:
            obj = scene.objects[0]
            sentences.append(self._pick("object_location").format(
                object=obj.name,
                position=self._t_pos(obj.position),
                distance=self._t_dist(obj.distance),
            ))

        if not sentences:
            sentences.append(self.t["no_objects"])

        return " ".join(sentences[: self.max_sentences])

    def format_unclear(self, assessment: FrameAssessment) -> str:
        key = {
            FrameQuality.BLURRY: "blurry",
            FrameQuality.DARK: "dark",
            FrameQuality.OVEREXPOSED: "dark",
            FrameQuality.STATIC: "low_confidence",
        }.get(assessment.quality, "low_confidence")
        return self.t["unclear_image"].get(key, self.t["unclear_image"]["low_confidence"])

    def format_reacquisition(self, hint: ReacquisitionHint) -> str:
        if hint.direction == "steady":
            return self.t["unclear_image"]["blurry"]
        translated = self._t_pos(hint.direction)
        return self.t["unclear_image"]["occluded"].format(hint=translated)

    # ---------- Helpers ----------

    def _pick(self, key: str) -> str:
        opts = self.t.get(key, [])
        if not opts:
            return ""
        return random.choice(opts) if isinstance(opts, list) else str(opts)

    def _t_pos(self, position: str) -> str:
        return self.t.get("position_translation", {}).get(position, position)

    def _t_dist(self, distance: str) -> str:
        return self.t.get("distance_translation", {}).get(distance, distance)

    def _format_position_for_object(self, scene: GroundedScene, hazard_name: str) -> str:
        for o in scene.objects:
            if o.name == hazard_name:
                return self._t_pos(o.position)
        return self.t.get("position_translation", {}).get("front", "phía trước")
