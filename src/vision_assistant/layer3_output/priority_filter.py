"""Priority filter — order content as Safety > Goal > Context (README §5.3.2)."""

from __future__ import annotations

from .spatial_grounding import GroundedScene


class PriorityFilter:
    def __init__(
        self,
        priority_order: list[str] | None = None,
        max_sentences: int = 2,
        safety_keywords_vi: list[str] | None = None,
    ) -> None:
        self.priority_order = priority_order or ["safety", "goal", "context"]
        self.max_sentences = max_sentences
        self.safety_keywords_vi = [k.lower() for k in (safety_keywords_vi or [])]
        self.last_segments: list[tuple[str, str]] = []

    def filter(self, scene: GroundedScene) -> GroundedScene:
        """In-place reorder; mutates `scene.objects` by hazard-priority."""
        # Move hazards to the front of the object list for downstream formatter.
        hazardous = [o for o in scene.objects if o.name.lower() in self.safety_keywords_vi]
        non_hazard = [o for o in scene.objects if o.name.lower() not in self.safety_keywords_vi]
        scene.objects = hazardous + non_hazard
        return scene
