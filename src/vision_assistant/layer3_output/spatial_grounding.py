"""Hybrid spatial grounding — merge VLM textual claims with YOLO bbox evidence.

Strategy:
    1. For each VLM-claimed object, attempt name match (substring or synonym) with
       a YOLO detection.
    2. If matched, OVERWRITE position/distance using bbox geometry:
         - position from x-center (left/center/right) and y-center (above/below)
         - distance approximated from bbox area relative to image area
    3. Hazards from YOLO safety classes are unioned with VLM-mentioned hazards.

Depth Anything V2 is intentionally NOT used in Phase 1 due to RAM budget (see
README §6.4). Distance is heuristic from bbox area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception.yolo_detector import YoloDetection
from ..schemas import DetectedObject, VLMRawResponse


@dataclass
class GroundedScene:
    scene: str
    objects: list[DetectedObject]
    hazards: list[str]
    raw_answer: str
    raw_status: str


# Light synonym mapping (extend incrementally; see eval/synonyms.yaml in future).
_SYNONYMS = {
    "cup": ["mug", "glass", "cốc", "ly"],
    "knife": ["dao", "blade"],
    "scissors": ["kéo"],
    "bottle": ["chai"],
    "chair": ["ghế"],
    "person": ["man", "woman", "người"],
}


def _normalize(name: str) -> str:
    return name.lower().strip()


def _names_match(a: str, b: str) -> bool:
    a, b = _normalize(a), _normalize(b)
    if a == b or a in b or b in a:
        return True
    for canonical, syns in _SYNONYMS.items():
        if a == canonical and b in syns:
            return True
        if b == canonical and a in syns:
            return True
    return False


def _bbox_to_position(x_center_ratio: float, y_center_ratio: float) -> str:
    """Map normalized centre to a human-friendly position label."""
    if y_center_ratio < 0.25:
        return "above"
    if y_center_ratio > 0.75:
        return "below"
    if x_center_ratio < 0.33:
        return "left"
    if x_center_ratio > 0.66:
        return "right"
    return "center"


def _bbox_to_distance(area_ratio: float) -> str:
    if area_ratio >= 0.18:
        return "near"
    if area_ratio >= 0.05:
        return "mid"
    return "far"


class SpatialGrounder:
    def __init__(self, safety_classes: list[str] | None = None) -> None:
        self.safety_classes = set(safety_classes or [])

    def merge(
        self,
        vlm: VLMRawResponse,
        detections: list[YoloDetection],
        image_shape: tuple[int, int, int],
    ) -> GroundedScene:
        h, w = image_shape[:2]
        img_area = float(h * w)

        objects: list[DetectedObject] = []
        used_det_ids: set[int] = set()

        # 1. Update VLM objects with YOLO evidence where possible.
        for vobj in vlm.objects:
            match_idx = None
            for i, det in enumerate(detections):
                if i in used_det_ids:
                    continue
                if _names_match(vobj.name, det.class_name):
                    match_idx = i
                    break
            if match_idx is not None:
                det = detections[match_idx]
                used_det_ids.add(match_idx)
                x_c = (det.x1 + det.x2) / 2 / w
                y_c = (det.y1 + det.y2) / 2 / h
                area_ratio = ((det.x2 - det.x1) * (det.y2 - det.y1)) / img_area
                objects.append(
                    DetectedObject(
                        name=vobj.name,
                        position=_bbox_to_position(x_c, y_c),
                        distance=_bbox_to_distance(area_ratio),
                    )
                )
            else:
                objects.append(vobj)

        # 2. Add unmatched YOLO detections of safety interest.
        hazards = set(vlm.hazards)
        for i, det in enumerate(detections):
            if i in used_det_ids:
                continue
            if det.class_name in self.safety_classes:
                x_c = (det.x1 + det.x2) / 2 / w
                y_c = (det.y1 + det.y2) / 2 / h
                area_ratio = ((det.x2 - det.x1) * (det.y2 - det.y1)) / img_area
                objects.append(
                    DetectedObject(
                        name=det.class_name,
                        position=_bbox_to_position(x_c, y_c),
                        distance=_bbox_to_distance(area_ratio),
                    )
                )
                hazards.add(det.class_name)

        return GroundedScene(
            scene=vlm.scene,
            objects=objects,
            hazards=sorted(hazards),
            raw_answer=vlm.answer,
            raw_status=vlm.status,
        )
