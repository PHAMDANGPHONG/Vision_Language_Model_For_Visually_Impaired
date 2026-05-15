"""Re-acquisition advisor — when confidence is low, suggest a camera action.

NOTE on terminology (README §5.2.3): this is NOT Active Vision in the Bajcsy
(1988) sense. It is Confidence-Guided Re-acquisition / Human-in-the-loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import VLMRawResponse


@dataclass
class ReacquisitionHint:
    direction: str   # left|right|up|down|closer|farther|steady
    reason: str
    text_en: str


_DIRECTION_KEYWORDS = {
    "left": ["left", "left side", "left edge"],
    "right": ["right", "right side", "right edge"],
    "up": ["top", "above", "upper"],
    "down": ["bottom", "below", "lower"],
    "closer": ["far", "too small", "distant"],
    "farther": ["too close", "too large", "cropped"],
}


class ReacquisitionAdvisor:
    def suggest(self, response: VLMRawResponse) -> ReacquisitionHint:
        text = (response.suggested_action or response.reason or response.answer or "").lower()

        for direction, keys in _DIRECTION_KEYWORDS.items():
            if any(k in text for k in keys):
                return ReacquisitionHint(
                    direction=direction,
                    reason=text,
                    text_en=f"Please tilt the camera {direction}.",
                )

        if re.search(r"\b(blur|shake|motion)\b", text):
            return ReacquisitionHint(
                direction="steady",
                reason="motion_blur",
                text_en="Please hold the camera still.",
            )

        return ReacquisitionHint(
            direction="steady",
            reason="low_confidence",
            text_en="The image is unclear; please try a different angle.",
        )
