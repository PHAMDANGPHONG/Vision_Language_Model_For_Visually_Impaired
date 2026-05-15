"""VLM Engine — thin wrapper around llama-cpp-python for GGUF VLMs.

Supports multiple backends through subclasses (Moondream, SmolVLM, Qwen2-VL).
Returns a parsed VLMRawResponse with token-level logprobs for downstream
confidence estimation.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..schemas import DetectedObject, VLMRawResponse


@dataclass
class GenerationParams:
    max_new_tokens: int = 128
    temperature: float = 0.4
    top_p: float = 0.9
    seed: Optional[int] = None
    return_logprobs: bool = True


class VLMEngine(abc.ABC):
    """Abstract VLM backend. Subclass per model family."""

    def __init__(
        self,
        text_model_path: str | Path,
        mmproj_path: str | Path,
        system_prompt: str,
        n_threads: int = 6,
        n_ctx: int = 2048,
        n_batch: int = 256,
        gen_params: Optional[GenerationParams] = None,
    ) -> None:
        self.text_model_path = Path(text_model_path)
        self.mmproj_path = Path(mmproj_path)
        self.system_prompt = system_prompt
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.gen_params = gen_params or GenerationParams()
        self._handle: Any = None

    # ---------- Lifecycle ----------

    @abc.abstractmethod
    def load(self) -> None:
        """Lazy-load the underlying llama.cpp model + mmproj."""

    def unload(self) -> None:
        self._handle = None

    # ---------- Inference ----------

    @abc.abstractmethod
    def _generate_raw(
        self,
        image_bgr: np.ndarray,
        query: str,
        gen_params: GenerationParams,
    ) -> tuple[str, list[float]]:
        """Return (raw_text, per_token_logprobs)."""

    def generate(
        self,
        image_bgr: np.ndarray,
        query: str,
        gen_params: Optional[GenerationParams] = None,
    ) -> VLMRawResponse:
        if self._handle is None:
            self.load()
        params = gen_params or self.gen_params
        raw_text, logprobs = self._generate_raw(image_bgr, query, params)
        response = self._parse_json(raw_text)
        response.raw_text = raw_text
        # Stash logprobs through the dataclass via .__dict__ to avoid schema bloat.
        response.__dict__["_logprobs"] = logprobs
        return response

    # ---------- Parsing ----------

    @staticmethod
    def _parse_json(text: str) -> VLMRawResponse:
        """Robust JSON extraction — small VLMs frequently leak prose."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.debug("No JSON in VLM output; returning unclear.")
            return VLMRawResponse(status="unclear", reason="non_json", answer=text.strip())
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.debug("JSON parse failed: {}", e)
            return VLMRawResponse(status="unclear", reason="malformed_json", answer=text.strip())

        objects = [
            DetectedObject(
                name=str(o.get("name", "")),
                position=str(o.get("position", "unknown")),
                distance=str(o.get("distance", "unknown")),
            )
            for o in payload.get("objects", [])
        ]
        return VLMRawResponse(
            status=str(payload.get("status", "ok")),
            scene=str(payload.get("scene", "")),
            objects=objects,
            hazards=[str(h) for h in payload.get("hazards", [])],
            answer=str(payload.get("answer", "")),
            reason=str(payload.get("reason", "")),
            suggested_action=str(payload.get("suggested_action", "")),
        )


# =========================================================
#  Backend: Moondream2
# =========================================================


class MoondreamEngine(VLMEngine):
    """Moondream 2 via llama-cpp-python.

    Implementation notes (TODO Week 1-2):
        - Use `Llama` with `clip_model_path` parameter for mmproj.
        - llama-cpp-python ≥0.2.90 added VLM helpers; verify API on install.
    """

    def load(self) -> None:
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MoondreamChatHandler
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python not available; install with appropriate backend."
            ) from e

        logger.info("Loading Moondream2 from {}", self.text_model_path)
        chat_handler = MoondreamChatHandler(clip_model_path=str(self.mmproj_path))
        self._handle = Llama(
            model_path=str(self.text_model_path),
            chat_handler=chat_handler,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            logits_all=True,            # required for token-level logprobs
            verbose=False,
        )

    def _generate_raw(
        self,
        image_bgr: np.ndarray,
        query: str,
        gen_params: GenerationParams,
    ) -> tuple[str, list[float]]:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image as JPEG.")
        image_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": query},
                ],
            },
        ]

        result = self._handle.create_chat_completion(
            messages=messages,
            max_tokens=gen_params.max_new_tokens,
            temperature=gen_params.temperature,
            top_p=gen_params.top_p,
            seed=gen_params.seed,
            logprobs=gen_params.return_logprobs,
        )
        text = result["choices"][0]["message"]["content"]
        logprobs: list[float] = []
        if "logprobs" in result["choices"][0] and result["choices"][0]["logprobs"]:
            tok_logprobs = result["choices"][0]["logprobs"].get("token_logprobs") or []
            logprobs = [float(lp) for lp in tok_logprobs if lp is not None]
        return text, logprobs


# =========================================================
#  Backend: Qwen2-VL (placeholder)
# =========================================================


class Qwen2VLEngine(VLMEngine):
    """Qwen2-VL-2B via llama-cpp-python.

    TODO Week 3-4: verify chat handler — Qwen2-VL uses a specific image-token
    interleaving; may require a custom chat-format or use the official
    `transformers` pipeline behind a thin adapter.
    """

    def load(self) -> None:
        raise NotImplementedError(
            "Qwen2VLEngine: implement in Week 3-4 after benchmark decision."
        )

    def _generate_raw(self, image_bgr, query, gen_params):  # type: ignore[override]
        raise NotImplementedError
