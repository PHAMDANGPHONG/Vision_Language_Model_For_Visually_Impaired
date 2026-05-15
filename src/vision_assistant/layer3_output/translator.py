"""Offline EN→VI translator.

Two implementations:
    - Argos Translate (default; smaller, installable model)
    - NLLB-200-distilled-600M via transformers (heavier, better quality)

Used only when the active VLM does not support Vietnamese output.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from .spatial_grounding import GroundedScene


class OfflineTranslator:
    def __init__(self, backend: str = "argos", from_code: str = "en", to_code: str = "vi") -> None:
        self.backend = backend
        self.from_code = from_code
        self.to_code = to_code
        self._impl: Any = None
        self._enabled: bool = True

    def is_needed(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    # ---------- Lazy load ----------

    def _ensure_loaded(self) -> None:
        if self._impl is not None:
            return
        if self.backend == "argos":
            self._impl = _ArgosImpl(self.from_code, self.to_code)
        elif self.backend == "nllb":
            self._impl = _NllbImpl()
        else:
            raise ValueError(f"Unknown translator backend: {self.backend}")
        self._impl.load()

    # ---------- Public API ----------

    def translate_text(self, text_en: str) -> str:
        if not text_en.strip():
            return ""
        self._ensure_loaded()
        try:
            return self._impl.translate(text_en)
        except Exception as e:  # pragma: no cover
            logger.warning("Translation failed; returning original. err={}", e)
            return text_en

    def translate_response(self, scene: GroundedScene) -> GroundedScene:
        scene.raw_answer = self.translate_text(scene.raw_answer)
        for o in scene.objects:
            o.name = self.translate_text(o.name)
        scene.hazards = [self.translate_text(h) for h in scene.hazards]
        return scene


# ---------- Implementations ----------


class _ArgosImpl:
    def __init__(self, from_code: str, to_code: str) -> None:
        self.from_code = from_code
        self.to_code = to_code
        self._translate_fn: Optional[Any] = None

    def load(self) -> None:
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError as e:
            raise RuntimeError("argostranslate not installed") from e

        # TODO Week 10: download `translate-en_vi.argosmodel` to data/models/translate/
        # and call argostranslate.package.install_from_path(...) once.
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed if l.code == self.from_code), None)
        to_lang = next((l for l in installed if l.code == self.to_code), None)
        if not (from_lang and to_lang):
            raise RuntimeError(
                f"Argos {self.from_code}->{self.to_code} model not installed."
            )
        translation = from_lang.get_translation(to_lang)
        self._translate_fn = translation.translate

    def translate(self, text: str) -> str:
        if self._translate_fn is None:
            raise RuntimeError("Argos translator not loaded.")
        return self._translate_fn(text)


class _NllbImpl:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def load(self) -> None:  # pragma: no cover
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("transformers not installed") from e

        name = "facebook/nllb-200-distilled-600M"
        self._tokenizer = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(name)

    def translate(self, text: str) -> str:  # pragma: no cover
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("NLLB not loaded.")
        inputs = self._tokenizer(text, return_tensors="pt")
        out = self._model.generate(
            **inputs,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids("vie_Latn"),
            max_new_tokens=80,
        )
        return self._tokenizer.batch_decode(out, skip_special_tokens=True)[0]
