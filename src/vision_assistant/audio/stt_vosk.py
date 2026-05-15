"""Vietnamese STT via Vosk (offline)."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Iterator

from loguru import logger


class VoskRecognizer:
    def __init__(self, model_path: str | Path, sample_rate: int = 16000) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self._model = None
        self._recognizer = None

    def load(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as e:
            raise RuntimeError("vosk not installed") from e

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {self.model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )
        logger.info("Loading Vosk model from {}", self.model_path)
        self._model = Model(str(self.model_path))
        self._recognizer = KaldiRecognizer(self._model, self.sample_rate)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """One-shot transcription on a full audio chunk (16-bit PCM)."""
        if self._recognizer is None:
            self.load()
        self._recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(self._recognizer.FinalResult())
        return result.get("text", "")

    def stream(self, audio_queue: "queue.Queue[bytes]") -> Iterator[str]:
        """Yields partial transcriptions as audio chunks arrive."""
        if self._recognizer is None:
            self.load()
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            if self._recognizer.AcceptWaveform(chunk):
                res = json.loads(self._recognizer.Result())
                if text := res.get("text", "").strip():
                    yield text
            # else: partial — ignore for now
