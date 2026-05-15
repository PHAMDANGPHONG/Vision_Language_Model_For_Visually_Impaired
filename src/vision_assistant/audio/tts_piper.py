"""Vietnamese TTS via Piper (offline, low-latency, streaming-friendly)."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from loguru import logger


class PiperSpeaker:
    def __init__(
        self,
        voice_model: str | Path,
        voice_config: Optional[str | Path] = None,
        sentence_silence: float = 0.25,
    ) -> None:
        self.voice_model = Path(voice_model)
        self.voice_config = Path(voice_config) if voice_config else None
        self.sentence_silence = sentence_silence
        self._voice = None

    def load(self) -> None:
        try:
            from piper import PiperVoice  # piper-tts package
        except ImportError as e:
            raise RuntimeError("piper-tts not installed") from e

        if not self.voice_model.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {self.voice_model}. "
                "Download a vi_VN voice from https://github.com/rhasspy/piper/releases"
            )
        logger.info("Loading Piper voice {}", self.voice_model)
        self._voice = PiperVoice.load(str(self.voice_model), config_path=str(self.voice_config or ""))

    def synthesize(self, text: str) -> bytes:
        """Returns 16-bit PCM mono WAV bytes."""
        if self._voice is None:
            self.load()
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wav:
                self._voice.synthesize(text, wav)
            return buf.getvalue()

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yields PCM int16 chunks suitable for sounddevice playback."""
        if self._voice is None:
            self.load()
        for audio_bytes in self._voice.synthesize_stream_raw(text):
            yield np.frombuffer(audio_bytes, dtype=np.int16)

    def speak(self, text: str) -> None:
        """Blocking playback via sounddevice."""
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError("sounddevice not installed") from e
        if self._voice is None:
            self.load()
        for chunk in self.synthesize_stream(text):
            sd.play(chunk, samplerate=self._voice.config.sample_rate, blocking=True)
