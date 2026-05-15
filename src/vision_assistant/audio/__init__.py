"""Audio I/O — Vietnamese STT (Vosk) and TTS (Piper)."""

from .stt_vosk import VoskRecognizer
from .tts_piper import PiperSpeaker

__all__ = ["VoskRecognizer", "PiperSpeaker"]
