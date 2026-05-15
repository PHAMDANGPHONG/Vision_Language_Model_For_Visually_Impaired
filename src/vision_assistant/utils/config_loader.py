"""YAML config loader with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_config(config_dir: str | Path) -> dict[str, Any]:
    """Load and merge default.yaml + models.yaml + thresholds.yaml. Applies .env."""
    load_dotenv(override=False)
    config_dir = Path(config_dir)
    cfg: dict[str, Any] = {}
    for name in ("default.yaml", "models.yaml", "thresholds.yaml"):
        path = config_dir / name
        if path.exists():
            cfg = _deep_merge(cfg, load_yaml(path))

    # Common ENV overrides
    if v := os.environ.get("VLM_MODEL_PATH"):
        active = cfg.get("vlm", {}).get("active", "moondream2")
        cfg.setdefault("vlm", {}).setdefault(active, {})["text_model"] = v
    if v := os.environ.get("VLM_MMPROJ_PATH"):
        active = cfg.get("vlm", {}).get("active", "moondream2")
        cfg.setdefault("vlm", {}).setdefault(active, {})["mmproj"] = v
    if v := os.environ.get("YOLO_MODEL_PATH"):
        cfg.setdefault("perception", {}).setdefault("yolo", {})["weights"] = v
    if v := os.environ.get("VOSK_MODEL_PATH"):
        cfg.setdefault("stt", {}).setdefault("vosk", {})["model_path"] = v
    if v := os.environ.get("PIPER_VOICE_PATH"):
        cfg.setdefault("tts", {}).setdefault("piper", {})["voice"] = v
    if v := os.environ.get("LLAMA_N_THREADS"):
        cfg.setdefault("hardware", {})["n_threads"] = int(v)
    if v := os.environ.get("LLAMA_N_CTX"):
        cfg.setdefault("hardware", {})["n_ctx"] = int(v)

    return cfg
