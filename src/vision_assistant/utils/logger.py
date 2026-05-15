"""Loguru-based logging configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logger(level: str = "INFO", log_dir: str | Path | None = "logs") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(log_dir) / "vision_assistant_{time:YYYYMMDD}.log",
            level=level,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
        )
