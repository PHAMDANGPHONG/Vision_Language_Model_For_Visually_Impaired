"""CLI entry point.

Usage:
    python -m vision_assistant.main --config configs/default.yaml
    vision-assistant run --query "Có vật gì trên bàn?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from .utils.config_loader import load_app_config
from .utils.logger import configure_logger


@click.group()
@click.option("--config", "config_dir", type=click.Path(exists=True), default="configs")
@click.option("--log-level", default="INFO")
@click.pass_context
def cli(ctx: click.Context, config_dir: str, log_level: str) -> None:
    """Vision Assistant — Multi-layer VLM for visually impaired users."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_app_config(Path(config_dir))
    configure_logger(level=log_level)


@cli.command("run")
@click.option("--query", default="Mô tả những gì bạn nhìn thấy.", help="Câu hỏi gửi tới VLM.")
@click.option("--camera/--image", default=True, help="Use camera stream or single image file.")
@click.option("--image-path", default=None, type=click.Path(exists=True))
@click.pass_context
def run(ctx: click.Context, query: str, camera: bool, image_path: str | None) -> None:
    """Run the assistant once or in live camera mode."""
    logger.info("Vision Assistant starting…")
    logger.warning("TODO: wire up the pipeline factory (planned for Week 1-2).")
    # Pipeline assembly is delegated to a factory module to keep CLI thin.
    # from .factory import build_pipeline
    # pipeline = build_pipeline(ctx.obj["config"])
    # ...
    click.echo("Skeleton ready. Implement pipeline assembly to start serving queries.")


@cli.command("doctor")
def doctor() -> None:
    """Diagnostic: check model files, RAM, and dependency versions."""
    from .utils.metrics import system_snapshot

    snap = system_snapshot()
    for k, v in snap.items():
        click.echo(f"  {k:<24} {v}")


def main() -> None:
    try:
        cli(obj={})
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
