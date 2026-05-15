"""End-to-end demo: take one snapshot, run pipeline, speak result.

Useful for quick sanity-checks during Week 5+ development.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from loguru import logger

from vision_assistant.camera.capture import CameraStream
from vision_assistant.utils.config_loader import load_app_config
from vision_assistant.utils.logger import configure_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs")
    parser.add_argument("--query", default="Mô tả những gì bạn nhìn thấy.")
    parser.add_argument("--image", default=None, help="Use a static image instead of camera.")
    parser.add_argument("--save-frame", default=None)
    args = parser.parse_args()

    configure_logger(level="INFO")
    cfg = load_app_config(Path(args.config))

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(args.image)
    else:
        cam_cfg = cfg.get("camera", {})
        with CameraStream(
            index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 15),
        ).session() as cam:
            time.sleep(1.0)  # warm-up
            frame = cam.read()

    if frame is None:
        logger.error("No frame captured.")
        return

    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        logger.info("Saved frame to {}", args.save_frame)

    logger.warning("TODO Week 5-11: assemble pipeline and run frame through it.")
    logger.info("Demo skeleton OK. Frame shape: {}", frame.shape)


if __name__ == "__main__":
    main()
