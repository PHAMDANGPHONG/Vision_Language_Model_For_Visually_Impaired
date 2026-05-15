"""Latency & RAM benchmark for the chosen VLM.

Used in Week 3-4 to decide between Moondream / SmolVLM / Qwen2-VL-2B.
Produces a CSV row per query: model, p50_ms, p90_ms, peak_rss_mb, output.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import psutil
from loguru import logger


def measure_once(generate_fn, frame, query: str) -> tuple[float, float, str]:
    rss_before = psutil.Process().memory_info().rss / 1024**2
    t0 = time.perf_counter()
    out = generate_fn(image_bgr=frame, query=query)
    dt_ms = (time.perf_counter() - t0) * 1000
    rss_after = psutil.Process().memory_info().rss / 1024**2
    return dt_ms, max(rss_before, rss_after), getattr(out, "raw_text", str(out))


def benchmark(
    generate_fn,
    image_paths: list[Path],
    query: str,
    warmup: int = 1,
) -> dict[str, Any]:
    times: list[float] = []
    peak_rss = 0.0
    for _ in range(warmup):
        if image_paths:
            f = cv2.imread(str(image_paths[0]))
            measure_once(generate_fn, f, query)

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Skipping unreadable image: {}", path)
            continue
        dt, rss, _ = measure_once(generate_fn, frame, query)
        times.append(dt)
        peak_rss = max(peak_rss, rss)
        gc.collect()

    times.sort()
    return {
        "n": len(times),
        "p50_ms": statistics.median(times) if times else 0,
        "p90_ms": times[int(0.9 * len(times)) - 1] if times else 0,
        "p99_ms": times[int(0.99 * len(times)) - 1] if times else 0,
        "mean_ms": statistics.mean(times) if times else 0,
        "peak_rss_mb": round(peak_rss, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Folder of test images")
    parser.add_argument("--query", default="Describe what you see in one sentence.")
    parser.add_argument("--model", default="moondream2", choices=["moondream2", "smolvlm", "qwen2vl_2b"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("eval/results/benchmark.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.images.glob("*.jpg"))[: args.limit]
    logger.info("Benchmarking {} on {} images", args.model, len(image_paths))

    # TODO Week 3-4: instantiate the correct backend from config
    raise NotImplementedError("Wire up VLMEngine factory before running.")


if __name__ == "__main__":
    main()
