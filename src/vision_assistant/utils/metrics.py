"""Lightweight perf & resource metrics."""

from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from typing import Any, Iterator

import psutil


class LatencyTimer:
    """Context manager that accumulates wall-clock latencies into a dict."""

    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records[label] = (time.perf_counter() - t0) * 1000

    def total_ms(self) -> float:
        return sum(self.records.values())


def system_snapshot() -> dict[str, Any]:
    """Return a small dict useful for the `vision-assistant doctor` CLI."""
    vm = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_used_pct": vm.percent,
        "process_rss_mb": round(psutil.Process().memory_info().rss / (1024**2), 1),
    }
