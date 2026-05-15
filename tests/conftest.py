"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sharp_frame() -> np.ndarray:
    """A high-contrast 640x480 BGR frame — should pass blur check."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, ::20] = 255
    img[::20, :] = 255
    return img


@pytest.fixture
def blurry_frame() -> np.ndarray:
    """Uniform gray frame — Laplacian variance ≈ 0."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def dark_frame() -> np.ndarray:
    return np.full((480, 640, 3), 5, dtype=np.uint8)
