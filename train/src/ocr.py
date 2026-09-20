"""Bounding-box helpers for LayoutLMv3 inputs."""

from __future__ import annotations

from typing import Sequence


def normalize_box(box: Sequence[int] | Sequence[float], width: int, height: int) -> list[int]:
    """Normalize a pixel box to the LayoutLM 0–1000 coordinate space."""
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * float(x0) / max(width, 1)))),
        max(0, min(1000, int(1000 * float(y0) / max(height, 1)))),
        max(0, min(1000, int(1000 * float(x1) / max(width, 1)))),
        max(0, min(1000, int(1000 * float(y1) / max(height, 1)))),
    ]


def normalize_boxes(
    boxes: Sequence[Sequence[int] | Sequence[float]], width: int, height: int
) -> list[list[int]]:
    return [normalize_box(b, width, height) for b in boxes]
