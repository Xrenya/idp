"""Shared payload helpers for Podder processes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from podder_task_foundation import Payload
from podder_task_foundation.objects import Object


def get_pil_image(payload: Payload, name: str = "image") -> Image.Image | None:
    obj = payload.get(name=name)
    if obj is None:
        return None
    data = obj.data
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    path = getattr(obj, "path", None)
    if path is not None:
        return Image.open(path).convert("RGB")
    if isinstance(data, str):
        p = Path(data)
        if p.is_file():
            return Image.open(p).convert("RGB")
    return None


def put_pil_image(payload: Payload, image: Image.Image, name: str = "image") -> None:
    payload.add(Object(data=image.convert("RGB"), name=name), name=name)


def forward(input_payload: Payload, output_payload: Payload, *names: str) -> None:
    for name in names:
        obj = input_payload.get(name=name)
        if obj is not None:
            output_payload.add(obj, name=name)
