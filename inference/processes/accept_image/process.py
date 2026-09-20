"""Stage 1: accept an image path / file into the payload."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase

from processes.utils import get_pil_image, put_pil_image

__version__ = "0.1.0"


class Process(ProcessBase):
    def execute(
        self, input_payload: Payload, output_payload: Payload, context: Context
    ) -> None:
        meta = dict(input_payload.get_data("request") or {})
        image = get_pil_image(input_payload, "image")

        if image is None:
            path = meta.get("image_path")
            if not path:
                # CLI: -i image=path.png → File object without PIL yet
                obj = input_payload.get(name="image")
                if obj is not None and getattr(obj, "path", None) is not None:
                    path = obj.path
            if not path:
                raise ValueError(
                    "accept_image requires -i image=<path> or request.image_path"
                )
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path).convert("RGB")
            meta["image_path"] = str(path.resolve())
        else:
            image = image.convert("RGB")

        meta["width"], meta["height"] = image.size
        put_pil_image(output_payload, image, name="image")
        output_payload.add_dictionary(meta, name="request")
        context.logger.info(f"Accepted image {meta['width']}x{meta['height']}")
