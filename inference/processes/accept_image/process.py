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

        path = meta.get("image_path")

        meta["image_path"] = str(path)
        meta["width"], meta["height"] = image.size
        put_pil_image(output_payload, image, name="image")
        output_payload.add_dictionary(meta, name="request")
        context.logger.info(f"Accepted image {meta['width']}x{meta['height']}")
