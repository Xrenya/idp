"""Stage 3: OCR words, boxes for LayoutLM input"""

from __future__ import annotations

from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase

from .ocr import run_ocr
from processes.utils import forward, get_pil_image, put_pil_image

__version__ = "0.1.0"


class Process(ProcessBase):
    def execute(
        self, input_payload: Payload, output_payload: Payload, context: Context
    ) -> None:
        cfg = context.config.get("config") or {}
        image = get_pil_image(input_payload, "image")

        words, boxes = run_ocr(
            image,
            lang=str(cfg.get("lang", "eng")),
            min_conf=float(cfg.get("min_conf", 0.0)),
            normalize=False,
            dpi=int(cfg.get("dpi", 200)),
            max_side=int(cfg.get("max_side", 1280)),
            fast=bool(cfg.get("fast", True)),
        )

        width, height = image.size
        ocr_meta = {
            "n_words": len(words),
            "width": width,
            "height": height,
            "box_space": "pixel",
            "engine": "pytesseract",
            "words": words,
            "boxes": boxes,
        }
        put_pil_image(output_payload, image, name="image")
        output_payload.add_dictionary(ocr_meta, name="ocr")
        forward(input_payload, output_payload, "request", "orientation")
        context.logger.info(f"OCR extracted {len(words)} words")
