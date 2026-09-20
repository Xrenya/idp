"""Stage 2: detect page orientation and rotate using config threshold"""

from __future__ import annotations

from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase

from .orientation import OrientationResult, correct_orientation
from processes.utils import forward, get_pil_image, put_pil_image

__version__ = "0.1.0"


class Process(ProcessBase):
    def execute(
        self, input_payload: Payload, output_payload: Payload, context: Context
    ) -> None:
        cfg = context.config.get("config") or {}
        image = get_pil_image(input_payload, "image")
        if image is None:
            raise ValueError("orient stage requires payload 'image'")

        backend = cfg.get("backend", "osd")
        min_conf = float(cfg.get("min_osd_confidence", 1.5))
        validate = bool(cfg.get("validate_with_ocr_score", True))

        upright, result = correct_orientation(
            image,
            backend=backend,
            min_osd_confidence=min_conf,
            validate_with_ocr_score=validate,
        )
        if (
            result.backend in {"osd", "auto:osd"}
            and float(result.confidence) < min_conf
            and int(result.rotate_degrees) % 360 != 0
        ):
            upright = image.convert("RGB")
            result = OrientationResult(
                rotate_degrees=0,
                detected_orientation=result.detected_orientation,
                confidence=result.confidence,
                backend="osd_skipped_low_conf",
                script=result.script,
                raw=result.raw,
            )

        applied = int(result.rotate_degrees) % 360
        orient_meta = result.to_dict()
        orient_meta["min_osd_confidence"] = min_conf
        orient_meta["applied_rotation"] = applied

        put_pil_image(output_payload, upright, name="image")
        output_payload.add_dictionary(orient_meta, name="orientation")
        forward(input_payload, output_payload, "request")
        context.logger.info(
            f"Orientation backend={orient_meta.get('backend')} "
            f"conf={float(orient_meta.get('confidence') or 0.0):.2f} "
            f"rotate={applied} (threshold={min_conf:.2f})"
        )
