from __future__ import annotations

from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase

from .infer import predict_document
from processes.utils import forward, get_pil_image

__version__ = "0.1.0"


class Process(ProcessBase):
    def execute(
        self, input_payload: Payload, output_payload: Payload, context: Context
    ) -> None:
        cfg = context.config.get("config") or {}
        long_cfg = context.shared_config.get("long_document") or {}
        output_cfg = context.shared_config.get("output") or {}
        labels = list(cfg.get("target_labels") or ["letter", "form", "email", "resume"])
        threshold = float(
            cfg.get("label_threshold") or output_cfg.get("label_threshold", 0.5)
        )

        image = get_pil_image(input_payload, "image")
        ocr = input_payload.get_data("ocr") or {}
        if image is None:
            raise ValueError("layoutlm stage requires payload 'image'")

        prediction = predict_document(
            image=image,
            words=list(ocr.get("words") or []),
            boxes=list(ocr.get("boxes") or []),
            cfg=cfg,
            long_cfg=long_cfg,
            target_labels=labels,
            threshold=threshold,
            n_ocr_words=ocr.get("n_words"),
        )

        output_payload.add_dictionary(prediction, name="prediction")
        forward(input_payload, output_payload, "request", "orientation", "ocr", "image")
        context.logger.info(
            f"Prediction labels={prediction['labels']} is_none={prediction['is_none']} "
            f"(stage={prediction.get('stage')} ckpt={prediction.get('checkpoint')})"
        )
