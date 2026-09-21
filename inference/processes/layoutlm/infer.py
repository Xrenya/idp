from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import LayoutLMv3Processor

from .boxes import normalize_boxes
from .model import LayoutLMv3MultiLabel
from .postprocess import prediction_to_json, word_chunk_spans

MODEL_CACHE: dict[str, Any] | None = None


def load_layoutlm_bundle(
    *,
    cfg: dict[str, Any],
    long_cfg: dict[str, Any],
    num_labels: int,
) -> dict[str, Any]:
    global MODEL_CACHE
    checkpoint = Path(str(cfg.get("checkpoint", "weights/layoutlm/best_stage2.pt")))
    device = torch.device(str(cfg.get("device", "cuda")))
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    model_name = str(cfg.get("model_name", "microsoft/layoutlmv3-base"))
    stage = int(cfg.get("stage", 2))
    model = LayoutLMv3MultiLabel(
        model_name=model_name,
        num_labels=num_labels,
        dropout=float(cfg.get("dropout", 0.1)),
        chunk_size=int(long_cfg.get("chunk_size", 400)),
        chunk_stride=int(long_cfg.get("chunk_stride", 100)),
        max_chunks=int(long_cfg.get("max_chunks", 16)),
        stage=stage,
        freeze_encoder=True,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    processor = LayoutLMv3Processor.from_pretrained(
        model_name, apply_ocr=False, local_files_only=True
    )

    MODEL_CACHE = {
        "model": model,
        "processor": processor,
        "checkpoint": str(checkpoint),
        "stage": stage,
        "device": device,
    }
    return MODEL_CACHE


def _encode_stage2_chunks(
    *,
    image: Image.Image,
    words: list[str],
    boxes: list[list[float]],
    processor,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
) -> dict[str, torch.Tensor]:
    """Encode word windows independently so each chunk has CLS at index 0."""
    width, height = image.size
    tokenizer = getattr(processor, "tokenizer", processor)
    spans = word_chunk_spans(
        words,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        max_chunks=max_chunks,
    )
    if not spans:
        spans = [(0, max(1, len(words)))]

    chunk_ids: list[torch.Tensor] = []
    chunk_mask_tok: list[torch.Tensor] = []
    chunk_bbox: list[torch.Tensor] = []
    pixel_values: torch.Tensor | None = None
    valid: list[float] = []

    for start, end in spans:
        w = words[start:end] or ["[UNK]"]
        b = boxes[start:end] or [
            [0.0, 0.0, float(max(width - 1, 1)), float(max(height - 1, 1))]
        ]
        norm_boxes = normalize_boxes(b, width, height)
        encoding = processor(
            image,
            w,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=chunk_size,
        )
        enc = {k: v.squeeze(0) for k, v in encoding.items()}
        chunk_ids.append(enc["input_ids"])
        chunk_mask_tok.append(enc["attention_mask"])
        chunk_bbox.append(enc["bbox"])
        if pixel_values is None:
            pixel_values = enc["pixel_values"]
        valid.append(1.0)

    assert pixel_values is not None
    return {
        "input_ids": torch.stack(chunk_ids, dim=0).unsqueeze(0),
        "attention_mask": torch.stack(chunk_mask_tok, dim=0).unsqueeze(0),
        "bbox": torch.stack(chunk_bbox, dim=0).unsqueeze(0),
        "pixel_values": pixel_values.unsqueeze(0),
        "chunk_mask": torch.tensor([valid], dtype=torch.float32),
    }


def predict_document(
    *,
    image: Image.Image,
    words: list[str],
    boxes: list[list[float]],
    cfg: dict[str, Any],
    long_cfg: dict[str, Any],
    target_labels: list[str],
    threshold: float,
    n_ocr_words: int | None = None,
) -> dict[str, Any]:
    chunk_size = int(long_cfg.get("chunk_size", 400))
    chunk_stride = int(long_cfg.get("chunk_stride", 100))
    max_chunks = int(long_cfg.get("max_chunks", 16))

    image = image.convert("RGB")
    width, height = image.size

    words = list(words)
    boxes = [list(map(float, b)) for b in boxes]
    if not words:
        words = ["[UNK]"]
        boxes = [[0.0, 0.0, float(max(width - 1, 1)), float(max(height - 1, 1))]]

    bundle = load_layoutlm_bundle(
        cfg=cfg,
        long_cfg=long_cfg,
        num_labels=len(target_labels),
    )
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]
    stage = int(bundle["stage"])

    if stage == 2:
        batch = _encode_stage2_chunks(
            image=image,
            words=words,
            boxes=boxes,
            processor=processor,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            max_chunks=max_chunks,
        )
        n_tokens = int(batch["attention_mask"].sum().item())
        n_chunks = int(batch["chunk_mask"].sum().item())
    else:
        # Stage 1: single 512-token window (head).
        max_tokens = int(long_cfg.get("max_tokens", 512))
        norm_boxes = normalize_boxes(boxes, width, height)
        encoding = processor(
            image,
            words,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_tokens,
        )
        batch = {k: v for k, v in encoding.items()}
        n_tokens = int(batch["attention_mask"].sum().item())
        n_chunks = 1

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        out = model(**batch)
        probs = torch.sigmoid(out["logits"][0]).detach().cpu().float().numpy()

    prediction = prediction_to_json(
        probs, target_names=target_labels, threshold=threshold
    )
    prediction["checkpoint"] = bundle["checkpoint"]
    prediction["stage"] = stage
    prediction["device"] = str(device)
    prediction["n_ocr_words"] = int(
        n_ocr_words if n_ocr_words is not None else len(words)
    )
    prediction["n_tokens"] = n_tokens
    prediction["n_chunks"] = n_chunks
    prediction["chunk_size"] = chunk_size
    prediction["chunk_stride"] = chunk_stride
    prediction["max_chunks"] = max_chunks
    return prediction
