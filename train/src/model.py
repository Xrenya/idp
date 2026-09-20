"""Two-stage LayoutLMv3 classifier for short and chunked documents."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model


class AttentionAggregator(nn.Module):
    """Pool chunk CLS embeddings with learned attention weights."""

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.query = nn.Parameter(torch.randn(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk_embs: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(chunk_embs))
        scores = torch.matmul(h, self.query)
        scores = scores.masked_fill(chunk_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.sum(chunk_embs * weights.unsqueeze(-1), dim=1)


def sliding_windows(seq_len: int, chunk_size: int, stride: int) -> list[tuple[int, int]]:
    if seq_len <= chunk_size:
        return [(0, seq_len)]
    spans = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        spans.append((start, end))
        if end >= seq_len:
            break
        start += stride
    return spans


class LayoutLMv3MultiLabel(nn.Module):
    """Run flat classification in stage 1 and chunk aggregation in stage 2."""

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        num_labels: int = 4,
        dropout: float = 0.1,
        *,
        chunk_size: int = 400,
        chunk_stride: int = 100,
        max_chunks: int = 16,
        stage: int = 1,
        freeze_encoder: bool | None = None,
    ) -> None:
        super().__init__()
        self.encoder = LayoutLMv3Model.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.hidden_size = hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

        self.chunk_size = int(chunk_size)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = int(max_chunks)
        self.stage = int(stage)
        self.aggregator = (
            AttentionAggregator(hidden, dropout=dropout) if self.stage == 2 else None
        )

        if freeze_encoder is None:
            freeze_encoder = self.stage == 2
        self._encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def agg_parameters(self):
        """Yield the parameters trained in stage 2."""
        if self.aggregator is None:
            raise RuntimeError("No aggregator: build model with stage=2")
        yield from self.aggregator.parameters()
        yield from self.classifier.parameters()
        yield from self.dropout.parameters()

    def load_stage1_checkpoint(self, path: str, strict: bool = False) -> None:
        """Load stage-1 weights; aggregator weights may be absent."""
        state = torch.load(path, map_location="cuda", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=strict)
        missing = [key for key in missing if not key.startswith("aggregator.")]
        if missing:
            warnings.warn(f"Missing checkpoint keys: {missing[:8]}", stacklevel=2)
        if unexpected:
            warnings.warn(f"Unused checkpoint keys: {unexpected[:8]}", stacklevel=2)

    def encode(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values,
        **kwargs,
    ) -> torch.Tensor:
        if self._encoder_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    **kwargs,
                )
                return outputs.last_hidden_state[:, 0].detach()
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            **kwargs,
        )
        return outputs.last_hidden_state[:, 0]

    def extract_chunk_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 2:
            input_ids, attention_mask, bbox, chunk_mask = self._make_chunks(
                input_ids, attention_mask, bbox
            )
        elif chunk_mask is None:
            chunk_mask = (attention_mask.sum(dim=-1) > 0).to(attention_mask.dtype)

        b, c, length = input_ids.shape
        flat_ids = input_ids.reshape(b * c, length)
        flat_mask = attention_mask.reshape(b * c, length)
        flat_bbox = bbox.reshape(b * c, length, 4)
        flat_pixels = (
            pixel_values.unsqueeze(1)
            .expand(-1, c, -1, -1, -1)
            .reshape(b * c, *pixel_values.shape[1:])
        )
        cls = self.encode(flat_ids, flat_mask, flat_bbox, flat_pixels, **kwargs)
        return cls.view(b, c, -1), chunk_mask

    def forward_overhead(
        self,
        chunk_embs: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> dict:
        pooled = self.aggregator(chunk_embs, chunk_mask)
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits, "pooled": pooled}

    def _make_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _ = input_ids.shape
        device = input_ids.device
        lengths = attention_mask.sum(dim=1).tolist()
        per_spans: list[list[tuple[int, int]]] = []
        max_c = 1
        for n in lengths:
            spans = sliding_windows(int(n), self.chunk_size, self.chunk_stride)[
                : self.max_chunks
            ]
            per_spans.append(spans)
            max_c = max(max_c, len(spans))

        chunk_ids, chunk_attn, chunk_bbox, chunk_valid = [], [], [], []
        for i, spans in enumerate(per_spans):
            ids_i, mask_i, bbox_i, valid_i = [], [], [], []
            for start, end in spans:
                win = end - start
                pad = self.chunk_size - win
                ids = input_ids[i, start:end]
                m = attention_mask[i, start:end]
                bb = bbox[i, start:end]
                if pad > 0:
                    ids = torch.nn.functional.pad(ids, (0, pad), value=0)
                    m = torch.nn.functional.pad(m, (0, pad), value=0)
                    bb = torch.nn.functional.pad(bb, (0, 0, 0, pad), value=0)
                ids_i.append(ids)
                mask_i.append(m)
                bbox_i.append(bb)
                valid_i.append(1)
            while len(ids_i) < max_c:
                ids_i.append(torch.zeros(self.chunk_size, dtype=input_ids.dtype, device=device))
                mask_i.append(
                    torch.zeros(self.chunk_size, dtype=attention_mask.dtype, device=device)
                )
                bbox_i.append(torch.zeros(self.chunk_size, 4, dtype=bbox.dtype, device=device))
                valid_i.append(0)
            chunk_ids.append(torch.stack(ids_i))
            chunk_attn.append(torch.stack(mask_i))
            chunk_bbox.append(torch.stack(bbox_i))
            chunk_valid.append(
                torch.tensor(valid_i, device=device, dtype=attention_mask.dtype)
            )
        return (
            torch.stack(chunk_ids),
            torch.stack(chunk_attn),
            torch.stack(chunk_bbox),
            torch.stack(chunk_valid),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        bbox=None,
        pixel_values=None,
        labels=None,
        chunk_mask=None,
        chunk_embs=None,
        **kwargs,
    ):
        if input_ids is None and chunk_embs is None:
            raise ValueError("Need input_ids or chunk_embs")

        if chunk_embs is not None:
            return self.forward_overhead(chunk_embs, chunk_mask)

        if self.stage == 1:
            pooled = self.encode(
                input_ids, attention_mask, bbox, pixel_values, **kwargs
            )
            logits = self.classifier(self.dropout(pooled))
            return {"logits": logits}

        feats, cmask = self.extract_chunk_features(
            input_ids,
            attention_mask,
            bbox,
            pixel_values,
            chunk_mask=chunk_mask,
            **kwargs,
        )
        return self.forward_overhead(feats, cmask)


def build_model_from_config(config: dict, num_labels: int = 4) -> LayoutLMv3MultiLabel:
    train_cfg = config.get("train", {})
    long_cfg = config.get("long_document", {})
    stage = int(train_cfg.get("stage", long_cfg.get("stage", 1)))
    return LayoutLMv3MultiLabel(
        model_name=train_cfg.get("model_name", "microsoft/layoutlmv3-base"),
        num_labels=num_labels,
        dropout=float(train_cfg.get("dropout", 0.1)),
        chunk_size=int(long_cfg.get("chunk_size", 400)),
        chunk_stride=int(long_cfg.get("chunk_stride", 100)),
        max_chunks=int(long_cfg.get("max_chunks", 16)),
        stage=stage,
        freeze_encoder=bool(long_cfg.get("freeze_encoder", stage == 2)),
    )
