from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.ocr import normalize_boxes
from src.transforms_albu import apply_transforms, transforms_from_config

TARGET_LABELS = ["letter", "form", "email", "resume"]
Mode = Literal["head", "random_window", "random_boxes"]


def stage2_token_budget(
    *,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
    num_special_tokens: int = 2,
) -> int:
    """Max subword tokens a Stage-2 sample can cover across sliding chunks."""
    chunk_size = int(chunk_size)
    chunk_stride = int(chunk_stride)
    max_chunks = max(1, int(max_chunks))
    if max_chunks == 1:
        cover = chunk_size
    else:
        cover = chunk_size + (max_chunks - 1) * chunk_stride
    return max(1, cover - int(num_special_tokens))


def sliding_word_windows(
    tok_lens: Sequence[int],
    *,
    budget: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Contiguous word spans whose subword counts fit in ``budget`` each."""
    n = len(tok_lens)
    if n == 0:
        return [(0, 0)]
    budget = max(1, int(budget))
    stride = max(1, int(stride))
    if int(sum(tok_lens)) <= budget:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        acc = 0
        end = start
        while end < n and acc + int(tok_lens[end]) <= budget:
            acc += int(tok_lens[end])
            end += 1
        if end == start:
            end = start + 1
        spans.append((start, end))
        if end >= n:
            break
        advanced = 0
        new_start = start
        while new_start < end and advanced < stride:
            advanced += int(tok_lens[new_start])
            new_start += 1
        if new_start <= start:
            new_start = start + 1
        start = new_start
    return spans


def select_word_windows(
    spans: list[tuple[int, int]],
    *,
    max_chunks: int,
    training: bool,
) -> list[tuple[int, int]]:
    max_chunks = max(1, int(max_chunks))
    if len(spans) <= max_chunks:
        return spans
    if training:
        offset = random.randint(0, len(spans) - max_chunks)
        return spans[offset : offset + max_chunks]
    return spans[:max_chunks]


def sample_words_boxes_for_context(
    words: list[str],
    boxes: list[list[float]],
    *,
    tokenizer,
    max_length: int,
    mode: Mode = "random_window",
    training: bool = True,
    num_special_tokens: int = 2,
) -> tuple[list[str], list[list[float]]]:
    n = len(words)
    if n == 0:
        return words, boxes

    budget = max(1, int(max_length) - int(num_special_tokens))
    tok_lens = [max(1, len(tokenizer.tokenize(str(w)))) for w in words]
    total = int(sum(tok_lens))
    if total <= budget:
        return words, boxes

    def take_contiguous(start: int) -> tuple[list[str], list[list[float]]]:
        acc = 0
        end = start
        while end < n and acc + tok_lens[end] <= budget:
            acc += tok_lens[end]
            end += 1
        if end == start:
            end = start + 1
        return words[start:end], boxes[start:end]

    if (not training) or mode == "head":
        return take_contiguous(0)

    if mode == "random_boxes":
        order = list(range(n))
        random.shuffle(order)
        chosen: list[int] = []
        acc = 0
        for i in order:
            if acc + tok_lens[i] <= budget:
                chosen.append(i)
                acc += tok_lens[i]
            if acc >= budget:
                break
        if not chosen:
            chosen = [order[0]]
        chosen.sort(key=lambda i: (boxes[i][1], boxes[i][0], i))
        return [words[i] for i in chosen], [boxes[i] for i in chosen]

    max_start = n - 1
    start = random.randint(0, max_start)
    return take_contiguous(start)


def decode_image(value: Any) -> Image.Image:
    raw = value["bytes"] if isinstance(value, dict) else value
    return Image.open(io.BytesIO(raw)).convert("RGB")


def train_val_index_split(
    n: int,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """A deterministic train and validation index split"""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n).tolist()
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val
    train_idx = sorted(perm[:n_train])
    val_idx = sorted(perm[n_train:])
    return train_idx, val_idx


class PreparedParquetDataset(Dataset):
    def __init__(
        self,
        parquet_dir: str | Path,
        processor,
        *,
        target_labels: Sequence[str] = TARGET_LABELS,
        transform=None,
        max_length: int = 512,
        mode_sampling: Mode = "random_window",
        training: bool = True,
        max_samples: int | None = None,
        indices: Sequence[int] | None = None,
        stage: int = 1,
        chunk_size: int = 400,
        chunk_stride: int = 100,
        max_chunks: int = 16,
    ) -> None:
        path = Path(parquet_dir)
        self.files = sorted(p for p in path.glob("*.parquet") if p.stat().st_size > 0)
        counts = [int(pq.ParquetFile(f).metadata.num_rows) for f in self.files]
        self.cumulative = np.cumsum([0] + counts).tolist()
        self.n_total = int(self.cumulative[-1])
        if self.n_total == 0:
            raise RuntimeError(f"Empty parquet source: {parquet_dir}")

        if indices is not None:
            self.indices = [int(i) for i in indices]
        else:
            self.indices = list(range(self.n_total))
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.target_labels = list(target_labels)
        self.transform = transform
        self.max_length = int(max_length)
        self.mode_sampling: Mode = mode_sampling
        self.training = bool(training)
        self.stage = int(stage)
        self.chunk_size = int(chunk_size)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = max(1, int(max_chunks))

        self._cache_shard: int | None = None
        self._cache_table = None

    def __len__(self) -> int:
        return len(self.indices)

    def _locate(self, global_idx: int) -> tuple[int, int]:
        shard = int(np.searchsorted(self.cumulative, global_idx, side="right") - 1)
        local = global_idx - self.cumulative[shard]
        return shard, local

    def _get_row(self, global_idx: int) -> dict[str, Any]:
        shard, local = self._locate(global_idx)
        if self._cache_shard != shard:
            self._cache_table = pl.read_parquet(self.files[shard])
            self._cache_shard = shard
        return self._cache_table.row(local, named=True)

    def _labels_tensor(self, row: dict[str, Any]) -> torch.FloatTensor:
        labels = row.get("labels")
        if labels is not None and len(labels) == len(self.target_labels):
            return torch.tensor(list(labels), dtype=torch.float32)
        name = row.get("rvl_label_name")
        vec = [0.0] * len(self.target_labels)
        if name in self.target_labels:
            vec[self.target_labels.index(name)] = 1.0
        return torch.tensor(vec, dtype=torch.float32)

    def _prepare_words_boxes(
        self, row: dict[str, Any]
    ) -> tuple[Image.Image, list[str], list[list[float]]]:
        image = decode_image(row["image"])
        width, height = image.size

        words = list(row.get("words") or [])
        boxes = [list(map(float, b)) for b in (row.get("boxes") or [])]
        clipped: list[list[float]] = []
        kept_words: list[str] = []
        for w, b in zip(words, boxes):
            x0, y0, x1, y1 = b
            x0 = float(np.clip(x0, 0, width - 1))
            y0 = float(np.clip(y0, 0, height - 1))
            x1 = float(np.clip(x1, x0 + 1, width))
            y1 = float(np.clip(y1, y0 + 1, height))
            if not str(w).strip():
                continue
            clipped.append([x0, y0, x1, y1])
            kept_words.append(str(w))

        arr = np.asarray(image, dtype=np.uint8)
        if self.transform is not None:
            arr, clipped, kept_words = apply_transforms(
                arr, clipped, kept_words, self.transform
            )
        image = Image.fromarray(arr)
        width, height = image.size

        if not kept_words:
            kept_words = ["[UNK]"]
            clipped = [[0.0, 0.0, float(max(width - 1, 1)), float(max(height - 1, 1))]]
        return image, kept_words, clipped

    def _encode_stage1(
        self, image: Image.Image, words: list[str], boxes: list[list[float]]
    ) -> dict[str, torch.Tensor]:
        width, height = image.size
        words, boxes = sample_words_boxes_for_context(
            words,
            boxes,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode=self.mode_sampling,
            training=self.training,
        )
        norm_boxes = normalize_boxes(boxes, width, height)
        encoding = self.processor(
            image,
            words,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

    def _encode_stage2(
        self, image: Image.Image, words: list[str], boxes: list[list[float]]
    ) -> dict[str, torch.Tensor]:
        width, height = image.size
        word_budget = max(1, self.chunk_size - 2)
        stride_budget = max(1, self.chunk_stride)
        tok_lens = [max(1, len(self.tokenizer.tokenize(str(w)))) for w in words]
        spans = sliding_word_windows(
            tok_lens, budget=word_budget, stride=stride_budget
        )
        spans = select_word_windows(
            spans, max_chunks=self.max_chunks, training=self.training
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
            encoding = self.processor(
                image,
                w,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.chunk_size,
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
            "input_ids": torch.stack(chunk_ids, dim=0),
            "attention_mask": torch.stack(chunk_mask_tok, dim=0),
            "bbox": torch.stack(chunk_bbox, dim=0),
            "pixel_values": pixel_values,
            "chunk_mask": torch.tensor(valid, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(self.indices[idx])
        image, words, boxes = self._prepare_words_boxes(row)
        if self.stage == 2:
            item = self._encode_stage2(image, words, boxes)
        else:
            item = self._encode_stage1(image, words, boxes)
        item["labels"] = self._labels_tensor(row)
        item["index"] = int(row.get("index", self.indices[idx]))
        return item


def _pad_chunk_dim(tensor: torch.Tensor, max_c: int, pad_value: int | float = 0) -> torch.Tensor:
    c = int(tensor.shape[0])
    if c == max_c:
        return tensor
    if c > max_c:
        return tensor[:max_c]
    pad_shape = (max_c - c, *tensor.shape[1:])
    pad = tensor.new_full(pad_shape, pad_value)
    return torch.cat([tensor, pad], dim=0)


def prepared_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    first = batch[0]["input_ids"]
    if first.dim() == 2:
        max_c = max(int(b["input_ids"].shape[0]) for b in batch)
        out: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "bbox", "chunk_mask"):
            out[key] = torch.stack(
                [_pad_chunk_dim(b[key], max_c, pad_value=0) for b in batch], dim=0
            )
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)
        out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
        return out

    keys = [k for k in batch[0].keys() if k != "index"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return out


def create_prepared_dataloaders(
    *,
    train_parquet: str | Path,
    processor,
    config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    train_cfg = config.get("train", {})
    long_cfg = config.get("long_document", {})

    stage = int(train_cfg.get("stage", long_cfg.get("stage", 1)))
    max_length = int(long_cfg.get("max_tokens", 512))
    chunk_size = int(long_cfg.get("chunk_size", 400))
    chunk_stride = int(long_cfg.get("chunk_stride", 100))
    max_chunks = int(long_cfg.get("max_chunks", 16))
    mode_sampling = long_cfg.get("mode_sampling", "random_window")
    target_labels = data_cfg.get("target_labels", TARGET_LABELS)
    seed = int(data_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.2))

    common = dict(
        processor=processor,
        target_labels=target_labels,
        max_length=max_length,
        stage=stage,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        max_chunks=max_chunks,
    )

    probe = PreparedParquetDataset(
        train_parquet,
        transform=None,
        mode_sampling="head",
        training=False,
        **common,
    )
    train_idx, val_idx = train_val_index_split(
        probe.n_total, val_ratio=val_ratio, seed=seed
    )
    if train_cfg.get("max_train_samples") is not None:
        train_idx = train_idx[: int(train_cfg["max_train_samples"])]
    if train_cfg.get("max_val_samples") is not None:
        val_idx = val_idx[: int(train_cfg["max_val_samples"])]

    train_ds = PreparedParquetDataset(
        train_parquet,
        transform=transforms_from_config(aug_cfg, train=True),
        mode_sampling=mode_sampling,
        training=True,
        indices=train_idx,
        **common,
    )
    val_ds = PreparedParquetDataset(
        train_parquet,
        transform=transforms_from_config(aug_cfg, train=False),
        mode_sampling="head",
        training=False,
        indices=val_idx,
        **common,
    )
    if rank == 0:
        if stage == 2:
            cover = stage2_token_budget(
                chunk_size=chunk_size,
                chunk_stride=chunk_stride,
                max_chunks=max_chunks,
            )
            print(
                f"Dataset split ({train_parquet}): "
                f"{len(train_ds)} train, {len(val_ds)} validation, seed {seed}"
            )
            print(
                f"Stage 2 long-doc loader: chunk_size={chunk_size}, "
                f"stride={chunk_stride}, max_chunks={max_chunks}, "
                f"~{cover} token cover"
            )
        else:
            print(
                f"Dataset split ({train_parquet}): "
                f"{len(train_ds)} train, {len(val_ds)} validation, seed {seed}"
            )

    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=prepared_collate,
        drop_last=world_size > 1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=prepared_collate,
    )
    return train_loader, val_loader
