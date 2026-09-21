from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

TARGET_LABELS = ["letter", "form", "email", "resume"]


def prediction_to_json(
    probs: Sequence[float] | np.ndarray | torch.Tensor,
    target_names: Sequence[str] = TARGET_LABELS,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Format thresholded class probabilities as JSON data."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().float().numpy()
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = [name for name, p in zip(target_names, probs) if float(p) >= threshold]
    return {
        "labels": labels,  # empty == none-of-the-above
        "scores": {name: float(p) for name, p in zip(target_names, probs)},
        "is_none": len(labels) == 0,
    }


def stage2_token_budget(
    *,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
    num_special_tokens: int = 2,
) -> int:
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
) -> list[tuple[int, int]]:
    """Deterministic head windows (inference / validation)."""
    max_chunks = max(1, int(max_chunks))
    if len(spans) <= max_chunks:
        return spans
    return spans[:max_chunks]


def word_chunk_spans(
    words: Sequence[str],
    *,
    tokenizer,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
    num_special_tokens: int = 2,
) -> list[tuple[int, int]]:
    """Stage-2 word windows matching the training long-doc loader."""
    if not words:
        return [(0, 0)]
    word_budget = max(1, int(chunk_size) - int(num_special_tokens))
    tok_lens = [max(1, len(tokenizer.tokenize(str(w)))) for w in words]
    spans = sliding_word_windows(
        tok_lens, budget=word_budget, stride=max(1, int(chunk_stride))
    )
    return select_word_windows(spans, max_chunks=max_chunks)
