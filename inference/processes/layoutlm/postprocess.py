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


def truncate_words_for_stage2(
    words: list[str],
    boxes: list[list[float]],
    *,
    tokenizer,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
    num_special_tokens: int = 2,
) -> tuple[list[str], list[list[float]]]:
    n = len(words)
    if n == 0:
        return words, boxes

    budget = stage2_token_budget(
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        max_chunks=max_chunks,
        num_special_tokens=num_special_tokens,
    )
    tok_lens = [max(1, len(tokenizer.tokenize(str(w)))) for w in words]
    if int(sum(tok_lens)) <= budget:
        return words, boxes

    acc = 0
    end = 0
    while end < n and acc + tok_lens[end] <= budget:
        acc += tok_lens[end]
        end += 1
    if end == 0:
        end = 1
    return words[:end], boxes[:end]
