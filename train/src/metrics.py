"""Multi-label validation metrics (sklearn)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def multilabel_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Compute micro/macro/per-class F1 and Hamming loss."""
    yt = _to_numpy(y_true).astype(np.int32)
    yp = (_to_numpy(y_prob) >= threshold).astype(np.int32)
    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
        yp = yp.reshape(1, -1)

    n_classes = yt.shape[1]
    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(n_classes)]

    per_f1 = f1_score(yt, yp, average=None, zero_division=0)
    per_p = precision_score(yt, yp, average=None, zero_division=0)
    per_r = recall_score(yt, yp, average=None, zero_division=0)
    support = yt.sum(axis=0).astype(int)

    per_class = {
        name: {
            "f1": float(per_f1[i]),
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }

    return {
        "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(yt, yp)),
        "per_class_f1": {k: v["f1"] for k, v in per_class.items()},
        "per_class": per_class,
    }
