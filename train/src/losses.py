from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | torch.Tensor | None = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        log_pt = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if self.alpha is None:
            alpha_t = 1.0
        elif self.alpha.ndim == 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        else:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha.view(1, -1) * targets + (1.0 - alpha.view(1, -1)) * (
                1.0 - targets
            )

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
