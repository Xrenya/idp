"""Post-training quantization for LayoutLMv3MultiLabel."""

from __future__ import annotations

import copy
import io
import time
from typing import Any, Iterable, Literal

import bitsandbytes as bnb
import torch
import torch.nn as nn

QuantBackend = Literal["hf_split", "bitsandbytes"]

LAYOUTLM_SKIP = (
    "rel_pos_bias",
    "rel_pos_x_bias",
    "rel_pos_y_bias",
)


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return int(t.storage().nbytes()) if t.storage() is not None else 0


def model_size_mb(model: nn.Module) -> float:
    """Approximate resident weight size in MB."""
    seen: set[int] = set()
    total = 0

    def _add(t: torch.Tensor) -> None:
        nonlocal total
        if not torch.is_tensor(t):
            return
        key = t.data_ptr()
        if key in seen or key == 0:
            return
        seen.add(key)
        total += _tensor_nbytes(t)

    for p in model.parameters(recurse=True):
        _add(p.data)
    for b in model.buffers(recurse=True):
        _add(b)
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if w is not None and hasattr(w, "CB") and getattr(w, "CB", None) is not None:
            _add(w.CB)
            if getattr(w, "SCB", None) is not None:
                _add(w.SCB)
    return total / (1024.0 * 1024.0)


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized state_dict footprint (closer to on-disk checkpoint size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024.0 * 1024.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _skip_linear(qualified_name: str) -> bool:
    name = qualified_name.lower()
    return any(s in name for s in LAYOUTLM_SKIP)


def _replace_linear_bnb(
    module: nn.Module,
    *,
    threshold: float = 6.0,
    prefix: str = "",
) -> nn.Module:
    """Replace eligible Linear layers with bitsandbytes Linear8bitLt modules."""
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and not isinstance(
            child, getattr(bnb.nn, "Linear8bitLt", type(None))
        ):
            if _skip_linear(full):
                continue
            has_bias = child.bias is not None
            new = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=threshold,
            )
            new.weight = bnb.nn.Int8Params(
                child.weight.data.detach().clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if has_bias:
                assert new.bias is not None
                new.bias.data = child.bias.data.detach().clone()
            setattr(module, name, new)
        else:
            _replace_linear_bnb(child, threshold=threshold, prefix=full)
    return module


def quantize_hf_split(
    model: nn.Module,
    device: torch.device,
    *,
    threshold: float = 6.0,
) -> nn.Module:
    """Quantize the LayoutLMv3 encoder and keep the stage-2 head in FP32."""
    if device.type != "cuda":
        raise RuntimeError("hf_split quantization requires CUDA")
    if not hasattr(model, "encoder"):
        raise RuntimeError("hf_split expects a model with an encoder")

    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    _replace_linear_bnb(model.encoder, threshold=threshold)
    model.to(device)
    model.eval()
    return model


def quantize_bitsandbytes(
    model: nn.Module,
    device: torch.device,
    *,
    threshold: float = 6.0,
) -> nn.Module:
    """Quantize every eligible Linear layer with bitsandbytes."""
    if device.type != "cuda":
        raise RuntimeError("bitsandbytes quantization requires CUDA")

    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    _replace_linear_bnb(model, threshold=threshold)
    model.to(device)
    model.eval()
    return model


def quantize_model(
    model: nn.Module,
    *,
    backend: QuantBackend = "hf_split",
    device: torch.device | None = None,
    threshold: float = 6.0,
) -> tuple[nn.Module, str]:
    """Return the quantized model and selected backend name."""
    device = device or torch.device("cuda")
    if backend == "hf_split":
        return quantize_hf_split(model, device, threshold=threshold), "hf_split"
    if backend == "bitsandbytes":
        return quantize_bitsandbytes(model, device, threshold=threshold), "bitsandbytes"
    raise ValueError(f"Unknown backend: {backend}")


@torch.inference_mode()
def benchmark_latency_ms(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    warmup: int = 5,
    runs: int = 20,
) -> dict[str, Any]:
    """Average inference latency (ms) for one forward on a fixed batch."""
    model.eval()
    model.to(device)
    moved = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
        if k not in {"labels", "index"}
    }

    def _step() -> None:
        out = model(**moved)
        _ = out["logits"]
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        _step()

    times: list[float] = []
    for _ in range(max(1, runs)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        times.append((time.perf_counter() - t0) * 1000.0)

    times_sorted = sorted(times)
    return {
        "avg_ms": float(sum(times) / len(times)),
        "std_ms": float(
            (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
        ),
        "min_ms": float(times_sorted[0]),
        "max_ms": float(times_sorted[-1]),
        "p50_ms": float(times_sorted[len(times_sorted) // 2]),
        "warmup": float(warmup),
        "runs": float(runs),
        "batch_size": float(next(iter(moved.values())).shape[0]),
        "device": str(device),
    }


@torch.inference_mode()
def collect_probs_labels(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect class probabilities and labels for a loader."""
    model.eval()
    model.to(device)

    all_probs, all_labels = [], []
    for batch in loader:
        labels = batch["labels"]
        feed = {
            k: v.to(device)
            for k, v in batch.items()
            if k not in {"labels", "index"} and torch.is_tensor(v)
        }
        logits = model(**feed)["logits"]
        all_probs.append(torch.sigmoid(logits.float()).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
