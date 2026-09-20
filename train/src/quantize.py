"""Post-training quantization for LayoutLMv3MultiLabel (CUDA).

Default strategy (HF-style split)
--------------------------------
* **Encoder** — bitsandbytes ``Linear8bitLt`` (same as
  ``transformers.BitsAndBytesConfig(load_in_8bit=True)`` on the backbone).
* **Head** (attention aggregator + classifier) — kept in **FP32** on CUDA
  (small, accuracy-sensitive; not INT8).

``torch_dynamic`` is CPU-only and must not be compared to GPU FP32 latency.
"""

from __future__ import annotations

import copy
import io
import time
from typing import Any, Iterable, Literal

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig

QuantBackend = Literal["hf_split", "bitsandbytes", "torch_dynamic"]

# LayoutLMv3 indexes ``rel_pos_bias.weight`` directly (``.weight.t()[idx]``).
_LAYOUTLM_SKIP_LINEAR_SUBSTRINGS = (
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
    """Approximate resident weight size in MB (params, buffers, packed INT8)."""
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
        if hasattr(mod, "_packed_params"):
            try:
                w, bias = mod._packed_params._weight_bias()  # type: ignore[attr-defined]
                _add(w)
                if bias is not None:
                    _add(bias)
            except Exception:
                pass
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
    return any(s in name for s in _LAYOUTLM_SKIP_LINEAR_SUBSTRINGS)


def _replace_linear_bnb(
    module: nn.Module,
    *,
    threshold: float = 6.0,
    prefix: str = "",
) -> nn.Module:
    """Recursively replace ``nn.Linear`` with ``bitsandbytes.nn.Linear8bitLt``."""
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
    """HF BitsAndBytes-style INT8 on LayoutLMv3 encoder; FP32 aggregator + classifier.

    Equivalent to ``BitsAndBytesConfig(load_in_8bit=True)`` applied only to the
    backbone. The Stage-2 head stays full precision (separate from encoder INT8).
    """
    if device.type != "cuda":
        raise RuntimeError("hf_split quantization requires CUDA")
    # Document / validate config shape used by transformers.
    _ = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=threshold)

    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    if not hasattr(model, "encoder"):
        raise RuntimeError("Expected LayoutLMv3MultiLabel with .encoder")
    # INT8 LayoutLM only — leave aggregator + classifier as nn.Linear (FP32).
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
    """INT8 every eligible Linear (encoder + head) via bitsandbytes."""
    if device.type != "cuda":
        raise RuntimeError("bitsandbytes Linear8bitLt requires CUDA")

    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    _replace_linear_bnb(model, threshold=threshold)
    model.to(device)
    model.eval()
    return model


class _FP32Linear(nn.Module):
    """Drop-in Linear that is *not* ``nn.Linear`` so dynamic quant skips it."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "_FP32Linear":
        m = cls(linear.in_features, linear.out_features, linear.bias is not None)
        with torch.no_grad():
            m.weight.copy_(linear.weight)
            if linear.bias is not None and m.bias is not None:
                m.bias.copy_(linear.bias)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


def _protect_skip_linears(module: nn.Module, prefix: str = "") -> None:
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _skip_linear(full):
            setattr(module, name, _FP32Linear.from_linear(child))
        else:
            _protect_skip_linears(child, full)


def quantize_torch_dynamic(model: nn.Module) -> nn.Module:
    """CPU-only dynamic INT8 on eligible ``nn.Linear`` layers.

    Do **not** use this for GPU latency comparisons — kernels run on CPU.
    """
    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    _protect_skip_linears(model)
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized


def quantize_model(
    model: nn.Module,
    *,
    backend: QuantBackend = "hf_split",
    device: torch.device | None = None,
    threshold: float = 6.0,
) -> tuple[nn.Module, str]:
    """Return ``(quantized_model, backend_used)``."""
    device = device or torch.device("cuda")
    if backend == "hf_split":
        return quantize_hf_split(model, device, threshold=threshold), "hf_split"
    if backend == "bitsandbytes":
        return quantize_bitsandbytes(model, device, threshold=threshold), "bitsandbytes"
    if backend == "torch_dynamic":
        return quantize_torch_dynamic(model), "torch_dynamic"
    raise ValueError(f"Unknown backend: {backend}")


def _is_torch_dynamic_quant(model: nn.Module) -> bool:
    return any("quantized" in type(m).__module__ for m in model.modules())


@torch.inference_mode()
def benchmark_latency_ms(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    warmup: int = 5,
    runs: int = 20,
) -> dict[str, float]:
    """Average inference latency (ms) for one forward on a fixed batch."""
    model.eval()
    # torch.ao dynamic INT8 is CPU-only; keep CUDA path for bnb / hf_split.
    run_device = torch.device("cpu") if _is_torch_dynamic_quant(model) else device
    model.to(run_device)
    moved = {
        k: v.to(run_device) if torch.is_tensor(v) else v
        for k, v in batch.items()
        if k not in {"labels", "index"}
    }

    def _step() -> None:
        out = model(**moved)
        _ = out["logits"]
        if run_device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        _step()

    times: list[float] = []
    for _ in range(max(1, runs)):
        if run_device.type == "cuda":
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
        "device": str(run_device),
    }


@torch.inference_mode()
def collect_probs_labels(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model over a loader; return ``(probs [N,C], labels [N,C])``."""
    model.eval()
    run_device = torch.device("cpu") if _is_torch_dynamic_quant(model) else device
    model.to(run_device)

    all_probs, all_labels = [], []
    for batch in loader:
        labels = batch["labels"]
        feed = {
            k: v.to(run_device)
            for k, v in batch.items()
            if k not in {"labels", "index"} and torch.is_tensor(v)
        }
        logits = model(**feed)["logits"]
        all_probs.append(torch.sigmoid(logits.float()).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
