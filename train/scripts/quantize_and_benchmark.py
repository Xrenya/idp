#!/usr/bin/env python3
"""Quantize and benchmark a Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Subset
from transformers import LayoutLMv3Processor

from src.metrics import multilabel_metrics
from src.model import LayoutLMv3MultiLabel
from src.prepared_dataset import create_prepared_dataloaders, prepared_collate
from src.quantize import (
    benchmark_latency_ms,
    collect_probs_labels,
    count_parameters,
    model_size_mb,
    quantize_model,
    state_dict_size_mb,
)

DEFAULT_CHECKPOINT = "Xrenya/layoutlmv3_stage2"
DEFAULT_WEIGHT_FILE = "best_stage2.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="HF repo id (default) or local .pt path",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quantization"),
    )
    p.add_argument(
        "--backend",
        choices=["hf_split", "bitsandbytes"],
        default="hf_split",
        help=(
            "hf_split (default): INT8 encoder + FP32 head on CUDA; "
            "bitsandbytes: INT8 all Linear layers"
        ),
    )
    p.add_argument("--batch-size", type=int, default=16, help="Benchmark batch size")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=256,
        help="Val samples for macro-F1 comparison. Use 0 for full val.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_checkpoint(checkpoint: str) -> Path:
    """Resolve a local path or download the checkpoint from Hugging Face."""
    path = Path(checkpoint)
    if path.is_file():
        return path
    return Path(
        hf_hub_download(repo_id=checkpoint, filename=DEFAULT_WEIGHT_FILE)
    )


def load_stage2_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    train_cfg = cfg.get("train", {})
    long_cfg = cfg.get("long_document", {})
    data_cfg = cfg.get("data", {})
    labels = list(data_cfg.get("target_labels", ["letter", "form", "email", "resume"]))

    model = LayoutLMv3MultiLabel(
        model_name=str(train_cfg.get("model_name", "microsoft/layoutlmv3-base")),
        num_labels=len(labels),
        dropout=float(train_cfg.get("dropout", 0.1)),
        chunk_size=int(long_cfg.get("chunk_size", 400)),
        chunk_stride=int(long_cfg.get("chunk_stride", 100)),
        max_chunks=int(long_cfg.get("max_chunks", 16)),
        stage=2,
        freeze_encoder=True,
    )
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, labels, missing, unexpected


def build_val_loader(
    cfg: dict[str, Any],
    processor,
    *,
    batch_size: int,
    max_eval_samples: int,
    seed: int,
) -> DataLoader:
    train_cfg = cfg.get("train", {})
    train_parquet = train_cfg.get("train_parquet", "data")

    _, val_loader = create_prepared_dataloaders(
        train_parquet=train_parquet,
        processor=processor,
        config=cfg,
        rank=0,
        world_size=1,
    )
    ds = val_loader.dataset
    n = len(ds)
    if max_eval_samples and max_eval_samples > 0 and max_eval_samples < n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[: int(max_eval_samples)].tolist()
        ds = Subset(ds, idx)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=prepared_collate,
    )


def take_one_batch(loader: DataLoader) -> dict[str, torch.Tensor]:
    batch = next(iter(loader))
    if batch["input_ids"].shape[0] < loader.batch_size:
        raise RuntimeError(
            f"Need a full batch of {loader.batch_size} for latency benchmark; "
            f"got {batch['input_ids'].shape[0]}. Increase --max-eval-samples."
        )
    return batch


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for quantization / benchmarking")

    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")
    model_fp32, labels, missing, unexpected = load_stage2_model(
        cfg, ckpt_path, device
    )
    if missing:
        warnings.warn(f"Missing checkpoint keys: {missing[:6]}", stacklevel=2)
    if unexpected:
        warnings.warn(f"Unused checkpoint keys: {unexpected[:6]}", stacklevel=2)

    train_cfg = cfg.get("train", {})
    model_name = str(train_cfg.get("model_name", "microsoft/layoutlmv3-base"))
    try:
        processor = LayoutLMv3Processor.from_pretrained(
            model_name, apply_ocr=False, local_files_only=True
        )
    except Exception:
        processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)

    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("long_document", {})["mode_sampling"] = "head"
    cfg.setdefault("train", {})["stage"] = 2
    cfg.setdefault("long_document", {})["stage"] = 2

    loader = build_val_loader(
        cfg,
        processor,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )
    bench_batch = take_one_batch(loader)
    threshold = float(cfg.get("output", {}).get("label_threshold", 0.5))

    print("Benchmarking the FP32 model")
    fp_size = model_size_mb(model_fp32)
    fp_sd = state_dict_size_mb(model_fp32)
    fp_lat = benchmark_latency_ms(
        model_fp32,
        bench_batch,
        device,
        warmup=args.warmup,
        runs=args.runs,
    )
    fp_probs, fp_labels = collect_probs_labels(model_fp32, loader, device)
    fp_metrics = multilabel_metrics(
        fp_labels, fp_probs, threshold=threshold, class_names=labels
    )

    print(f"Quantizing with {args.backend}")
    model_int8, backend_used = quantize_model(
        model_fp32, backend=args.backend, device=device
    )
    q_size = model_size_mb(model_int8)
    q_sd = state_dict_size_mb(model_int8)
    q_lat = benchmark_latency_ms(
        model_int8,
        bench_batch,
        device,
        warmup=args.warmup,
        runs=args.runs,
    )
    q_probs, q_labels = collect_probs_labels(model_int8, loader, device)
    q_metrics = multilabel_metrics(
        q_labels, q_probs, threshold=threshold, class_names=labels
    )

    q_path = args.output_dir / f"best_stage2_int8_{backend_used}.pt"
    try:
        torch.save(
            {
                "backend": backend_used,
                "state_dict": model_int8.state_dict(),
                "source_checkpoint": args.checkpoint,
            },
            q_path,
        )
        print(f"Saved checkpoint: {q_path}")
    except Exception as exc:
        warnings.warn(f"Could not save the quantized checkpoint: {exc}", stacklevel=2)
        q_path = None

    size_red = 100.0 * (1.0 - q_size / max(fp_size, 1e-9))
    speedup = fp_lat["avg_ms"] / max(q_lat["avg_ms"], 1e-9)
    f1_drop = float(fp_metrics["macro_f1"] - q_metrics["macro_f1"])
    f1_drop_pct = 100.0 * f1_drop / max(abs(fp_metrics["macro_f1"]), 1e-9)

    report = {
        "checkpoint": args.checkpoint,
        "weights_path": str(ckpt_path),
        "backend": backend_used,
        "batch_size": args.batch_size,
        "max_eval_samples": args.max_eval_samples,
        "quantized_checkpoint": str(q_path) if q_path else None,
        "fp32": {
            "device": str(device),
            "num_params": count_parameters(model_fp32),
            "size_mb": fp_size,
            "state_dict_mb": fp_sd,
            "latency": fp_lat,
            "throughput_docs_per_s": 1000.0
            * args.batch_size
            / max(fp_lat["avg_ms"], 1e-9),
            "metrics": fp_metrics,
        },
        "int8": {
            "device": q_lat["device"],
            "num_params": count_parameters(model_int8),
            "size_mb": q_size,
            "state_dict_mb": q_sd,
            "latency": q_lat,
            "throughput_docs_per_s": 1000.0
            * args.batch_size
            / max(q_lat["avg_ms"], 1e-9),
            "metrics": q_metrics,
        },
        "tradeoff": {
            "size_reduction_pct": size_red,
            "speedup_x": speedup,
            "macro_f1_drop": f1_drop,
            "macro_f1_drop_pct": f1_drop_pct,
            "primary_metric": "macro_f1",
        },
    }

    json_path = args.output_dir / "quantization_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["tradeoff"], indent=2))
    print(f"Report: {json_path}")


if __name__ == "__main__":
    main()
