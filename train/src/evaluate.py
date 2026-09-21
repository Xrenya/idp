#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

from src.metrics import multilabel_metrics
from src.model import LayoutLMv3MultiLabel
from src.prepared_dataset import TARGET_LABELS, create_prepared_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--train-parquet", type=str, default=None)
    p.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
    )
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=64,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float,
    class_names: list[str],
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="eval"):
        labels = batch.pop("labels").to(device)
        batch.pop("index", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return multilabel_metrics(
        labels, probs, threshold=threshold, class_names=class_names
    )


def load_checkpoint(model: LayoutLMv3MultiLabel, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})
    long_cfg = cfg.setdefault("long_document", {})

    stage = int(args.stage)
    train_cfg["stage"] = stage
    long_cfg["stage"] = stage
    if stage == 2:
        long_cfg["freeze_encoder"] = True

    if args.max_eval_samples is not None:
        train_cfg["max_val_samples"] = int(args.max_eval_samples)
    if args.batch_size is not None:
        data_cfg["batch_size"] = int(args.batch_size)

    device = torch.device("cuda")

    train_parquet = Path(
        args.train_parquet or train_cfg.get("train_parquet", "data")
    )
    model_name = train_cfg.get("model_name", "microsoft/layoutlmv3-base")
    target_labels = list(data_cfg.get("target_labels", TARGET_LABELS))
    threshold = float(cfg.get("output", {}).get("label_threshold", 0.5))

    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    _, val_loader = create_prepared_dataloaders(
        train_parquet=train_parquet,
        processor=processor,
        config=cfg,
        rank=0,
        world_size=1,
    )

    model = LayoutLMv3MultiLabel(
        model_name=model_name,
        num_labels=len(target_labels),
        dropout=float(train_cfg.get("dropout", 0.0)),
        chunk_size=int(long_cfg.get("chunk_size", 400)),
        chunk_stride=int(long_cfg.get("chunk_stride", 100)),
        max_chunks=int(long_cfg.get("max_chunks", 16)),
        stage=stage,
        freeze_encoder=(stage == 2),
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    metrics = run_eval(model, val_loader, device, threshold, target_labels)
    printable = {k: v for k, v in metrics.items() if k != "per_class"}
    print(json.dumps(printable, indent=2))

    path = Path(args.checkpoint)
    out_path = Path(
        args.output
        if args.output
        else path.with_name(f"eval_metrics_stage{stage}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "checkpoint": str(path),
        "train_parquet": str(train_parquet),
        "n_eval": int(len(val_loader.dataset)),
        "threshold": threshold,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
