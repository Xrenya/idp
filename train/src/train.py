#!/usr/bin/env python3
"""Multi-GPU LayoutLMv3 training (PyTorch DDP, CUDA only).

Example (2 GPUs):
  torchrun --nproc_per_node=2 src/train.py --config config.yaml

Single GPU:
  python src/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiLabelFocalLoss
from src.metrics import multilabel_metrics
from src.model import LayoutLMv3MultiLabel
from src.prepared_dataset import TARGET_LABELS, create_prepared_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LayoutLMv3 multi-label (DDP)")
    p.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    p.add_argument("--train-parquet", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="1=fine-tune LayoutLMv3; 2=freeze encoder, train attention aggregator only",
    )
    p.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Stage-1 checkpoint (best.pt) required for --stage 2",
    )
    p.add_argument("--local_rank", type=int, default=-1)  # legacy
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float,
    class_names: list[str],
    world_size: int,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch.pop("index", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if world_size > 1:
        gathered_p = [None] * world_size
        gathered_y = [None] * world_size
        dist.all_gather_object(gathered_p, probs)
        dist.all_gather_object(gathered_y, labels)
        if dist.get_rank() == 0:
            probs = torch.cat(gathered_p, dim=0)
            labels = torch.cat(gathered_y, dim=0)
        else:
            return {}

    return multilabel_metrics(labels, probs, threshold=threshold, class_names=class_names)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    rank,
    epoch,
    use_amp: bool,
) -> float:
    model.train()
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    n_steps = 0
    pbar = tqdm(loader, disable=not is_main(rank), desc=f"train epoch {epoch}")
    for batch in pbar:
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch.pop("index", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(**batch)["logits"]
            loss = loss_fn(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1
        if is_main(rank):
            pbar.set_postfix(loss=total_loss / n_steps)
    return total_loss / max(n_steps, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})

    rank, local_rank, world_size, device = setup_distributed()

    train_parquet = Path(
        args.train_parquet
        or train_cfg.get("train_parquet", "data")
    )

    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "outputs/layoutlmv3"))
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    model_name = train_cfg.get("model_name", "microsoft/layoutlmv3-base")
    target_labels = list(data_cfg.get("target_labels", TARGET_LABELS))
    epochs = int(args.epochs or train_cfg.get("epochs", 3))
    lr = float(args.lr or train_cfg.get("lr", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    threshold = float(cfg.get("output", {}).get("label_threshold", 0.5))
    use_amp = bool(train_cfg.get("amp", True))

    long_cfg = cfg.setdefault("long_document", {})
    stage = int(args.stage or train_cfg.get("stage", 1))
    train_cfg["stage"] = stage
    long_cfg["stage"] = stage
    if stage == 2:
        long_cfg["freeze_encoder"] = True

    if is_main(rank):
        print(f"train_parquet={train_parquet}")
        print(f"stage={stage}")
        print(f"world_size={world_size} device={device}")

    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    train_loader, val_loader = create_prepared_dataloaders(
        train_parquet=train_parquet,
        processor=processor,
        config=cfg,
        rank=rank if world_size > 1 else 0,
        world_size=world_size,
    )

    model = LayoutLMv3MultiLabel(
        model_name=model_name,
        num_labels=len(target_labels),
        dropout=float(train_cfg.get("dropout", 0.1)),
        chunk_size=int(long_cfg.get("chunk_size", 400)),
        chunk_stride=int(long_cfg.get("chunk_stride", 100)),
        max_chunks=int(long_cfg.get("max_chunks", 16)),
        stage=stage,
        freeze_encoder=(stage == 2),
    ).to(device)

    enc_ckpt = args.encoder_checkpoint or train_cfg.get("encoder_checkpoint")
    if stage == 2:
        if not enc_ckpt:
            raise SystemExit(
                "Stage 2 requires --encoder-checkpoint path/to/stage1/best.pt"
            )
        raw = model.module if isinstance(model, DDP) else model
        raw.load_stage1_checkpoint(enc_ckpt, strict=False)
        raw.freeze_encoder()
        if is_main(rank):
            print(f"Loaded Stage-1 encoder from {enc_ckpt}; overhead params only")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    loss_cfg = train_cfg.get("focal", {})
    loss_fn = MultiLabelFocalLoss(
        gamma=float(loss_cfg.get("gamma", 2.0)),
        alpha=loss_cfg.get("alpha", 0.25),
    ).to(device)

    raw_model = model.module if isinstance(model, DDP) else model
    if stage == 2:
        params = list(raw_model.overhead_parameters())
        lr = float(args.lr or train_cfg.get("stage2_lr", lr * 10))
    else:
        params = list(raw_model.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_macro = -1.0
    stage_tag = f"stage{stage}"
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, rank, epoch, use_amp
        )
        metrics = evaluate(
            model.module if isinstance(model, DDP) else model,
            val_loader,
            device,
            threshold,
            target_labels,
            world_size,
        )
        if is_main(rank):
            row = {"epoch": epoch, "stage": stage, "train_loss": train_loss, **metrics}
            printable = {
                k: v
                for k, v in row.items()
                if k not in {"per_class"}
            }
            print(json.dumps(printable, indent=2))
            history.append(row)
            (output_dir / f"history_{stage_tag}.json").write_text(
                json.dumps(history, indent=2)
            )

            state = (model.module if isinstance(model, DDP) else model).state_dict()
            torch.save(state, output_dir / f"last_{stage_tag}.pt")
            if stage == 1:
                torch.save(state, output_dir / "last.pt")
            macro = float(metrics.get("macro_f1", -1.0)) if metrics else -1.0
            if macro >= best_macro:
                best_macro = macro
                torch.save(state, output_dir / f"best_{stage_tag}.pt")
                if stage == 1:
                    torch.save(state, output_dir / "best.pt")
                if metrics:
                    (output_dir / f"best_metrics_{stage_tag}.json").write_text(
                        json.dumps(metrics, indent=2)
                    )

        if world_size > 1:
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
