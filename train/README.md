# ADCS training

Self-contained package for **Stage 1 / Stage 2** LayoutLMv3 training, plus **INT8 quantization and benchmarking**.

## Layout

```
train/
  config.yaml
  requirements.txt
  data/                 prepared parquet shards (you create this)
  outputs/              checkpoints, quantization reports
  src/
    train.py            DDP training entrypoint
    model.py
    prepared_dataset.py
    losses.py / metrics.py / transforms_albu.py / ocr.py / quantize.py
  scripts/
    train_stage1.sh
    train_stage2.sh
    train_ddp.sh
    quantize.sh
    quantize_and_benchmark.py
```

## Setup

```bash
cd train
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put parquet shards under `data/prepared/train_fixed/` (or set `train.train_parquet` / `--train-parquet`):

```bash
mkdir -p data/prepared/train_fixed
# copy one or more *.parquet shards into that directory
```

## Stage 1 — fine-tune encoder

```bash
# 2 GPUs (default)
bash scripts/train_stage1.sh

# single GPU
NPROC=1 bash scripts/train_stage1.sh

# custom
NPROC=4 OUT=outputs/layoutlmv3 bash scripts/train_stage1.sh --epochs 5
```

Writes `outputs/layoutlmv3/best.pt` (and `best_stage1.pt`).
Train/val is always an 80/20 split of `train_parquet` (no separate val file).

## Stage 2 — freeze encoder, train attention aggregator

```bash
bash scripts/train_stage2.sh
# or: ENC=outputs/layoutlmv3/best.pt OUT=outputs/layoutlmv3_stage2 bash scripts/train_stage2.sh
```

Writes `outputs/layoutlmv3_stage2/best_stage2.pt`.

## Quantize + benchmark (CUDA / bitsandbytes)

```bash
bash scripts/quantize.sh
# or: CKPT=outputs/layoutlmv3_stage2/best_stage2.pt bash scripts/quantize.sh --batch-size 16 --max-eval-samples 256
```

Writes `outputs/quantization/` (INT8 weights + JSON report: size, latency, macro-F1).

## Manual entrypoints

```bash
python src/train.py --config config.yaml --stage 1 --output-dir outputs/layoutlmv3
torchrun --nproc_per_node=2 src/train.py --config config.yaml --stage 2 \
  --encoder-checkpoint outputs/layoutlmv3/best.pt \
  --output-dir outputs/layoutlmv3_stage2
python scripts/quantize_and_benchmark.py --checkpoint outputs/layoutlmv3_stage2/best_stage2.pt
```
