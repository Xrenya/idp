#!/usr/bin/env bash
# Multi-GPU training launcher (PyTorch DDP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-config.yaml}"

torchrun --nproc_per_node="${NPROC}" src/train.py \
  --config "${CONFIG}" \
  "$@"
