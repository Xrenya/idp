#!/usr/bin/env bash
# Stage 1: fine-tune LayoutLMv3 encoder + overhead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-config.yaml}"
OUT="${OUT:-outputs/layoutlmv3}"

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" src/train.py \
    --config "${CONFIG}" \
    --stage 1 \
    --output-dir "${OUT}" \
    "$@"
else
  python src/train.py \
    --config "${CONFIG}" \
    --stage 1 \
    --output-dir "${OUT}" \
    "$@"
fi
