#!/usr/bin/env bash
# Stage 2: freeze encoder, train attention aggregator only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-config.yaml}"
OUT="${OUT:-outputs/layoutlmv3_stage2}"
ENC="${ENC:-outputs/layoutlmv3/best.pt}"

if [[ ! -f "${ENC}" ]]; then
  echo "Stage-1 checkpoint not found: ${ENC}" >&2
  echo "Train stage 1 first, or set ENC=/path/to/best.pt" >&2
  exit 1
fi

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" src/train.py \
    --config "${CONFIG}" \
    --stage 2 \
    --encoder-checkpoint "${ENC}" \
    --output-dir "${OUT}" \
    "$@"
else
  python src/train.py \
    --config "${CONFIG}" \
    --stage 2 \
    --encoder-checkpoint "${ENC}" \
    --output-dir "${OUT}" \
    "$@"
fi
