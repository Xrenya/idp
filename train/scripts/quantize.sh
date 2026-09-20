#!/usr/bin/env bash
# Post-training INT8 quantization + size / latency / accuracy benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-config.yaml}"
CKPT="${CKPT:-Xrenya/layoutlmv3_stage2}"
OUT="${OUT:-outputs/quantization}"

python scripts/quantize_and_benchmark.py \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}" \
  "$@"
