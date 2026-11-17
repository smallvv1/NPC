#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/data/CC120K}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/logs}"
MODEL_CACHE="${MODEL_CACHE:-${PROJECT_ROOT}/pre-trained_models}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/main_CLIP.py" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset cc \
  --num_anns "${NUM_ANNS:-1}" \
  --dev_length "${DEV_LENGTH:-1000}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --epochs "${EPOCHS:-5}" \
  --lr "${LR:-5e-7}" \
  --vision_model "${VISION_MODEL:-ViT-B/32}" \
  --noise_ratio "${NOISE_RATIO:-0}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --checkpoint_path "${CHECKPOINT_ROOT}" \
  --model_cache "${MODEL_CACHE}" \
  --experiments "${RUN_NAME}" \
  --name "${NAME:-CLIP-CC120K}" \
  --device "${DEVICE:-auto}" \
  "$@"
