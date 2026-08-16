#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/liang_wu/syy/MVP/models/Qwen3-VL-2B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-teacher}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18008}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES

exec /opt/conda/envs/model_deploy/bin/vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code
