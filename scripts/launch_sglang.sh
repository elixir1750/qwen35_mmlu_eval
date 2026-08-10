#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/model/Qwen3.5-4B}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
RANDOM_SEED="${RANDOM_SEED:-42}"
LOG_PATH="${SGLANG_LOG:-$PROJECT_DIR/logs/sglang.log}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_torchinductor_${SLURM_JOB_ID:-local}}"
CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.8}"

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export TORCHINDUCTOR_CACHE_DIR
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
exec "$PROJECT_DIR/.venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --context-length "$CONTEXT_LENGTH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --random-seed "$RANDOM_SEED" \
  --reasoning-parser qwen3 \
  >>"$LOG_PATH" 2>&1
