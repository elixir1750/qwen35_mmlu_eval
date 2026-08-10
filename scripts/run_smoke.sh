#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

export PORT="${PORT:-8000}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export SGLANG_LOG="${SGLANG_LOG:-$PROJECT_DIR/logs/sglang.log}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_torchinductor_smoke_${SLURM_JOB_ID:-local}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p "$PROJECT_DIR/outputs/smoke/mmlu_pro" "$PROJECT_DIR/outputs/smoke/mmlu_redux" "$PROJECT_DIR/logs"

echo "smoke_start=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

bash scripts/launch_sglang.sh &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > outputs/smoke/api_models.json 2>/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SGLang exited before readiness" >&2
    tail -n 200 "$SGLANG_LOG" >&2 || true
    exit 1
  fi
  sleep 5
  if [[ "$attempt" == 240 ]]; then
    echo "SGLang readiness timeout" >&2
    tail -n 200 "$SGLANG_LOG" >&2 || true
    exit 1
  fi
done

echo "models_endpoint_ready=$(date -Is)"
cat outputs/smoke/api_models.json
OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  API_SMOKE_OUTPUT=outputs/smoke/api_smoke.json \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  .venv/bin/python scripts/api_smoke.py | tee outputs/smoke/api_smoke_summary.json

GEN_CONFIG='{"max_tokens":32768,"temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":1.5,"repetition_penalty":1.0,"seed":42,"timeout":600,"retries":3,"retry_interval":2,"stream":false}'

echo "mmlu_pro_smoke_start=$(date -Is)"
.venv/bin/evalscope eval \
  --model Qwen/Qwen3.5-4B \
  --model-id Qwen-Qwen3.5-4B-sglang-0.5.10 \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --datasets mmlu_pro \
  --dataset-hub modelscope \
  --limit 3 \
  --eval-batch-size 4 \
  --generation-config "$GEN_CONFIG" \
  --seed 42 \
  --work-dir outputs/smoke/mmlu_pro \
  --no-timestamp \
  --collect-perf \
  2>&1 | tee logs/evalscope-mmlu-pro-smoke.log

echo "mmlu_redux_smoke_start=$(date -Is)"
.venv/bin/evalscope eval \
  --model Qwen/Qwen3.5-4B \
  --model-id Qwen-Qwen3.5-4B-sglang-0.5.10 \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --datasets mmlu_redux \
  --dataset-hub modelscope \
  --dataset-args '{"mmlu_redux":{"subset_list":["abstract_algebra"]}}' \
  --limit 40 \
  --eval-batch-size 4 \
  --generation-config "$GEN_CONFIG" \
  --seed 42 \
  --work-dir outputs/smoke/mmlu_redux \
  --no-timestamp \
  --collect-perf \
  2>&1 | tee logs/evalscope-mmlu-redux-smoke.log

echo "smoke_end=$(date -Is)"
