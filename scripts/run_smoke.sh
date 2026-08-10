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
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
export SMOKE_DIR="${SMOKE_DIR:-$PROJECT_DIR/outputs/smoke/$RUN_ID}"
export SGLANG_LOG="${SGLANG_LOG:-$PROJECT_DIR/logs/sglang-smoke-${RUN_ID}.log}"
export EVAL_LOG_PRO="${EVAL_LOG_PRO:-$PROJECT_DIR/logs/evalscope-mmlu-pro-smoke-${RUN_ID}.log}"
export EVAL_LOG_REDUX="${EVAL_LOG_REDUX:-$PROJECT_DIR/logs/evalscope-mmlu-redux-smoke-${RUN_ID}.log}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_torchinductor_smoke_${SLURM_JOB_ID:-local}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
if [[ -d "$SMOKE_DIR" ]] && find "$SMOKE_DIR" -mindepth 1 -print -quit | grep -q . && [[ "${RESUME:-0}" != "1" ]]; then
  echo "refusing to reuse non-empty smoke directory without RESUME=1: $SMOKE_DIR" >&2
  exit 2
fi
mkdir -p "$SMOKE_DIR/mmlu_pro" "$SMOKE_DIR/mmlu_redux" "$PROJECT_DIR/logs"

MANIFEST="$SMOKE_DIR/run_manifest.json"
SETTINGS_JSON="{\"model\":\"Qwen/Qwen3.5-4B\",\"port\":$PORT,\"tp_size\":1,\"mem_fraction_static\":$MEM_FRACTION_STATIC,\"context_length\":$CONTEXT_LENGTH,\"random_seed\":42,\"max_tokens\":32768,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"presence_penalty\":1.5,\"repetition_penalty\":1.0}"
.venv/bin/python scripts/write_manifest.py start \
  --manifest "$MANIFEST" --model-path "$PROJECT_DIR/model/Qwen3.5-4B" \
  --output-dir "$SMOKE_DIR" --kind smoke --backend sglang \
  --command "bash scripts/run_smoke.sh" --settings "$SETTINGS_JSON" >/dev/null

echo "smoke_start=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

bash scripts/launch_sglang.sh &
SERVER_PID=$!
cleanup() {
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    return
  fi
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
finish() {
  local status=$?
  cleanup
  .venv/bin/python scripts/write_manifest.py finish --manifest "$MANIFEST" --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT INT TERM

for attempt in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$SMOKE_DIR/api_models.json" 2>/dev/null; then
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

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B","prompt":"ready","max_tokens":4,"temperature":0,"stream":false}' \
  > "$SMOKE_DIR/api_readiness.json"; then
  echo "SGLang generation readiness failed" >&2
  tail -n 200 "$SGLANG_LOG" >&2 || true
  exit 1
fi

echo "models_endpoint_ready=$(date -Is)"
cat "$SMOKE_DIR/api_models.json"
OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  API_SMOKE_OUTPUT="$SMOKE_DIR/api_smoke.json" \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  .venv/bin/python scripts/api_smoke.py | tee "$SMOKE_DIR/api_smoke_summary.json"

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
  --work-dir "$SMOKE_DIR/mmlu_pro" \
  --no-timestamp \
  --collect-perf \
  2>&1 | tee "$EVAL_LOG_PRO"

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
  --work-dir "$SMOKE_DIR/mmlu_redux" \
  --no-timestamp \
  --collect-perf \
  2>&1 | tee "$EVAL_LOG_REDUX"

echo "smoke_end=$(date -Is)"
