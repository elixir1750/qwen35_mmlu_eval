#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

BENCHMARK="${1:?usage: run_full_eval.sh mmlu_pro|mmlu_redux}"
case "$BENCHMARK" in
  mmlu_pro|mmlu_redux) ;;
  *) echo "unsupported benchmark: $BENCHMARK" >&2; exit 2 ;;
esac

export PORT="${PORT:-8000}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export CACHE_MODE="${CACHE_MODE:-enabled}"
export DISABLE_RADIX_CACHE=0
[[ "$CACHE_MODE" == "disabled" ]] && export DISABLE_RADIX_CACHE=1
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
export RUN_DIR="${RUN_DIR:-$PROJECT_DIR/outputs/full/$BENCHMARK/$RUN_ID}"
export SGLANG_LOG="${SGLANG_LOG:-$PROJECT_DIR/logs/sglang-full-${BENCHMARK}-${RUN_ID}.log}"
export EVAL_LOG="${EVAL_LOG:-$PROJECT_DIR/logs/evalscope-${BENCHMARK}-full-${RUN_ID}.log}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_torchinductor_full_${BENCHMARK}_${SLURM_JOB_ID:-local}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

FULL_DIR="$RUN_DIR"
if [[ -d "$FULL_DIR" ]] && find "$FULL_DIR" -mindepth 1 -print -quit | grep -q . && [[ "${RESUME:-0}" != "1" ]]; then
  echo "refusing to reuse non-empty output directory without RESUME=1: $FULL_DIR" >&2
  exit 2
fi
mkdir -p "$FULL_DIR" "$PROJECT_DIR/logs"

MANIFEST="$FULL_DIR/run_manifest.json"
SETTINGS_JSON="{\"model\":\"Qwen/Qwen3.5-4B\",\"port\":$PORT,\"tp_size\":1,\"mem_fraction_static\":$MEM_FRACTION_STATIC,\"context_length\":$CONTEXT_LENGTH,\"random_seed\":$RANDOM_SEED,\"eval_batch_size\":$EVAL_BATCH_SIZE,\"max_tokens\":32768,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"presence_penalty\":1.5,\"repetition_penalty\":1.0,\"timeout_seconds\":600,\"retries\":3,\"stream\":false,\"cache_mode\":\"$CACHE_MODE\",\"disable_radix_cache\":$DISABLE_RADIX_CACHE}"
.venv/bin/python scripts/write_manifest.py start \
  --manifest "$MANIFEST" --model-path "$PROJECT_DIR/model/Qwen3.5-4B" \
  --output-dir "$FULL_DIR" --kind evaluation --backend sglang --benchmark "$BENCHMARK" \
  --command "bash scripts/run_full_eval.sh $BENCHMARK" --settings "$SETTINGS_JSON" >/dev/null

echo "full_start=$(date -Is)"
echo "benchmark=$BENCHMARK"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=$(hostname)"
echo "port=$PORT"
echo "eval_batch_size=$EVAL_BATCH_SIZE"
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
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$FULL_DIR/api_models.json" 2>/dev/null; then
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
  > "$FULL_DIR/api_readiness.json"; then
  echo "SGLang generation readiness failed" >&2
  tail -n 200 "$SGLANG_LOG" >&2 || true
  exit 1
fi

echo "models_endpoint_ready=$(date -Is)"
cat "$FULL_DIR/api_models.json"

GEN_CONFIG='{"max_tokens":32768,"temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":1.5,"repetition_penalty":1.0,"seed":42,"timeout":600,"retries":3,"retry_interval":2,"stream":false}'
CACHE_ARGS=()
if [[ "${USE_CACHE:-0}" == "1" ]]; then
  CACHE_ARGS=(--use-cache "$FULL_DIR")
fi

echo "eval_start=$(date -Is)"
.venv/bin/evalscope eval \
  --model Qwen/Qwen3.5-4B \
  --model-id Qwen-Qwen3.5-4B-sglang-0.5.10 \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --datasets "$BENCHMARK" \
  --dataset-hub modelscope \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --generation-config "$GEN_CONFIG" \
  --seed 42 \
  --work-dir "$FULL_DIR" \
  --no-timestamp \
  --collect-perf \
  "${CACHE_ARGS[@]}" \
  2>&1 | tee "$EVAL_LOG"

echo "full_end=$(date -Is)"
