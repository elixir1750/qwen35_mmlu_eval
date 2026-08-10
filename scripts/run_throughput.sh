#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
ORIGINAL_ARGS=("$@")
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

BACKEND=all
MODE=quick
PORT=8000
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/model/Qwen3.5-4B}"
CONTEXT_LENGTH=65536
CONCURRENCY=
INPUT_LENGTH=
OUTPUT_LENGTH=
REPEATS=3
CACHE_MODE=enabled
DRY_RUN=0
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"

usage() {
  sed -n '1,80p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --input-length) INPUT_LENGTH="$2"; shift 2 ;;
    --output-length) OUTPUT_LENGTH="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --cache-mode) CACHE_MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$BACKEND" in sglang|transformers|all) ;; *) echo "invalid --backend: $BACKEND" >&2; exit 2 ;; esac
case "$CACHE_MODE" in disabled|enabled) ;; *) echo "invalid --cache-mode: $CACHE_MODE" >&2; exit 2 ;; esac

if [[ "$MODE" == "full" ]]; then
  INPUT_LENGTH="${INPUT_LENGTH:-1024}"
  OUTPUT_LENGTH="${OUTPUT_LENGTH:-256,1024}"
  CONCURRENCY="${CONCURRENCY:-1,4,8,16}"
else
  INPUT_LENGTH="${INPUT_LENGTH:-256}"
  OUTPUT_LENGTH="${OUTPUT_LENGTH:-64}"
  CONCURRENCY="${CONCURRENCY:-1,4}"
fi

IFS=',' read -r -a INPUTS <<< "$INPUT_LENGTH"
IFS=',' read -r -a OUTPUTS <<< "$OUTPUT_LENGTH"
IFS=',' read -r -a CONCURRENCIES <<< "$CONCURRENCY"
INPUT_ARGS=(--input-length "${INPUTS[@]}")
OUTPUT_ARGS=(--output-length "${OUTPUTS[@]}")
CONCURRENCY_ARGS=(--concurrency "${CONCURRENCIES[@]}")

RAW_DIR="$PROJECT_DIR/outputs/throughput/raw/$RUN_ID"
LOG_DIR="$PROJECT_DIR/outputs/throughput/logs/$RUN_ID"
MANIFEST_DIR="$PROJECT_DIR/outputs/throughput/manifests"
mkdir -p "$RAW_DIR" "$LOG_DIR" "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/$RUN_ID.json"

SETTINGS_JSON="{\"mode\":\"$MODE\",\"port\":$PORT,\"model_path\":\"$MODEL_PATH\",\"context_length\":$CONTEXT_LENGTH,\"input_length\":\"$INPUT_LENGTH\",\"output_length\":\"$OUTPUT_LENGTH\",\"concurrency\":\"$CONCURRENCY\",\"repeats\":$REPEATS,\"cache_mode\":\"$CACHE_MODE\",\"mtp\":false,\"speculative_decoding\":false}"
FULL_COMMAND="bash scripts/run_throughput.sh ${ORIGINAL_ARGS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  dry_backends=("$BACKEND")
  [[ "$BACKEND" == "all" ]] && dry_backends=(sglang transformers)
  for dry_backend in "${dry_backends[@]}"; do
    .venv/bin/python scripts/throughput_benchmark.py --backend "$dry_backend" --layer online \
      --model-path "$MODEL_PATH" --port "$PORT" --context-length "$CONTEXT_LENGTH" \
      "${INPUT_ARGS[@]}" "${OUTPUT_ARGS[@]}" "${CONCURRENCY_ARGS[@]}" \
      --repeats "$REPEATS" --cache-mode "$CACHE_MODE" --output "$RAW_DIR/dry-run-${dry_backend}.json" --dry-run
  done
  echo "raw_dir=$RAW_DIR"
  echo "manifest=$MANIFEST"
  exit 0
fi

.venv/bin/python scripts/write_manifest.py start \
  --manifest "$MANIFEST" --model-path "$MODEL_PATH" --output-dir "$RAW_DIR" \
  --kind throughput --backend "$BACKEND" --command "$FULL_COMMAND" --settings "$SETTINGS_JSON" >/dev/null

SERVER_PID=""
cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    # SGLang may keep a streaming request alive after TERM. The PID is the
    # server launched by this script, so force only this owned process after
    # the bounded graceful-shutdown window.
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
finish() {
  local status=$?
  cleanup_server
  .venv/bin/python scripts/write_manifest.py finish --manifest "$MANIFEST" --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT INT TERM

common=(--model-path "$MODEL_PATH" --context-length "$CONTEXT_LENGTH" "${INPUT_ARGS[@]}" "${OUTPUT_ARGS[@]}" --repeats "$REPEATS" --cache-mode "$CACHE_MODE" --project-dir "$PROJECT_DIR")

run_sglang() {
  local cache_flag=0
  [[ "$CACHE_MODE" == "disabled" ]] && cache_flag=1
  export MODEL_PATH PORT CONTEXT_LENGTH
  export TP_SIZE="${TP_SIZE:-1}"
  export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
  export RANDOM_SEED=42
  export DISABLE_RADIX_CACHE="$cache_flag"
  export SGLANG_LOG="$LOG_DIR/sglang.log"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_throughput_torchinductor_${SLURM_JOB_ID:-local}}"
  bash scripts/launch_sglang.sh &
  SERVER_PID=$!
  for attempt in $(seq 1 180); do
    if .venv/bin/python scripts/check_api_ready.py --port "$PORT" > "$RAW_DIR/sglang_readiness.json" 2> "$LOG_DIR/readiness.err"; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SGLang exited before readiness" >&2
      tail -n 200 "$SGLANG_LOG" >&2 || true
      return 1
    fi
    sleep 5
    if [[ "$attempt" == 180 ]]; then
      echo "SGLang readiness timeout" >&2
      tail -n 200 "$SGLANG_LOG" >&2 || true
      return 1
    fi
  done
  .venv/bin/python scripts/throughput_benchmark.py --backend sglang --layer single --port "$PORT" --output "$RAW_DIR/sglang_single.json" "${common[@]}" 2>&1 | tee "$LOG_DIR/sglang_single.log"
  .venv/bin/python scripts/throughput_benchmark.py --backend sglang --layer online --port "$PORT" --output "$RAW_DIR/sglang_online.json" "${common[@]}" "${CONCURRENCY_ARGS[@]}" --requests "$([[ "$MODE" == full ]] && echo 20 || echo 8)" 2>&1 | tee "$LOG_DIR/sglang_online.log"
  cleanup_server
}

run_transformers() {
  # Direct model.generate is deliberately one persistent process and one stream.
  # There is no supported equivalent Transformers online server in this venv.
  .venv/bin/python scripts/throughput_benchmark.py --backend transformers --layer single --port "$PORT" --output "$RAW_DIR/transformers_single.json" "${common[@]}" --concurrency 1 2>&1 | tee "$LOG_DIR/transformers_single.log"
}

case "$BACKEND" in
  sglang) run_sglang ;;
  transformers) run_transformers ;;
  all)
    run_sglang
    run_transformers
    ;;
esac

echo "throughput_run_complete=$RAW_DIR"
