#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ $# -eq 1 && "$1" == "--help" ]]; then
  cat <<'EOF'
usage: bash scripts/run_benchmark.sh MODEL_NAME BENCHMARK_NAME
Environment overrides: RUN_MODE=smoke|full MODE=thinking|non_thinking|both RESUME=0|1
NO_AUTO_DOWNLOAD=0|1 PORT TP_SIZE EVAL_BATCH_SIZE MEM_FRACTION_STATIC CONTEXT_LENGTH
EOF
  exit 0
fi
if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/run_benchmark.sh MODEL_NAME BENCHMARK_NAME" >&2
  exit 2
fi
MODEL_NAME="$1"
BENCHMARK="$2"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing project Python: $PYTHON" >&2
  exit 2
fi

RUN_MODE="${RUN_MODE:-full}"
MODE_OVERRIDE="${MODE:-}"
RESUME="${RESUME:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
NO_AUTO_DOWNLOAD="${NO_AUTO_DOWNLOAD:-0}"
PORT_BASE="${PORT:-8000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-$("$PYTHON" -c 'import json; print(json.load(open("configs/resources.json"))["context_length"])')}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$("$PYTHON" -c 'import json; print(json.load(open("configs/resources.json"))["mem_fraction_static"])')}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_MODE="${CACHE_MODE:-enabled}"
TP_SIZE_OVERRIDE="${TP_SIZE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
MAX_TOKENS_OVERRIDE="${MAX_TOKENS:-}"
SGLANG_LOG_OVERRIDE="${SGLANG_LOG:-}"
EVAL_LOG_OVERRIDE="${EVAL_LOG:-}"

case "$RUN_MODE" in
  smoke|full) ;;
  *) echo "RUN_MODE must be smoke or full" >&2; exit 2 ;;
esac
case "$CACHE_MODE" in
  enabled|disabled) ;;
  *) echo "CACHE_MODE must be enabled or disabled" >&2; exit 2 ;;
esac
if [[ -n "$MAX_TOKENS_OVERRIDE" && "$MAX_TOKENS_OVERRIDE" -gt "$CONTEXT_LENGTH" ]]; then
  echo "MAX_TOKENS=$MAX_TOKENS_OVERRIDE exceeds CONTEXT_LENGTH=$CONTEXT_LENGTH; choose a compatible value" >&2
  exit 2
fi

resolve_json="$("$PYTHON" scripts/benchmark_config.py resolve "$MODEL_NAME" --benchmark "$BENCHMARK")"
model_tag="$(printf '%s' "$resolve_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["model"]["model_tag"])')"
checkpoint_type="$(printf '%s' "$resolve_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["model"]["checkpoint_type"])')"
size="$(printf '%s' "$resolve_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["model"]["size"])')"
default_path="$(printf '%s' "$resolve_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["model"]["local_path"])')"
expected_total="$(printf '%s' "$resolve_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["benchmark"]["expected_total"])')"

declare -a SETTINGS=()
if [[ "$checkpoint_type" == "base" ]]; then
  SETTINGS=(base)
elif [[ -z "$MODE_OVERRIDE" || "$MODE_OVERRIDE" == "both" ]]; then
  SETTINGS=(thinking non_thinking)
else
  case "$MODE_OVERRIDE" in
    thinking|non_thinking|non-thinking) SETTINGS=("${MODE_OVERRIDE//-/_}") ;;
    *) echo "MODE must be both, thinking, non_thinking, or non-thinking" >&2; exit 2 ;;
  esac
fi

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')" || gpu_count=0
fi
if [[ "$TP_SIZE_OVERRIDE" == "auto" ]]; then
  TP_SIZE=1
  [[ "$size" == "9B" && "$gpu_count" -ge 2 ]] && TP_SIZE=2
else
  TP_SIZE="$TP_SIZE_OVERRIDE"
fi

if [[ -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
  MODEL_PATH="$MODEL_PATH_OVERRIDE"
else
  MODEL_PATH="$PROJECT_DIR/$default_path"
fi
export MODEL_PATH SERVED_MODEL_NAME="$MODEL_NAME" TP_SIZE MEM_FRACTION_STATIC CONTEXT_LENGTH RANDOM_SEED
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

for index in "${!SETTINGS[@]}"; do
  SETTING="${SETTINGS[$index]}"
  if [[ "$TP_SIZE_OVERRIDE" == "auto" && "$size" != "9B" ]]; then
    export TP_SIZE=1
  else
    export TP_SIZE
  fi
  if [[ "$SETTING" == "thinking" ]]; then
    export REASONING_PARSER=qwen3
  else
    export REASONING_PARSER=""
  fi
  if [[ "$CACHE_MODE" == "disabled" ]]; then
    export DISABLE_RADIX_CACHE=1
  else
    export DISABLE_RADIX_CACHE=0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry-run: MODEL_NAME=$MODEL_NAME BENCHMARK=$BENCHMARK SETTING=$SETTING MODEL_PATH=$MODEL_PATH TP_SIZE=$TP_SIZE CONTEXT_LENGTH=$CONTEXT_LENGTH"
    continue
  fi

  run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_$SETTING}"
  if [[ "${#SETTINGS[@]}" -gt 1 && -n "${RUN_ID:-}" ]]; then
    run_id="${RUN_ID}_$SETTING"
  fi
  run_dir="$PROJECT_DIR/outputs/$RUN_MODE/$model_tag/$SETTING/$BENCHMARK/$run_id"
  if [[ "${#SETTINGS[@]}" -gt 1 && -n "$SGLANG_LOG_OVERRIDE" ]]; then
    server_log="${SGLANG_LOG_OVERRIDE}.${SETTING}"
  else
    server_log="${SGLANG_LOG_OVERRIDE:-$PROJECT_DIR/logs/${model_tag}-${SETTING}-${BENCHMARK}-${run_id}.log}"
  fi
  if [[ "${#SETTINGS[@]}" -gt 1 && -n "$EVAL_LOG_OVERRIDE" ]]; then
    eval_log="${EVAL_LOG_OVERRIDE}.${SETTING}"
  else
    eval_log="${EVAL_LOG_OVERRIDE:-$PROJECT_DIR/logs/evalscope-${model_tag}-${SETTING}-${BENCHMARK}-${run_id}.log}"
  fi
  manifest="$run_dir/run_manifest.json"
  mkdir -p "$run_dir" "$PROJECT_DIR/logs"

  if [[ -f "$manifest" && "$SKIP_COMPLETED" == "1" ]] && "$PYTHON" - "$manifest" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("status") == "finished" and data.get("exit_code") == 0 else 1)
PY
  then
    echo "skip completed run: $run_dir"
    continue
  fi
  if [[ -n "$(find "$run_dir" -mindepth 1 -print -quit 2>/dev/null)" && "$RESUME" != "1" ]]; then
    echo "refusing to reuse non-empty output directory without RESUME=1: $run_dir" >&2
    exit 2
  fi
  if [[ -f "$manifest" && "$RESUME" == "1" ]]; then
    revision=$("$PYTHON" - "$MODEL_NAME" <<'PY'
import json, sys
for item in json.load(open("configs/models.json", encoding="utf-8"))["models"]:
    if item["model_name"] == sys.argv[1]:
        print(item["revision"])
        break
PY
)
    "$PYTHON" scripts/validate_resume.py "$manifest" --model "$MODEL_NAME" --setting "$SETTING" --benchmark "$BENCHMARK" --revision "$revision"
  fi

  if [[ "$NO_AUTO_DOWNLOAD" == "1" ]]; then
    "$PYTHON" scripts/ensure_model.py "$MODEL_NAME" --path "$MODEL_PATH" --no-auto-download >/dev/null
  else
    "$PYTHON" scripts/ensure_model.py "$MODEL_NAME" --path "$MODEL_PATH" >/dev/null
  fi
  hash_file="$PROJECT_DIR/env/model_hashes/${model_tag}.json"
  generation_args=(generation "$MODEL_NAME" "$SETTING" --evalscope-json)
  if [[ -n "$MAX_TOKENS_OVERRIDE" ]]; then
    generation_args+=(--max-tokens "$MAX_TOKENS_OVERRIDE")
  fi
  generation_json=$("$PYTHON" scripts/benchmark_config.py "${generation_args[@]}")
  settings_json=$("$PYTHON" - "$generation_json" "$PORT_BASE" "$TP_SIZE" "$MEM_FRACTION_STATIC" "$CONTEXT_LENGTH" "$RANDOM_SEED" "$EVAL_BATCH_SIZE" "$CACHE_MODE" <<'PY'
import json, sys
generation=json.loads(sys.argv[1])
result={**generation, "port":int(sys.argv[2]), "tp_size":int(sys.argv[3]), "mem_fraction_static":float(sys.argv[4]), "context_length":int(sys.argv[5]), "random_seed":int(sys.argv[6]), "eval_batch_size":int(sys.argv[7]), "cache_mode":sys.argv[8], "backend":"sglang", "precision":"bfloat16", "mtp":False, "speculative_decoding":False}
print(json.dumps(result, ensure_ascii=False))
PY
)
  metadata_json=$("$PYTHON" - "$MODEL_NAME" "$model_tag" "$size" "$checkpoint_type" "$SETTING" "$BENCHMARK" "$hash_file" "$generation_json" "$run_dir" <<'PY'
import json, os, sys
model_name, model_tag, size, checkpoint_type, setting, benchmark, hash_file, generation, run_dir = sys.argv[1:]
hashes=json.load(open(hash_file, encoding="utf-8")) if os.path.exists(hash_file) else {}
print(json.dumps({
  "model_repo": model_name, "model_name": model_name, "model_tag": model_tag,
  "served_model_name": model_name, "evalscope_model_id": f"{model_tag}-{setting}-sglang",
  "model_size": size, "checkpoint_type": checkpoint_type, "setting": setting,
  "benchmark_name": benchmark, "model_revision": hashes.get("revision"),
  "model_hash_manifest": hashes, "generation_config": json.loads(generation),
  "reasoning_parser": "qwen3" if setting == "thinking" else None,
  "prompt_config": "EvalScope built-in standard adapter; see configs/benchmarks.json",
  "dataset_config_path": "configs/benchmarks.json", "resource_config_path": "configs/resources.json",
  "resource_selection_source": "local CUDA_VISIBLE_DEVICES" if os.environ.get("CUDA_VISIBLE_DEVICES") else "runtime nvidia-smi/Slurm",
  "run_directory": run_dir, "resume": os.environ.get("RESUME", "0") == "1"
}, ensure_ascii=False))
PY
)
  "$PYTHON" scripts/write_manifest.py start --manifest "$manifest" --model-path "$MODEL_PATH" \
    --output-dir "$run_dir" --kind "$RUN_MODE" --backend sglang --benchmark "$BENCHMARK" \
    --command "bash scripts/run_benchmark.sh $MODEL_NAME $BENCHMARK" --settings "$settings_json" --metadata "$metadata_json" >/dev/null

  export PORT="$((PORT_BASE + index))"
  export SGLANG_LOG="$server_log"
  export EVAL_LOG="$eval_log"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/qwen35_torchinductor_${SLURM_JOB_ID:-local}_${model_tag}_${SETTING}}"
  export SERVED_MODEL_NAME="$MODEL_NAME"
  bash scripts/launch_sglang.sh &
  server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  finish() {
    status=$?
    cleanup
    "$PYTHON" scripts/write_manifest.py finish --manifest "$manifest" --exit-code "$status" >/dev/null || true
    trap - EXIT INT TERM
    exit "$status"
  }
  trap finish EXIT INT TERM

  ready=0
  for attempt in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$run_dir/api_models.json" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "SGLang exited before readiness; log=$server_log" >&2
      tail -n 200 "$server_log" >&2 || true
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != "1" ]]; then
    echo "SGLang readiness timeout; log=$server_log" >&2
    tail -n 200 "$server_log" >&2 || true
    exit 1
  fi
  "$PYTHON" scripts/check_api_ready.py --port "$PORT" --model "$MODEL_NAME" --output "$run_dir/api_readiness.json"

  "$PYTHON" scripts/api_smoke.py --model "$MODEL_NAME" --mode "$SETTING" \
    --base-url "http://127.0.0.1:${PORT}/v1" --output "$run_dir/mode_smoke_raw.json" \
    --max-tokens "${MAX_TOKENS_OVERRIDE:-32768}" --seed "$RANDOM_SEED" | tee "$run_dir/mode_smoke.json"

  eval_args=(
    --model "$MODEL_NAME" --model-id "${model_tag}-${SETTING}-sglang" \
    --api-url "http://127.0.0.1:${PORT}/v1" --api-key EMPTY --datasets "$BENCHMARK" \
    --dataset-hub modelscope --eval-batch-size "$EVAL_BATCH_SIZE" \
    --generation-config "$generation_json" --seed "$RANDOM_SEED" --work-dir "$run_dir" \
    --no-timestamp --collect-perf
  )
  if [[ "$RUN_MODE" == "smoke" ]]; then
    if [[ "$BENCHMARK" == "mmlu_pro" ]]; then
      eval_args+=(--limit "${SMOKE_LIMIT_PRO:-3}")
    else
      eval_args+=(--limit "${SMOKE_LIMIT_REDUX:-30}")
    fi
    if [[ "$BENCHMARK" == "mmlu_redux" ]]; then
      eval_args+=(--dataset-args '{"mmlu_redux":{"subset_list":["abstract_algebra"]}}')
    fi
  fi
  echo "evalscope_command=evalscope eval ${eval_args[*]}" | tee "$eval_log"
  "$PROJECT_DIR/.venv/bin/evalscope" eval "${eval_args[@]}" 2>&1 | tee -a "$eval_log"
  summary_args=(
    "$BENCHMARK" --output-dir "$run_dir" --eval-log "$eval_log"
    --run-manifest "$manifest" --out "$run_dir/summary.json"
  )
  if [[ "$RUN_MODE" == "full" ]]; then
    summary_args+=(--expected-total "$expected_total")
  fi
  "$PYTHON" scripts/summarize_eval.py "${summary_args[@]}"
  cleanup
  "$PYTHON" scripts/write_manifest.py finish --manifest "$manifest" --exit-code 0 >/dev/null
  trap - EXIT INT TERM
done
