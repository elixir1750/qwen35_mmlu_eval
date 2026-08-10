#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DRY_RUN=0
INCLUDE_9B="${INCLUDE_9B:-0}"
SUBMIT_SLURM="${SUBMIT_SLURM:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME="${RESUME:-0}"
RUN_MODE="${RUN_MODE:-full}"
MAX_TOKENS="${MAX_TOKENS:-}"
MODEL_FILTER=""
BENCHMARK_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --include-9b) INCLUDE_9B=1; shift ;;
    --exclude-9b) INCLUDE_9B=0; shift ;;
    --no-submit) SUBMIT_SLURM=0; shift ;;
    --skip-completed) SKIP_COMPLETED=1; shift ;;
    --resume) RESUME=1; shift ;;
    --run-mode) RUN_MODE="$2"; shift 2 ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --benchmarks) BENCHMARK_FILTER="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
usage: bash scripts/run_benchmark_matrix.sh [options]
  --dry-run              print all planned commands without running
  --include-9b           include 9B entries when suitable BF16 resources exist
  --exclude-9b           run the required 0.8B/2B/4B matrix only
  --no-submit            never submit Slurm jobs; run only on a local GPU
  --skip-completed       pass SKIP_COMPLETED=1 (default)
  --resume               resume incomplete run directories after manifest validation
  --run-mode smoke|full  select underlying run mode
  --models CSV           restrict model names
  --benchmarks CSV       restrict mmlu_pro,mmlu_redux
EOF
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mapfile -t ALL_MODELS < <("$PROJECT_DIR/.venv/bin/python" - <<'PY'
import json
for item in json.load(open("configs/models.json", encoding="utf-8"))["models"]:
    print(item["model_name"])
PY
)
mapfile -t ALL_BENCHMARKS < <("$PROJECT_DIR/.venv/bin/python" - <<'PY'
import json
for name in json.load(open("configs/benchmarks.json", encoding="utf-8"))["benchmarks"]:
    print(name)
PY
)

contains_csv() {
  local value="$1" needle="$2" item
  IFS=',' read -ra items <<< "$value"
  for item in "${items[@]}"; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}

models=()
for model in "${ALL_MODELS[@]}"; do
  [[ "$INCLUDE_9B" == "1" || "$model" != *Qwen3.5-9B* ]] || continue
  [[ -z "$MODEL_FILTER" ]] || contains_csv "$MODEL_FILTER" "$model" || continue
  models+=("$model")
done
benchmarks=()
for benchmark in "${ALL_BENCHMARKS[@]}"; do
  [[ -z "$BENCHMARK_FILTER" ]] || contains_csv "$BENCHMARK_FILTER" "$benchmark" || continue
  benchmarks+=("$benchmark")
done

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')" || gpu_count=0
fi
gres="$("$PROJECT_DIR/.venv/bin/python" -c 'import json; print(json.load(open("configs/resources.json"))["slurm"]["gres"])')"
cpus="$("$PROJECT_DIR/.venv/bin/python" -c 'import json; print(json.load(open("configs/resources.json"))["slurm"]["cpus_per_task"])')"
memory="$("$PROJECT_DIR/.venv/bin/python" -c 'import json; print(json.load(open("configs/resources.json"))["slurm"]["memory"])')"
walltime="$("$PROJECT_DIR/.venv/bin/python" -c 'import json; print(json.load(open("configs/resources.json"))["slurm"]["time"])')"

for model in "${models[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    tag="$("$PROJECT_DIR/.venv/bin/python" scripts/benchmark_config.py resolve "$model" --benchmark "$benchmark" | "$PROJECT_DIR/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["model"]["model_tag"])')"
    size="$("$PROJECT_DIR/.venv/bin/python" scripts/benchmark_config.py resolve "$model" --benchmark "$benchmark" | "$PROJECT_DIR/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["model"]["size"])')"
    run_id="matrix_${tag}_${benchmark}"
    command="RUN_MODE=$RUN_MODE SKIP_COMPLETED=$SKIP_COMPLETED RESUME=$RESUME RUN_ID=$run_id"
    [[ -n "$MAX_TOKENS" ]] && command+=" MAX_TOKENS=$MAX_TOKENS"
    command+=" bash scripts/run_benchmark.sh $model $benchmark"
    echo "$command"
    if [[ "$DRY_RUN" == "1" ]]; then
      continue
    fi
    if [[ "$gpu_count" -gt 0 ]]; then
      eval "$command"
    elif [[ "$SUBMIT_SLURM" == "1" && -x "$(command -v sbatch || true)" ]]; then
      safe_name="qwen35-${tag}-$benchmark"
      requested_gres="$gres"
      [[ "$size" == "9B" ]] && requested_gres="gpu:2"
      job_id="$(sbatch --parsable --gres="$requested_gres" --cpus-per-task="$cpus" --mem="$memory" --time="$walltime" --job-name="$safe_name" --output="$PROJECT_DIR/logs/slurm-$safe_name-%j.out" --wrap="cd $PROJECT_DIR && $command")"
      echo "submitted job_id=$job_id model=$model benchmark=$benchmark"
    else
      echo "no local GPU and Slurm submission unavailable; command not run: $command" >&2
      exit 1
    fi
  done
done
