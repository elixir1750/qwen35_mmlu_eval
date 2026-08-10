#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "usage: MODEL_NAME=Qwen/Qwen3.5-<size> bash scripts/run_full_eval.sh mmlu_pro|mmlu_redux" >&2
  exit 2
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required; use scripts/run_benchmark.sh MODEL_NAME BENCHMARK_NAME" >&2
  exit 2
fi
RUN_MODE=full exec bash "$PROJECT_DIR/scripts/run_benchmark.sh" "$MODEL_NAME" "$1"
