#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-}"
BENCHMARK="${BENCHMARK:-mmlu_pro}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "MODEL_NAME is required; use RUN_MODE=smoke bash scripts/run_benchmark.sh MODEL_NAME BENCHMARK_NAME" >&2
  exit 2
fi
RUN_MODE=smoke exec bash "$PROJECT_DIR/scripts/run_benchmark.sh" "$MODEL_NAME" "$BENCHMARK"
