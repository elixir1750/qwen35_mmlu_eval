#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

usage() {
  echo "usage: $0 PRO_RUN_DIR REDUX_RUN_DIR [REPORT_OUT]" >&2
  echo "example: $0 outputs/full/mmlu_pro/<RUN_ID> outputs/full/mmlu_redux/<RUN_ID> REPORT_<RUN_ID>.md" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

PRO_DIR="$1"
REDUX_DIR="$2"
REPORT_OUT="${3:-REPORT_$(basename "$PRO_DIR")_$(basename "$REDUX_DIR").md}"
PRO_ID="$(basename "$PRO_DIR")"
REDUX_ID="$(basename "$REDUX_DIR")"
PRO_SUMMARY="env/mmlu_pro_${PRO_ID}_summary.json"
REDUX_SUMMARY="env/mmlu_redux_${REDUX_ID}_summary.json"
PYTHON="$PROJECT_DIR/.venv/bin/python"

[[ -d "$PRO_DIR" ]] || { echo "missing MMLU-Pro run directory: $PRO_DIR" >&2; exit 1; }
[[ -d "$REDUX_DIR" ]] || { echo "missing MMLU-Redux run directory: $REDUX_DIR" >&2; exit 1; }

"$PYTHON" scripts/summarize_eval.py mmlu_pro \
  --output-dir "$PRO_DIR" \
  --eval-log "logs/evalscope-mmlu_pro-full-${PRO_ID}.log" \
  --run-manifest "$PRO_DIR/run_manifest.json" \
  --out "$PRO_SUMMARY"
"$PYTHON" scripts/summarize_eval.py mmlu_redux \
  --output-dir "$REDUX_DIR" \
  --eval-log "logs/evalscope-mmlu_redux-full-${REDUX_ID}.log" \
  --run-manifest "$REDUX_DIR/run_manifest.json" \
  --out "$REDUX_SUMMARY"
"$PYTHON" scripts/finalize_report.py \
  --pro-summary "$PRO_SUMMARY" \
  --redux-summary "$REDUX_SUMMARY" \
  --out "$REPORT_OUT"
