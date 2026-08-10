#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/.venv/bin/python"
"$PYTHON" scripts/summarize_eval.py mmlu_pro \
  --output-dir outputs/full/mmlu_pro \
  --eval-log logs/evalscope-mmlu_pro-full.log \
  --out env/mmlu_pro_full_summary.json
"$PYTHON" scripts/summarize_eval.py mmlu_redux \
  --output-dir outputs/full/mmlu_redux \
  --eval-log logs/evalscope-mmlu_redux-full.log \
  --out env/mmlu_redux_full_summary.json
"$PYTHON" scripts/finalize_report.py \
  --pro-summary env/mmlu_pro_full_summary.json \
  --redux-summary env/mmlu_redux_full_summary.json \
  --out REPORT.md
