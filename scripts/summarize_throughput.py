#!/usr/bin/env python3
"""Summarize structured throughput JSON without parsing log text."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_runs(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith("readiness.json") or path.name == "dry-run.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_file"] = str(path)
        rows.extend({**case, "backend": data.get("backend"), "layer": data.get("layer"), "cache_mode": data.get("cache_mode"), "gpu": data.get("gpu"), "model_revision": data.get("model_revision"), "context_length": data.get("context_length"), "backend_versions": data.get("backend_versions")} for case in data.get("runs", []))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("env/throughput_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("THROUGHPUT_REPORT.md"))
    args = parser.parse_args()
    raw_dirs = [path.resolve() for path in args.raw_dir]
    rows: list[dict[str, Any]] = []
    for raw_dir in raw_dirs:
        rows.extend(load_runs(raw_dir))
    if not rows:
        raise SystemExit(f"no structured throughput JSON found in {raw_dirs}")
    sglang_single = {(r.get("cache_mode"), r["input_length_requested"], r["output_length_requested"]): r for r in rows if r["backend"] == "sglang" and r["layer"] == "single" and r["concurrency"] == 1}
    transformers_single = {(r.get("cache_mode"), r["input_length_requested"], r["output_length_requested"]): r for r in rows if r["backend"] == "transformers" and r["layer"] == "single"}
    speedups: list[dict[str, Any]] = []
    for key, sglang in sorted(sglang_single.items()):
        transformer = transformers_single.get(key)
        if transformer is None:
            continue
        s_out = sglang.get("output_tokens_per_second")
        t_out = transformer.get("output_tokens_per_second")
        s_req = sglang.get("request_per_second")
        t_req = transformer.get("request_per_second")
        speedups.append({
            "cache_mode": key[0], "input_length": key[1], "output_length": key[2],
            "sglang_output_tokens_per_second": s_out,
            "transformers_output_tokens_per_second": t_out,
            "speedup_output_tps": round(s_out / t_out, 6) if isinstance(s_out, (int, float)) and isinstance(t_out, (int, float)) and t_out else None,
            "speedup_request_tps": round(s_req / t_req, 6) if isinstance(s_req, (int, float)) and isinstance(t_req, (int, float)) and t_req else None,
            "same_input_output_length": bool(sglang.get("same_input_output_length") and transformer.get("same_input_output_length")),
        })
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_dirs": [str(path) for path in raw_dirs],
        "run_count": len(rows),
        "runs": rows,
        "speedups": speedups,
        "comparison_policy": "speedups are emitted only for matching input/output lengths and single-stream runs; no Transformers concurrent-service speedup is claimed",
        "transformers_online_service": {"available": False, "reason": "transformers serve --help fails before startup because the installed transformers 5.3.0 and huggingface_hub 1.26.1 CLI annotations are incompatible"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gpu = next((r.get("gpu") for r in rows if r.get("gpu")), {})
    lines = [
        "# Qwen3.5-4B Throughput Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        "- Model: `Qwen/Qwen3.5-4B`, the same local original safetensors checkpoint for both backends.",
        "- Both paths use BF16, the same token-ID prompts, requested output lengths, context length, seed 42, and no MTP/speculative decoding.",
        "- SGLang uses the OpenAI-compatible completions API. Transformers uses one persistent `model.generate` process; it is not an online concurrent server.",
        f"- GPU: {gpu.get('name', 'n/a')}; driver: {gpu.get('driver', 'n/a')}; raw structured data: `{', '.join(str(path) for path in raw_dirs)}`.",
        "",
        "## Fixed-length single-stream results",
        "",
        "| Backend | Cache | Input | Requested output | Completed/failed | Output tok/s | Total tok/s | E2E P50 ms | E2E P95 ms | Avg output tokens | P95 output tokens | Same lengths | Peak GPU MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row.get("layer") != "single":
            continue
        e2e = row.get("e2e_latency_ms") or {}
        output = row.get("actual_output_tokens") or {}
        lines.append(f"| {row.get('backend')} | {row.get('cache_mode')} | {row.get('input_length_requested')} | {row.get('output_length_requested')} | {row.get('completed_requests')}/{row.get('failed_requests')} | {fmt(row.get('output_tokens_per_second'))} | {fmt(row.get('total_tokens_per_second'))} | {fmt(e2e.get('median'))} | {fmt(e2e.get('p95'))} | {fmt(output.get('mean'))} | {fmt(output.get('p95'))} | {row.get('same_input_output_length')} | {fmt(row.get('peak_gpu_memory_mib'), 0)} |")
    lines.extend([
        "",
        "## SGLang online concurrency",
        "",
        "| Cache | Concurrency | Input | Output | Completed/failed | Request/s | Output tok/s | TTFT P50/P95 ms | TPOT P50/P95 ms | E2E P50/P95 ms | Same lengths |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        if row.get("backend") != "sglang" or row.get("layer") != "online":
            continue
        ttft, tpot, e2e = row.get("ttft_ms") or {}, row.get("tpot_ms") or {}, row.get("e2e_latency_ms") or {}
        lines.append(f"| {row.get('cache_mode')} | {row.get('concurrency')} | {row.get('input_length_requested')} | {row.get('output_length_requested')} | {row.get('completed_requests')}/{row.get('failed_requests')} | {fmt(row.get('request_per_second'))} | {fmt(row.get('output_tokens_per_second'))} | {fmt(ttft.get('median'))}/{fmt(ttft.get('p95'))} | {fmt(tpot.get('median'))}/{fmt(tpot.get('p95'))} | {fmt(e2e.get('median'))}/{fmt(e2e.get('p95'))} | {row.get('same_input_output_length')} |")
    lines.extend([
        "",
        "## Same-workload speedups",
        "",
        "| Cache | Input | Output | SGLang output tok/s | Transformers output tok/s | Output-tok speedup | Request speedup | Same lengths |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in speedups:
        lines.append(f"| {row['cache_mode']} | {row['input_length']} | {row['output_length']} | {fmt(row['sglang_output_tokens_per_second'])} | {fmt(row['transformers_output_tokens_per_second'])} | {fmt(row['speedup_output_tps'])}x | {fmt(row['speedup_request_tps'])}x | {row['same_input_output_length']} |")
    lines.extend([
        "",
        "## Reliability and limitations",
        "",
        "- Every case records completed/failed requests and explicitly checks actual input/output token counts. A `false` same-length value must not be used for request/s comparisons.",
        "- Direct Transformers single-stream calls report E2E latency but no TTFT/TPOT because there is no streaming endpoint in `model.generate`.",
        "- `transformers serve --help` was tested and failed during CLI construction due to the installed dependency mismatch; no concurrent Transformers service or concurrent speedup is claimed.",
        "- SGLang radix-cache `enabled` and `disabled` are separate runs selected by `--cache-mode`; MTP and speculative decoding are disabled.",
        "",
        "## Re-run",
        "",
        "```bash",
        "cd /home/lhzhang/qwen35_mmlu_eval",
        "bash scripts/run_throughput.sh --backend all --quick --cache-mode enabled",
        ".venv/bin/python scripts/summarize_throughput.py --raw-dir outputs/throughput/raw/<RUN_ID>",
        "```",
    ])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.out), "report": str(args.report), "runs": len(rows), "speedups": len(speedups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
