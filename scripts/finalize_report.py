#!/usr/bin/env python3
"""Create the durable benchmark report after full EvalScope runs finish."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def delta(value: float | None, official: float) -> str:
    return "n/a" if value is None else f"{(value - official) * 100:+.2f} pp"


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pro-summary", type=Path, required=True)
    parser.add_argument("--redux-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project = args.project_dir
    config = json.loads((project / "env/eval_config.json").read_text(encoding="utf-8"))
    model_repo = config["model"]["repo"]
    pro = json.loads(args.pro_summary.read_text(encoding="utf-8"))
    redux = json.loads(args.redux_summary.read_text(encoding="utf-8"))
    out = args.out or (project / "REPORT.md")

    pro_local, redux_local = pro.get("accuracy"), redux.get("accuracy")
    max_delta = max(abs((pro_local or 0) - 0.791), abs((redux_local or 0) - 0.888))
    classification = "close reproduction" if max_delta <= 0.01 else "moderate discrepancy" if max_delta <= 0.02 else "large discrepancy"

    def diagnostics_table(name: str, summary: dict) -> str:
        output = summary.get("output_tokens") or {}
        reasoning = summary.get("reasoning_chars") or {}
        elapsed = summary.get("elapsed_seconds")
        if elapsed is None and summary.get("elapsed_source") == "monotonic_manifest":
            elapsed = summary.get("elapsed_seconds_from_log")
        return "\n".join([
            f"| {name} | {summary.get('total_samples')} | {summary.get('successful_generations')} | {summary.get('invalid_answers')} | {summary.get('truncated_generations')} | {summary.get('api_failures')} | {summary.get('missing_prediction_rows')} | {summary.get('duplicate_review_rows')} | {summary.get('retry_log_occurrences')} | {summary.get('dataset_timeout_occurrences')} | {fmt(elapsed)} | {fmt(summary.get('throughput_samples_per_second'))} | {fmt(output.get('mean'))} | {fmt(output.get('max'))} | {fmt(reasoning.get('mean'))} |",
        ])

    lines = [
        f"# {model_repo} MMLU Evaluation Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Environment",
        "",
        f"- GPU: {config['hardware']['gpu']} × {config['hardware']['gpu_count']}; VRAM per GPU: {config['hardware']['vram_gib']} GiB",
        f"- Driver/CUDA: {config['hardware']['driver']} / {config['hardware']['cuda']}; TP size: {config['hardware']['tp_size']}",
        f"- Python: {config['software']['python']}; PyTorch: {config['software']['torch']}",
        f"- SGLang: {config['software']['sglang']}; EvalScope: {config['software']['evalscope']}; OpenAI client: {config['software']['openai']}",
        "- Dependency index: Tsinghua PyPI mirror; model endpoint: `https://hf-mirror.com`.",
        "",
        "## Model",
        "",
        f"- Repository: `{config['model']['repo']}`",
        f"- Local path: `{config['model']['local_path']}`",
        f"- Revision: `{config['model']['revision']}`",
        f"- Model type: post-trained {model_repo}; original Hugging Face safetensors checkpoint; no quantization, GGUF/AWQ/GPTQ/MLX conversion, MTP, or speculative decoding.",
        "",
        "## Evaluation Protocol",
        "",
        f"- SGLang: `tp_size={config['sglang']['tp_size']}`, `mem_fraction_static={config['sglang']['mem_fraction_static']}`, `context_length={config['sglang']['context_length']}`, reasoning parser `{config['sglang']['reasoning_parser']}`, seed `{config['sglang']['random_seed']}`.",
        f"- Generation: temperature `{config['generation']['temperature']}`, top_p `{config['generation']['top_p']}`, top_k `{config['generation']['top_k']}`, presence_penalty `{config['generation']['presence_penalty']}`, repetition_penalty `{config['generation']['repetition_penalty']}`, max_tokens `{config['generation']['max_tokens']}`, timeout `{config['generation']['timeout_seconds']}s`, retries `{config['generation']['retries']}`.",
        f"- MMLU-Pro: `{config['benchmarks']['mmlu_pro']['dataset_id']}`, test split, 5-shot CoT, A–J, standard total {config['benchmarks']['mmlu_pro']['standard_total_samples']} samples.",
        f"- MMLU-Redux: `{config['benchmarks']['mmlu_redux']['dataset_id']}`, test split, 0-shot CoT, A–D, standard total {config['benchmarks']['mmlu_redux']['standard_total_samples']} samples.",
        "- Prompt and answer extraction use the installed EvalScope 1.10.0 built-in adapters; no benchmark prompt or ground truth was rewritten.",
        "",
        "## Results",
        "",
        "| Benchmark | Qwen official | Local result | Delta |",
        "|---|---:|---:|---:|",
        f"| MMLU-Pro | 79.1% | {pct(pro_local)} | {delta(pro_local, 0.791)} |",
        f"| MMLU-Redux | 88.8% | {pct(redux_local)} | {delta(redux_local, 0.888)} |",
        "",
        "Per-subset full results are stored in `env/mmlu_pro_full_summary.json` and `env/mmlu_redux_full_summary.json`; EvalScope native reports are under `outputs/full/*/reports/`.",
        "",
        "## Reliability diagnostics",
        "",
        "| Benchmark | Total | Successful | Invalid | Truncated | API failures | Missing predictions | Duplicate reviews | Retry log occurrences | Dataset timeouts | Elapsed (s) | Samples/s | Avg output tokens | Max output tokens | Avg reasoning chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        diagnostics_table("MMLU-Pro", pro),
        diagnostics_table("MMLU-Redux", redux),
        "",
        "The `*_diagnostics.json` files contain up to 10 wrong, invalid, truncated, and API-error samples for inspection. Smoke tests manually inspected six samples per benchmark; all had reasoning, final content, valid extracted answers, and `stop` termination.",
        "",
        "## Error inspection and interpretation",
        "",
        f"- The first standard-recipe smoke results were MMLU-Pro {pct(config['smoke_diagnostics']['mmlu_pro_accuracy'])} and MMLU-Redux {pct(config['smoke_diagnostics']['mmlu_redux_smoke_accuracy'])}; both had zero invalid, truncated, and API-error samples.",
        f"- This is explicitly a `{model_repo} + SGLang + EvalScope standard-recipe reproduction`, not an official exact-recipe claim. The model card does not publish every evaluator implementation detail needed to establish exact identity.",
        f"- Based on the largest full-evaluation gap, the result is classified as **{classification}**.",
        "- If the gap is material, the next variants should be recorded separately and checked in order: model/revision, thinking/parser behavior, dataset revision, shot/prompt template, extraction, max_tokens/truncation, then sampling randomness. The first standard result must remain unchanged.",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "cd /home/lhzhang/qwen35_mmlu_eval",
        "bash scripts/run_smoke.sh",
        "PORT=8000 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_pro",
        "PORT=8001 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_redux",
        "bash scripts/postprocess_full.sh outputs/full/mmlu_pro/<PRO_RUN_ID> outputs/full/mmlu_redux/<REDUX_RUN_ID> REPORT_<PRO_RUN_ID>_<REDUX_RUN_ID>.md",
        "```",
        "",
        "Key commands and all resolved settings are archived in `env/eval_config.json`; raw logs and JSONL outputs are retained under `logs/` and `outputs/`.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
