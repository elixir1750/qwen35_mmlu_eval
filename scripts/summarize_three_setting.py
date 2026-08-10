#!/usr/bin/env python3
"""Build the auditable Base/Thinking/Non-Thinking MMLU matrix summary."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_config import ROOT, load_benchmarks, load_models, resolve_benchmark


HISTORICAL = {
    ("qwen35_4b_posttrained", "thinking", "mmlu_pro"): {
        "correct": 9417, "total": 12032, "invalid_answers": 4, "truncated_generations": 0,
        "api_failures": 0, "elapsed_seconds": None, "provenance": "historical_reused",
        "source": "env/mmlu_pro_full_summary.json",
    },
    ("qwen35_4b_posttrained", "thinking", "mmlu_redux"): {
        "correct": 5055, "total": 5700, "invalid_answers": 2, "truncated_generations": 0,
        "api_failures": 0, "elapsed_seconds": None, "provenance": "historical_reused",
        "source": "env/mmlu_redux_full_summary.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def software_versions() -> dict[str, str]:
    path = ROOT / "env/software_versions.txt"
    result: dict[str, str] = {}
    if not path.exists():
        return result
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.endswith(":") and not line.startswith(" "):
            current = line[:-1]
        elif "=" in line and line.split("=", 1)[0].startswith("torch.version"):
            key, value = line.split("=", 1)
            result[key] = value.strip()
        elif current and line.strip() and current not in result:
            result[current] = line.strip()
    return result


def latest_run(model_tag: str, setting: str, benchmark: str) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    return latest_run_in(ROOT / "outputs/full", model_tag, setting, benchmark)


def latest_run_in(root: Path, model_tag: str, setting: str, benchmark: str) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    base = root / model_tag / setting / benchmark
    if not base.exists():
        return None
    candidates: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        manifest_path = directory / "run_manifest.json"
        summary_path = directory / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "finished" and manifest.get("exit_code") == 0:
            candidates.append((str(directory), directory, manifest, summary))
    if not candidates:
        return None
    _, directory, manifest, summary = sorted(candidates)[-1]
    return directory, manifest, summary


def smoke_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in load_models()["models"]:
        for setting in model["supported_settings"]:
            for benchmark in load_benchmarks()["benchmarks"]:
                found = latest_run_in(ROOT / "outputs/smoke", model["model_tag"], setting, benchmark)
                if not found:
                    continue
                directory, manifest, summary = found
                rows.append({
                    "model_repo": model["model_name"], "size": model["size"], "setting": setting,
                    "benchmark": benchmark, "samples": summary.get("total_samples"),
                    "accuracy": round(float(summary["accuracy"]) * 100, 4) if summary.get("accuracy") is not None else None,
                    "invalid_answers": summary.get("invalid_answers"),
                    "truncated": summary.get("truncated_generations"),
                    "api_failures": summary.get("api_failures"),
                    "run_dir": str(directory.relative_to(ROOT)),
                    "job_id": manifest.get("environment", {}).get("SLURM_JOB_ID"),
                })
    return rows


def row(model: dict[str, Any], setting: str, benchmark: str) -> dict[str, Any]:
    benchmark_config = resolve_benchmark(benchmark)
    official = ((model.get("official_scores") or {}).get(setting) or {}).get(benchmark)
    key = (model["model_tag"], setting, benchmark)
    historical = HISTORICAL.get(key)
    run = None if historical else latest_run(model["model_tag"], setting, benchmark)
    if historical:
        summary = historical
        manifest: dict[str, Any] = {}
        run_dir = historical["source"]
        provenance = historical["provenance"]
        status = "completed_historical"
    elif run:
        directory, manifest, summary = run
        run_dir = str(directory.relative_to(ROOT))
        provenance = "official_reproduction" if model["checkpoint_type"] == "posttrained" else "protocol_transfer"
        status = "completed"
    else:
        summary = {}
        manifest = {}
        run_dir = None
        provenance = "official_reproduction" if model["checkpoint_type"] == "posttrained" else "protocol_transfer"
        status = "pending"

    total = summary.get("total_samples") or summary.get("total")
    correct = summary.get("correct")
    accuracy = summary.get("accuracy")
    if accuracy is not None:
        local = round(float(accuracy) * 100, 4) if float(accuracy) <= 1 else round(float(accuracy), 4)
    elif total and correct is not None:
        local = round(100 * int(correct) / int(total), 4)
    else:
        local = None
    delta = round(local - official, 4) if local is not None and official is not None else None
    output_stats = summary.get("output_tokens") or {}
    reasoning_stats = summary.get("reasoning_chars") or {}
    return {
        "size": model["size"],
        "model_repo": model["model_name"],
        "model_tag": model["model_tag"],
        "checkpoint_type": model["checkpoint_type"],
        "setting": setting,
        "benchmark": benchmark,
        "official": official,
        "local": local,
        "delta": delta,
        "correct": correct,
        "total": total,
        "correct_total": f"{correct}/{total}" if correct is not None and total is not None else None,
        "invalid_answers": summary.get("invalid_answers"),
        "truncated": summary.get("truncated_generations"),
        "api_failures": summary.get("api_failures"),
        "successful_generations": summary.get("successful_generations"),
        "duplicate_predictions": summary.get("duplicate_prediction_rows"),
        "duplicate_reviews": summary.get("duplicate_review_rows"),
        "avg_output_tokens": output_stats.get("mean"),
        "p95_output_tokens": output_stats.get("p90"),
        "avg_reasoning_chars": reasoning_stats.get("mean"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "throughput_samples_per_second": summary.get("throughput_samples_per_second"),
        "gpu": (manifest.get("hardware") or {}).get("gpu_name") or manifest.get("gpu_name"),
        "tp_size": (manifest.get("settings") or {}).get("tp_size") or manifest.get("tp_size"),
        "model_revision": manifest.get("model_revision") or model["revision"],
        "run_dir": run_dir,
        "status": status,
        "provenance": provenance,
        "expected_total": benchmark_config["expected_total"],
    }


def render_report(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Qwen3.5 三种设定 MMLU 评测报告",
        "",
        f"> 生成日期：{date.today().isoformat()}；代码提交：`{payload.get('git_commit') or 'unknown'}`",
        "",
        "## Technical summary",
        "",
        "本报告比较同尺寸 Qwen3.5 Base、Post-trained Thinking 与 Post-trained Non-Thinking。",
        "Post-trained 行使用官方 model card 公开的对应分数作为参考；Base 是将已验证的生成式 EvalScope 协议迁移到 Base checkpoint，标记为 `protocol_transfer`，不宣称官方复现。",
        "",
        "## Environment",
        "",
        f"- Python: `{payload['software'].get('python', 'n/a')}`; PyTorch: `{payload['software'].get('torch', 'n/a')}`; CUDA reported by PyTorch: `{payload['software'].get('torch.version.cuda', 'n/a')}`.",
        f"- SGLang: `{payload['software'].get('sglang', 'n/a')}`; EvalScope: `{payload['software'].get('evalscope', 'n/a')}`.",
        "- GPU name, driver, VRAM, TP size and Slurm job are taken from each run manifest; they are not inferred from the login node.",
        "",
        "## 结果总表",
        "",
        "| Size | Setting | Benchmark | Official | Local | Delta | Correct/Total | Invalid | Truncated | API failures | Avg output tokens | Avg reasoning chars | Elapsed(s) | Revision | Provenance | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in rows:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else str(value)
        lines.append("| " + " | ".join(fmt(item.get(key)) for key in (
            "size", "setting", "benchmark", "official", "local", "delta", "correct_total",
            "invalid_answers", "truncated", "api_failures", "avg_output_tokens",
            "avg_reasoning_chars", "elapsed_seconds", "model_revision", "provenance", "status",
        )) + " |")

    lines += [
        "",
        "## Protocol and data provenance",
        "",
        f"- Model card URL/read date: registry entries under `configs/models.json`, read on `{payload['official_card_read_date']}`.",
        "- Checkpoints: fixed Hugging Face commit revisions in `configs/models.json`; runtime manifests record local file SHA256 values.",
        "- Precision: BF16; SGLang; MTP and speculative decoding disabled; reasoning parser is enabled only for Thinking.",
        "- Thinking uses `chat_template_kwargs.enable_thinking=true`; Non-Thinking uses `false`; Base sends neither field and does not use a reasoning parser.",
        "- MMLU-Pro: EvalScope built-in adapter, test split, 5-shot from validation, expected 12,032 samples.",
        "- MMLU-Redux: EvalScope built-in adapter, test split, 0-shot, expected 5,700 samples.",
        "- Prompt rendering and answer extraction are not rewritten; actual installed EvalScope adapter is the source of truth.",
        "",
        "## Same-size comparisons",
        "",
        "| Size | Benchmark | Thinking - Base | Non-Thinking - Base | Thinking - Non-Thinking |",
        "|---|---|---:|---:|---:|",
    ]
    by_key = {(item["size"], item["benchmark"], item["setting"]): item for item in rows}
    for size in ("0.8B", "2B", "4B", "9B"):
        for benchmark in ("mmlu_pro", "mmlu_redux"):
            base = by_key.get((size, benchmark, "base"), {}).get("local")
            thinking = by_key.get((size, benchmark, "thinking"), {}).get("local")
            nonthinking = by_key.get((size, benchmark, "non_thinking"), {}).get("local")
            diff = lambda a, b: "n/a" if a is None or b is None else f"{a - b:.4f}"
            lines.append(f"| {size} | {benchmark} | {diff(thinking, base)} | {diff(nonthinking, base)} | {diff(thinking, nonthinking)} |")

    lines += [
        "",
        "## Smoke verification",
        "",
        "Smoke 结果只用于验证数据加载、API、模式字段、parser 和答案抽取，不替代 full accuracy。",
        "",
        "| Model | Size | Setting | Benchmark | Samples | Accuracy | Invalid | Truncated | API failures | Slurm job | Run |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in payload.get("smoke_rows", []):
        lines.append("| " + " | ".join("n/a" if item.get(key) is None else str(item.get(key)) for key in (
            "model_repo", "size", "setting", "benchmark", "samples", "accuracy",
            "invalid_answers", "truncated", "api_failures", "job_id", "run_dir",
        )) + " |")

    lines += [
        "",
        "## Reliability diagnostics",
        "",
        "每个 completed run 的 `summary.json` 和 `summary_diagnostics.json` 位于对应 run directory。统计中的 API failure 包括缺失 prediction；重复 prediction/review 保留原始重复计数并使用确定性的最后一条记录计分。elapsed 只接受 run manifest 的 monotonic clock。",
        "",
        "## Limitations and next steps",
        "",
        "- 官方 model card 并未公开完整的 MMLU 评测 recipe，因此本项目的首要定义是 `Qwen3.5 + SGLang + EvalScope standard-recipe reproduction`。",
        "- 4B 的 78.27% / 88.68% 是此前审计过的历史 Thinking 结果，原始输出没有被覆盖；新矩阵结果与其分开标记。",
        "- 9B 是否已在其他位置完成过两个结果，当前仓库没有发现可复查的 prediction/review/summary 资产，因此这里不代填；9B 配置已固定，资源足够时可独立运行。",
        "- 若官方复现 delta 超过 1--2 个百分点，应先核查 revision、thinking 字段、dataset cache、shot、prompt、answer extraction、max_tokens 和随机性，并另起 variant run。",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "RUN_MODE=smoke bash scripts/run_benchmark.sh Qwen/Qwen3.5-0.8B mmlu_pro",
        "bash scripts/run_benchmark_matrix.sh --exclude-9b",
        "python scripts/summarize_three_setting.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / "env/mmlu_three_setting_summary.json")
    parser.add_argument("--output-report", type=Path, default=ROOT / "MMLU_THREE_SETTING_REPORT.md")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for model in load_models()["models"]:
        for setting in model["supported_settings"]:
            for benchmark in load_benchmarks()["benchmarks"]:
                rows.append(row(model, setting, benchmark))
    payload = {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "official_card_read_date": load_models()["official_card_read_date"],
        "software": software_versions(),
        "models_config": "configs/models.json",
        "benchmarks_config": "configs/benchmarks.json",
        "rows": rows,
        "smoke_rows": smoke_rows(),
        "pending": [item for item in rows if item["status"] == "pending"],
        "comparisons": "computed only when both local scores exist; percentages are absolute points",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_json), "report": str(args.output_report), "rows": len(rows), "completed": len(rows)-len(payload["pending"]), "pending": len(payload["pending"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
