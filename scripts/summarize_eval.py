#!/usr/bin/env python3
"""Summarize EvalScope prediction/review JSONL files without changing them."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "p90", "max")}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": round(ordered[p90_index], 3),
        "max": round(max(values), 3),
    }


def text_from_content(content: Any) -> tuple[str, str]:
    reasoning: list[str] = []
    final: list[str] = []
    if isinstance(content, str):
        return "", content
    if not isinstance(content, list):
        return "", ""
    for part in content:
        if isinstance(part, str):
            final.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "")).lower()
        value = part.get("reasoning") if part_type in {"reasoning", "thinking"} else part.get("text")
        if value is None:
            value = part.get("content")
        if not isinstance(value, str):
            continue
        if part_type in {"reasoning", "thinking"}:
            reasoning.append(value)
        else:
            final.append(value)
    return "\n".join(reasoning), "\n".join(final)


def prediction_details(row: dict[str, Any]) -> dict[str, Any]:
    model_output = row.get("model_output") or {}
    choices = model_output.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    reasoning, final = text_from_content(message.get("content"))
    usage = model_output.get("usage") or {}
    stop_reason = choice.get("stop_reason") or choice.get("finish_reason")
    error = model_output.get("error")
    return {
        "index": row.get("index"),
        "error": error,
        "stop_reason": stop_reason,
        "reasoning_chars": len(reasoning),
        "final_chars": len(final),
        "reasoning": reasoning,
        "final": final,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def find_files(root: Path, kind: str) -> list[Path]:
    return sorted((root / kind).rglob("*.jsonl"))


def subset_from_path(path: Path, benchmark: str) -> str:
    prefix = benchmark + "_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def manifest_elapsed(manifest_path: Path | None) -> tuple[float | None, str]:
    """Read elapsed time from a structured monotonic-clock manifest.

    Older runs only have log timestamps.  Deliberately do not infer duration
    from those timestamps: wall-clock logging can be sparse, reordered, or
    rounded to seconds.  Such runs are reported with a missing duration.
    """
    if manifest_path is None or not manifest_path.exists():
        return None, "missing_manifest"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = data.get("monotonic_duration_seconds")
    if isinstance(value, (int, float)) and value >= 0:
        return float(value), "monotonic_manifest"
    return None, "missing_monotonic_duration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("mmlu_pro", "mmlu_redux"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-log", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prediction_rows: dict[tuple[str, Any], dict[str, Any]] = {}
    duplicate_prediction_rows = 0
    for path in find_files(args.output_dir, "predictions"):
        subset = subset_from_path(path, args.benchmark)
        for row in load_jsonl(path):
            key = (subset, row.get("index"))
            if key in prediction_rows:
                duplicate_prediction_rows += 1
            # EvalScope may leave a partial duplicate after interruption. Reviews are
            # the canonical completed-sample set; keep the latest prediction by index.
            prediction_rows[key] = row

    review_rows: dict[tuple[str, Any], dict[str, Any]] = {}
    duplicate_review_rows = 0
    for path in find_files(args.output_dir, "reviews"):
        subset = subset_from_path(path, args.benchmark)
        for row in load_jsonl(path):
            row["_subset"] = subset
            key = (subset, row.get("index"))
            if key in review_rows:
                duplicate_review_rows += 1
            # Keep the last row in deterministic path/line traversal order. This
            # makes interrupted/resumed EvalScope outputs auditable.
            review_rows[key] = row
    reviews = [review_rows[key] for key in sorted(review_rows, key=lambda item: (item[0], str(item[1])))]
    if not reviews:
        raise RuntimeError(f"no review JSONL files found below {args.output_dir}")

    allowed = set("ABCDEFGHIJ" if args.benchmark == "mmlu_pro" else "ABCD")
    records: list[dict[str, Any]] = []
    for review in reviews:
        score = (review.get("sample_score") or {}).get("score") or {}
        sample_score = review.get("sample_score") or {}
        sample_metadata = sample_score.get("sample_metadata") or {}
        extracted = score.get("extracted_prediction")
        target = review.get("target")
        value = score.get("value")
        if isinstance(value, dict):
            accuracy_value = value.get("acc")
        else:
            accuracy_value = value
        prediction_key = (review.get("_subset", ""), review.get("index"))
        prediction_row = prediction_rows.get(prediction_key)
        missing_prediction = prediction_row is None
        prediction = prediction_details(prediction_row or {})
        invalid = not isinstance(extracted, str) or extracted.upper() not in allowed
        truncated = str(prediction.get("stop_reason", "")).lower() in {"length", "max_tokens", "max_output_tokens"}
        api_error = missing_prediction or prediction.get("error") not in (None, "", {})
        # A review score cannot turn an API failure or missing prediction into a
        # correct generation. This keeps arithmetic and reliability counts aligned.
        correct = bool(not api_error and (accuracy_value == 1 or (not invalid and extracted.upper() == str(target).upper())))
        records.append({
            "index": review.get("index"),
            "subject": sample_metadata.get("subject") or review.get("_subset"),
            "target": target,
            "extracted_prediction": extracted,
            "correct": correct,
            "invalid_answer": invalid,
            "truncated": truncated,
            "api_error": api_error,
            "missing_prediction": missing_prediction,
            **{key: prediction[key] for key in (
                "stop_reason", "reasoning_chars", "final_chars", "reasoning", "final",
                "input_tokens", "output_tokens", "total_tokens",
            )},
        })

    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in records if isinstance(row.get(key), (int, float))]

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_subject[str(row.get("subject") or "unknown")].append(row)
    subsets = {}
    for subject, rows in sorted(by_subject.items()):
        subsets[subject] = {
            "num": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 6),
            "invalid_answers": sum(row["invalid_answer"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "api_errors": sum(row["api_error"] for row in rows),
        }

    elapsed, elapsed_source = manifest_elapsed(args.run_manifest)
    output_tokens = numeric("output_tokens")
    summary = {
        "benchmark": args.benchmark,
        "output_dir": str(args.output_dir),
        "total_samples": len(records),
        "successful_generations": sum(not row["api_error"] for row in records),
        "correct": sum(row["correct"] for row in records),
        "accuracy": round(sum(row["correct"] for row in records) / len(records), 6),
        "invalid_answers": sum(row["invalid_answer"] for row in records),
        "truncated_generations": sum(row["truncated"] for row in records),
        "api_failures": sum(row["api_error"] for row in records),
        "missing_prediction_rows": sum(row["missing_prediction"] for row in records),
        "duplicate_prediction_rows": duplicate_prediction_rows,
        "duplicate_review_rows": duplicate_review_rows,
        "elapsed_seconds": elapsed,
        "elapsed_source": elapsed_source,
        "elapsed_seconds_from_log": None,
        "throughput_samples_per_second": round(len(records) / elapsed, 6) if elapsed and elapsed > 0 else None,
        "input_tokens": quantiles(numeric("input_tokens")),
        "output_tokens": quantiles(output_tokens),
        "total_tokens": quantiles(numeric("total_tokens")),
        "reasoning_chars": quantiles(numeric("reasoning_chars")),
        "final_chars": quantiles(numeric("final_chars")),
        "stop_reasons": dict(Counter(str(row["stop_reason"]) for row in records)),
        "retry_log_occurrences": None,
        "subsets": subsets,
    }
    if args.eval_log and args.eval_log.exists():
        log_text = args.eval_log.read_text(encoding="utf-8", errors="replace")
        summary["retry_log_occurrences"] = len(re.findall(r"(?im)\b(?:retrying|retry\s*#\d+|retry\s+request)\b", log_text))
        summary["dataset_network_retry_occurrences"] = len(re.findall(r"(?im)Retrying \(Retry", log_text))
        summary["dataset_timeout_occurrences"] = len(re.findall(r"(?im)(?:Read timed out|ConnectTimeout|ReadTimeoutError)", log_text))
    else:
        summary["dataset_network_retry_occurrences"] = None
        summary["dataset_timeout_occurrences"] = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics = {
        "wrong_samples": [
            {key: row[key] for key in ("index", "subject", "target", "extracted_prediction", "stop_reason", "reasoning", "final")}
            for row in records if not row["correct"] and not row["invalid_answer"] and not row["api_error"]
        ][:10],
        "invalid_samples": [row for row in records if row["invalid_answer"]][:10],
        "truncated_samples": [row for row in records if row["truncated"]][:10],
        "api_error_samples": [row for row in records if row["api_error"]][:10],
        "missing_prediction_samples": [row for row in records if row["missing_prediction"]][:10],
    }
    diagnostics_path = args.out.with_name(args.out.stem + "_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.out), "diagnostics": str(diagnostics_path), **{key: summary[key] for key in (
        "total_samples", "accuracy", "invalid_answers", "truncated_generations", "api_failures",
    )}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
