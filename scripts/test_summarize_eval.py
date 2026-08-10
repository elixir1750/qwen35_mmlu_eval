#!/usr/bin/env python3
"""Small regression fixtures for summary arithmetic and failure accounting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prediction(index: int, text: str = "ANSWER: A", *, finish: str = "stop", error: dict | None = None) -> dict:
    model_output: dict = {"choices": [{"message": {"content": [{"type": "reasoning", "text": "short reasoning"}, {"type": "text", "text": text}]}, "finish_reason": finish}], "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}}
    if error:
        model_output = {"error": error}
    return {"index": index, "model_output": model_output}


def review(index: int, answer: str = "A", target: str = "A") -> dict:
    return {"index": index, "target": target, "sample_score": {"score": {"extracted_prediction": answer, "value": {"acc": int(answer == target)}}, "sample_metadata": {"subject": "fixture_subject"}}}


class SummarizeFixtureTest(unittest.TestCase):
    def test_duplicates_missing_api_error_truncation_and_zero_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictions").mkdir()
            (root / "reviews").mkdir()
            prediction_rows = [
                prediction(0, "ANSWER: B"),
                prediction(0, "ANSWER: A"),  # duplicate; latest row wins
                prediction(1, "ANSWER: B"),  # invalid extraction for A-D? B is valid, but target is A
                prediction(3, error={"message": "timeout"}),
                prediction(4, "ANSWER: A", finish="length"),
                prediction(5, "ANSWER: Z"),
            ]
            review_rows = [
                review(0),
                review(0, answer="B", target="B"),  # duplicate; latest review wins
                review(1),
                review(2),  # missing prediction
                review(3),  # API error prediction
                review(4),  # truncated prediction
                review(5, answer="Z"),  # invalid answer
            ]
            pred_path = root / "predictions/mmlu_pro_fixture_subject.jsonl"
            review_path = root / "reviews/mmlu_pro_fixture_subject.jsonl"
            pred_path.write_text("\n".join(json.dumps(row) for row in prediction_rows) + "\n", encoding="utf-8")
            review_path.write_text("\n".join(json.dumps(row) for row in review_rows) + "\n", encoding="utf-8")
            manifest = root / "run_manifest.json"
            manifest.write_text(json.dumps({"monotonic_duration_seconds": 0}), encoding="utf-8")
            output = root / "summary.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/summarize_eval.py"), "mmlu_pro",
                "--output-dir", str(root), "--run-manifest", str(manifest), "--out", str(output),
            ], check=True)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_samples"], 6)
            self.assertEqual(summary["duplicate_prediction_rows"], 1)
            self.assertEqual(summary["duplicate_review_rows"], 1)
            self.assertEqual(summary["missing_prediction_rows"], 1)
            self.assertEqual(summary["api_failures"], 2)  # missing + explicit API error
            self.assertEqual(summary["truncated_generations"], 1)
            self.assertEqual(summary["invalid_answers"], 1)
            self.assertIsNone(summary["throughput_samples_per_second"])
            self.assertEqual(summary["elapsed_source"], "monotonic_manifest")

    def test_missing_manifest_does_not_guess_from_log_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictions").mkdir()
            (root / "reviews").mkdir()
            (root / "predictions/mmlu_pro_subject.jsonl").write_text(json.dumps(prediction(0)) + "\n", encoding="utf-8")
            (root / "reviews/mmlu_pro_subject.jsonl").write_text(json.dumps(review(0)) + "\n", encoding="utf-8")
            log = root / "eval.log"
            log.write_text("2026-08-10 00:00:00 start\n2026-08-10 00:01:00 end\n", encoding="utf-8")
            output = root / "summary.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/summarize_eval.py"), "mmlu_pro",
                "--output-dir", str(root), "--eval-log", str(log), "--out", str(output),
            ], check=True)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(summary["elapsed_seconds"])
            self.assertEqual(summary["elapsed_source"], "missing_manifest")
            self.assertIsNone(summary["throughput_samples_per_second"])


if __name__ == "__main__":
    unittest.main()
