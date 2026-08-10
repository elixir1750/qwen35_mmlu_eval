#!/usr/bin/env python3
"""Unit tests for model registry and mode selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_config import generation_config, resolve_benchmark, resolve_model, settings_for_model


class BenchmarkConfigTest(unittest.TestCase):
    def test_posttrained_has_two_explicit_modes(self) -> None:
        model = resolve_model("Qwen/Qwen3.5-2B")
        self.assertEqual(settings_for_model(model), ["thinking", "non_thinking"])
        self.assertEqual(settings_for_model(model, "non-thinking"), ["non_thinking"])

    def test_base_only_has_base_and_no_thinking_field(self) -> None:
        model = resolve_model("Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(settings_for_model(model), ["base"])
        config = generation_config(model, "base")
        self.assertEqual(config["extra_body"], {})
        self.assertIsNone(config["thinking_field"])
        self.assertEqual(config["recipe_source"], "posttrained_non_thinking_transfer")

    def test_posttrained_api_field_and_recipes(self) -> None:
        model = resolve_model("Qwen/Qwen3.5-2B")
        thinking = generation_config(model, "thinking")
        non_thinking = generation_config(model, "non_thinking")
        self.assertTrue(thinking["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(non_thinking["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(thinking["temperature"], 1.0)
        self.assertEqual(non_thinking["temperature"], 0.7)

    def test_benchmark_registry(self) -> None:
        pro = resolve_benchmark("mmlu_pro")
        redux = resolve_benchmark("mmlu_redux")
        self.assertEqual((pro["eval_split"], pro["few_shot"], pro["expected_total"]), ("test", 5, 12032))
        self.assertEqual((redux["eval_split"], redux["few_shot"], redux["expected_total"]), ("test", 0, 5700))


if __name__ == "__main__":
    unittest.main()
