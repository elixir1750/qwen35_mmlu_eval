#!/usr/bin/env python3
"""Resolve the parameterized Qwen3.5 MMLU evaluation configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_models() -> dict[str, Any]:
    return load_json(ROOT / "configs/models.json")


def load_benchmarks() -> dict[str, Any]:
    return load_json(ROOT / "configs/benchmarks.json")


def load_resources() -> dict[str, Any]:
    return load_json(ROOT / "configs/resources.json")


def supported_models() -> list[str]:
    return [item["model_name"] for item in load_models()["models"]]


def resolve_model(model_name: str) -> dict[str, Any]:
    for item in load_models()["models"]:
        if item["model_name"] == model_name:
            return dict(item)
    supported = ", ".join(supported_models())
    raise ValueError(f"unknown MODEL_NAME {model_name!r}; supported: {supported}")


def resolve_benchmark(benchmark_name: str) -> dict[str, Any]:
    try:
        value = load_benchmarks()["benchmarks"][benchmark_name]
    except KeyError as exc:
        supported = ", ".join(sorted(load_benchmarks()["benchmarks"]))
        raise ValueError(f"unknown BENCHMARK_NAME {benchmark_name!r}; supported: {supported}") from exc
    result = dict(value)
    result["name"] = benchmark_name
    return result


def normalize_setting(setting: str) -> str:
    aliases = {
        "base": "base",
        "thinking": "thinking",
        "non-thinking": "non_thinking",
        "non_thinking": "non_thinking",
        "nonthinking": "non_thinking",
    }
    try:
        return aliases[setting.lower()]
    except KeyError as exc:
        raise ValueError("setting must be base, thinking, or non_thinking") from exc


def settings_for_model(model: dict[str, Any], requested: str | None = None) -> list[str]:
    if requested:
        setting = normalize_setting(requested)
        if setting not in model["supported_settings"]:
            raise ValueError(f"{model['model_name']} does not support setting {setting}")
        return [setting]
    return list(model["supported_settings"])


def generation_config(model: dict[str, Any], setting: str) -> dict[str, Any]:
    setting = normalize_setting(setting)
    if setting == "base" and model["checkpoint_type"] != "base":
        raise ValueError(f"Base setting requires a -Base checkpoint: {model['model_name']}")
    if setting != "base" and model["checkpoint_type"] == "base":
        raise ValueError(f"Base checkpoint only supports setting=base: {model['model_name']}")
    if setting == "base":
        source = model.get("recipes", {}).get("non_thinking")
        if source is None:
            raise ValueError(f"Base model {model['model_name']} has no transfer recipe")
        recipe_source = model.get("base_recipe_source", "posttrained_non_thinking_transfer")
    else:
        source = model.get("recipes", {}).get(setting)
        if source is None:
            raise ValueError(f"{model['model_name']} has no recipe for {setting}")
        recipe_source = "official_model_card_general_recipe" if setting == "thinking" else "official_model_card_non_thinking_recipe"

    result = dict(source)
    result["recipe_source"] = recipe_source
    result["timeout"] = 600
    result["checkpoint_type"] = model["checkpoint_type"]
    if model["checkpoint_type"] == "posttrained":
        result["extra_body"] = {"chat_template_kwargs": {"enable_thinking": setting == "thinking"}}
        result["thinking_field"] = "chat_template_kwargs.enable_thinking"
    else:
        result["extra_body"] = {}
        result["thinking_field"] = None
    return result


def resolved(model_name: str, benchmark_name: str | None = None, setting: str | None = None) -> dict[str, Any]:
    model = resolve_model(model_name)
    result: dict[str, Any] = {
        "model": model,
        "settings": settings_for_model(model, setting),
        "resources": load_resources(),
    }
    if benchmark_name:
        result["benchmark"] = resolve_benchmark(benchmark_name)
    result["generation_configs"] = {
        item: generation_config(model, item) for item in result["settings"]
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("model")
    resolve_parser.add_argument("--benchmark")
    resolve_parser.add_argument("--setting")
    resolve_parser.add_argument("--json", action="store_true")
    generation_parser = sub.add_parser("generation")
    generation_parser.add_argument("model")
    generation_parser.add_argument("setting")
    generation_parser.add_argument("--json", action="store_true")
    generation_parser.add_argument("--evalscope-json", action="store_true")
    generation_parser.add_argument("--max-tokens", type=int)
    benchmark_parser = sub.add_parser("benchmark")
    benchmark_parser.add_argument("benchmark")
    benchmark_parser.add_argument("--json", action="store_true")
    sub.add_parser("list-models")
    sub.add_parser("list-benchmarks")
    args = parser.parse_args()

    try:
        if args.command == "resolve":
            value = resolved(args.model, args.benchmark, args.setting)
        elif args.command == "generation":
            value = generation_config(resolve_model(args.model), args.setting)
            if args.evalscope_json:
                allowed = {
                    "temperature", "top_p", "top_k", "presence_penalty",
                    "repetition_penalty", "max_tokens", "seed", "extra_body",
                    "timeout",
                }
                value = {key: value[key] for key in value if key in allowed}
            if args.max_tokens is not None:
                value["max_tokens"] = args.max_tokens
        elif args.command == "benchmark":
            value = resolve_benchmark(args.benchmark)
        elif args.command == "list-models":
            value = supported_models()
        else:
            value = sorted(load_benchmarks()["benchmarks"])
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
