#!/usr/bin/env python3
"""Readiness check: model listing plus a short real generation."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    models = request(f"http://127.0.0.1:{args.port}/v1/models")
    result = request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        {"model": args.model, "prompt": "ready", "max_tokens": 4, "temperature": 0, "stream": False},
    )
    choices = result.get("choices") or []
    if result.get("error") or not choices:
        raise SystemExit(f"generation readiness failed: {json.dumps(result, ensure_ascii=False)[:1000]}")
    print(json.dumps({"models": models, "generation": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
