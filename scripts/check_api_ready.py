#!/usr/bin/env python3
"""Readiness check: model listing plus a short real generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request


def request(url: str, payload: dict | None = None, timeout: float = 180.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    models = request(f"http://127.0.0.1:{args.port}/v1/models", timeout=args.timeout)
    result = request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        {"model": args.model, "prompt": "ready", "max_tokens": 4, "temperature": 0, "stream": False},
        timeout=args.timeout,
    )
    choices = result.get("choices") or []
    if result.get("error") or not choices:
        raise SystemExit(f"generation readiness failed: {json.dumps(result, ensure_ascii=False)[:1000]}")
    payload = {"models": models, "generation": result, "model": args.model, "port": args.port}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
