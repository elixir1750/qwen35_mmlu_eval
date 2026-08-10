#!/usr/bin/env python3
"""Reject a resume that changes model, protocol, revision, or GPU identity."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def current_gpu_names() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = sorted({line.strip() for line in output.splitlines() if line.strip()})
    return ", ".join(names) if names else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = {
        "model_repo": args.model,
        "setting": args.setting,
        "benchmark_name": args.benchmark,
        "model_revision": args.revision,
    }
    failures = [f"{key}: manifest={data.get(key)!r}, current={value!r}" for key, value in checks.items() if data.get(key) != value]
    old_gpu = (data.get("hardware") or {}).get("gpu_name")
    new_gpu = current_gpu_names()
    if old_gpu and new_gpu and old_gpu != new_gpu:
        failures.append(f"gpu_name: manifest={old_gpu!r}, current={new_gpu!r}")
    if failures:
        raise SystemExit("unsafe resume; " + "; ".join(failures))
    print(json.dumps({"resume_valid": True, "gpu_name": new_gpu or old_gpu}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
