#!/usr/bin/env python3
"""Create and finish auditable run manifests.

The manifest intentionally stores a monotonic-clock duration.  Wall-clock
timestamps are useful for humans, but are not used for throughput arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_metadata(project: Path, model_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"path": str(model_path.resolve())}
    info_path = project / "env/model_info.txt"
    if info_path.exists():
        for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("hf_revision="):
                metadata["revision"] = line.split("=", 1)[1]
                break
    files: dict[str, str] = {}
    for relative in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ):
        digest = sha256_file(model_path / relative)
        if digest:
            files[relative] = digest
    # The large shard hashes were already verified and recorded during model
    # download. Reusing them avoids hashing 8.8 GB at every benchmark start.
    if info_path.exists():
        for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if " sha256=" in line:
                name, digest = line.split(" sha256=", 1)
                files[name.split()[0]] = digest.strip()
    metadata["file_sha256"] = files
    return metadata


def selected_environment() -> dict[str, str]:
    names = (
        "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "HF_ENDPOINT", "HF_HUB_DISABLE_XET",
        "NO_PROXY", "SGLANG_ATTENTION_BACKEND", "TORCHINDUCTOR_CACHE_DIR",
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR",
    )
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def start(args: argparse.Namespace) -> None:
    project = args.project_dir.resolve()
    model_path = args.model_path.resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "kind": args.kind,
        "backend": args.backend,
        "benchmark": args.benchmark,
        "project_dir": str(project),
        "output_dir": str(args.output_dir.resolve()),
        "command": args.command,
        "argv": sys.argv,
        "wall_clock_start": iso_now(),
        "wall_clock_start_ns": time.time_ns(),
        "monotonic_start_ns": time.monotonic_ns(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "git_commit": git_commit(project),
        "model": model_metadata(project, model_path),
        "dataset_cache_manifest": {
            "path": str((project / "env/dataset_cache_info.json").resolve()),
            "sha256": sha256_file(project / "env/dataset_cache_info.json"),
        },
        "environment": selected_environment(),
        "settings": json.loads(args.settings) if args.settings else {},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.manifest)


def finish(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    end_ns = time.monotonic_ns()
    start_ns = manifest.get("monotonic_start_ns")
    duration = (end_ns - start_ns) / 1_000_000_000 if isinstance(start_ns, int) else None
    manifest.update({
        "status": "finished" if args.exit_code == 0 else "failed",
        "exit_code": args.exit_code,
        "wall_clock_end": iso_now(),
        "wall_clock_end_ns": time.time_ns(),
        "monotonic_end_ns": end_ns,
        "monotonic_duration_seconds": duration,
    })
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "exit_code": args.exit_code, "monotonic_duration_seconds": duration}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--manifest", type=Path, required=True)
    start_parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    start_parser.add_argument("--model-path", type=Path, required=True)
    start_parser.add_argument("--output-dir", type=Path, required=True)
    start_parser.add_argument("--kind", default="evaluation")
    start_parser.add_argument("--backend", default="sglang")
    start_parser.add_argument("--benchmark")
    start_parser.add_argument("--command", required=True)
    start_parser.add_argument("--settings")
    start_parser.set_defaults(func=start)

    finish_parser = sub.add_parser("finish")
    finish_parser.add_argument("--manifest", type=Path, required=True)
    finish_parser.add_argument("--exit-code", type=int, required=True)
    finish_parser.set_defaults(func=finish)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
