#!/usr/bin/env python3
"""Download/validate a fixed Qwen3.5 checkpoint and record file hashes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmark_config import ROOT, resolve_model


REQUIRED = ("config.json", "tokenizer_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and (item.name.endswith(".safetensors") or item.name in REQUIRED or item.name in {"model.safetensors.index.json", "tokenizer.json", "chat_template.jinja"})
    )


def validate(path: Path) -> None:
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if not (path / "model.safetensors.index.json").is_file() and not (path / "model.safetensors").is_file():
        missing.append("model.safetensors.index.json or model.safetensors")
    incomplete = sorted(str(item.relative_to(path)) for item in path.rglob("*.incomplete")) if path.exists() else []
    shards = list(path.glob("*.safetensors")) if path.exists() else []
    if missing or incomplete or not shards:
        details = {"path": str(path), "missing": missing, "incomplete": incomplete, "safetensors": len(shards)}
        raise RuntimeError(f"checkpoint validation failed: {json.dumps(details, ensure_ascii=False)}")


def inspect(model_name: str, path_override: str | None = None) -> dict[str, Any]:
    model = resolve_model(model_name)
    path = Path(path_override).expanduser() if path_override else ROOT / model["local_path"]
    path = path.resolve()
    validate(path)
    files = {str(item.relative_to(path)): sha256(item) for item in model_files(path)}
    state = {
        "model_name": model_name,
        "repo": model["repo"],
        "revision": model["revision"],
        "model_tag": model["model_tag"],
        "path": str(path),
        "file_sha256": files,
        "safetensors_shards": sorted(name for name in files if name.endswith(".safetensors")),
    }
    output = ROOT / "env" / "model_hashes" / f"{model['model_tag']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def download(model_name: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # The mirror's regular HTTP files are usable without Hub Xet credentials.
    # Explicitly disable Xet so an inherited Hub client cannot switch to CAS.
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Avoid an inherited proxy silently changing the source used for a fixed revision.
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(name, None)
    model = resolve_model(model_name)
    command = [
        str(ROOT / ".venv/bin/hf"), "download", model["repo"],
        "--revision", model["revision"], "--local-dir", str(path),
    ]
    print("download_command=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--path")
    parser.add_argument("--no-auto-download", action="store_true")
    args = parser.parse_args()
    model = resolve_model(args.model)
    path = Path(args.path).expanduser() if args.path else ROOT / model["local_path"]
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.download.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not path.exists() or not any(path.glob("*.safetensors")):
            if args.no_auto_download:
                raise SystemExit(f"checkpoint is absent and NO_AUTO_DOWNLOAD=1: {path}")
            download(args.model, path)
        state = inspect(args.model, str(path))
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
