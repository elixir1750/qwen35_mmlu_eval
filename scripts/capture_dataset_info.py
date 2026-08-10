#!/usr/bin/env python3
"""Capture the locally materialized EvalScope/ModelScope dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dir(path: Path) -> dict[str, Any]:
    info = json.loads((path / "dataset_info.json").read_text(encoding="utf-8"))
    checksums = list((info.get("download_checksums") or {}).keys())
    upstream = checksums[0] if checksums else None
    revision = upstream.split("@", 1)[1].split("/", 1)[0] if upstream and "@" in upstream else None
    files = {}
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix in {".arrow", ".parquet", ".json"}:
            files[child.name] = {"bytes": child.stat().st_size, "sha256": sha256(child)}
    return {
        "cache_dir": str(path),
        "dataset_name": info.get("dataset_name"),
        "split": list((info.get("splits") or {}).keys()),
        "num_examples": sum(int(value.get("num_examples", 0)) for value in (info.get("splits") or {}).values()),
        "upstream_uri": upstream,
        "upstream_revision": revision,
        "files": files,
        "dataset_info_sha256": sha256(path / "dataset_info.json"),
        "state_sha256": sha256(path / "state.json") if (path / "state.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/modelscope/hub/datasets/datasets")
    parser.add_argument("--out", type=Path, default=Path("env/dataset_cache_info.json"))
    args = parser.parse_args()
    entries = []
    for path in sorted(args.cache_root.glob("TIGER-Lab_MMLU-Pro-*")):
        info_path = path / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if (info.get("splits") or {}).get("test", {}).get("num_examples") == 12032:
                entries.append({"benchmark": "mmlu_pro", **inspect_dir(path)})
    for path in sorted(args.cache_root.glob("AI-ModelScope_mmlu-redux-2.0-*")):
        info_path = path / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if (info.get("splits") or {}).get("test", {}).get("num_examples") == 100:
                entries.append({"benchmark": "mmlu_redux", **inspect_dir(path)})
    by_benchmark: dict[str, list[dict[str, Any]]] = {"mmlu_pro": [], "mmlu_redux": []}
    for entry in entries:
        by_benchmark[entry["benchmark"]].append(entry)
    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "cache_root": str(args.cache_root.resolve()),
        "source": "local ModelScope cache materialized by EvalScope 1.10.0",
        "upstream_revision_status": "the cache metadata points to @master, not an immutable upstream commit; content hashes below are the reproducibility anchor for this run",
        "benchmarks": {
            name: {
                "cache_entry_count": len(rows),
                "total_examples": sum(row["num_examples"] for row in rows),
                "upstream_revisions": sorted({row["upstream_revision"] for row in rows}),
                "entries": rows,
            }
            for name, rows in by_benchmark.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "mmlu_pro_entries": len(by_benchmark["mmlu_pro"]), "mmlu_redux_entries": len(by_benchmark["mmlu_redux"])}))


if __name__ == "__main__":
    main()
