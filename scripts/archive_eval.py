#!/usr/bin/env python3
"""Archive an evaluation/throughput run and write checksums beside it."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = [path.resolve() for path in args.input_dir]
    archive = args.archive.resolve()
    for source in sources:
        if not source.is_dir():
            raise SystemExit(f"input directory does not exist: {source}")
    files = [(source, path) for source in sources for path in sorted(source.rglob("*")) if path.is_file()]
    entries = [{"source_dir": str(source), "path": str(path.relative_to(source)), "bytes": path.stat().st_size} for source, path in files]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_dirs": [str(source) for source in sources],
        "archive": str(archive),
        "file_count": len(files),
        "files": entries,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Never add the archive itself when it is placed below the input directory.
    with tarfile.open(archive, "w:gz") as handle:
        for source, path in files:
            if path == archive:
                continue
            handle.add(path, arcname=Path(source.name) / path.relative_to(source), recursive=False)
    manifest["archive_sha256"] = sha256(archive)
    manifest_path = args.manifest or archive.with_suffix(archive.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "manifest": str(manifest_path), "file_count": len(files)}))


if __name__ == "__main__":
    main()
