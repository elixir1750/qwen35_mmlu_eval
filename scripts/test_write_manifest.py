#!/usr/bin/env python3
"""Regression tests for checkpoint-specific manifest metadata."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_manifest import model_metadata


class WriteManifestTest(unittest.TestCase):
    def test_uses_hash_manifest_for_matching_checkpoint_not_legacy_model_info(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            model_path = project / "model/Qwen3.5-0.8B"
            model_path.mkdir(parents=True)
            config = model_path / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            expected_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            (project / "env/model_hashes").mkdir(parents=True)
            (project / "env/model_hashes/qwen35_0_8b.json").write_text(
                json.dumps({
                    "path": str(model_path.resolve()),
                    "revision": "0.8b-revision",
                    "file_sha256": {"model.safetensors-00001-of-00001.safetensors": "0.8b-shard"},
                }),
                encoding="utf-8",
            )
            (project / "env/model_info.txt").parent.mkdir(parents=True, exist_ok=True)
            (project / "env/model_info.txt").write_text("hf_revision=wrong-4b-revision\n", encoding="utf-8")

            metadata = model_metadata(project, model_path)

            self.assertEqual(metadata["revision"], "0.8b-revision")
            self.assertEqual(metadata["file_sha256"]["model.safetensors-00001-of-00001.safetensors"], "0.8b-shard")
            self.assertEqual(metadata["file_sha256"]["config.json"], expected_hash)


if __name__ == "__main__":
    unittest.main()
