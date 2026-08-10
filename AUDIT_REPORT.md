# Qwen3.5-4B Evaluation Pipeline Audit

Generated after the repository audit and quick throughput validation on 2026-08-10.

## Conclusion

The original MMLU chain is runnable and its arithmetic is internally consistent, but
the original runner was not fully reproducible: fixed `--no-timestamp` work directories,
log-derived elapsed time, and static-only runtime metadata could silently obscure an
interrupted or resumed run. These issues are fixed for new runs. The historical full
results remain unchanged: MMLU-Pro 9417/12032 = 78.27%, MMLU-Redux 5055/5700 = 88.68%.

## Findings and fixes

| Severity | Finding | Status |
|---|---|---|
| High | Full/smoke outputs could be reused with `--no-timestamp`, mixing old JSONL with a new run. | Fixed: timestamped run directories; non-empty reuse requires `RESUME=1`. |
| High | Elapsed time was inferred from the earliest/latest log timestamps. | Fixed for new summaries: `time.monotonic_ns()` values are stored in `run_manifest.json`; missing/zero duration is explicit. |
| High | Review/prediction duplicates and missing predictions were not fully auditable. | Fixed: deterministic last-row deduplication and counters for duplicate reviews/predictions, missing predictions, API failures, and truncation. |
| High | The old post-processing command targeted shared benchmark roots and could summarize the wrong run. | Fixed: `scripts/postprocess_full.sh` now requires the two explicit run directories and writes run-specific summaries/report paths. |
| Medium | Runtime parameters and model checksums were not guaranteed to be recorded with each run. | Fixed: manifest records command, selected environment, git commit, model revision, small-file hashes, verified shard hashes, and dataset-cache manifest hash. |
| Medium | Raw outputs/logs had no repository-provided archive/checksum workflow. | Fixed: `scripts/archive_eval.py`; quick throughput raw JSON and logs were archived with a manifest. |
| Medium | Dataset metadata resolves to `@master`, not an immutable upstream commit. | Explicitly recorded as an unresolved reproducibility limitation; local ModelScope data/state/info hashes are captured in `env/dataset_cache_info.json`. |
| Medium | `transformers serve` cannot be used in this environment. | Verified failure is recorded; direct persistent Transformers `generate` is used and no concurrent-service speedup is claimed. |

## Verification

- `bash -n scripts/*.sh`
- `.venv/bin/python -m compileall -q scripts`
- `.venv/bin/python scripts/test_summarize_eval.py -v` — 2/2 passed.
- All new `--help` commands and throughput `--dry-run` passed.
- Full-summary arithmetic checked: 9417/12032 = 0.7826628989 and 5055/5700 = 0.8868421053.
- GPU smoke/readiness and quick throughput ran on `node2`, RTX3090, with local proxy disabled.

## Limitations kept visible

- No full MMLU rerun was started during this audit; existing raw results were not modified.
- The standard 1024/256 single-stream plus SGLang online c=1 case was executed and is
  included in `THROUGHPUT_REPORT.md`; the optional 1024/1024 sweep was not run.
- Transformers had no streaming endpoint, so TTFT/TPOT are unavailable for that direct
  baseline. SGLang online TTFT/TPOT are measured from SSE events.
