# Qwen3.5-4B Throughput Report

Generated: 2026-08-10T12:52:02+08:00

## Scope

- Model: `Qwen/Qwen3.5-4B`, the same local original safetensors checkpoint for both backends.
- Both paths use BF16, the same token-ID prompts, requested output lengths, context length, seed 42, and no MTP/speculative decoding.
- SGLang uses the OpenAI-compatible completions API. Transformers uses one persistent `model.generate` process; it is not an online concurrent server.
- GPU: NVIDIA GeForce RTX 3090; driver: 570.153.02; raw structured data: `/home/lhzhang/qwen35_mmlu_eval/outputs/throughput/raw/20260810_123322_94445, /home/lhzhang/qwen35_mmlu_eval/outputs/throughput/raw/20260810_123626_94447, /home/lhzhang/qwen35_mmlu_eval/outputs/throughput/raw/20260810_123837_94448, /home/lhzhang/qwen35_mmlu_eval/outputs/throughput/raw/20260810_124649_94451`.

## Fixed-length single-stream results

| Backend | Cache | Input | Requested output | Completed/failed | Output tok/s | Total tok/s | E2E P50 ms | E2E P95 ms | Avg output tokens | P95 output tokens | Same lengths | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sglang | enabled | 256 | 64 | 3/0 | 71.800 | 358.998 | 892.361 | 893.569 | 64.000 | 64.000 | True | 19472 |
| transformers | enabled | 256 | 64 | 3/0 | 21.394 | 106.968 | 2985.831 | 2986.304 | 64.000 | 64.000 | True | 9217 |
| sglang | disabled | 256 | 64 | 3/0 | 77.263 | 386.315 | 827.888 | 829.295 | 64.000 | 64.000 | True | 19520 |
| sglang | enabled | 1024 | 256 | 3/0 | 72.933 | 364.666 | 3512.599 | 3513.680 | 256.000 | 256.000 | True | 19470 |
| transformers | enabled | 1024 | 256 | 3/0 | 22.548 | 112.742 | 11363.328 | 11366.023 | 256.000 | 256.000 | True | 9447 |

## SGLang online concurrency

| Cache | Concurrency | Input | Output | Completed/failed | Request/s | Output tok/s | TTFT P50/P95 ms | TPOT P50/P95 ms | E2E P50/P95 ms | Same lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enabled | 1 | 256 | 64 | 8/0 | 1.118 | 71.525 | 48.750/49.882 | 13.447/13.518 | 896.078/900.108 | True |
| enabled | 4 | 256 | 64 | 20/0 | 3.783 | 242.081 | 159.393/326.729 | 13.737/15.467 | 1024.205/1196.483 | True |
| disabled | 1 | 256 | 64 | 8/0 | 1.202 | 76.925 | 55.937/69.621 | 12.306/12.313 | 830.127/843.591 | True |
| disabled | 4 | 256 | 64 | 20/0 | 4.076 | 260.858 | 144.100/289.207 | 12.457/12.919 | 928.853/1074.069 | True |
| enabled | 1 | 1024 | 256 | 20/0 | 0.284 | 72.652 | 134.655/137.039 | 13.300/13.317 | 3526.041/3531.420 | True |

## Same-workload speedups

| Cache | Input | Output | SGLang output tok/s | Transformers output tok/s | Output-tok speedup | Request speedup | Same lengths |
|---|---:|---:|---:|---:|---:|---:|---|
| enabled | 256 | 64 | 71.800 | 21.394 | 3.356x | 3.356x | True |
| enabled | 1024 | 256 | 72.933 | 22.548 | 3.235x | 3.235x | True |

## Reliability and limitations

- Every case records completed/failed requests and explicitly checks actual input/output token counts. A `false` same-length value must not be used for request/s comparisons.
- Direct Transformers single-stream calls report E2E latency but no TTFT/TPOT because there is no streaming endpoint in `model.generate`.
- `transformers serve --help` was tested and failed during CLI construction due to the installed dependency mismatch; no concurrent Transformers service or concurrent speedup is claimed.
- SGLang radix-cache `enabled` and `disabled` are separate runs selected by `--cache-mode`; MTP and speculative decoding are disabled.

## Re-run

```bash
cd /home/lhzhang/qwen35_mmlu_eval
bash scripts/run_throughput.sh --backend all --quick --cache-mode enabled
.venv/bin/python scripts/summarize_throughput.py --raw-dir outputs/throughput/raw/<RUN_ID>
```
