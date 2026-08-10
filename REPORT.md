# Qwen3.5-4B MMLU Evaluation Report

Generated: 2026-08-09T01:48:35+08:00

## Environment

- GPU: NVIDIA GeForce RTX 3090 × 1; VRAM per GPU: 24 GiB
- Driver/CUDA: 570.153.02 / 12.8; TP size: 1
- Python: 3.12.8; PyTorch: 2.9.1+cu128
- SGLang: 0.5.10; EvalScope: 1.10.0; OpenAI client: 2.6.1
- Dependency index: Tsinghua PyPI mirror; model endpoint: `https://hf-mirror.com`.

## Model

- Repository: `Qwen/Qwen3.5-4B`
- Local path: `/home/lhzhang/qwen35_mmlu_eval/model/Qwen3.5-4B`
- Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Model type: post-trained Qwen3.5-4B; original Hugging Face safetensors checkpoint; no quantization, GGUF/AWQ/GPTQ/MLX conversion, MTP, or speculative decoding.

## Evaluation Protocol

- SGLang: `tp_size=1`, `mem_fraction_static=0.75`, `context_length=65536`, reasoning parser `qwen3`, seed `42`.
- Generation: temperature `1.0`, top_p `0.95`, top_k `20`, presence_penalty `1.5`, repetition_penalty `1.0`, max_tokens `32768`, timeout `600s`, retries `3`.
- MMLU-Pro: `TIGER-Lab/MMLU-Pro`, test split, 5-shot CoT, A–J, standard total 12032 samples.
- MMLU-Redux: `AI-ModelScope/mmlu-redux-2.0`, test split, 0-shot CoT, A–D, standard total 5700 samples.
- Prompt and answer extraction use the installed EvalScope 1.10.0 built-in adapters; no benchmark prompt or ground truth was rewritten.

## Results

| Benchmark | Qwen official | Local result | Delta |
|---|---:|---:|---:|
| MMLU-Pro | 79.1% | 78.27% | -0.83 pp |
| MMLU-Redux | 88.8% | 88.68% | -0.12 pp |

Per-subset full results are stored in `env/mmlu_pro_full_summary.json` and `env/mmlu_redux_full_summary.json`; EvalScope native reports are under `outputs/full/*/reports/`.

## Reliability diagnostics

| Benchmark | Total | Successful | Invalid | Truncated | API failures | Retry log occurrences | Dataset timeouts | Elapsed (s) | Samples/s | Avg output tokens | Max output tokens | Avg reasoning chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 12032 | 12032 | 4 | 0 | 0 | 0 | 0 | 109829.000 | 0.110 | 4636.415 | 28840.000 | 14426.202 |
| MMLU-Redux | 5700 | 5700 | 2 | 0 | 0 | 4 | 11 | 35573.000 | 0.160 | 3036.973 | 24221.000 | 10488.092 |

The `*_diagnostics.json` files contain up to 10 wrong, invalid, truncated, and API-error samples for inspection. Smoke tests manually inspected six samples per benchmark; all had reasoning, final content, valid extracted answers, and `stop` termination.

## Error inspection and interpretation

- The first standard-recipe smoke results were MMLU-Pro 78.57% and MMLU-Redux 92.50%; both had zero invalid, truncated, and API-error samples.
- This is explicitly a `Qwen3.5-4B + SGLang + EvalScope standard-recipe reproduction`, not an official exact-recipe claim. The model card does not publish every evaluator implementation detail needed to establish exact identity.
- Based on the largest full-evaluation gap, the result is classified as **close reproduction**.
- If the gap is material, the next variants should be recorded separately and checked in order: model/revision, thinking/parser behavior, dataset revision, shot/prompt template, extraction, max_tokens/truncation, then sampling randomness. The first standard result must remain unchanged.

## Reproduction commands

```bash
cd /home/lhzhang/qwen35_mmlu_eval
bash scripts/run_smoke.sh
PORT=8000 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_pro
PORT=8001 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_redux
```

Key commands and all resolved settings are archived in `env/eval_config.json`; raw logs and JSONL outputs are retained under `logs/` and `outputs/`.
