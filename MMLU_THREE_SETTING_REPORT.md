# Qwen3.5 三种设定 MMLU 评测报告

> 生成日期：2026-08-10；代码提交：`690dc68ff1c1f7263d739462b6cefbc4bad4d6f3`

## Technical summary

本报告比较同尺寸 Qwen3.5 Base、Post-trained Thinking 与 Post-trained Non-Thinking。
Post-trained 行使用官方 model card 公开的对应分数作为参考；Base 是将已验证的生成式 EvalScope 协议迁移到 Base checkpoint，标记为 `protocol_transfer`，不宣称官方复现。

## Environment

- Python: `Python 3.12.8`; PyTorch: `2.9.1+cu128`; CUDA reported by PyTorch: `12.8`.
- SGLang: `0.5.10`; EvalScope: `1.10.0`.
- GPU name, driver, VRAM, TP size and Slurm job are taken from each run manifest; they are not inferred from the login node.

## 结果总表

| Size | Setting | Benchmark | Official | Local | Delta | Correct/Total | Invalid | Truncated | API failures | Avg output tokens | Avg reasoning chars | Elapsed(s) | Revision | Provenance | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0.8B | thinking | mmlu_pro | 42.3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 2fc06364715b967f1860aea9cf38778875588b17 | official_reproduction | pending |
| 0.8B | thinking | mmlu_redux | 59.5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 2fc06364715b967f1860aea9cf38778875588b17 | official_reproduction | pending |
| 0.8B | non_thinking | mmlu_pro | 29.7 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 2fc06364715b967f1860aea9cf38778875588b17 | official_reproduction | pending |
| 0.8B | non_thinking | mmlu_redux | 48.5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 2fc06364715b967f1860aea9cf38778875588b17 | official_reproduction | pending |
| 0.8B | base | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 | protocol_transfer | pending |
| 0.8B | base | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 | protocol_transfer | pending |
| 2B | thinking | mmlu_pro | 66.5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 15852e8c16360a2fea060d615a32b45270f8a8fc | official_reproduction | pending |
| 2B | thinking | mmlu_redux | 79.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 15852e8c16360a2fea060d615a32b45270f8a8fc | official_reproduction | pending |
| 2B | non_thinking | mmlu_pro | 55.3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 15852e8c16360a2fea060d615a32b45270f8a8fc | official_reproduction | pending |
| 2B | non_thinking | mmlu_redux | 69.2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 15852e8c16360a2fea060d615a32b45270f8a8fc | official_reproduction | pending |
| 2B | base | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | b1485b2fa6dfa1287294f269f5fb618e03d52d7c | protocol_transfer | pending |
| 2B | base | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | b1485b2fa6dfa1287294f269f5fb618e03d52d7c | protocol_transfer | pending |
| 4B | thinking | mmlu_pro | 79.1 | 78.2663 | -0.8337 | 9417/12032 | 4 | 0 | 0 | n/a | n/a | n/a | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | historical_reused | completed_historical |
| 4B | thinking | mmlu_redux | 88.8 | 88.6842 | -0.1158 | 5055/5700 | 2 | 0 | 0 | n/a | n/a | n/a | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | historical_reused | completed_historical |
| 4B | non_thinking | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | official_reproduction | pending |
| 4B | non_thinking | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | official_reproduction | pending |
| 4B | base | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1001bb4d826a52d1f399e183466143f4da7b741b | protocol_transfer | pending |
| 4B | base | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1001bb4d826a52d1f399e183466143f4da7b741b | protocol_transfer | pending |
| 9B | thinking | mmlu_pro | 82.5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | c202236235762e1c871ad0ccb60c8ee5ba337b9a | official_reproduction | pending |
| 9B | thinking | mmlu_redux | 91.1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | c202236235762e1c871ad0ccb60c8ee5ba337b9a | official_reproduction | pending |
| 9B | non_thinking | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | c202236235762e1c871ad0ccb60c8ee5ba337b9a | official_reproduction | pending |
| 9B | non_thinking | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | c202236235762e1c871ad0ccb60c8ee5ba337b9a | official_reproduction | pending |
| 9B | base | mmlu_pro | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 68c46c4b3498877f3ef123c856ecfde50c39f404 | protocol_transfer | pending |
| 9B | base | mmlu_redux | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 68c46c4b3498877f3ef123c856ecfde50c39f404 | protocol_transfer | pending |

## Protocol and data provenance

- Model card URL/read date: registry entries under `configs/models.json`, read on `2026-08-10`.
- Checkpoints: fixed Hugging Face commit revisions in `configs/models.json`; runtime manifests record local file SHA256 values.
- Precision: BF16; SGLang; MTP and speculative decoding disabled; reasoning parser is enabled only for Thinking.
- Thinking uses `chat_template_kwargs.enable_thinking=true`; Non-Thinking uses `false`; Base sends neither field and does not use a reasoning parser.
- MMLU-Pro: EvalScope built-in adapter, test split, 5-shot from validation, expected 12,032 samples.
- MMLU-Redux: EvalScope built-in adapter, test split, 0-shot, expected 5,700 samples.
- Prompt rendering and answer extraction are not rewritten; actual installed EvalScope adapter is the source of truth.

## Same-size comparisons

| Size | Benchmark | Thinking - Base | Non-Thinking - Base | Thinking - Non-Thinking |
|---|---|---:|---:|---:|
| 0.8B | mmlu_pro | n/a | n/a | n/a |
| 0.8B | mmlu_redux | n/a | n/a | n/a |
| 2B | mmlu_pro | n/a | n/a | n/a |
| 2B | mmlu_redux | n/a | n/a | n/a |
| 4B | mmlu_pro | n/a | n/a | n/a |
| 4B | mmlu_redux | n/a | n/a | n/a |
| 9B | mmlu_pro | n/a | n/a | n/a |
| 9B | mmlu_redux | n/a | n/a | n/a |

## Smoke verification

Smoke 结果只用于验证数据加载、API、模式字段、parser 和答案抽取，不替代 full accuracy。

| Model | Size | Setting | Benchmark | Samples | Accuracy | Invalid | Truncated | API failures | Slurm job | Run |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen/Qwen3.5-0.8B | 0.8B | thinking | mmlu_pro | 42 | 45.2381 | 0 | 0 | 0 | 94609 | outputs/smoke/qwen35_0_8b_posttrained/thinking/mmlu_pro/smoke_0p8b_thinking_80k_131k_pro |
| Qwen/Qwen3.5-0.8B | 0.8B | non_thinking | mmlu_pro | 420 | 34.7619 | 9 | 9 | 0 | 94561 | outputs/smoke/qwen35_0_8b_posttrained/non_thinking/mmlu_pro/smoke_0p8b_pro_retry_non_thinking |
| Qwen/Qwen3.5-0.8B | 0.8B | non_thinking | mmlu_redux | 30 | 36.6667 | 2 | 2 | 0 | 94619 | outputs/smoke/qwen35_0_8b_posttrained/non_thinking/mmlu_redux/smoke_0p8b_non_thinking_redux |
| Qwen/Qwen3.5-0.8B-Base | 0.8B | base | mmlu_pro | 42 | 35.7143 | 0 | 0 | 0 | 94619 | outputs/smoke/qwen35_0_8b_base/base/mmlu_pro/smoke_0p8b_base_pro |
| Qwen/Qwen3.5-0.8B-Base | 0.8B | base | mmlu_redux | 30 | 46.6667 | 2 | 0 | 0 | 94619 | outputs/smoke/qwen35_0_8b_base/base/mmlu_redux/smoke_0p8b_base_redux |
| Qwen/Qwen3.5-2B | 2B | thinking | mmlu_pro | 42 | 57.1429 | 0 | 0 | 0 | 94690 | outputs/smoke/qwen35_2b_posttrained/thinking/mmlu_pro/smoke_2b_matrix_pro_r1_thinking |
| Qwen/Qwen3.5-2B | 2B | thinking | mmlu_redux | 3 | 66.6667 | 0 | 0 | 0 | 94691 | outputs/smoke/qwen35_2b_posttrained/thinking/mmlu_redux/smoke_2b_matrix_redux_r1_thinking |
| Qwen/Qwen3.5-2B | 2B | non_thinking | mmlu_pro | 42 | 47.619 | 0 | 0 | 0 | 94690 | outputs/smoke/qwen35_2b_posttrained/non_thinking/mmlu_pro/smoke_2b_matrix_pro_r1_non_thinking |
| Qwen/Qwen3.5-2B | 2B | non_thinking | mmlu_redux | 3 | 33.3333 | 0 | 0 | 0 | 94691 | outputs/smoke/qwen35_2b_posttrained/non_thinking/mmlu_redux/smoke_2b_matrix_redux_r1_non_thinking |
| Qwen/Qwen3.5-2B-Base | 2B | base | mmlu_pro | 42 | 47.619 | 3 | 1 | 0 | 94699 | outputs/smoke/qwen35_2b_base/base/mmlu_pro/smoke_2b_base_pro_r1 |
| Qwen/Qwen3.5-2B-Base | 2B | base | mmlu_redux | 3 | 33.3333 | 0 | 0 | 0 | 94700 | outputs/smoke/qwen35_2b_base/base/mmlu_redux/smoke_2b_base_redux_r1 |
| Qwen/Qwen3.5-4B | 4B | thinking | mmlu_pro | 42 | 80.9524 | 0 | 0 | 0 | 94486 | outputs/smoke/qwen35_4b_posttrained/thinking/mmlu_pro/smoke_validation_4b_thinking2 |
| Qwen/Qwen3.5-4B | 4B | thinking | mmlu_redux | 3 | 66.6667 | 0 | 0 | 0 | 94693 | outputs/smoke/qwen35_4b_posttrained/thinking/mmlu_redux/smoke_4b_matrix_redux_r1_thinking |
| Qwen/Qwen3.5-4B | 4B | non_thinking | mmlu_pro | 42 | 76.1905 | 0 | 0 | 0 | 94707 | outputs/smoke/qwen35_4b_posttrained/non_thinking/mmlu_pro/smoke_4b_matrix_pro_non_thinking_r2 |
| Qwen/Qwen3.5-4B | 4B | non_thinking | mmlu_redux | 3 | 33.3333 | 0 | 0 | 0 | 94693 | outputs/smoke/qwen35_4b_posttrained/non_thinking/mmlu_redux/smoke_4b_matrix_redux_r1_non_thinking |
| Qwen/Qwen3.5-4B-Base | 4B | base | mmlu_pro | 42 | 73.8095 | 0 | 0 | 0 | 94701 | outputs/smoke/qwen35_4b_base/base/mmlu_pro/smoke_4b_base_pro_r1 |
| Qwen/Qwen3.5-4B-Base | 4B | base | mmlu_redux | 3 | 66.6667 | 0 | 0 | 0 | 94702 | outputs/smoke/qwen35_4b_base/base/mmlu_redux/smoke_4b_base_redux_r1 |

## Reliability diagnostics

每个 completed run 的 `summary.json` 和 `summary_diagnostics.json` 位于对应 run directory。统计中的 API failure 包括缺失 prediction；重复 prediction/review 保留原始重复计数并使用确定性的最后一条记录计分。elapsed 只接受 run manifest 的 monotonic clock。

## Limitations and next steps

- 官方 model card 并未公开完整的 MMLU 评测 recipe，因此本项目的首要定义是 `Qwen3.5 + SGLang + EvalScope standard-recipe reproduction`。
- 4B 的 78.27% / 88.68% 是此前审计过的历史 Thinking 结果，原始输出没有被覆盖；新矩阵结果与其分开标记。
- 9B 是否已在其他位置完成过两个结果，当前仓库没有发现可复查的 prediction/review/summary 资产，因此这里不代填；9B 配置已固定，资源足够时可独立运行。
- 若官方复现 delta 超过 1--2 个百分点，应先核查 revision、thinking 字段、dataset cache、shot、prompt、answer extraction、max_tokens 和随机性，并另起 variant run。

## Reproduction commands

```bash
RUN_MODE=smoke bash scripts/run_benchmark.sh Qwen/Qwen3.5-0.8B mmlu_pro
bash scripts/run_benchmark_matrix.sh --exclude-9b
python scripts/summarize_three_setting.py
```
