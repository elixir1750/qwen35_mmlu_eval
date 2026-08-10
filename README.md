# Qwen3.5 MMLU 三种设定评测

本仓库提供 Qwen3.5 Base、Post-trained Thinking、Post-trained Non-Thinking 的可复查生成式 MMLU-Pro/MMLU-Redux pipeline。模型 registry、固定 Hugging Face revision、EvalScope adapter 信息和资源策略分别在 `configs/models.json`、`configs/benchmarks.json`、`configs/resources.json`。

模型权重、`.venv`、raw prediction/review JSONL、运行日志和 Slurm 临时输出不提交 Git。checkpoint 必须是固定 revision 的原始 Hugging Face safetensors，不使用量化、GGUF、MTP 或 speculative decoding。

## 唯一必要入口

普通用户只需提供模型名和 benchmark：

```bash
RUN_MODE=smoke bash scripts/run_benchmark.sh Qwen/Qwen3.5-0.8B mmlu_pro
bash scripts/run_benchmark.sh Qwen/Qwen3.5-2B-Base mmlu_redux
```

Post-trained 模型默认顺序运行 Thinking 和 Non-Thinking；名称以 `-Base` 结尾的模型只运行 Base。默认 `RUN_MODE=full`、BF16、seed 42、context length 65536。可用 `MODE=thinking|non_thinking|both`、`RESUME=1`、`NO_AUTO_DOWNLOAD=1`、`TP_SIZE`、`EVAL_BATCH_SIZE` 等环境变量覆盖。

输出为：

```text
outputs/<smoke|full>/<model_tag>/<setting>/<benchmark>/<run_id>/
```

每个 run 保存 manifest、API readiness、mode smoke、EvalScope 配置、summary 和 diagnostics；Thinking 的服务端使用 `qwen3` reasoning parser，Non-Thinking 和 Base 不使用 parser。

## 矩阵

```bash
bash scripts/run_benchmark_matrix.sh --dry-run --exclude-9b
bash scripts/run_benchmark_matrix.sh --exclude-9b
```

矩阵入口仍然调用同一个两参数底层入口。9B 默认不提交，只有存在合适 BF16 GPU 时显式使用 `--include-9b`；不会自动量化。

## 汇总和测试

```bash
python scripts/summarize_three_setting.py
python -m unittest -v scripts.test_summarize_eval scripts.test_benchmark_config
bash -n scripts/*.sh
```

统一结果见 `env/mmlu_three_setting_summary.json` 和 `MMLU_THREE_SETTING_REPORT.md`。历史 4B 结果仍保留在原来的 `REPORT.md`、`env/mmlu_*_full_summary.json`，不会被新矩阵覆盖。

旧的 `run_smoke.sh` / `run_full_eval.sh` 只作为兼容 wrapper，要求显式设置 `MODEL_NAME`；新实验请直接使用 `run_benchmark.sh`。
