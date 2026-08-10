# Qwen3.5-4B MMLU Evaluation

可复查的 `Qwen/Qwen3.5-4B` + SGLang + EvalScope benchmark pipeline，覆盖 MMLU-Pro 和 MMLU-Redux。

## Repository contents

- `scripts/`: system inspection, SGLang launch, smoke/full evaluation and report-generation scripts.
- `env/`: resolved hardware/software/model/benchmark metadata and full-evaluation summaries.
- `REPORT.md`: completed local reproduction report.

为避免 GitHub 大文件和凭据风险，模型权重、`.venv`、raw prediction/review JSONL、运行日志和 Slurm 临时日志不纳入 Git。模型必须单独下载为原始 `Qwen/Qwen3.5-4B` safetensors。

## Re-run

```bash
cd qwen35_mmlu_eval
uv venv .venv --python 3.12
# 依照 env/software_versions.txt 安装依赖

export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/hf download Qwen/Qwen3.5-4B --local-dir model/Qwen3.5-4B

bash scripts/run_smoke.sh
PORT=8000 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_pro
PORT=8001 EVAL_BATCH_SIZE=8 bash scripts/run_full_eval.sh mmlu_redux
```

详细配置、版本、dataset ID、shot、generation 参数和 job 信息见 `env/eval_config.json`。
