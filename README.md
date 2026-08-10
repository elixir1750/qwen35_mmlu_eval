# Qwen3.5-4B MMLU Evaluation

可复查的 `Qwen/Qwen3.5-4B` + SGLang + EvalScope benchmark pipeline，覆盖 MMLU-Pro 和 MMLU-Redux。

## Repository contents

- `scripts/`: system inspection, SGLang launch, smoke/full evaluation, throughput benchmark and report-generation scripts.
- `env/`: resolved hardware/software/model/benchmark metadata and full-evaluation summaries.
- `REPORT.md`: completed local reproduction report.
- `THROUGHPUT_REPORT.md`: structured SGLang versus direct-Transformers throughput report.

为避免 GitHub 大文件和凭据风险，模型权重、`.venv`、raw prediction/review JSONL、运行日志和 Slurm 临时日志不纳入 Git。模型必须单独下载为原始 `Qwen/Qwen3.5-4B` safetensors。

## Re-run

```bash
cd qwen35_mmlu_eval
uv venv .venv --python 3.12
# 依照 env/software_versions.txt 安装依赖

export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/hf download Qwen/Qwen3.5-4B --local-dir model/Qwen3.5-4B

bash scripts/run_smoke.sh
PRO_ID="$(date +%Y%m%d_%H%M%S)_pro"
REDUX_ID="$(date +%Y%m%d_%H%M%S)_redux"
PORT=8000 EVAL_BATCH_SIZE=8 RUN_ID="$PRO_ID" bash scripts/run_full_eval.sh mmlu_pro
PORT=8001 EVAL_BATCH_SIZE=8 RUN_ID="$REDUX_ID" bash scripts/run_full_eval.sh mmlu_redux
bash scripts/postprocess_full.sh \
  "outputs/full/mmlu_pro/$PRO_ID" "outputs/full/mmlu_redux/$REDUX_ID" \
  "REPORT_${PRO_ID}_${REDUX_ID}.md"

# quick throughput: SGLang online + persistent direct Transformers single-stream
bash scripts/run_throughput.sh --backend all --quick --cache-mode enabled
.venv/bin/python scripts/summarize_throughput.py --raw-dir outputs/throughput/raw/<RUN_ID>
```

`run_smoke.sh` 和 `run_full_eval.sh` 默认生成带时间戳的独立目录，并写入
`run_manifest.json`；重复使用已有目录必须显式设置 `RESUME=1`。完整评测的旧结果
不会被新运行覆盖。吞吐测试的 raw JSON、日志和 manifest 位于
`outputs/throughput/{raw,logs,manifests}/`，不会提交 Git。

`evalscope perf` 是当前安装版本的实际性能入口，但固定 token-ID 输入、固定输出长度、
TTFT/TPOT 和逐请求长度校验由 `scripts/throughput_benchmark.py` 直接完成，以避免
自由生成长度不同导致错误的 request/s 比较。当前环境的 `transformers serve` CLI
因 `transformers==5.3.0` 与 `huggingface_hub==1.26.1` 注解兼容问题无法启动，因此
Transformers 并发在线服务不冒充可比结果；报告只计算同长度单流 output-tok speedup。

详细配置、版本、dataset ID、shot、generation 参数和 job 信息见 `env/eval_config.json`。
