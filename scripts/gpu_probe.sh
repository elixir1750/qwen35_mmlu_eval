#!/usr/bin/env bash
set -euxo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
echo "timestamp=$(date -Is)"
hostname
nvidia-smi
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
TORCHINDUCTOR_CACHE_DIR="/tmp/qwen35_torchinductor_probe_${SLURM_JOB_ID:-local}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" "$PROJECT_DIR/.venv/bin/python" -c 'import torch; print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("cuda_available",torch.cuda.is_available()); print("device_count",torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]'
TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" "$PROJECT_DIR/.venv/bin/python" -c 'import sglang; print("sglang",getattr(sglang,"__version__","<none>"),sglang.__file__)'
ls -lh "$PROJECT_DIR/model/Qwen3.5-4B"/model.safetensors*
