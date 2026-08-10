#!/usr/bin/env bash
set +e

echo "timestamp=$(date -Is)"
echo "hostname=$(hostname)"
echo
echo "== python =="
python --version
echo
echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi: not found on this login node"
fi
echo
echo "== nvcc =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc: not found on this login node"
fi
echo
echo "== cuda locations =="
for cuda_dir in /data1/public/cuda/cuda-*; do
  if [ -x "$cuda_dir/bin/nvcc" ]; then
    echo "[$cuda_dir]"
    "$cuda_dir/bin/nvcc" --version
  fi
done
echo
echo "== memory =="
free -h
echo
echo "== disk =="
df -h .
echo
echo "== devices =="
ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia* on this login node"
echo
echo "== slurm gpu partitions =="
if command -v sinfo >/dev/null 2>&1; then
  timeout 30s sinfo -a -o '%P|%a|%D|%G|%m|%l|%N' || true
else
  echo "sinfo: not found"
fi

exit 0
