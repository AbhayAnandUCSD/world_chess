#!/usr/bin/env bash
# LR sweep on dev games. Usage: scripts/sweep_lr.sh GPU_A GPU_B
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
for lr in 3e-6 1e-5 3e-5; do
  for p in karpov kasparov; do
    g=$([ $p = karpov ] && echo $1 || echo $2)
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/03_finetune.py $p --lr $lr --steps 300 --dev_every 20 --tag _lr$lr > runs/sweep_${p}_lr$lr.log 2>&1 &
  done
done
wait
