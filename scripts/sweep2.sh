#!/usr/bin/env bash
# Sweep 2: lr x dropout, both players, spread over given GPUs. Usage: scripts/sweep2.sh 0 1 2
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
GPUS=("$@"); i=0
for lr in 2e-6 3e-6 5e-6 1e-5; do for dp in 0.0 0.1; do for p in karpov kasparov; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/03_finetune.py $p --lr $lr --dropout $dp --steps 400 --dev_every 20 --tag _s2_lr${lr}_dp${dp} > runs/sweep2_${p}_lr${lr}_dp${dp}.log 2>&1 &
done; done; done
wait
