#!/usr/bin/env bash
# Launch P1 GPU jobs on a shared server. Usage: scripts/remote_launch.sh GPU_PROBE GPU_KARPOV GPU_KASPAROV
set -euo pipefail
cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
mkdir -p runs results
CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python scripts/02_probe_base.py      > runs/02_probe_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 nohup .venv/bin/python scripts/03_finetune.py karpov   > runs/03_finetune_karpov.log 2>&1 &
CUDA_VISIBLE_DEVICES=$3 nohup .venv/bin/python scripts/03_finetune.py kasparov > runs/03_finetune_kasparov.log 2>&1 &
sleep 2; pgrep -af "scripts/0[23]_" || true
