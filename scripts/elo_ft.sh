#!/usr/bin/env bash
# Launch one Elo-experiment fine-tune with arbitrary 03_finetune.py args on this host.
# Usage: scripts/elo_ft.sh GPU BAND [03_finetune args ...]      (log: exp_elo/runs/ft_<band><tag>.log)
set -uo pipefail; cd "$(dirname "$0")/.."
export WC_ROOT=exp_elo WC_CONFIG=elo WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=$1; BAND=$2; shift 2
TAG=""; prev=""; for a in "$@"; do [ "$prev" = "--tag" ] && TAG=$a; prev=$a; done
CUDA_VISIBLE_DEVICES=$GPU setsid nohup .venv/bin/python scripts/03_finetune.py $BAND "$@" \
    > exp_elo/runs/ft_${BAND}${TAG}.log 2>&1 < /dev/null &
echo "launched $BAND$TAG gpu=$GPU"
