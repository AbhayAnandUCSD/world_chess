#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export WC_ROOT=exp_elo WC_CONFIG=elo WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1 MAXSEQ=256
name=$2
checkpoint=$3
trap 'echo "FAILED $name" >&2' ERR
mkdir -p exp_elo/results/19
.venv/bin/python scripts/18b_style_policy.py "$name" "$checkpoint"
.venv/bin/python scripts/19_rollouts.py "$name" "$checkpoint" 300 30
echo "COMPLETE $name"
