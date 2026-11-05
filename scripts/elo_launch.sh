#!/usr/bin/env bash
# Launch Elo-experiment fine-tunes on this host, one per spec "band:lr:gpu[:extra args with commas]".
# e.g. scripts/elo_launch.sh low:1e-06:0 high:1e-06:0 low:3e-06:1
set -uo pipefail; cd "$(dirname "$0")/.."
export WC_ROOT=exp_elo WC_CONFIG=elo WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p exp_elo/runs exp_elo/results exp_elo/ckpts
SW=$(.venv/bin/python -c "from wc import config; print(config.load()['finetune']['schedule_steps'])")
DE=$(.venv/bin/python -c "from wc import config; print(config.load()['finetune']['sweep_dev_every'])")
for spec in "$@"; do
  IFS=: read -r band lr gpu extra <<< "$spec"
  tag=_s3_lr${lr}_dp0.0
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup .venv/bin/python scripts/03_finetune.py $band --lr $lr --dropout 0.0 \
      --steps $SW --schedule_steps $SW --dev_every $DE --tag $tag ${extra//,/ } \
      > exp_elo/runs/ft_${band}${tag}.log 2>&1 < /dev/null &
  echo "launched $band lr=$lr gpu=$gpu"
done
