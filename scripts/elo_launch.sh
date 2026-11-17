#!/usr/bin/env bash
# Launch Elo-experiment fine-tunes on this host, one per spec.
#   stage 1 (lr sweep):        band:lr:gpu                -> tag _s3_lr<lr>_dp0.0, SW steps
#   stage 2 (batch x wd sweep): band:lr:gpu:bs:wd          -> tag _s4_lr<lr>_dp0.0_bs<bs>_wd<wd>, SW*16/bs steps (same games seen)
# e.g. scripts/elo_launch.sh low:1e-06:0 high:3e-06:0:32:0.01
set -uo pipefail; cd "$(dirname "$0")/.."
export WC_ROOT=exp_elo WC_CONFIG=elo WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p exp_elo/runs exp_elo/results exp_elo/ckpts
SW=$(.venv/bin/python -c "from wc import config; print(config.load()['finetune']['schedule_steps'])")
DE=$(.venv/bin/python -c "from wc import config; print(config.load()['finetune']['sweep_dev_every'])")
for spec in "$@"; do
  IFS=: read -r band lr gpu bs wd <<< "$spec"
  if [ -n "${bs:-}" ]; then
    S=$((SW * 16 / bs)); D=$((DE * 16 / bs)); tag=_s4_lr${lr}_dp0.0_bs${bs}_wd${wd}; extra="--batch_games $bs --wd $wd"
  else
    S=$SW; D=$DE; tag=_s3_lr${lr}_dp0.0; extra=""
  fi
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup .venv/bin/python scripts/03_finetune.py $band --lr $lr --dropout 0.0 \
      --steps $S --schedule_steps $S --dev_every $D --tag $tag $extra \
      > exp_elo/runs/ft_${band}${tag}.log 2>&1 < /dev/null &
  echo "launched $band$tag gpu=$gpu steps=$S"
done
