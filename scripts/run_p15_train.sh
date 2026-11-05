#!/usr/bin/env bash
# P1.5 fine-tunes: seeds, data subsamples, dose trajectory, hot (overfit) run. Usage: run_p15_train.sh G0 G1 G2
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1; PY=.venv/bin/python; G=("$@"); i=0
run() { g=${G[$((i % ${#G[@]}))]}; i=$((i+1)); CUDA_VISIBLE_DEVICES=$g $PY scripts/03_finetune.py "$@" --dev_every 1000 > runs/p15_$(echo "$@" | tr ' ' '_' | tr -d '-' | cut -c1-80).log 2>&1 & }
for p in karpov kasparov; do
  run $p --steps 400 --schedule_steps 400 --ckpt_every 20 --seed 0 --tag _p15_dose
  run $p --lr 3e-5 --steps 400 --schedule_steps 400 --ckpt_every 40 --seed 0 --tag _p15_hot
  for s in 1 2 3 4; do run $p --steps 60 --schedule_steps 400 --ckpt_every 60 --seed $s --tag _p15_s$s; done
  for d in 1 2; do run $p --steps 60 --schedule_steps 400 --ckpt_every 60 --seed 0 --data_seed $d --tag _p15_d$d; done
done
wait; echo TRAIN_DONE
