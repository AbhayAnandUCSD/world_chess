#!/usr/bin/env bash
# P1.5 analysis after run_p15_train.sh. Usage: run_p15_analysis.sh G0 G1 G2
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1; PY=.venv/bin/python
until grep -q TRAIN_DONE runs/p15_train.log; do sleep 15; done
K=ckpts/ft_karpov_p15; S=ckpts/ft_kasparov_p15; BASE=lichess_8layers_ckpt_no_optimizer.pt
traj() { p=$1; echo ckpts/ft_${p}_p15_dose/step{20,40,60,100,200,400}.pt ckpts/ft_${p}_p15_hot/step{40,80,200,400}.pt; }
CUDA_VISIBLE_DEVICES=$1 $PY scripts/09_analysis.py traj_karpov $BASE $(traj karpov) > runs/p15_09_traj_karpov.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 $PY scripts/09_analysis.py traj_kasparov $(traj kasparov) > runs/p15_09_traj_kasparov.log 2>&1 &
( CUDA_VISIBLE_DEVICES=$3 $PY scripts/09_analysis.py seeds $(for p in karpov kasparov; do for t in s1 s2 s3 s4 d1 d2; do echo ckpts/ft_${p}_p15_$t/step60.pt; done; done) > runs/p15_09_seeds.log 2>&1
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/10_causal.py p15 $BASE ${K}_dose/step60.pt ${S}_dose/step60.pt ${K}_hot/step400.pt ${S}_hot/step400.pt > runs/p15_10.log 2>&1 ) &
wait; echo ANALYSIS_DONE
