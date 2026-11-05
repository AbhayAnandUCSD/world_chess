#!/usr/bin/env bash
# Generic experiment pipeline (P1 recipe + P1.5 analyses) for the player pair in $WC_CONFIG under $WC_ROOT.
# Usage: WC_ROOT=exp_p2 WC_CONFIG=p2 scripts/run_exp.sh STAGE GPU [GPU ...]
#   stage sweep  : probe refit (02) + dev LR sweep, then select+lock lr/steps
#   stage final  : final fine-tunes, dose + hot runs, evals 04/05/06/07
#   stage seeds  : seed + data-subsample fine-tunes and their 09 analysis
#   stage analyze: 09 trajectories, 10 causal, 11 geometry
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID; PY=$PWD/.venv/bin/python   # PCI order = nvidia-smi numbering
STAGE=$1; shift; G=("$@"); i=0
R=${WC_ROOT:-.}; mkdir -p $R/runs $R/results $R/ckpts
read -r A B < <($PY -c "from wc import config; print(*config.load()['players'])")
cfg() { $PY -c "from wc import config; c=config.load()['finetune']; print(c['$1'])"; }
# NOTE: call nextgpu (sets $g) directly, never inside $(...) — a subshell would lose the counter
nextgpu() { g=${G[$((i % ${#G[@]}))]}; i=$((i+1)); }
MAXJOBS=${MAXJOBS:-99}   # cap on concurrent fine-tunes (each needs ~12 GB at batch 16 x 1023 chars)
ft() { while [ $(jobs -rp | wc -l) -ge $MAXJOBS ]; do wait -n; done; nextgpu; log=$R/runs/ft_$(echo "$@" | tr ' ' '_' | tr -d '-' | cut -c1-90).log; CUDA_VISIBLE_DEVICES=$g $PY scripts/03_finetune.py "$@" > $log 2>&1 & }
BASE=lichess_8layers_ckpt_no_optimizer.pt; SW=$(cfg schedule_steps)
echo "[$(date +%T)] stage=$STAGE players=$A,$B root=$R gpus=${G[*]}"
case $STAGE in
sweep)
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/02_probe_base.py > $R/runs/02_probe_base.log 2>&1 &
  P02=$!
  LRS=$($PY -c "from wc import config; print(*config.load()['finetune'].get('sweep_lrs', ['3e-6','5e-6','1e-5']))")
  DE=$($PY -c "from wc import config; print(config.load()['finetune'].get('sweep_dev_every', 20))")
  for lr in $LRS; do for p in $A $B; do ft $p --lr $lr --dropout 0.0 --steps $SW --schedule_steps $SW --dev_every $DE --tag _s3_lr${lr}_dp0.0; done; done
  for j in $(jobs -p); do [ "$j" != "$P02" ] && wait $j; done
  $PY scripts/select_config.py s3 | tee $R/runs/select.log
  wait $P02 ;;
final)
  ST=$(cfg steps)
  ft $A; ft $B
  ft $A --steps $SW --ckpt_every 20 --seed 0 --tag _p15_dose; ft $B --steps $SW --ckpt_every 20 --seed 0 --tag _p15_dose
  HS=$((SW / 3)); HE=$((HS / 10))   # 'hot' overfit run: lr 3e-5 for a third of the sweep horizon (P1: 400 steps)
  ft $A --lr 3e-5 --steps $HS --schedule_steps $HS --ckpt_every $HE --tag _p15_hot; ft $B --lr 3e-5 --steps $HS --schedule_steps $HS --ckpt_every $HE --tag _p15_hot
  wait
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/04_crossover.py test $ST > $R/runs/04_test.log 2>&1 &
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/04_crossover.py h2h $ST > $R/runs/04_h2h.log 2>&1 &
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/06_sensitivity.py > $R/runs/06.log 2>&1 &
  wait
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/05_probe_transfer.py test > $R/runs/05_test.log 2>&1
  $PY scripts/07_style.py > $R/runs/07.log 2>&1 ;;
seeds)
  ST=$(cfg steps)
  for p in $A $B; do for s in 1 2 3 4; do ft $p --seed $s --ckpt_every $ST --tag _p15_s$s; done
                     for d in 1 2; do ft $p --seed 0 --data_seed $d --ckpt_every $ST --tag _p15_d$d; done; done
  wait
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/09_analysis.py seeds $(for p in $A $B; do for t in s1 s2 s3 s4 d1 d2; do echo $R/ckpts/ft_${p}_p15_$t/step$ST.pt; done; done) > $R/runs/09_seeds.log 2>&1 ;;
analyze)
  ST=$(cfg steps)
  traj() { p=$1; for s in $(printf "%s\n" 20 40 60 100 200 400 800 $ST $SW | sort -nu); do echo $R/ckpts/ft_${p}_p15_dose/step$s.pt; done; HS=$((SW / 3)); HE=$((HS / 10)); for s in $HE $((2*HE)) $((5*HE)) $HS; do echo $R/ckpts/ft_${p}_p15_hot/step$s.pt; done; }
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/09_analysis.py traj_$A $BASE $(traj $A) > $R/runs/09_traj_$A.log 2>&1 &
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/09_analysis.py traj_$B $(traj $B) > $R/runs/09_traj_$B.log 2>&1 &
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/10_causal.py p15 $BASE $R/ckpts/ft_$A/step$ST.pt $R/ckpts/ft_$B/step$ST.pt $R/ckpts/ft_${A}_p15_hot/step$((SW/3)).pt $R/ckpts/ft_${B}_p15_hot/step$((SW/3)).pt > $R/runs/10.log 2>&1 &
  nextgpu; CUDA_VISIBLE_DEVICES=$g $PY scripts/11_geometry.py p15 $R/ckpts/ft_$A/step$ST.pt $R/ckpts/ft_$B/step$ST.pt $R/ckpts/ft_${A}_p15_hot/step$((SW/3)).pt $R/ckpts/ft_${B}_p15_hot/step$((SW/3)).pt > $R/runs/11.log 2>&1 &
  wait ;;
esac
echo "[$(date +%T)] stage=$STAGE DONE"
