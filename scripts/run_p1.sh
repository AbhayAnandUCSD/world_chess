#!/usr/bin/env bash
# Full P1 pipeline on a GPU server. Usage: scripts/run_p1.sh GPU_A GPU_B GPU_C
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; A=$1; B=$2; C=$3
echo "[$(date +%T)] probe refit + sweep"
CUDA_VISIBLE_DEVICES=$A $PY scripts/02_probe_base.py > runs/02_probe_base.log 2>&1 &
PROBE=$!
i=0
for lr in 3e-6 5e-6 1e-5; do for p in karpov kasparov; do
  g=$([ $((i % 2)) = 0 ] && echo $B || echo $C); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/03_finetune.py $p --lr $lr --dropout 0.0 --steps 400 --dev_every 20 --tag _s3_lr${lr}_dp0.0 > runs/sweep3_${p}_lr$lr.log 2>&1 &
done; done
wait $(jobs -p | grep -v "^$PROBE$") 2>/dev/null || true
echo "[$(date +%T)] select"
$PY scripts/select_config.py s3
echo "[$(date +%T)] final fine-tunes"
CUDA_VISIBLE_DEVICES=$B $PY scripts/03_finetune.py karpov > runs/03_final_karpov.log 2>&1 &
CUDA_VISIBLE_DEVICES=$C $PY scripts/03_finetune.py kasparov > runs/03_final_kasparov.log 2>&1 &
wait
echo "[$(date +%T)] evals"
STEPS=$(grep -E "^  steps:" configs/p1.yaml | awk '{print $2}')
CUDA_VISIBLE_DEVICES=$A $PY scripts/04_crossover.py test $STEPS > runs/04_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B $PY scripts/04_crossover.py h2h $STEPS > runs/04_h2h.log 2>&1 &
CUDA_VISIBLE_DEVICES=$C $PY scripts/06_sensitivity.py > runs/06.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=$A $PY scripts/05_probe_transfer.py test > runs/05_test.log 2>&1
$PY scripts/07_style.py > runs/07.log 2>&1
echo "[$(date +%T)] DONE"; ls results
