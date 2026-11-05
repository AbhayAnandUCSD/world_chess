#!/usr/bin/env bash
# Final stage only (after selection locked lr/steps in configs/p1.yaml).
set -euo pipefail; cd "$(dirname "$0")/.."
export WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; A=$1; B=$2; C=$3
echo "[$(date +%T)] final fine-tunes"
CUDA_VISIBLE_DEVICES=$B $PY scripts/03_finetune.py karpov > runs/03_final_karpov.log 2>&1 &
CUDA_VISIBLE_DEVICES=$C $PY scripts/03_finetune.py kasparov > runs/03_final_kasparov.log 2>&1 &
wait
STEPS=$(grep -E "^  steps:" configs/p1.yaml | awk '{print $2}')
echo "[$(date +%T)] evals (steps=$STEPS)"
CUDA_VISIBLE_DEVICES=$A $PY scripts/04_crossover.py test $STEPS > runs/04_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B $PY scripts/04_crossover.py h2h $STEPS > runs/04_h2h.log 2>&1 &
CUDA_VISIBLE_DEVICES=$C $PY scripts/06_sensitivity.py > runs/06.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=$A $PY scripts/05_probe_transfer.py test > runs/05_test.log 2>&1
$PY scripts/07_style.py > runs/07.log 2>&1
echo "[$(date +%T)] DONE"; ls results
