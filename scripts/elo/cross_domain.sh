#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu=$1
band=$2
domain=$3
wait_name=$4
while ! grep -q "^COMPLETE $wait_name$" "exp_elo/runs/full_$wait_name.log"; do
  if grep -q "^FAILED $wait_name$" "exp_elo/runs/full_$wait_name.log"; then
    exit 1
  fi
  sleep 20
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu WC_KEYFILE_ONLY=1 PYTHONUNBUFFERED=1
if [ "$domain" = otb ]; then
  export WC_ROOT=. WC_CONFIG=p1
else
  export WC_ROOT=exp_p2 WC_CONFIG=p2
fi
checkpoints=()
for step in 100 200 400 800 1600 3200 6000; do
  checkpoints+=("exp_elo/ckpts/ft_${band}_p15_dose/step$step.pt")
done
for step in 200 400 800 1600 2000; do
  checkpoints+=("exp_elo/ckpts/ft_${band}_p15_hot/step$step.pt")
done
.venv/bin/python scripts/09_analysis.py "xdom_$band" "${checkpoints[@]}"
echo "COMPLETE $domain $band"
