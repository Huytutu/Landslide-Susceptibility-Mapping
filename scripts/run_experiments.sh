#!/usr/bin/env bash
# Train every model with several seeds, then print the mean ± std table.
#   bash scripts/run_experiments.sh            # seeds 0-4
#   SEEDS="0 1 2" bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
MODELS=${MODELS:-"UNet ResUNet MUNet ResMUNet"}

for seed in $SEEDS; do
  for model in $MODELS; do
    if [[ -f runs/$model/seed$seed/results.json ]]; then
      echo "skip $model seed$seed (done)"; continue
    fi
    python train.py --model "$model" --seed "$seed" "$@"
  done
done

python summarize.py --compare ResMUNet ResUNet
python summarize.py --compare MUNet UNet
