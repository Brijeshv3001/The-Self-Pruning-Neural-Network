#!/usr/bin/env bash
set -e
echo "=== Rehabilitation Study ==="
echo "Testing gate rehabilitation with aggressive pruning + self-healing"
python src/train.py --lambda_val 1e-3 --epochs 40 --rehab_thresh 0.05 \
  --rehab_epochs 3 --save_dir experiments/rehab_study
echo "Rehabilitation study complete."
