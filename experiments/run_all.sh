#!/usr/bin/env bash
# File: experiments/run_all.sh
set -euo pipefail

python src/train.py --lambda_val 1e-5 --epochs 30
python src/train.py --lambda_val 1e-4 --epochs 30
python src/train.py --lambda_val 1e-3 --epochs 30
