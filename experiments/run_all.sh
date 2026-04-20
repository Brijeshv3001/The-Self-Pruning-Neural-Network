#!/bin/bash
# File: experiments/run_all.sh

echo "Starting Self-Pruning Network Experiments..."

# Experiment 1: Low sparsity penalty
echo "Running Experiment 1: lambda = 1e-5"
python src/train.py --lambda_val 1e-5 --epochs 30

# Experiment 2: Medium sparsity penalty
echo "Running Experiment 2: lambda = 1e-4"
python src/train.py --lambda_val 1e-4 --epochs 30

# Experiment 3: High sparsity penalty
echo "Running Experiment 3: lambda = 1e-3"
python src/train.py --lambda_val 1e-3 --epochs 30

echo "All experiments completed!"
