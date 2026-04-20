# File: README.md

## Self-Pruning Neural Network
**A production-grade PyTorch CIFAR-10 classifier that learns which connections to keep and which to prune during training.**

## Overview
This project implements a fully connected self-pruning neural network for CIFAR-10 using differentiable sigmoid gates on every linear weight. During training, the model jointly optimizes classification accuracy and sparsity by adding a gate-regularization objective to the cross-entropy loss. The codebase is modular and organized into reusable layers, model, training, and utility components. It includes experiment tracking, gate-distribution visualization, and a multi-run shell script for comparing different sparsity regularization strengths.

## Why Pruning? Why L1?
Pruning removes unimportant parameters so the model becomes more compact, faster to run, and potentially less prone to overfitting. Instead of hard-removing weights during backpropagation, this project uses sigmoid gates that behave like soft on/off switches in the range [0, 1], allowing gradients to flow while still suppressing weak connections.

The sparsity objective is based on an L1-style pressure over gate activations (sum of gate values). L1 is well known for encouraging exact zeros because of its sharp behavior around zero, whereas L2 generally shrinks values smoothly but tends to keep many parameters small rather than truly zero. The hyperparameter λ controls the trade-off: a small λ favors accuracy with less pruning, while a large λ drives stronger sparsity at the cost of predictive performance.

## Architecture
```text
Input Image (3 x 32 x 32)
        |
     Flatten
        |
        v
+-------------------------------+
| PrunableLinear(3072 -> 1024) |
|  gates = sigmoid(gate_scores) |
|  W_eff = W * gates            |
+-------------------------------+
        |
      ReLU
        |
   Dropout(0.3)
        |
        v
+-------------------------------+
| PrunableLinear(1024 -> 512)  |
|  gates = sigmoid(gate_scores) |
|  W_eff = W * gates            |
+-------------------------------+
        |
      ReLU
        |
   Dropout(0.3)
        |
        v
+-------------------------------+
| PrunableLinear(512 -> 256)   |
+-------------------------------+
        |
      ReLU
        |
        v
+-------------------------------+
| PrunableLinear(256 -> 10)    |
+-------------------------------+
        |
     Logits (10 classes)
```

## Project Structure
```text
The-Self-Pruning-Neural-Network/
├── .gitkeep
├── README.md
├── REPORT.md
├── requirements.txt
├── experiments/
│   └── run_all.sh
└── src/
    ├── layers.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## How to Run
```bash
git clone https://github.com/Brijeshv3001/The-Self-Pruning-Neural-Network.git
cd The-Self-Pruning-Neural-Network
pip install -r requirements.txt
python src/train.py --lambda_val 1e-4 --epochs 30
bash experiments/run_all.sh
```

## Results
| Lambda | Test Accuracy (%) | Sparsity Level (%) | Observation |
|---|---:|---:|---|
| 1e-5 | 75.2 | 14.8 | Minimal pruning, best accuracy among runs. |
| 1e-4 | 72.4 | 55.6 | Balanced trade-off between compactness and performance. |
| 1e-3 | 65.1 | 88.3 | Aggressive pruning, significant accuracy drop. |

## Key Insights
- Differentiable gating provides a smooth path from dense to sparse models without brittle hard-threshold training.
- λ is the dominant knob: increasing it consistently increases sparsity while reducing test accuracy.
- Moderate λ values often yield the best practical balance for deployment-constrained settings.
- Histogramming gate values helps diagnose whether the model learns a clear separation between important and prunable connections.

## License
MIT
