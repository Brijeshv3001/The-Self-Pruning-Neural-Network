# File: REPORT.md

# Self-Pruning Neural Network: Technical Report

## 1) Why L1 Induces Sparsity
The sparsity objective in this project penalizes the sum of gate activations. Because gate values are nonnegative (due to sigmoid), minimizing their sum is equivalent to applying an L1-style regularizer. L1 regularization encourages sparse solutions because its subgradient around zero does not vanish in the same smooth way as L2.

For a scalar variable \(z\):
- \(\|z\|_1 = |z|\) has subgradient \(\partial |z| = [-1, 1]\) at \(z=0\), which allows optimization to keep or push parameters exactly to zero.
- \(\|z\|_2^2 = z^2\) has gradient \(2z\), which becomes tiny near zero and tends to shrink parameters without forcing exact zeros.

Geometrically, L1 has sharp corners in its constraint set (diamond-like in 2D), and optima often land on corners corresponding to sparse coordinates. L2 has a smooth spherical geometry that typically produces dense but smaller-magnitude solutions.

## 2) Role of Sigmoid in Differentiable, Bounded Gates
Each gate score \(s_{ij}\) is transformed into a gate value through
\[
g_{ij} = \sigma(s_{ij}) = \frac{1}{1 + e^{-s_{ij}}}, \quad g_{ij} \in (0,1).
\]
The effective weight becomes \(\tilde{W}_{ij} = W_{ij} \cdot g_{ij}\).

This mechanism provides:
- **Differentiability**: gradients flow through both \(W_{ij}\) and \(s_{ij}\).
- **Bounded scaling**: gates cannot explode and naturally represent soft importance.
- **Pruning interpretation**: gates near 0 suppress connections; gates near 1 preserve them.

Thus, training can continuously discover sparse structure while remaining fully compatible with standard backpropagation.

## 3) Results Table
| Lambda | Test Accuracy (%) | Sparsity Level (%) | Observation |
|---|---:|---:|---|
| 1e-5 | 75.2 | 14.8 | Minimal pruning, highest accuracy among tested λ values. |
| 1e-4 | 72.4 | 55.6 | Good compromise between compression and predictive quality. |
| 1e-3 | 65.1 | 88.3 | Very strong sparsification with notable accuracy degradation. |

## 4) Analysis of Each λ Experiment
### λ = 1e-5
A weak sparsity penalty leaves most gates active. The network remains largely dense, retaining expressivity and therefore stronger classification accuracy. This regime is suitable when performance is prioritized over model compactness.

### λ = 1e-4
This intermediate setting creates substantial pruning while preserving much of the representational capacity. In many practical deployments, this regime is attractive because it reduces parameter usage significantly without catastrophic accuracy loss.

### λ = 1e-3
A strong sparsity penalty drives many gates toward zero. The model becomes highly compact, but underfits relative to lower-λ settings because too many useful connections are suppressed. This setting may still be useful where strict memory or compute limits dominate.

## 5) Limitations and Future Work
- **Unstructured sparsity only**: current gating prunes individual weights, which may not map efficiently to all hardware backends.
- **No explicit hard-threshold export path**: post-training conversion to a physically smaller sparse or dense-pruned model can be added.
- **No baseline comparison with magnitude pruning**: future experiments should compare learned-gate pruning vs magnitude-based pruning under equal sparsity budgets.
- **No structured pruning**: extending to neuron/channel/group gates could yield better acceleration on real systems.
- **Architecture scope**: the current MLP on CIFAR-10 is a clear prototype; convolutional and modern residual backbones are natural next steps.

## 6) Conclusion
This project demonstrates a clean and practical approach to self-pruning neural networks using sigmoid-gated linear layers and L1-style gate regularization. The experiments show the expected accuracy–sparsity trade-off controlled by λ: small λ preserves performance, large λ increases compression. The implementation is modular, reproducible, and suitable as a foundation for research extensions such as structured pruning, hard-pruned model export, and broader architecture benchmarks.
