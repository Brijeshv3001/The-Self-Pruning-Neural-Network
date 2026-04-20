# File: src/layers.py
"""Custom neural network layers with differentiable pruning gates."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PrunableLinear(nn.Module):
    """Linear layer augmented with learnable sigmoid gates for self-pruning.

    The layer maintains a standard weight matrix and bias vector, along with
    trainable gate scores. During the forward pass, gate scores are transformed
    through a sigmoid to obtain gate values in ``[0, 1]``. The effective weight
    matrix is then computed as an elementwise product of weights and gates.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize a prunable linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters for stable training."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated linear transformation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Compute fraction of gate values below a pruning threshold.

        Args:
            threshold: Gate value below which a connection is considered pruned.

        Returns:
            Fraction of gated connections considered pruned.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return float((gates < threshold).float().mean().item())

    def get_all_gates(self) -> Tensor:
        """Return all current gate values as a detached flattened tensor.

        Returns:
            A 1D tensor containing all gate values.
        """
        return torch.sigmoid(self.gate_scores).detach().flatten()
