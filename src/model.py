# File: src/model.py
"""Model definitions for the self-pruning CIFAR-10 classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from layers import PrunableLinear


class SelfPruningNet(nn.Module):
    """Fully connected neural network with learnable pruning gates."""

    def __init__(self, dropout_p: float = 0.3) -> None:
        """Initialize network layers.

        Args:
            dropout_p: Dropout probability for hidden layers.
        """
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits for CIFAR-10 input batch.

        Args:
            x: Input image tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Logits tensor of shape ``(batch_size, 10)``.
        """
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def compute_sparsity_loss(self) -> Tensor:
        """Sum all gate values across every prunable layer.

        Returns:
            Scalar tensor used as differentiable sparsity regularization term.
        """
        sparsity_loss: Tensor = torch.zeros((), device=self.fc1.weight.device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                sparsity_loss = sparsity_loss + torch.sigmoid(module.gate_scores).sum()
        return sparsity_loss

    def get_network_sparsity(self, threshold: float = 1e-2) -> float:
        """Compute overall fraction of gates below threshold across network.

        Args:
            threshold: Gate value below which a connection is considered pruned.

        Returns:
            Fraction of all network gates considered pruned.
        """
        total_params = 0
        total_pruned = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    total_params += gates.numel()
                    total_pruned += int((gates < threshold).sum().item())

        if total_params == 0:
            return 0.0
        return float(total_pruned / total_params)
