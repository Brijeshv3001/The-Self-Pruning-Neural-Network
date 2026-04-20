# =================================================================
# File: src/layers.py
# Project: The Self-Pruning Neural Network
# Description: PrunableLinear and PrunableConv2d with temperature-annealed gates
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A customizable Linear layer with learned gating mechanisms for self-pruning.
    Gates are derived via a temperature-scaled sigmoid applied to learned gate scores.
    """
    def __init__(self, in_features: int, out_features: int, temperature: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Kaiming Uniform, and zeros out biases and gate scores."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass by applying the temperature-scaled sigmoid gates 
        to the dense weights, effectively pruning them dynamically.
        """
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Calculates the percentage of weights that have been pruned (gate < threshold)."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        active_weights = (gates > threshold).sum().item()
        total_weights = self.weight.numel()
        return 100.0 * (1.0 - active_weights / total_weights)

    def get_all_gates(self) -> torch.Tensor:
        """Returns a detached, flattened tensor of all gate values."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        return gates.detach().flatten()

    def set_temperature(self, temp: float) -> None:
        """Sets the temperature controlling the sharpness of the sigmoid gates."""
        self.temperature = temp

    def count_active_weights(self, threshold: float = 1e-2) -> int:
        """Returns the absolute count of unpruned weights."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        return (gates > threshold).sum().item()

    def extra_repr(self) -> str:
        """Custom representation for the prunable layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, temperature={self.temperature:.4f}"


class PrunableConv2d(nn.Module):
    """
    A customizable Conv2d layer with learned gating mechanisms for self-pruning.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, temperature: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.gate_scores = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Kaiming Uniform, and zeros out biases and gate scores."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies convolution using the dynamically pruned kernel weights."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weight = self.weight * gates
        return F.conv2d(x, pruned_weight, self.bias, self.stride, self.padding)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Calculates the percentage of weights that have been pruned."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        active_weights = (gates > threshold).sum().item()
        total_weights = self.weight.numel()
        return 100.0 * (1.0 - active_weights / total_weights)

    def get_all_gates(self) -> torch.Tensor:
        """Returns a detached, flattened tensor of all gate values."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        return gates.detach().flatten()

    def set_temperature(self, temp: float) -> None:
        """Sets the temperature controlling the sharpness of the sigmoid gates."""
        self.temperature = temp

    def count_active_weights(self, threshold: float = 1e-2) -> int:
        """Returns the absolute count of unpruned weights."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        return (gates > threshold).sum().item()

    def extra_repr(self) -> str:
        """Custom representation for the prunable layer."""
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"temperature={self.temperature:.4f}")


def anneal_temperature(model: nn.Module, epoch: int, total_epochs: int, 
                       t_start: float = 1.0, t_end: float = 0.01) -> None:
    """
    Linearly anneals the temperature of all Prunable layers from t_start to t_end.
    Should be called once per epoch in the training loop.
    
    A lower temperature creates a sharper sigmoid, forcing the soft gates to harden 
    into a more binary state, ultimately deciding what to keep or prune permanently.
    """
    progress = (epoch - 1) / max(1, total_epochs - 1)
    current_temp = t_start - progress * (t_start - t_end)
    current_temp = max(current_temp, t_end)
    
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            module.set_temperature(current_temp)
