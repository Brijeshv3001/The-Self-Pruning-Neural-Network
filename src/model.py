# =================================================================
# File: src/model.py
# Project: The Self-Pruning Neural Network
# Description: SelfPruningNet (ResNet-18 style with prunable layers)
# =================================================================

import torch
import torch.nn as nn
from typing import Tuple
from src.layers import PrunableConv2d, PrunableLinear

class SparseBlock(nn.Module):
    """
    A foundational building block for SelfPruningNet using PrunableConv2d layers.
    Incorporates residual connections.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, temperature: float = 1.0):
        super().__init__()
        self.conv1 = PrunableConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, temperature=temperature)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PrunableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, temperature=temperature)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                PrunableConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, temperature=temperature),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SelfPruningNet(nn.Module):
    """
    A ResNet-18 inspired architecture constructed primarily with dynamically 
    prunable convolutions and linear layers to enable structural sparsity.
    """
    def __init__(self, num_classes: int = 10, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
        self.prep = nn.Sequential(
            PrunableConv2d(3, 64, kernel_size=3, stride=1, padding=1, temperature=temperature),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            SparseBlock(64, 64, stride=1, temperature=temperature),
            SparseBlock(64, 64, stride=1, temperature=temperature)
        )
        self.layer2 = nn.Sequential(
            SparseBlock(64, 128, stride=2, temperature=temperature),
            SparseBlock(128, 128, stride=1, temperature=temperature)
        )
        self.layer3 = nn.Sequential(
            SparseBlock(128, 256, stride=2, temperature=temperature),
            SparseBlock(256, 256, stride=1, temperature=temperature)
        )
        self.layer4 = nn.Sequential(
            SparseBlock(256, 512, stride=2, temperature=temperature),
            SparseBlock(512, 512, stride=1, temperature=temperature)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = PrunableLinear(512, num_classes, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass through the entire network."""
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Computes the differentiable L1 sum over all gate scores in the network.
        Directly uses torch.sigmoid(module.gate_scores) to maintain the computation graph.
        """
        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                # Differentiable gates, explicitly NOT detached.
                gates = torch.sigmoid(module.gate_scores / module.temperature)
                sparsity_loss += torch.sum(gates)
        return sparsity_loss

    def get_network_sparsity(self, threshold: float = 1e-2) -> float:
        """Calculates the overall structural sparsity of the network as a percentage."""
        total_weights = 0
        active_weights = 0
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                total_weights += module.weight.numel()
                active_weights += module.count_active_weights(threshold)
        if total_weights == 0:
            return 0.0
        return 100.0 * (1.0 - active_weights / total_weights)

    def set_temperature(self, temp: float) -> None:
        """Updates the temperature across all prunable layers."""
        self.temperature = temp
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                module.set_temperature(temp)

    def count_parameters(self) -> int:
        """Counts the total dense parameters in the prunable layers."""
        total = 0
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                total += module.weight.numel()
        return total

    def count_active_parameters(self, threshold: float = 1e-2) -> int:
        """Counts only the surviving (unpruned) parameters in the prunable layers."""
        active = 0
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                active += module.count_active_weights(threshold)
        return active

    def compute_flops(self, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32), threshold: float = 1e-2) -> float:
        """
        Computes the effective structural GFLOPs of the current sparsely-activated network.
        Uses manual heuristics for convolution and linear parameter sizes multplied by active fraction.
        """
        gflops = 0.0
        dummy_input = torch.zeros(input_size)
        
        current_h, current_w = input_size[2], input_size[3]
        
        def calculate_flops_for_module(module, h, w):
            if isinstance(module, PrunableConv2d):
                active_frac = module.count_active_weights(threshold) / module.weight.numel()
                kH, kW = module.kernel_size, module.kernel_size
                in_ch, out_ch = module.in_channels, module.out_channels
                ops = 2 * in_ch * out_ch * kH * kW * h * w
                return (ops * active_frac) / 1e9  # GFLOPs
            elif isinstance(module, PrunableLinear):
                active_frac = module.count_active_weights(threshold) / module.weight.numel()
                in_f, out_f = module.in_features, module.out_features
                ops = 2 * in_f * out_f
                return (ops * active_frac) / 1e9
            return 0.0

        for name, module in self.named_modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                if 'prep' in name or 'layer1' in name:
                    h, w = 32, 32
                elif 'layer2.0.conv1' in name or 'layer2.0.shortcut' in name:
                    h, w = 32, 32
                elif 'layer2' in name:
                    h, w = 16, 16
                elif 'layer3.0.conv1' in name or 'layer3.0.shortcut' in name:
                    h, w = 16, 16
                elif 'layer3' in name:
                    h, w = 8, 8
                elif 'layer4.0.conv1' in name or 'layer4.0.shortcut' in name:
                    h, w = 8, 8
                elif 'layer4' in name:
                    h, w = 4, 4
                else:
                    h, w = 1, 1 
                
                gflops += calculate_flops_for_module(module, h, w)
                
        return gflops

    def get_compression_ratio(self, threshold: float = 1e-2) -> float:
        """
        Returns the ultimate structural compression ratio tracking total parameters to active parameters.
        Example: If total parameters is 11,000,000 and valid is 72,000, returns ~152.0.
        """
        total = self.count_parameters()
        active = self.count_active_parameters(threshold)
        if active == 0:
            return float('inf')
        return total / active
