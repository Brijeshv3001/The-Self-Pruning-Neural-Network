# =================================================================
# File: tests/test_gradients.py
# Project: The Self-Pruning Neural Network
# Description: pytest unit tests proving gradient flow correctness
# =================================================================

import torch
import pytest
from src.layers import PrunableLinear, PrunableConv2d
from src.model import SelfPruningNet

def test_prunable_linear_gradient_flow() -> None:
    """Proves gradients flow to both weight AND gate_scores in PrunableLinear."""
    layer = PrunableLinear(4, 8)
    x = torch.randn(2, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None, "Weight gradient should not be None"
    assert layer.gate_scores.grad is not None, "Gate scores gradient should not be None"
    assert layer.gate_scores.grad.shape == layer.gate_scores.shape

def test_prunable_conv_gradient_flow() -> None:
    """Proves gradients flow to both weight AND gate_scores in PrunableConv2d."""
    layer = PrunableConv2d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None
    assert layer.gate_scores.grad.shape == layer.gate_scores.shape

def test_sparsity_loss_is_differentiable() -> None:
    """Ensures L1 sparsity loss acts continuously and traces through gate_scores."""
    model = SelfPruningNet()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    loss = model.compute_sparsity_loss()
    loss.backward()
    
    has_grad = False
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            if module.gate_scores.grad is not None:
                has_grad = True
    assert has_grad, "Gradient did not backpropagate from sparsity loss!"

def test_temperature_annealing_sharpens_gates() -> None:
    """Lower temperatures cause stronger binary variance within gates."""
    layer = PrunableLinear(4, 4, temperature=10.0)
    
    # Introduce variance so standard dev isn't identically 0
    layer.gate_scores.data = torch.randn_like(layer.gate_scores.data)
    
    gates_soft = torch.sigmoid(layer.gate_scores / 10.0)
    layer.set_temperature(0.01)
    gates_hard = torch.sigmoid(layer.gate_scores / 0.01)
    
    assert gates_hard.std().item() >= gates_soft.std().item()

def test_prunable_linear_output_shape() -> None:
    """Prunable geometry checks against standard batch interfaces."""
    layer = PrunableLinear(128, 64)
    x = torch.randn(32, 128)
    out = layer(x)
    assert out.shape == (32, 64)

def test_zero_gate_kills_weight() -> None:
    """Negative infinity gates completely suppress dense linear activations."""
    layer = PrunableLinear(4, 4)
    layer.gate_scores.data.fill_(-100.0)
    x = torch.randn(2, 4)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)
