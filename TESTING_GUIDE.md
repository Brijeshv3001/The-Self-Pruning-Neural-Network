# =================================================================
# File: TESTING_GUIDE.md
# Project: The Self-Pruning Neural Network
# Description: Automated architecture verifications
# =================================================================

# Advanced Testing Logic

## Running Pytest
Execute via shell standard:
```bash
pytest tests/ -v --disable-warnings
```

## What each test proves concerning logic
1. `test_prunable_linear_gradient_flow()`: Absolute assurance that differentiation travels strictly up target paths touching gate scores natively ensuring graph isn't detached.
2. `test_prunable_conv_gradient_flow()`: Identical routing check ensuring 2D convolutional networks process sparsity loss identically mirroring Linear vectors.
3. `test_sparsity_loss_is_differentiable()`: Validates L1 parameter checks dynamically loop natively generating tensor operations yielding gradients reliably.
4. `test_temperature_annealing_sharpens_gates()`: Proves the Sigmoid probability shift creates wider standard deviations effectively separating random initialization into distinct 0 or 1 boundaries locking the final network.
5. `test_prunable_linear_output_shape()`: Confirms mathematical boundaries process batch input matrixes reliably mirroring PyTorch's native `nn.Linear`.
6. `test_zero_gate_kills_weight()`: Direct forward confirmation mapping absolute suppression to functional 0% output structures natively.

## How to add a new test
Create a localized method prefixed strictly with `test_` dropping into `tests/` directories triggering standard PyTest discovery protocols. Add precise comments documenting what the function proves against the system specifications.

## Expected Test Output
It should uniformly pass 6 test markers in < 1.0 seconds yielding complete green outputs mapping full functionality.
