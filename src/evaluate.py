# =================================================================
# File: src/evaluate.py
# Project: The Self-Pruning Neural Network
# Description: standalone evaluation + FLOP counting
# =================================================================

import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from src.model import SelfPruningNet
from src.utils import plot_gate_distribution

def evaluate_checkpoint() -> None:
    """Invokes evaluation isolated from the training loop strictly for deployed analytical testing."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint model map")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SelfPruningNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # In evaluation, gates are completely sharp and set to t_end
    model.set_temperature(0.01)
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    sparsity = model.get_network_sparsity()
    flops_dense = model.compute_flops(threshold=-1.0)
    flops_sparse = model.compute_flops()
    cr = model.get_compression_ratio()

    print(f"Results for Checkpoint: {args.checkpoint}")
    print(f"Test Accuracy     : {acc:.2f}%")
    print(f"Network Sparsity  : {sparsity:.2f}%")
    print(f"Dense GFLOPs      : {flops_dense:.4f}")
    print(f"Sparse GFLOPs     : {flops_sparse:.4f}")
    print(f"Compression Ratio : {cr:.2f}x")
    
    plot_gate_distribution(model, lambda_val=-1, save_dir=".", epoch=999)

if __name__ == "__main__":
    evaluate_checkpoint()
