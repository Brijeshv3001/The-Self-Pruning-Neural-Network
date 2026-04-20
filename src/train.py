# File: src/train.py
"""Training script for the Self-Pruning Neural Network on CIFAR-10."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SelfPruningNet
from utils import plot_gate_distribution, save_results, setup_logging


@dataclass
class EpochMetrics:
    """Container for evaluation metrics."""

    accuracy: float
    sparsity: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train self-pruning network on CIFAR-10")
    parser.add_argument("--lambda_val", type=float, default=1e-4, help="Sparsity penalty weight")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="experiments", help="Directory for artifacts")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders for CIFAR-10."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def evaluate(model: SelfPruningNet, data_loader: DataLoader, device: torch.device) -> EpochMetrics:
    """Evaluate model accuracy and sparsity on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total else 0.0
    sparsity = 100.0 * model.get_network_sparsity()
    return EpochMetrics(accuracy=accuracy, sparsity=sparsity)


def train() -> None:
    """Run end-to-end training, evaluation, logging, and artifact generation."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    logger = setup_logging(save_dir)

    set_seed(args.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    train_loader, test_loader = build_dataloaders(args.batch_size)

    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    final_accuracy = 0.0
    final_sparsity = 0.0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            ce_loss = criterion(logits, labels)
            sparsity_loss = model.compute_sparsity_loss()
            total_loss = ce_loss + args.lambda_val * sparsity_loss
            total_loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logger.info(
                    "Epoch [%d/%d] Batch [%d/%d] | ce_loss=%.4f | sparsity_loss=%.4f | total_loss=%.4f",
                    epoch,
                    args.epochs,
                    batch_idx,
                    len(train_loader),
                    ce_loss.item(),
                    sparsity_loss.item(),
                    total_loss.item(),
                )

        metrics = evaluate(model, test_loader, device)
        final_accuracy = metrics.accuracy
        final_sparsity = metrics.sparsity
        logger.info(
            "Epoch [%d/%d] complete | test_accuracy=%.2f%% | network_sparsity=%.2f%%",
            epoch,
            args.epochs,
            metrics.accuracy,
            metrics.sparsity,
        )

    save_results(args.lambda_val, final_accuracy, final_sparsity, save_dir)
    plot_gate_distribution(model, args.lambda_val, save_dir)

    summary = (
        "\nFinal Summary\n"
        "------------------------------------------\n"
        f"Lambda           : {args.lambda_val}\n"
        f"Test Accuracy (%) : {final_accuracy:.2f}\n"
        f"Sparsity Level (%) : {final_sparsity:.2f}\n"
        "------------------------------------------"
    )
    logger.info(summary)


if __name__ == "__main__":
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    train()
