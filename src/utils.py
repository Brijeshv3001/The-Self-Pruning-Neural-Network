# File: src/utils.py
"""Utility helpers for logging, experiment persistence, and visualization."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from layers import PrunableLinear


def setup_logging(save_dir: str | Path, log_filename: str = "train.log") -> logging.Logger:
    """Configure logging to both console and a file.

    Args:
        save_dir: Directory where logs should be stored.
        log_filename: Name of the log file.

    Returns:
        Configured root logger instance.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler], force=True)
    return logging.getLogger(__name__)


def save_results(lambda_val: float, accuracy: float, sparsity: float, save_dir: str | Path) -> None:
    """Append experiment summary metrics to a CSV file.

    Args:
        lambda_val: Sparsity-penalty coefficient used in training.
        accuracy: Final test accuracy percentage.
        sparsity: Final network sparsity percentage.
        save_dir: Directory where results CSV should be located.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["lambda", "test_accuracy", "sparsity_level", "timestamp"])
        writer.writerow(
            [
                lambda_val,
                accuracy,
                sparsity,
                datetime.now(timezone.utc).isoformat(),
            ]
        )


def plot_gate_distribution(model: torch.nn.Module, lambda_val: float, save_dir: str | Path) -> None:
    """Plot and save a histogram of all gate values in the model.

    Args:
        model: Trained model containing ``PrunableLinear`` layers.
        lambda_val: Sparsity-penalty coefficient used for filename/title.
        save_dir: Directory where the plot should be saved.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_gates: list[np.ndarray] = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(module.get_all_gates().cpu().numpy())

    if not all_gates:
        raise ValueError("No PrunableLinear layers found in model; cannot plot gates.")

    gate_values = np.concatenate(all_gates)

    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=80, range=(0.0, 1.0), color="tab:blue", alpha=0.85)
    plt.title(f"Gate value distribution (λ={lambda_val})")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"gate_dist_lambda_{lambda_val}.png", dpi=150)
    plt.close()
