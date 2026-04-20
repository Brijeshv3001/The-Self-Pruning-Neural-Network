# =================================================================
# File: src/utils.py
# Project: The Self-Pruning Neural Network
# Description: logging, plotting, CSV saving
# =================================================================

import logging
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.layers import PrunableLinear, PrunableConv2d

def setup_logging(save_dir: str, log_filename: str = "train.log") -> logging.Logger:
    """Configures multi-handler logging to console and file output."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("self_pruning")
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(os.path.join(save_dir, log_filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def save_results(lambda_val: float, accuracy: float, sparsity: float, flops_dense: float,
                 flops_sparse: float, compression_ratio: float, save_dir: str) -> None:
    """Appends experiment metrics to a cumulative CSV tracker."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "results.csv")
    
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["lambda", "test_accuracy", "sparsity_level", 
                             "flops_dense_gflops", "flops_sparse_gflops", 
                             "compression_ratio", "timestamp"])
        writer.writerow([
            lambda_val,
            accuracy,
            sparsity,
            flops_dense,
            flops_sparse,
            compression_ratio,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

def plot_gate_distribution(model: nn.Module, lambda_val: float, save_dir: str, epoch: Optional[int] = None) -> None:
    """Visualizes the architectural gating mechanism distribution revealing sparsity tendencies."""
    os.makedirs(save_dir, exist_ok=True)
    
    all_gates = []
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            all_gates.append(module.get_all_gates().cpu().numpy())
            
    if not all_gates:
        return
        
    gates_array = np.concatenate(all_gates)
    
    plt.figure(figsize=(10, 6))
    plt.hist(gates_array, bins=80, range=(0, 1), color="#4C72B0", edgecolor="white")
    plt.axvline(x=0.01, color="red", linestyle="--", label="Pruning threshold")
    plt.title(f"Gate Distribution (Lambda: {lambda_val})")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    filename = f"gate_dist_lambda_{lambda_val}.png" if epoch is None else f"gate_dist_lambda_{lambda_val}_ep_{epoch}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def plot_accuracy_vs_flops(results: List[Dict[str, Any]], save_dir: str) -> None:
    """Generates an Efficiency Frontier plot contrasting performance with computational expenditure."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for res in results:
        flops = res["flops_sparse"]
        acc = res["accuracy"]
        lam = res["lambda"]
        cr = res["compression_ratio"]
        label = f"λ={lam}"
        plt.scatter(flops, acc, s=150, alpha=0.8, label=label, marker="o")
        plt.annotate(f"λ={lam}\n({cr:.1f}x)", (flops, acc), xytext=(5, -15), textcoords='offset points')
        
    plt.title("Accuracy vs. Computational Efficiency (GFLOPs)")
    plt.xlabel("GFLOPs (Sparse)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "accuracy_vs_flops.png"), dpi=150)
    plt.close()

def plot_pareto_frontier(results: List[Dict[str, Any]], save_dir: str) -> None:
    """Identifies and highlights the absolute optimal balance points for Accuracy x Sparsity."""
    os.makedirs(save_dir, exist_ok=True)
    
    if not results:
        return
        
    results.sort(key=lambda x: x["sparsity_level"])
    sparsities = [r["sparsity_level"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, accuracies, "o-", color="gray", alpha=0.5)
    
    # Simple Pareto optimum highlighting
    pareto_front = []
    max_acc = 0
    for s, a in zip(sparsities, accuracies):
        if a >= max_acc:
            pareto_front.append((s, a))
            max_acc = a
            
    if pareto_front:
        px, py = zip(*pareto_front)
        plt.plot(px, py, "ro-", linewidth=2, markersize=8, label="Pareto Frontier")
        
    plt.title("Pareto Frontier: Accuracy vs. Sparsity")
    plt.xlabel("Sparsity Level (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "pareto_frontier.png"), dpi=150)
    plt.close()

def create_gate_evolution_gif(gate_snapshots: List[np.ndarray], save_dir: str, lambda_val: float) -> None:
    """Combines epoch-level structural states into a visual proof of Gate Hardening."""
    if not gate_snapshots:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame: int) -> tuple:
        ax.clear()
        data = gate_snapshots[frame]
        ax.hist(data, bins=80, range=(0, 1), color="#00d4ff", edgecolor="black")
        ax.axvline(x=0.01, color="red", linestyle="--", label="Pruning threshold")
        ax.set_title(f"Gate Hardening - Epoch {frame + 1} (Lambda: {lambda_val})")
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, len(data) // 5)  # Scale roughly to display structure
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax,
        
    ani = animation.FuncAnimation(fig, update, frames=len(gate_snapshots), blit=False)
    gif_path = os.path.join(save_dir, f"gate_dist_lambda_{lambda_val}_evolution.gif")
    
    writer = animation.PillowWriter(fps=2)
    ani.save(gif_path, writer=writer)
    plt.close()
