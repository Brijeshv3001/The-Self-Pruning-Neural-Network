# =================================================================
# File: src/train.py
# Project: The Self-Pruning Neural Network
# Description: full training loop, argparse, logging
# =================================================================

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from src.model import SelfPruningNet
from src.layers import anneal_temperature, PrunableConv2d, PrunableLinear
from src.utils import (setup_logging, save_results, plot_gate_distribution, 
                       create_gate_evolution_gif, plot_accuracy_vs_flops)

def get_args() -> argparse.Namespace:
    """Parses application arguments and handles external config injection."""
    parser = argparse.ArgumentParser(description="Train Self-Pruning Neural Network")
    parser.add_argument("--lambda_val", type=float, default=1e-4, help="L1 sparsity regularization weight")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial gate temperature (annealing start)")
    parser.add_argument("--t_end", type=float, default=0.01, help="Final gate temperature after annealing")
    parser.add_argument("--rehab_thresh", type=float, default=0.05, help="Gates below this for N epochs get rehabilitated")
    parser.add_argument("--rehab_epochs", type=int, default=5, help="Epochs a gate must be dead before rehabilitation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml to override all args")
    return parser.parse_args()

def set_seed(seed: int) -> None:
    """Ensures absolute determinism across hardware layers."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_device() -> torch.device:
    """Discovers available acceleration hardware prioritizing CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Provides memory-efficient data conduits for the CIFAR-10 classification task."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def rehabilitate_dead_gates(model: SelfPruningNet, dead_tracker: dict, threshold: float, patience: int) -> int:
    """
    Implements the 'Self-Healing Intelligence / Resurrection Rate' feature.
    Gates that remain strictly below the viability threshold for too many epochs 
    are surgically revived (score set back to 0.0 yielding a neutral gate of 0.5) 
    to prevent absolute network collapse.
    """
    rehabilitated_count = 0
    idx = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                gates = torch.sigmoid(module.gate_scores / module.temperature)
                dead_mask = gates < threshold
                
                if idx not in dead_tracker:
                    dead_tracker[idx] = torch.zeros_like(gates, dtype=torch.int32)
                
                dead_tracker[idx] += dead_mask.int()
                dead_tracker[idx][~dead_mask] = 0
                
                rehab_mask = dead_tracker[idx] >= patience
                if rehab_mask.any():
                    # Reset the scores to 0.0, injecting life back into disconnected sub-graphs
                    module.gate_scores[rehab_mask] = 0.0
                    dead_tracker[idx][rehab_mask] = 0
                    rehabilitated_count += rehab_mask.sum().item()
                idx += 1
                
    return rehabilitated_count

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module, lambda_val: float, device: torch.device, 
                    epoch: int, logger: logging.Logger) -> tuple[float, float, float]:
    """Pushes a single iteration of training updates across the dataset."""
    model.train()
    ce_loss_sum = 0.0
    sp_loss_sum = 0.0
    tot_loss_sum = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)
        
        if hasattr(model, 'compute_sparsity_loss'):
            sp_loss = model.compute_sparsity_loss()
        else:
            sp_loss = torch.tensor(0.0, device=device)
            
        total_loss = ce_loss + lambda_val * sp_loss
        total_loss.backward()
        optimizer.step()
        
        ce_loss_sum += ce_loss.item()
        sp_loss_sum += sp_loss.item()
        tot_loss_sum += total_loss.item()
        
        if batch_idx % 50 == 0:
            logger.info(f"Epoch: {epoch} | Batch: {batch_idx}/{len(loader)} | "
                        f"CE Loss: {ce_loss.item():.4f} | Sparsity Loss: {sp_loss.item():.4f} | "
                        f"Total Loss: {total_loss.item():.4f}")
                        
    return ce_loss_sum / len(loader), sp_loss_sum / len(loader), tot_loss_sum / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Executes a non-differentiable evaluation pass mapping output accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / (total + 1e-8)

def main() -> None:
    """Entry point coordinating the complete neural training orchestration."""
    args = get_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            args.lambda_val = cfg.get('training', {}).get('lambda_val', args.lambda_val)
            args.epochs = cfg.get('training', {}).get('epochs', args.epochs)
            args.batch_size = cfg.get('training', {}).get('batch_size', args.batch_size)
            args.lr = cfg.get('training', {}).get('lr', args.lr)
            args.seed = cfg.get('training', {}).get('seed', args.seed)
            args.save_dir = cfg.get('training', {}).get('save_dir', args.save_dir)
            args.temperature = cfg.get('pruning', {}).get('temperature', args.temperature)
            args.t_end = cfg.get('pruning', {}).get('t_end', args.t_end)
            args.rehab_thresh = cfg.get('pruning', {}).get('rehab_thresh', args.rehab_thresh)
            args.rehab_epochs = cfg.get('pruning', {}).get('rehab_epochs', args.rehab_epochs)
            
    logger = setup_logging(args.save_dir)
    logger.info("=== The Self-Pruning Neural Network Protocol Started ===")
    logger.info(f"Arguments: {vars(args)}")
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Target Hardware Engine: {device}")
    
    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = SelfPruningNet(num_classes=10, temperature=args.temperature).to(device)
    
    params_dense = model.count_parameters()
    flops_dense = model.compute_flops(threshold=-1.0) # all active
    
    logger.info(f"Network Genesis Parameters: {params_dense:,}")
    logger.info(f"Initial Dense Capacity (GFLOPs): {flops_dense:.4f}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    dead_tracker = {}
    gate_snapshots = []
    
    best_acc = 0.0
    checkpoint_path = os.path.join(args.save_dir, f"model_lambda_{args.lambda_val}.pt")
    
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training Evolution"):
        train_one_epoch(model, train_loader, optimizer, criterion, args.lambda_val, device, epoch, logger)
        
        anneal_temperature(model, epoch, args.epochs, args.temperature, args.t_end)
        rehabbed = rehabilitate_dead_gates(model, dead_tracker, args.rehab_thresh, args.rehab_epochs)
        
        if rehabbed > 0:
            logger.info(f"Rehabilitated {rehabbed} structurally isolated gates.")
            
        accuracy = evaluate(model, test_loader, device)
        sparsity = model.get_network_sparsity()
        flops_sparse = model.compute_flops()
        compression = model.get_compression_ratio()
        
        logger.info(f"[Epoch {epoch}/{args.epochs}] Acc: {accuracy:.2f}% | Sparsity: {sparsity:.2f}% | "
                    f"Sparse GFLOPs: {flops_sparse:.4f} | Extracted Compress: {compression:.2f}x")
                    
        # Capture for animation
        all_gates = [m.get_all_gates().cpu().numpy() for m in model.modules() if isinstance(m, (PrunableLinear, PrunableConv2d))]
        if all_gates:
            gate_snapshots.append(np.concatenate(all_gates))
            
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            
    # Final Operations
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    final_acc = evaluate(model, test_loader, device)
    final_sparsity = model.get_network_sparsity()
    final_flops = model.compute_flops()
    final_cr = model.get_compression_ratio()
    
    save_results(args.lambda_val, final_acc, final_sparsity, flops_dense, final_flops, final_cr, args.save_dir)
    plot_gate_distribution(model, args.lambda_val, args.save_dir)
    create_gate_evolution_gif(gate_snapshots, args.save_dir, args.lambda_val)
    
    mock_result = {
        "lambda": args.lambda_val,
        "accuracy": final_acc,
        "flops_sparse": final_flops,
        "compression_ratio": final_cr,
        "sparsity_level": final_sparsity
    }
    plot_accuracy_vs_flops([mock_result], args.save_dir)
    
    logger.info("=== Execution Conclusion Summary ===")
    logger.info(f"Lambda        : {args.lambda_val}")
    logger.info(f"Test Accuracy : {final_acc:.2f}%")
    logger.info(f"Final Sparsity: {final_sparsity:.2f}%")
    logger.info(f"GFLOPs        : {flops_dense:.4f} -> {final_flops:.4f}")
    logger.info(f"Compression   : {final_cr:.2f}x")
    logger.info(f"Artifact path : {checkpoint_path}")

if __name__ == "__main__":
    main()
