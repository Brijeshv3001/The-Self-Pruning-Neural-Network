# =================================================================
# File: src/agent.py
# Project: The Self-Pruning Neural Network
# Description: AI Pruning Advisor agent (hardware-aware model selector)
# =================================================================

import argparse

def launch_advisor() -> None:
    """Intelligence layer parsing system environment variables to recommend optimized architectural models."""
    parser = argparse.ArgumentParser(description="AI Pruning Advisor")
    parser.add_argument("--latency", type=float, required=True, help="ms budget")
    parser.add_argument("--accuracy", type=float, required=True, help="min accuracy %")
    parser.add_argument("--memory", type=float, default=None, help="MB budget, optional")
    args = parser.parse_args()

    knowledge_base = [
        {"name": "Dense Baseline",    "accuracy": 89.43, "flops": 0.764, "sparsity": 0.0,  "latency_est": 12.0, "memory_mb": 45.0},
        {"name": "Optimized Sparse",  "accuracy": 89.51, "flops": 0.005, "sparsity": 99.3, "latency_est": 0.5,  "memory_mb": 0.3},
        {"name": "Annealed Dynamic",  "accuracy": 81.80, "flops": 0.037, "sparsity": 95.0, "latency_est": 1.2,  "memory_mb": 2.1},
        {"name": "Medium Sparse",     "accuracy": 85.20, "flops": 0.120, "sparsity": 84.3, "latency_est": 3.5,  "memory_mb": 7.5},
        {"name": "Light Sparse",      "accuracy": 87.90, "flops": 0.280, "sparsity": 63.4, "latency_est": 6.8,  "memory_mb": 18.0},
    ]

    valid = []
    for config in knowledge_base:
        if config["latency_est"] <= args.latency and config["accuracy"] >= args.accuracy:
            if args.memory is None or config["memory_mb"] <= args.memory:
                valid.append(config)

    if not valid:
        print("\n[!] ADVISOR WARNING: No structural configuration satisfies the strict constraints.")
        print("Suggest relaxing latency or min accuracy standards.")
        return

    # Advanced Multi-Objective Heuristics
    # Ranks by compound fitness equation utilizing precision scaling
    for conf in valid:
        norm_flops = max(conf["flops"], 0.001)
        conf["score"] = (conf["accuracy"] * 0.6) + ((1.0 / norm_flops) * 0.4)

    valid.sort(key=lambda x: x["score"], reverse=True)
    best = valid[0]

    print("\n" + "="*60)
    print("🤖 AI PRUNING ADVISOR REPORT")
    print("="*60)
    print(f"> RECOMMENDATION: {best['name']}")
    print(f"> Reasoning: Fulfills latency boundary ({best['latency_est']}ms <= {args.latency}ms) "
          f"while maintaining critical mission accuracy ({best['accuracy']}%).\n")
          
    print(f"%-20s %-10s %-10s %-15s %-10s" % ("Name", "Accuracy", "Sparsity", "Latency (ms)", "Mem (MB)"))
    print("-" * 60)
    for v in valid:
        mark = "*" if v["name"] == best["name"] else " "
        print(f"{mark} %-18s %-10.2f %-10.1f %-15.1f %-10.1f" % 
              (v["name"], v["accuracy"], v["sparsity"], v["latency_est"], v["memory_mb"]))
    print("="*60 + "\n")

if __name__ == "__main__":
    launch_advisor()
