#!/usr/bin/env python3
"""Analyze LoRA weight norms across all experiments.

Computes per-layer and total Frobenius norm of LoRA adapter weights (ΔW = B·A)
for every checkpoint. Then correlates with ORZ accuracy to test whether
performance degradation is driven by total parameter perturbation magnitude.

Key hypothesis: All experiments may collapse onto a single curve when plotted
as Accuracy vs ||ΔW||_F, regardless of sample count, learning rate, rank, etc.

No GPU required — just loads safetensor files and computes norms.
"""

import json
import os
import glob
import numpy as np
from safetensors import safe_open

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_lora_weights(adapter_path):
    """Load LoRA A and B matrices from a safetensors adapter file."""
    safetensor_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(safetensor_path):
        return None

    weights = {}
    with safe_open(safetensor_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def compute_lora_norms(weights):
    """Compute per-layer and total norms of LoRA weight updates.

    LoRA: ΔW = B · A (where A is r×d_in, B is d_out×r)
    The effective weight change has Frobenius norm ||B·A||_F

    Also computes:
    - Per-layer ||A||_F, ||B||_F, and ||B·A||_F
    - Total ||ΔW||_F across all layers
    - Spectral norm approximation
    """
    # Group A and B matrices by layer
    layers = {}
    for key, tensor in weights.items():
        # Keys like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        if "lora_A" in key:
            layer_key = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            if layer_key not in layers:
                layers[layer_key] = {}
            layers[layer_key]["A"] = tensor
        elif "lora_B" in key:
            layer_key = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            if layer_key not in layers:
                layers[layer_key] = {}
            layers[layer_key]["B"] = tensor

    per_layer = {}
    total_delta_norm_sq = 0.0
    total_a_norm_sq = 0.0
    total_b_norm_sq = 0.0

    for layer_key, mats in sorted(layers.items()):
        if "A" not in mats or "B" not in mats:
            continue
        A = mats["A"].astype(np.float32)  # shape: (r, d_in)
        B = mats["B"].astype(np.float32)  # shape: (d_out, r)

        # Compute ΔW = B · A
        delta_W = B @ A  # shape: (d_out, d_in)

        a_norm = np.linalg.norm(A)
        b_norm = np.linalg.norm(B)
        delta_norm = np.linalg.norm(delta_W)

        # Spectral norm (largest singular value) of ΔW
        # Use SVD of the smaller matrix for efficiency
        try:
            s = np.linalg.svd(delta_W, compute_uv=False)
            spectral_norm = float(s[0])
        except:
            spectral_norm = 0.0

        per_layer[layer_key] = {
            "A_shape": list(A.shape),
            "B_shape": list(B.shape),
            "delta_shape": list(delta_W.shape),
            "A_frobenius": float(a_norm),
            "B_frobenius": float(b_norm),
            "delta_frobenius": float(delta_norm),
            "spectral_norm": spectral_norm,
            "rank": int(A.shape[0]),
        }

        total_delta_norm_sq += delta_norm ** 2
        total_a_norm_sq += a_norm ** 2
        total_b_norm_sq += b_norm ** 2

    return {
        "per_layer": per_layer,
        "total_delta_frobenius": float(np.sqrt(total_delta_norm_sq)),
        "total_A_frobenius": float(np.sqrt(total_a_norm_sq)),
        "total_B_frobenius": float(np.sqrt(total_b_norm_sq)),
        "num_layers": len(per_layer),
    }


def load_experiment_metrics():
    """Load experiment log to get accuracy metrics."""
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        log = json.load(f)
    metrics = {}
    for entry in log:
        name = entry.get("experiment_name") or entry.get("name")
        if name:
            metrics[name] = entry
    return metrics


def main():
    print("=" * 70)
    print("LoRA Weight Norm Analysis Across All Experiments")
    print("=" * 70)

    # Load experiment metrics
    exp_metrics = load_experiment_metrics()
    print(f"Found {len(exp_metrics)} experiments in log")

    # Find all checkpoints with final adapters
    checkpoint_dirs = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "*/final_adapter")))
    print(f"Found {len(checkpoint_dirs)} checkpoints with final_adapter")

    results = []
    for adapter_path in checkpoint_dirs:
        exp_name = os.path.basename(os.path.dirname(adapter_path))
        print(f"\nAnalyzing: {exp_name}")

        # Load weights
        weights = load_lora_weights(adapter_path)
        if weights is None:
            print(f"  Skipping: no safetensors file")
            continue

        # Compute norms
        norms = compute_lora_norms(weights)

        # Get metrics if available
        metrics = exp_metrics.get(exp_name, {})

        # Parse experiment name for config
        config = parse_experiment_name(exp_name)

        entry = {
            "experiment_name": exp_name,
            "adapter_path": adapter_path,
            **config,
            "total_delta_frobenius": norms["total_delta_frobenius"],
            "total_A_frobenius": norms["total_A_frobenius"],
            "total_B_frobenius": norms["total_B_frobenius"],
            "num_lora_layers": norms["num_layers"],
            "per_layer_norms": norms["per_layer"],
            "orz_accuracy": metrics.get("orz_accuracy"),
            "orz_delta": metrics.get("orz_delta"),
            "sciknoweval_accuracy": metrics.get("sciknoweval_accuracy"),
            "valid": metrics.get("valid"),
        }
        results.append(entry)

        print(f"  ||ΔW||_F = {norms['total_delta_frobenius']:.4f}")
        if metrics.get("orz_accuracy") is not None:
            print(f"  ORZ accuracy = {metrics['orz_accuracy']:.4f}")

    # Save full results
    output_path = os.path.join(RESULTS_DIR, "lora_norm_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Experiment':<55} {'||ΔW||_F':>10} {'ORZ Acc':>10} {'ORZ Δ':>10} {'Valid':>6}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x.get("total_delta_frobenius", 0)):
        orz_acc = f"{r['orz_accuracy']:.4f}" if r.get("orz_accuracy") is not None else "N/A"
        orz_delta = f"{r['orz_delta']:+.4f}" if r.get("orz_delta") is not None else "N/A"
        valid = "YES" if r.get("valid") else ("NO" if r.get("valid") is not None else "N/A")
        print(f"  {r['experiment_name']:<53} {r['total_delta_frobenius']:10.4f} {orz_acc:>10} {orz_delta:>10} {valid:>6}")

    # Print per-layer analysis for a few key experiments
    print("\n\n" + "=" * 70)
    print("Per-Layer Analysis (selected experiments)")
    print("=" * 70)
    key_exps = ["sft_numinamath_n100_r8_lr5e-5_ep1", "sft_numinamath_n1000_r8_lr5e-5_ep1",
                "sft_numinamath_n10000_r8_lr5e-5_ep1", "sft_openr1_n1000_r8_lr5e-5_ep1"]
    for r in results:
        if r["experiment_name"] in key_exps:
            print(f"\n--- {r['experiment_name']} (||ΔW||_F = {r['total_delta_frobenius']:.4f}) ---")
            for layer_name, lnorm in sorted(r["per_layer_norms"].items()):
                print(f"  {layer_name:<60} ||ΔW||_F={lnorm['delta_frobenius']:.6f}  σ_max={lnorm['spectral_norm']:.6f}")

    # Compute correlation between ||ΔW||_F and ORZ accuracy
    valid_pairs = [(r["total_delta_frobenius"], r["orz_accuracy"])
                   for r in results if r.get("orz_accuracy") is not None]
    if len(valid_pairs) >= 3:
        norms_arr = np.array([p[0] for p in valid_pairs])
        accs_arr = np.array([p[1] for p in valid_pairs])
        corr = np.corrcoef(norms_arr, accs_arr)[0, 1]
        print(f"\n\nCorrelation(||ΔW||_F, ORZ_accuracy) = {corr:.4f}")
        print(f"  (N={len(valid_pairs)} experiments)")

        # Rank correlation (Spearman)
        from scipy import stats
        rho, p_val = stats.spearmanr(norms_arr, accs_arr)
        print(f"  Spearman ρ = {rho:.4f}, p = {p_val:.6f}")

    # Group by data source to analyze separately
    print("\n\n" + "=" * 70)
    print("Norm vs Accuracy by Data Source")
    print("=" * 70)
    by_source = {}
    for r in results:
        src = r.get("data_source", "unknown")
        if r.get("orz_accuracy") is not None:
            by_source.setdefault(src, []).append(r)

    for src, exps in sorted(by_source.items()):
        norms_arr = np.array([e["total_delta_frobenius"] for e in exps])
        accs_arr = np.array([e["orz_accuracy"] for e in exps])
        if len(exps) >= 3:
            corr = np.corrcoef(norms_arr, accs_arr)[0, 1]
            print(f"\n  {src} (N={len(exps)}): Correlation = {corr:.4f}")
        for e in sorted(exps, key=lambda x: x["total_delta_frobenius"]):
            print(f"    N={e.get('num_samples','?'):>6}  ||ΔW||={e['total_delta_frobenius']:.4f}  ORZ={e['orz_accuracy']:.4f}")


def parse_experiment_name(name):
    """Extract config from experiment name."""
    config = {}
    # Data source
    for src in ["numinamath_comp", "numinamath_hard", "numinamath", "openr1", "orz_reject", "orz_self"]:
        if f"_{src}_" in name or name.startswith(f"sft_{src}_") or name.startswith(f"sft_kl_{src}_"):
            config["data_source"] = src
            break

    # Sample count
    import re
    m = re.search(r"_n(\d+)", name)
    if m:
        config["num_samples"] = int(m.group(1))

    # Rank
    m = re.search(r"_r(\d+)", name)
    if m:
        config["lora_rank"] = int(m.group(1))

    # Learning rate
    m = re.search(r"_lr([\d.e-]+)", name)
    if m:
        config["lr"] = m.group(1)

    # Epochs
    m = re.search(r"_ep(\d+)", name)
    if m:
        config["epochs"] = int(m.group(1))

    # Seed
    m = re.search(r"_seed(\d+)", name)
    if m:
        config["seed"] = int(m.group(1))

    # KL weight
    m = re.search(r"_kl([\d.p]+)", name)
    if m:
        kl_str = m.group(1).replace("p", ".")
        config["kl_weight"] = float(kl_str)

    config["is_kl"] = "kl" in name and "kl_weight" not in name or config.get("kl_weight") is not None

    return config


if __name__ == "__main__":
    main()
