#!/usr/bin/env python3
"""Analyze LoRA weight perturbation across checkpoints.

For each checkpoint, loads the LoRA adapter and computes:
1. Frobenius norm of the effective weight update (A * B) per layer
2. Total perturbation magnitude
3. Effective rank (via SVD)
4. Layer-wise contribution analysis

This reveals the mechanism behind cross-domain robustness and in-domain degradation:
larger perturbations explain worse performance, and the layer-wise pattern shows
which model components are most affected.

Requires GPU to load model + adapters (or can run on CPU with enough memory).

Usage:
    python analyze_lora_weights.py

Output:
    results/lora_weight_analysis.json
"""

import json
import os
import sys
import glob
import numpy as np
import torch
from collections import defaultdict
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_adapter_weights(adapter_path):
    """Load LoRA adapter weights from safetensors or pytorch files."""
    # Try safetensors first
    safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
    if safetensor_files:
        weights = {}
        for f in safetensor_files:
            weights.update(load_file(f))
        return weights

    # Try pytorch
    pt_files = glob.glob(os.path.join(adapter_path, "adapter_model.bin"))
    if pt_files:
        return torch.load(pt_files[0], map_location="cpu")

    return None


def analyze_adapter(adapter_path, adapter_name):
    """Analyze a single LoRA adapter's weight perturbation."""
    weights = load_adapter_weights(adapter_path)
    if weights is None:
        print(f"  Could not load weights from {adapter_path}")
        return None

    # Group LoRA weights by layer
    # LoRA weights have names like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    layer_pairs = defaultdict(dict)
    for name, tensor in weights.items():
        if "lora_A" in name:
            key = name.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            layer_pairs[key]["A"] = tensor.float().cpu()
        elif "lora_B" in name:
            key = name.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            layer_pairs[key]["B"] = tensor.float().cpu()

    if not layer_pairs:
        print(f"  No LoRA A/B pairs found in {adapter_name}")
        return None

    results = {
        "adapter_name": adapter_name,
        "adapter_path": adapter_path,
        "num_lora_layers": len(layer_pairs),
        "layers": {},
    }

    total_frobenius_sq = 0
    total_params = 0
    layer_norms = []

    for layer_name, pair in sorted(layer_pairs.items()):
        if "A" not in pair or "B" not in pair:
            continue

        A = pair["A"]  # shape: (r, in_features)
        B = pair["B"]  # shape: (out_features, r)

        # Effective weight update: delta_W = B @ A
        delta_W = B @ A

        # Frobenius norm
        frob_norm = float(torch.norm(delta_W, p='fro').item())

        # Spectral norm (largest singular value)
        try:
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
            spectral_norm = float(S[0].item())
            # Effective rank: ||S||_1^2 / ||S||_2^2 (or nuclear/frobenius ratio)
            nuclear_norm = float(S.sum().item())
            effective_rank = float((nuclear_norm ** 2) / (frob_norm ** 2 + 1e-10))
            # Also compute stable rank
            stable_rank = float((frob_norm ** 2) / (spectral_norm ** 2 + 1e-10))
            singular_values = S[:min(16, len(S))].tolist()
        except Exception as e:
            spectral_norm = 0
            effective_rank = 0
            stable_rank = 0
            singular_values = []

        # Relative norm (compared to dimension)
        relative_norm = frob_norm / (delta_W.shape[0] * delta_W.shape[1]) ** 0.5

        layer_info = {
            "shape_A": list(A.shape),
            "shape_B": list(B.shape),
            "shape_deltaW": list(delta_W.shape),
            "frobenius_norm": frob_norm,
            "spectral_norm": spectral_norm,
            "relative_norm": relative_norm,
            "effective_rank": effective_rank,
            "stable_rank": stable_rank,
            "top_singular_values": singular_values,
            "lora_rank": A.shape[0],
        }

        # Extract layer number and type for grouping
        # e.g., "base_model.model.model.layers.15.self_attn.q_proj"
        import re
        layer_match = re.search(r'layers\.(\d+)\.(.+)', layer_name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            module_name = layer_match.group(2)
            layer_info["layer_num"] = layer_num
            layer_info["module_name"] = module_name

        results["layers"][layer_name] = layer_info
        total_frobenius_sq += frob_norm ** 2
        total_params += delta_W.numel()
        layer_norms.append(frob_norm)

    results["total_frobenius_norm"] = float(total_frobenius_sq ** 0.5)
    results["total_params_in_delta"] = total_params
    results["mean_layer_norm"] = float(np.mean(layer_norms))
    results["max_layer_norm"] = float(np.max(layer_norms))
    results["std_layer_norm"] = float(np.std(layer_norms))

    # Group by module type (q_proj, k_proj, v_proj, o_proj)
    module_norms = defaultdict(list)
    for layer_name, info in results["layers"].items():
        if "module_name" in info:
            module_norms[info["module_name"]].append(info["frobenius_norm"])

    results["module_type_norms"] = {
        mod: {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "total": float(np.sum(np.array(norms) ** 2) ** 0.5),
        }
        for mod, norms in module_norms.items()
    }

    # Group by layer number
    layer_num_norms = defaultdict(float)
    for layer_name, info in results["layers"].items():
        if "layer_num" in info:
            layer_num_norms[info["layer_num"]] += info["frobenius_norm"] ** 2
    results["layer_num_total_norms"] = {
        str(k): float(v ** 0.5)
        for k, v in sorted(layer_num_norms.items())
    }

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("LoRA Weight Analysis Across Checkpoints")
    print("=" * 70)

    # Find all checkpoints with final_adapter
    checkpoint_dirs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*/final_adapter")))
    print(f"Found {len(checkpoint_dirs)} checkpoints with final_adapter")

    # Focus on the standard config experiments for clean comparison
    # r=8, lr=5e-5, ep=1 across data sources and N values
    target_experiments = []
    all_experiments = []

    for adapter_path in checkpoint_dirs:
        exp_name = os.path.basename(os.path.dirname(adapter_path))
        all_experiments.append((exp_name, adapter_path))

        # Parse experiment name for filtering
        # e.g., sft_numinamath_n100_r8_lr5e-5_ep1
        parts = exp_name.split("_")
        if "r8" in parts and "lr5e-5" in parts and "ep1" in parts:
            # Check it's not a seed/kl variant
            if not any(p.startswith("seed") or p.startswith("kl") for p in parts):
                target_experiments.append((exp_name, adapter_path))

    print(f"Standard config experiments: {len(target_experiments)}")
    print(f"All experiments: {len(all_experiments)}")

    # Analyze standard config experiments
    all_results = []
    for exp_name, adapter_path in target_experiments:
        print(f"\n  Analyzing: {exp_name}")
        result = analyze_adapter(adapter_path, exp_name)
        if result:
            # Parse metadata from name
            import re
            n_match = re.search(r'_n(\d+)_', exp_name)
            src_match = re.search(r'sft_(\w+?)_n', exp_name)
            result["num_samples"] = int(n_match.group(1)) if n_match else 0
            result["data_source"] = src_match.group(1) if src_match else ""
            all_results.append(result)
            print(f"    Total Frobenius norm: {result['total_frobenius_norm']:.4f}")
            print(f"    Mean layer norm: {result['mean_layer_norm']:.6f}")

    # Also analyze a few non-standard configs for comparison
    special_configs = [
        "sft_numinamath_n100_r16_lr2e-4_ep3",
        "sft_numinamath_n100_r4_lr2e-5_ep1",
        "sft_numinamath_n1000_r16_lr2e-4_ep3",
    ]
    for exp_name in special_configs:
        adapter_path = os.path.join(CHECKPOINT_DIR, exp_name, "final_adapter")
        if os.path.exists(adapter_path):
            print(f"\n  Analyzing (special): {exp_name}")
            result = analyze_adapter(adapter_path, exp_name)
            if result:
                n_match = re.search(r'_n(\d+)_', exp_name)
                src_match = re.search(r'sft_(\w+?)_n', exp_name)
                result["num_samples"] = int(n_match.group(1)) if n_match else 0
                result["data_source"] = src_match.group(1) if src_match else ""
                result["special_config"] = True
                all_results.append(result)
                print(f"    Total Frobenius norm: {result['total_frobenius_norm']:.4f}")

    # Sort by data source and N
    all_results.sort(key=lambda x: (x["data_source"], x["num_samples"]))

    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Experiment':<50} {'N':>6} {'Tot Frob':>10} {'Mean Norm':>10} {'Max Norm':>10} {'Eff Rank':>10}")
    print("-" * 100)
    for r in all_results:
        avg_eff_rank = np.mean([l["effective_rank"] for l in r["layers"].values()])
        print(f"  {r['adapter_name']:<48} {r['num_samples']:6d} "
              f"{r['total_frobenius_norm']:10.4f} {r['mean_layer_norm']:10.6f} "
              f"{r['max_layer_norm']:10.6f} {avg_eff_rank:10.4f}")

    # Analyze trend: norm vs N for each data source
    print("\n--- Norm vs N by Data Source ---")
    source_trends = defaultdict(list)
    for r in all_results:
        if not r.get("special_config"):
            source_trends[r["data_source"]].append(
                (r["num_samples"], r["total_frobenius_norm"])
            )

    for src, points in sorted(source_trends.items()):
        points.sort()
        print(f"  {src}:")
        for n, norm in points:
            print(f"    N={n:>6}: total_frob_norm = {norm:.4f}")

    # Save results
    # Strip individual layer details for compact output, keep summary
    compact_results = []
    for r in all_results:
        compact = {k: v for k, v in r.items() if k != "layers"}
        # Keep just layer_num_total_norms and module_type_norms
        compact_results.append(compact)

    output = {
        "description": "LoRA weight perturbation analysis",
        "experiments": compact_results,
        "full_results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, "lora_weight_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
