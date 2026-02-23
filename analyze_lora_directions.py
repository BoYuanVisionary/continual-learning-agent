#!/usr/bin/env python3
"""Analyze LoRA weight perturbation DIRECTIONS across checkpoints.

Core analysis for the paper: computes pairwise cosine similarity between
LoRA weight update vectors (dW = B @ A) across all checkpoints.

Key findings expected:
- Intra-source (e.g., NuminaMath at different N): cosine > 0.8
- NuminaMath vs NuminaMath-Hard: cosine ~ 0.6-0.8
- NuminaMath vs OpenR1: cosine < 0.5

Also computes "effective perturbation" = ||dW|| * cos(dW, dW_ref)
to test whether direction-corrected norm unifies all data sources
into a single universal degradation curve.

Runs on CPU (~10 min for all checkpoints).

Usage:
    python analyze_lora_directions.py

Output:
    results/lora_direction_analysis.json
"""

import json
import os
import glob
import re
import numpy as np
import torch
from collections import defaultdict
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_adapter_weights(adapter_path):
    """Load LoRA adapter weights from safetensors or pytorch files."""
    safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
    if safetensor_files:
        weights = {}
        for f in safetensor_files:
            weights.update(load_file(f))
        return weights
    pt_files = glob.glob(os.path.join(adapter_path, "adapter_model.bin"))
    if pt_files:
        return torch.load(pt_files[0], map_location="cpu")
    return None


def compute_dW_vector(adapter_path, keep_layers=False):
    """Load adapter and compute flattened dW = B @ A vector for all layers.

    Args:
        adapter_path: Path to adapter directory
        keep_layers: If True, also return per-layer dW tensors (uses more memory)
    """
    weights = load_adapter_weights(adapter_path)
    if weights is None:
        return None, None

    # Group LoRA weights by layer
    layer_pairs = defaultdict(dict)
    for name, tensor in weights.items():
        if "lora_A" in name:
            key = name.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            layer_pairs[key]["A"] = tensor.float().cpu()
        elif "lora_B" in name:
            key = name.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            layer_pairs[key]["B"] = tensor.float().cpu()

    if not layer_pairs:
        return None, None

    # Compute dW for each layer
    layer_dWs = {} if keep_layers else None
    all_flat = []

    for layer_name in sorted(layer_pairs.keys()):
        pair = layer_pairs[layer_name]
        if "A" not in pair or "B" not in pair:
            continue
        A = pair["A"]
        B = pair["B"]
        dW = B @ A
        all_flat.append(dW.flatten())
        if keep_layers:
            layer_dWs[layer_name] = dW
        del A, B, dW

    # Concatenate all layer dWs into one big vector
    full_vector = torch.cat(all_flat)
    del all_flat
    # Free the raw weights
    del weights, layer_pairs

    return full_vector, layer_dWs


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def parse_experiment_name(name):
    """Parse experiment name into components."""
    n_match = re.search(r'_n(\d+)_', name)
    src_match = re.search(r'sft_(\w+?)_n', name)
    num_samples = int(n_match.group(1)) if n_match else 0
    data_source = src_match.group(1) if src_match else ""
    return data_source, num_samples


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("LoRA Direction Analysis — Cosine Similarity Between Weight Updates")
    print("=" * 70)

    # Find all checkpoints with final_adapter
    checkpoint_dirs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*/final_adapter")))
    print(f"Found {len(checkpoint_dirs)} checkpoints with final_adapter")

    # Pass 1: Load all adapters (vectors only, no per-layer data to save memory)
    import gc
    experiments = {}
    for adapter_path in checkpoint_dirs:
        exp_name = os.path.basename(os.path.dirname(adapter_path))
        data_source, num_samples = parse_experiment_name(exp_name)

        # Focus on standard config experiments for clean comparison
        parts = exp_name.split("_")
        is_standard = ("r8" in parts and "lr5e-5" in parts and "ep1" in parts
                       and not any(p.startswith("seed") or p.startswith("kl") for p in parts)
                       and "0p1" not in exp_name and "0p5" not in exp_name)

        print(f"  Loading: {exp_name} (standard={is_standard})")
        full_vec, _ = compute_dW_vector(adapter_path, keep_layers=False)
        gc.collect()

        if full_vec is not None:
            frob_norm = float(torch.norm(full_vec))
            experiments[exp_name] = {
                "full_vector": full_vec,
                "data_source": data_source,
                "num_samples": num_samples,
                "frob_norm": frob_norm,
                "is_standard": is_standard,
            }
            print(f"    ||dW||_F = {frob_norm:.4f}, dims = {full_vec.shape[0]}")

    print(f"\nLoaded {len(experiments)} experiments")

    # === 1. Pairwise cosine similarity for standard-config experiments ===
    standard_exps = {k: v for k, v in experiments.items() if v["is_standard"]}
    std_names = sorted(standard_exps.keys(),
                       key=lambda x: (standard_exps[x]["data_source"], standard_exps[x]["num_samples"]))

    print(f"\n{'='*70}")
    print(f"Pairwise Cosine Similarity ({len(std_names)} standard experiments)")
    print(f"{'='*70}")

    pairwise_cosine = {}
    for i, n1 in enumerate(std_names):
        for j, n2 in enumerate(std_names):
            if j <= i:
                continue
            cos = cosine_similarity(standard_exps[n1]["full_vector"],
                                    standard_exps[n2]["full_vector"])
            pairwise_cosine[f"{n1} vs {n2}"] = cos

    # === 2. Group by source pair type for summary statistics ===
    source_pair_cosines = defaultdict(list)
    for i, n1 in enumerate(std_names):
        for j, n2 in enumerate(std_names):
            if j <= i:
                continue
            s1 = standard_exps[n1]["data_source"]
            s2 = standard_exps[n2]["data_source"]
            cos = cosine_similarity(standard_exps[n1]["full_vector"],
                                    standard_exps[n2]["full_vector"])
            pair_key = tuple(sorted([s1, s2]))
            source_pair_cosines[f"{pair_key[0]} vs {pair_key[1]}"].append(cos)

    print("\n--- Average Cosine Similarity by Source Pair ---")
    source_pair_summary = {}
    for pair, cosines in sorted(source_pair_cosines.items()):
        mean = np.mean(cosines)
        std = np.std(cosines)
        min_c = np.min(cosines)
        max_c = np.max(cosines)
        print(f"  {pair}: mean={mean:.4f} std={std:.4f} min={min_c:.4f} max={max_c:.4f} (n={len(cosines)})")
        source_pair_summary[pair] = {
            "mean": float(mean), "std": float(std),
            "min": float(min_c), "max": float(max_c),
            "n": len(cosines), "values": [float(c) for c in cosines],
        }

    # === 3. Intra-source cosine at matching N ===
    print("\n--- Cosine Similarity at Matching N (across sources) ---")
    matching_n_cosines = {}
    sources = list(set(v["data_source"] for v in standard_exps.values()))

    for n_val in [100, 500, 1000, 2000, 5000, 10000]:
        exps_at_n = {k: v for k, v in standard_exps.items() if v["num_samples"] == n_val}
        if len(exps_at_n) < 2:
            continue
        names_at_n = sorted(exps_at_n.keys())
        for i, n1 in enumerate(names_at_n):
            for j, n2 in enumerate(names_at_n):
                if j <= i:
                    continue
                cos = cosine_similarity(exps_at_n[n1]["full_vector"],
                                        exps_at_n[n2]["full_vector"])
                s1 = exps_at_n[n1]["data_source"]
                s2 = exps_at_n[n2]["data_source"]
                key = f"N={n_val}: {s1} vs {s2}"
                matching_n_cosines[key] = float(cos)
                print(f"  {key}: cos={cos:.4f}")

    # === 4. Per-layer cosine similarity (load on demand to save memory) ===
    print("\n--- Per-Layer Cosine Similarity ---")
    per_layer_analysis = {}

    ref_pairs = [
        ("sft_numinamath_n10000_r8_lr5e-5_ep1", "sft_openr1_n2000_r8_lr5e-5_ep1"),
        ("sft_numinamath_n10000_r8_lr5e-5_ep1", "sft_numinamath_hard_n10000_r8_lr5e-5_ep1"),
        ("sft_numinamath_n1000_r8_lr5e-5_ep1", "sft_numinamath_n10000_r8_lr5e-5_ep1"),
        ("sft_openr1_n100_r8_lr5e-5_ep1", "sft_openr1_n2000_r8_lr5e-5_ep1"),
    ]

    for exp1_name, exp2_name in ref_pairs:
        if exp1_name not in experiments or exp2_name not in experiments:
            continue

        # Load per-layer data on demand
        path1 = os.path.join(CHECKPOINT_DIR, exp1_name, "final_adapter")
        path2 = os.path.join(CHECKPOINT_DIR, exp2_name, "final_adapter")
        _, layer_dWs1 = compute_dW_vector(path1, keep_layers=True)
        _, layer_dWs2 = compute_dW_vector(path2, keep_layers=True)

        if layer_dWs1 is None or layer_dWs2 is None:
            continue

        common_layers = set(layer_dWs1.keys()) & set(layer_dWs2.keys())
        layer_cosines = {}

        for layer_name in sorted(common_layers):
            dW1 = layer_dWs1[layer_name].flatten()
            dW2 = layer_dWs2[layer_name].flatten()
            cos = cosine_similarity(dW1, dW2)

            layer_match = re.search(r'layers\.(\d+)\.(.+)', layer_name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                module = layer_match.group(2)
                layer_cosines[f"L{layer_num}_{module}"] = float(cos)

        pair_key = f"{exp1_name} vs {exp2_name}"
        per_layer_analysis[pair_key] = layer_cosines

        module_cosines = defaultdict(list)
        for k, v in layer_cosines.items():
            module = k.split("_", 1)[1] if "_" in k else k
            module_cosines[module].append(v)

        print(f"\n  {pair_key}:")
        print(f"    Overall cos: {cosine_similarity(experiments[exp1_name]['full_vector'], experiments[exp2_name]['full_vector']):.4f}")
        for mod, vals in sorted(module_cosines.items()):
            print(f"    {mod}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")

        del layer_dWs1, layer_dWs2
        gc.collect()

    # === 5. Effective perturbation and universal curve ===
    print("\n--- Effective Perturbation (Projected Norm) ---")

    # Use NuminaMath N=10000 as reference direction
    ref_name = "sft_numinamath_n10000_r8_lr5e-5_ep1"
    if ref_name in experiments:
        ref_vec = experiments[ref_name]["full_vector"]
        ref_norm = torch.norm(ref_vec)
        ref_direction = ref_vec / ref_norm

        # Load ORZ accuracy from experiment log
        log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
        with open(log_path) as f:
            exp_log = json.load(f)
        accuracy_map = {e["experiment_name"]: e.get("orz_accuracy", None) for e in exp_log}

        effective_perturbations = []
        for exp_name, exp_data in sorted(standard_exps.items()):
            vec = exp_data["full_vector"]
            norm = exp_data["frob_norm"]
            cos_with_ref = cosine_similarity(vec, ref_vec)
            effective_perturb = norm * cos_with_ref
            orz_acc = accuracy_map.get(exp_name, None)

            entry = {
                "experiment": exp_name,
                "data_source": exp_data["data_source"],
                "num_samples": exp_data["num_samples"],
                "frob_norm": norm,
                "cos_with_ref": float(cos_with_ref),
                "effective_perturbation": float(effective_perturb),
                "orz_accuracy": orz_acc,
            }
            effective_perturbations.append(entry)
            print(f"  {exp_name}: ||dW||={norm:.4f}, cos={cos_with_ref:.4f}, "
                  f"eff={effective_perturb:.4f}, ORZ={orz_acc}")

        # Compute R^2 for different models
        # 1. Raw norm vs accuracy (within source)
        # 2. Raw norm vs accuracy (across sources)
        # 3. Effective perturbation vs accuracy (across sources)

        def compute_r2(x, y):
            """Compute R^2."""
            if len(x) < 3:
                return float('nan')
            x = np.array(x)
            y = np.array(y)
            correlation = np.corrcoef(x, y)[0, 1]
            return float(correlation ** 2)

        def compute_pearson(x, y):
            """Compute Pearson correlation."""
            if len(x) < 3:
                return float('nan')
            return float(np.corrcoef(np.array(x), np.array(y))[0, 1])

        # Filter to experiments with ORZ accuracy
        valid_entries = [e for e in effective_perturbations if e["orz_accuracy"] is not None]

        # Within-source R^2
        within_source_r2 = {}
        for source in set(e["data_source"] for e in valid_entries):
            source_entries = [e for e in valid_entries if e["data_source"] == source]
            if len(source_entries) >= 3:
                norms = [e["frob_norm"] for e in source_entries]
                accs = [e["orz_accuracy"] for e in source_entries]
                r2 = compute_r2(norms, accs)
                r = compute_pearson(norms, accs)
                within_source_r2[source] = {"r2": r2, "pearson_r": r, "n": len(source_entries)}
                print(f"\n  Within {source}: R^2={r2:.4f}, r={r:.4f} (n={len(source_entries)})")

        # Across-source: raw norm vs accuracy
        all_norms = [e["frob_norm"] for e in valid_entries]
        all_accs = [e["orz_accuracy"] for e in valid_entries]
        across_raw_r2 = compute_r2(all_norms, all_accs)
        across_raw_r = compute_pearson(all_norms, all_accs)
        print(f"\n  Across all sources (raw norm): R^2={across_raw_r2:.4f}, r={across_raw_r:.4f}")

        # Across-source: effective perturbation vs accuracy
        all_eff = [e["effective_perturbation"] for e in valid_entries]
        across_eff_r2 = compute_r2(all_eff, all_accs)
        across_eff_r = compute_pearson(all_eff, all_accs)
        print(f"  Across all sources (eff perturb): R^2={across_eff_r2:.4f}, r={across_eff_r:.4f}")

        r2_analysis = {
            "within_source": within_source_r2,
            "across_all_raw_norm": {"r2": across_raw_r2, "pearson_r": across_raw_r, "n": len(valid_entries)},
            "across_all_effective_perturbation": {"r2": across_eff_r2, "pearson_r": across_eff_r, "n": len(valid_entries)},
            "reference_direction": ref_name,
        }
    else:
        effective_perturbations = []
        r2_analysis = {}

    # === 6. Cosine similarity matrix for standard experiments ===
    cosine_matrix = {}
    for n1 in std_names:
        row = {}
        for n2 in std_names:
            if n1 == n2:
                row[n2] = 1.0
            else:
                row[n2] = float(cosine_similarity(
                    standard_exps[n1]["full_vector"],
                    standard_exps[n2]["full_vector"]
                ))
        cosine_matrix[n1] = row

    # === 7. Per-layer norm profiles (load on demand) ===
    print("\n--- Per-Layer Norm Profiles ---")
    layer_norm_profiles = {}
    # Only compute for a subset of key experiments to save memory
    profile_exps = [n for n in std_names if parse_experiment_name(n)[1] in [2000, 10000]]
    for exp_name in profile_exps:
        adapter_path = os.path.join(CHECKPOINT_DIR, exp_name, "final_adapter")
        _, layer_dWs = compute_dW_vector(adapter_path, keep_layers=True)
        if layer_dWs is None:
            continue
        profile = {}
        for layer_name, dW in layer_dWs.items():
            layer_match = re.search(r'layers\.(\d+)\.(.+)', layer_name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                module = layer_match.group(2)
                key = f"L{layer_num}_{module}"
                profile[key] = float(torch.norm(dW, p='fro'))
        layer_norm_profiles[exp_name] = profile
        del layer_dWs
        gc.collect()
        print(f"  Profile computed for: {exp_name}")

    # === Save results ===
    output = {
        "description": "LoRA direction (cosine similarity) analysis across checkpoints",
        "num_experiments_loaded": len(experiments),
        "num_standard_experiments": len(standard_exps),
        "source_pair_summary": source_pair_summary,
        "matching_n_cosines": matching_n_cosines,
        "per_layer_analysis": per_layer_analysis,
        "effective_perturbations": effective_perturbations,
        "r2_analysis": r2_analysis,
        "cosine_matrix": {
            "experiment_order": std_names,
            "matrix": cosine_matrix,
        },
        "layer_norm_profiles": layer_norm_profiles,
        "pairwise_cosine": pairwise_cosine,
    }

    out_path = os.path.join(RESULTS_DIR, "lora_direction_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # === Print key summary ===
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    if source_pair_summary:
        for pair, stats in sorted(source_pair_summary.items()):
            print(f"  {pair}: cos = {stats['mean']:.4f} +/- {stats['std']:.4f}")
    if r2_analysis:
        print(f"\n  Raw norm vs ORZ (all sources): R^2 = {r2_analysis.get('across_all_raw_norm', {}).get('r2', 'N/A')}")
        print(f"  Eff perturbation vs ORZ (all): R^2 = {r2_analysis.get('across_all_effective_perturbation', {}).get('r2', 'N/A')}")
        for src, stats in r2_analysis.get("within_source", {}).items():
            print(f"  Within {src}: R^2 = {stats['r2']:.4f}")


if __name__ == "__main__":
    main()
