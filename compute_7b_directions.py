#!/usr/bin/env python3
"""Compute direction analysis for 7B checkpoints (memory-efficient).

Computes pairwise cosine similarities between all 7B LoRA weight updates.
Loads only 2 vectors at a time to avoid OOM on CPU nodes.

Output: results/direction_analysis_7b.json
"""

import json
import os
import gc
import re
import torch
from collections import defaultdict
from analyze_lora_directions import compute_dW_vector, cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def find_7b_experiments():
    """Find all 7B experiment checkpoints."""
    experiments = []
    for d in sorted(os.listdir(CHECKPOINT_DIR)):
        if d.endswith("_7b") and os.path.isdir(os.path.join(CHECKPOINT_DIR, d, "final_adapter")):
            experiments.append(d)
    return experiments


def parse_experiment(name):
    """Parse experiment name -> (source, N)."""
    src_match = re.search(r'sft_(\w+?)_n', name)
    n_match = re.search(r'_n(\d+)_', name)
    source = src_match.group(1) if src_match else ""
    n_val = int(n_match.group(1)) if n_match else 0
    return source, n_val


def load_vector(exp_name):
    """Load dW vector for an experiment."""
    adapter_path = os.path.join(CHECKPOINT_DIR, exp_name, "final_adapter")
    vec, _ = compute_dW_vector(adapter_path, keep_layers=False)
    return vec


def main():
    import numpy as np
    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiments = find_7b_experiments()
    print(f"Found {len(experiments)} 7B experiments:")
    for e in experiments:
        print(f"  {e}")

    if not experiments:
        print("No 7B experiments found. Exiting.")
        return

    exp_names = sorted(experiments)
    n = len(exp_names)

    # Phase 1: Compute norms (one vector at a time)
    print(f"\nPhase 1: Computing norms...")
    norms = {}
    for exp_name in exp_names:
        vec = load_vector(exp_name)
        if vec is not None:
            norm = float(torch.norm(vec))
            norms[exp_name] = norm
            source, n_val = parse_experiment(exp_name)
            print(f"  {exp_name}: ||dW|| = {norm:.4f}, source={source}, N={n_val}")
        del vec
        gc.collect()

    # Filter to only experiments with valid vectors
    exp_names = [e for e in exp_names if e in norms]
    n = len(exp_names)

    # Phase 2: Compute pairwise cosines (load 2 at a time)
    print(f"\nPhase 2: Computing {n*(n-1)//2} pairwise cosines...")
    cos_matrix = {e: {} for e in exp_names}
    # Set diagonal
    for e in exp_names:
        cos_matrix[e][e] = 1.0

    for i in range(n):
        # Load vector i
        vec_i = load_vector(exp_names[i])
        for j in range(i + 1, n):
            # Load vector j
            vec_j = load_vector(exp_names[j])
            cos_val = cosine_similarity(vec_i, vec_j)
            cos_matrix[exp_names[i]][exp_names[j]] = cos_val
            cos_matrix[exp_names[j]][exp_names[i]] = cos_val
            s_i = parse_experiment(exp_names[i])[0]
            s_j = parse_experiment(exp_names[j])[0]
            print(f"  {exp_names[i]} vs {exp_names[j]}: {cos_val:.4f}")
            del vec_j
            gc.collect()
        del vec_i
        gc.collect()

    # Choose reference direction: largest NM-7b experiment
    nm_exps = [(parse_experiment(e)[1], e) for e in exp_names if parse_experiment(e)[0] == "numinamath"]
    if nm_exps:
        nm_exps.sort(reverse=True)
        ref_name = nm_exps[0][1]
    else:
        ref_name = exp_names[-1]
    print(f"\nReference direction: {ref_name}")

    # Effective perturbations
    eff_perturbs = []
    for exp_name in exp_names:
        source, n_val = parse_experiment(exp_name)
        cos_ref = cos_matrix[exp_name][ref_name]
        eff_perturbs.append({
            "experiment_name": exp_name,
            "data_source": source,
            "num_samples": n_val,
            "frob_norm": norms[exp_name],
            "cos_with_ref": cos_ref,
            "effective_perturbation": norms[exp_name] * cos_ref,
        })

    # Source-pair summary
    source_pair_cosines = defaultdict(list)
    for i, n1 in enumerate(exp_names):
        for j, n2 in enumerate(exp_names):
            if i >= j:
                continue
            s1 = parse_experiment(n1)[0]
            s2 = parse_experiment(n2)[0]
            pair = tuple(sorted([s1, s2]))
            source_pair_cosines[f"{pair[0]} vs {pair[1]}"].append(cos_matrix[n1][n2])

    source_pair_summary = {}
    for pair, cosines in source_pair_cosines.items():
        arr = np.array(cosines)
        source_pair_summary[pair] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(cosines),
        }

    # Matching-N cross-source cosines
    matching_n_cosines = {}
    for i, n1 in enumerate(exp_names):
        for j, n2 in enumerate(exp_names):
            if i >= j:
                continue
            s1, nv1 = parse_experiment(n1)
            s2, nv2 = parse_experiment(n2)
            if nv1 == nv2 and s1 != s2:
                key = f"N={nv1}: {s1} vs {s2}"
                matching_n_cosines[key] = cos_matrix[n1][n2]

    results = {
        "model_scale": "7B",
        "reference_direction": ref_name,
        "num_experiments": len(exp_names),
        "cosine_matrix": {
            "experiment_order": exp_names,
            "matrix": cos_matrix,
        },
        "effective_perturbations": eff_perturbs,
        "source_pair_summary": source_pair_summary,
        "matching_n_cosines": matching_n_cosines,
        "norms": norms,
    }

    output_path = os.path.join(RESULTS_DIR, "direction_analysis_7b.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n=== Source Pair Summary ===")
    for pair, stats in sorted(source_pair_summary.items()):
        print(f"  {pair}: mean={stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['count']})")

    print("\n=== Matching-N Cross-Source Cosines ===")
    for key, val in sorted(matching_n_cosines.items()):
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
