#!/usr/bin/env python3
"""Compute direction analysis for CodeAlpaca SFT checkpoints.

Computes cosine similarity between code SFT directions and math SFT directions.
Key prediction: code direction should be near-orthogonal (cos << 0.5) with math directions.

Output: results/direction_analysis_code.json
"""

import json
import os
import gc
import re
import torch
from analyze_lora_directions import compute_dW_vector, cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Code experiments
CODE_EXPERIMENTS = [
    "sft_codealpaca_n500_r8_lr5e-5_ep1",
    "sft_codealpaca_n1000_r8_lr5e-5_ep1",
    "sft_codealpaca_n2000_r8_lr5e-5_ep1",
    "sft_codealpaca_n5000_r8_lr5e-5_ep1",
]

# Math references for comparison (must use same 7 target modules as code experiments)
MATH_REFERENCES = [
    "sft_numinamath_n2000_r8_lr5e-5_ep1_seed3",  # NM with 7 target modules
    "sft_openr1_n2000_r8_lr5e-5_ep1_seed1",  # OR1 with 7 target modules
    "sft_openr1_n2000_r8_lr5e-5_ep1_seed2",
    "sft_openr1_n2000_r8_lr5e-5_ep1_seed3",
]


def load_dW(exp_name):
    adapter_path = os.path.join(CHECKPOINT_DIR, exp_name, "final_adapter")
    if not os.path.exists(adapter_path):
        print(f"  WARNING: {adapter_path} not found, skipping")
        return None, 0.0
    vec, _ = compute_dW_vector(adapter_path, keep_layers=False)
    gc.collect()
    if vec is None:
        return None, 0.0
    norm = float(torch.norm(vec))
    print(f"  {exp_name}: ||dW|| = {norm:.4f}")
    return vec, norm


def parse_n(name):
    m = re.search(r'_n(\d+)_', name)
    return int(m.group(1)) if m else 0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load math reference vectors
    print("Loading math reference vectors...")
    math_vecs = {}
    for name in MATH_REFERENCES:
        vec, norm = load_dW(name)
        if vec is not None:
            math_vecs[name] = {"vec": vec, "norm": norm}

    # Load code vectors
    print("\nLoading code experiment vectors...")
    code_vecs = {}
    for name in CODE_EXPERIMENTS:
        vec, norm = load_dW(name)
        if vec is not None:
            code_vecs[name] = {"vec": vec, "norm": norm}

    if not code_vecs:
        print("No code experiments found. Exiting.")
        return

    results = {
        "code_experiments": {},
        "code_vs_math_cosines": {},
        "code_vs_code_cosines": {},
    }

    # Code vs math cosines
    print("\n=== Code vs Math Direction Cosines ===")
    for code_name, code_data in sorted(code_vecs.items()):
        entry = {
            "frob_norm": code_data["norm"],
            "num_samples": parse_n(code_name),
            "cosines_with_math": {},
        }
        for math_name, math_data in sorted(math_vecs.items()):
            cos_val = cosine_similarity(code_data["vec"], math_data["vec"])
            entry["cosines_with_math"][math_name] = cos_val

            # Short labels
            m_src = "NM" if "numinamath" in math_name else "OR1"
            m_n = parse_n(math_name)
            c_n = parse_n(code_name)
            print(f"  Code-{c_n} vs {m_src}-{m_n}: {cos_val:.4f}")
            results["code_vs_math_cosines"][f"code_{c_n}_vs_{m_src.lower()}_{m_n}"] = cos_val

        results["code_experiments"][code_name] = entry

    # Code vs code cosines
    print("\n=== Code vs Code Direction Cosines ===")
    code_names = sorted(code_vecs.keys())
    for i, n1 in enumerate(code_names):
        for j, n2 in enumerate(code_names):
            if i >= j:
                continue
            cos_val = cosine_similarity(code_vecs[n1]["vec"], code_vecs[n2]["vec"])
            key = f"code_{parse_n(n1)}_vs_code_{parse_n(n2)}"
            results["code_vs_code_cosines"][key] = cos_val
            print(f"  Code-{parse_n(n1)} vs Code-{parse_n(n2)}: {cos_val:.4f}")

    # Math vs math for context (same-source vs cross-source)
    print("\n=== Math vs Math Reference Cosines ===")
    math_names = sorted(math_vecs.keys())
    math_cosines = {}
    for i, n1 in enumerate(math_names):
        for j, n2 in enumerate(math_names):
            if i >= j:
                continue
            cos_val = cosine_similarity(math_vecs[n1]["vec"], math_vecs[n2]["vec"])
            s1 = "NM" if "numinamath" in n1 else "OR1"
            s2 = "NM" if "numinamath" in n2 else "OR1"
            key = f"{s1}_{parse_n(n1)}_vs_{s2}_{parse_n(n2)}"
            math_cosines[key] = cos_val
            print(f"  {s1}-{parse_n(n1)} vs {s2}-{parse_n(n2)}: {cos_val:.4f}")
    results["math_vs_math_cosines"] = math_cosines

    # Summary statistics
    code_math_vals = list(results["code_vs_math_cosines"].values())
    code_code_vals = list(results["code_vs_code_cosines"].values())
    import numpy as np
    results["summary"] = {
        "code_vs_math_mean": float(np.mean(code_math_vals)) if code_math_vals else None,
        "code_vs_math_std": float(np.std(code_math_vals)) if code_math_vals else None,
        "code_vs_code_mean": float(np.mean(code_code_vals)) if code_code_vals else None,
        "code_vs_code_std": float(np.std(code_code_vals)) if code_code_vals else None,
    }

    print(f"\n=== Summary ===")
    print(f"  Code vs Math: mean={results['summary']['code_vs_math_mean']:.4f} +/- {results['summary']['code_vs_math_std']:.4f}")
    if results['summary']['code_vs_code_mean'] is not None:
        print(f"  Code vs Code: mean={results['summary']['code_vs_code_mean']:.4f} +/- {results['summary']['code_vs_code_std']:.4f}")

    output_path = os.path.join(RESULTS_DIR, "direction_analysis_code.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
