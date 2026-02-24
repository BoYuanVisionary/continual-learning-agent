#!/usr/bin/env python3
"""Compute direction analysis for new checkpoints (truncated OpenR1 + mixed).

Loads only the NM-10K reference vector and the 6 new checkpoints,
plus pure NM-2000 and OR1-2000 for pairwise comparison.
Much more memory-efficient than running the full analysis.

Output: results/new_direction_analysis.json
"""

import json
import os
import gc
import torch
from analyze_lora_directions import compute_dW_vector, cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Experiments we need direction data for
NEW_EXPERIMENTS = [
    "sft_openr1trunc_n1000_r8_lr5e-5_ep1",
    "sft_openr1trunc_n2000_r8_lr5e-5_ep1",
    "sft_openr1trunc_n5000_r8_lr5e-5_ep1",
    "sft_mixed75nm_25or_n2000_r8_lr5e-5_ep1",
    "sft_mixed50nm_50or_n2000_r8_lr5e-5_ep1",
    "sft_mixed25nm_75or_n2000_r8_lr5e-5_ep1",
]

# Reference experiments for cosine comparison
REFERENCE_EXPERIMENTS = [
    "sft_numinamath_n10000_r8_lr5e-5_ep1",  # NM reference
    "sft_numinamath_n2000_r8_lr5e-5_ep1",   # Pure NM at N=2000
    "sft_openr1_n2000_r8_lr5e-5_ep1",       # Pure OR1 at N=2000
]


def load_dW(exp_name):
    adapter_path = os.path.join(CHECKPOINT_DIR, exp_name, "final_adapter")
    if not os.path.exists(adapter_path):
        print(f"  WARNING: {adapter_path} not found")
        return None, 0.0
    vec, _ = compute_dW_vector(adapter_path, keep_layers=False)
    gc.collect()
    if vec is None:
        return None, 0.0
    norm = float(torch.norm(vec))
    print(f"  {exp_name}: ||dW|| = {norm:.4f}, dims = {vec.shape[0]}")
    return vec, norm


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading reference vectors...")
    refs = {}
    for name in REFERENCE_EXPERIMENTS:
        vec, norm = load_dW(name)
        if vec is not None:
            refs[name] = {"vec": vec, "norm": norm}

    nm_ref_vec = refs.get("sft_numinamath_n10000_r8_lr5e-5_ep1", {}).get("vec")
    nm2k_vec = refs.get("sft_numinamath_n2000_r8_lr5e-5_ep1", {}).get("vec")
    or2k_vec = refs.get("sft_openr1_n2000_r8_lr5e-5_ep1", {}).get("vec")

    results = {}

    print("\nLoading new experiment vectors...")
    for exp_name in NEW_EXPERIMENTS:
        vec, norm = load_dW(exp_name)
        if vec is None:
            continue

        entry = {
            "frob_norm": norm,
        }

        if nm_ref_vec is not None:
            cos_nm_ref = cosine_similarity(vec, nm_ref_vec)
            entry["cos_with_nm10k_ref"] = cos_nm_ref
            entry["effective_perturbation"] = norm * cos_nm_ref

        if nm2k_vec is not None:
            entry["cos_with_nm2k"] = cosine_similarity(vec, nm2k_vec)

        if or2k_vec is not None:
            entry["cos_with_or2k"] = cosine_similarity(vec, or2k_vec)

        results[exp_name] = entry
        del vec
        gc.collect()
        print(f"    cos(NM-ref)={entry.get('cos_with_nm10k_ref', 'N/A'):.4f}, "
              f"cos(NM-2k)={entry.get('cos_with_nm2k', 'N/A'):.4f}, "
              f"cos(OR-2k)={entry.get('cos_with_or2k', 'N/A'):.4f}")

    # Also compute reference self-cosines for context
    if nm2k_vec is not None and or2k_vec is not None:
        results["_reference_cosines"] = {
            "nm2k_vs_or2k": cosine_similarity(nm2k_vec, or2k_vec),
            "nm2k_vs_nm10k_ref": cosine_similarity(nm2k_vec, nm_ref_vec) if nm_ref_vec is not None else None,
            "or2k_vs_nm10k_ref": cosine_similarity(or2k_vec, nm_ref_vec) if nm_ref_vec is not None else None,
            "nm2k_norm": refs["sft_numinamath_n2000_r8_lr5e-5_ep1"]["norm"],
            "or2k_norm": refs["sft_openr1_n2000_r8_lr5e-5_ep1"]["norm"],
        }

    output_path = os.path.join(RESULTS_DIR, "new_direction_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Computed directions for {len([k for k in results if not k.startswith('_')])} new experiments")


if __name__ == "__main__":
    main()
