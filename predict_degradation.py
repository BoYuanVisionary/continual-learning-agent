#!/usr/bin/env python3
"""Predictive Direction Probes: Can dW geometry predict ORZ accuracy?

Experiment A for addressing reviewer W2 (predictive utility).

Tests whether the geometric framework has predictive power by:
1. Fitting linear models: ORZ_acc = a * metric + b
   - Raw norm model: metric = ||dW||
   - Effective perturbation model: metric = ||dW|| * cos(dW, dW_ref)
2. Train/test split: train on N<=1000, predict N>1000
3. Cross-source prediction: fit on NuminaMath, predict OpenR1
4. Leave-one-out cross-validation comparing both models
5. Direction stability: cos(dW_N100, dW_N10K) within source

Runs on CPU only (~10-20 min).

Usage:
    python predict_degradation.py

Output:
    results/predictive_probe_analysis.json
"""

import json
import os
import glob
import re
import numpy as np
import torch
from collections import defaultdict

# Reuse functions from analyze_lora_directions.py
from analyze_lora_directions import (
    compute_dW_vector, cosine_similarity, parse_experiment_name,
    CHECKPOINT_DIR, RESULTS_DIR
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fit_linear(x, y):
    """Fit y = a*x + b, return (a, b, r2, mae_train)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return 0.0, 0.0, 0.0, float('inf')
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b = result[0]
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y - y_pred))
    return float(a), float(b), float(r2), float(mae)


def predict_linear(x, a, b):
    """Predict y = a*x + b."""
    return np.array(x, dtype=float) * a + b


def leave_one_out_mae(x, y):
    """Leave-one-out cross-validation MAE."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return float('inf')
    errors = []
    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        a, b, _, _ = fit_linear(x_train, y_train)
        y_pred = a * x[i] + b
        errors.append(abs(y[i] - y_pred))
    return float(np.mean(errors))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import gc

    print("=" * 70)
    print("Predictive Direction Probes — Experiment A")
    print("=" * 70)

    # Load experiment log for ORZ accuracies
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path) as f:
        exp_log = json.load(f)
    accuracy_map = {e["experiment_name"]: e.get("orz_accuracy") for e in exp_log}

    # Load all standard-config checkpoints
    checkpoint_dirs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*/final_adapter")))
    print(f"Found {len(checkpoint_dirs)} checkpoints")

    experiments = {}
    for adapter_path in checkpoint_dirs:
        exp_name = os.path.basename(os.path.dirname(adapter_path))
        data_source, num_samples = parse_experiment_name(exp_name)

        parts = exp_name.split("_")
        is_standard = ("r8" in parts and "lr5e-5" in parts and "ep1" in parts
                       and not any(p.startswith("seed") or p.startswith("kl") for p in parts)
                       and "0p1" not in exp_name and "0p5" not in exp_name)

        if not is_standard:
            continue
        if data_source not in ("numinamath", "numinamath_hard", "openr1"):
            continue

        orz_acc = accuracy_map.get(exp_name)
        if orz_acc is None:
            continue

        print(f"  Loading: {exp_name}")
        full_vec, _ = compute_dW_vector(adapter_path, keep_layers=False)
        gc.collect()

        if full_vec is not None:
            frob_norm = float(torch.norm(full_vec))
            experiments[exp_name] = {
                "full_vector": full_vec,
                "data_source": data_source,
                "num_samples": num_samples,
                "frob_norm": frob_norm,
                "orz_accuracy": orz_acc,
            }

    print(f"\nLoaded {len(experiments)} standard experiments with ORZ accuracy")

    # Compute reference direction (NuminaMath N=10000)
    ref_name = "sft_numinamath_n10000_r8_lr5e-5_ep1"
    if ref_name not in experiments:
        print("ERROR: reference checkpoint not found")
        return
    ref_vec = experiments[ref_name]["full_vector"]

    # Compute effective perturbation for each experiment
    for exp_name, exp_data in experiments.items():
        vec = exp_data["full_vector"]
        cos_ref = cosine_similarity(vec, ref_vec)
        exp_data["cos_with_ref"] = float(cos_ref)
        exp_data["effective_perturbation"] = exp_data["frob_norm"] * cos_ref

    # === Analysis 1: Train/test split per source ===
    print("\n" + "=" * 70)
    print("Analysis 1: Train (N<=1000) -> Test (N>1000) prediction")
    print("=" * 70)

    per_source_results = {}
    for source in ["numinamath", "numinamath_hard", "openr1"]:
        source_exps = {k: v for k, v in experiments.items() if v["data_source"] == source}
        train_exps = {k: v for k, v in source_exps.items() if v["num_samples"] <= 1000}
        test_exps = {k: v for k, v in source_exps.items() if v["num_samples"] > 1000}

        if len(train_exps) < 2 or len(test_exps) < 1:
            print(f"  {source}: insufficient data (train={len(train_exps)}, test={len(test_exps)})")
            continue

        # Raw norm model
        train_norms = [v["frob_norm"] for v in train_exps.values()]
        train_accs = [v["orz_accuracy"] for v in train_exps.values()]
        a_raw, b_raw, r2_raw, mae_train_raw = fit_linear(train_norms, train_accs)

        test_norms = [v["frob_norm"] for v in test_exps.values()]
        test_accs = [v["orz_accuracy"] for v in test_exps.values()]
        pred_raw = predict_linear(test_norms, a_raw, b_raw)
        mae_test_raw = float(np.mean(np.abs(np.array(test_accs) - pred_raw)))

        # Effective perturbation model
        train_eff = [v["effective_perturbation"] for v in train_exps.values()]
        a_eff, b_eff, r2_eff, mae_train_eff = fit_linear(train_eff, train_accs)

        test_eff = [v["effective_perturbation"] for v in test_exps.values()]
        pred_eff = predict_linear(test_eff, a_eff, b_eff)
        mae_test_eff = float(np.mean(np.abs(np.array(test_accs) - pred_eff)))

        print(f"\n  {source}:")
        print(f"    Raw norm:  train R2={r2_raw:.3f}, test MAE={mae_test_raw:.4f}")
        print(f"    Eff perturb: train R2={r2_eff:.3f}, test MAE={mae_test_eff:.4f}")

        test_details = []
        for k, v in sorted(test_exps.items(), key=lambda x: x[1]["num_samples"]):
            idx = list(test_exps.keys()).index(k)
            test_details.append({
                "name": k,
                "N": v["num_samples"],
                "actual_orz": v["orz_accuracy"],
                "pred_raw": float(pred_raw[idx]),
                "pred_eff": float(pred_eff[idx]),
                "error_raw": float(abs(v["orz_accuracy"] - pred_raw[idx])),
                "error_eff": float(abs(v["orz_accuracy"] - pred_eff[idx])),
            })

        per_source_results[source] = {
            "n_train": len(train_exps),
            "n_test": len(test_exps),
            "raw_norm_model": {"a": a_raw, "b": b_raw, "r2_train": r2_raw, "mae_test": mae_test_raw},
            "eff_perturb_model": {"a": a_eff, "b": b_eff, "r2_train": r2_eff, "mae_test": mae_test_eff},
            "test_predictions": test_details,
        }

    # === Analysis 2: Cross-source prediction ===
    print("\n" + "=" * 70)
    print("Analysis 2: Cross-source prediction (fit NM, predict OpenR1)")
    print("=" * 70)

    nm_exps = {k: v for k, v in experiments.items() if v["data_source"] == "numinamath"}
    or_exps = {k: v for k, v in experiments.items() if v["data_source"] == "openr1"}

    cross_source_results = {}
    if len(nm_exps) >= 3 and len(or_exps) >= 2:
        # Fit on NuminaMath
        nm_norms = [v["frob_norm"] for v in nm_exps.values()]
        nm_accs = [v["orz_accuracy"] for v in nm_exps.values()]
        nm_eff = [v["effective_perturbation"] for v in nm_exps.values()]

        a_raw, b_raw, r2_raw, _ = fit_linear(nm_norms, nm_accs)
        a_eff, b_eff, r2_eff, _ = fit_linear(nm_eff, nm_accs)

        # Predict OpenR1
        or_norms = [v["frob_norm"] for v in or_exps.values()]
        or_accs = [v["orz_accuracy"] for v in or_exps.values()]
        or_eff = [v["effective_perturbation"] for v in or_exps.values()]

        pred_raw = predict_linear(or_norms, a_raw, b_raw)
        pred_eff = predict_linear(or_eff, a_eff, b_eff)

        mae_cross_raw = float(np.mean(np.abs(np.array(or_accs) - pred_raw)))
        mae_cross_eff = float(np.mean(np.abs(np.array(or_accs) - pred_eff)))

        print(f"  Fit on NuminaMath (n={len(nm_exps)}), predict OpenR1 (n={len(or_exps)}):")
        print(f"    Raw norm:    MAE = {mae_cross_raw:.4f}")
        print(f"    Eff perturb: MAE = {mae_cross_eff:.4f}")
        print(f"    Improvement: {(1 - mae_cross_eff / mae_cross_raw) * 100:.1f}%")

        cross_predictions = []
        for k, v in sorted(or_exps.items(), key=lambda x: x[1]["num_samples"]):
            idx = list(or_exps.keys()).index(k)
            cross_predictions.append({
                "name": k,
                "N": v["num_samples"],
                "actual_orz": v["orz_accuracy"],
                "pred_raw": float(pred_raw[idx]),
                "pred_eff": float(pred_eff[idx]),
            })

        cross_source_results = {
            "fit_source": "numinamath",
            "predict_source": "openr1",
            "n_fit": len(nm_exps),
            "n_predict": len(or_exps),
            "raw_norm_mae": mae_cross_raw,
            "eff_perturb_mae": mae_cross_eff,
            "improvement_pct": float((1 - mae_cross_eff / mae_cross_raw) * 100),
            "predictions": cross_predictions,
        }

    # === Analysis 3: Leave-one-out CV ===
    print("\n" + "=" * 70)
    print("Analysis 3: Leave-one-out cross-validation (all experiments)")
    print("=" * 70)

    all_norms = [v["frob_norm"] for v in experiments.values()]
    all_accs = [v["orz_accuracy"] for v in experiments.values()]
    all_eff = [v["effective_perturbation"] for v in experiments.values()]

    loo_mae_raw = leave_one_out_mae(all_norms, all_accs)
    loo_mae_eff = leave_one_out_mae(all_eff, all_accs)

    print(f"  All experiments (n={len(experiments)}):")
    print(f"    Raw norm LOO-MAE:    {loo_mae_raw:.4f}")
    print(f"    Eff perturb LOO-MAE: {loo_mae_eff:.4f}")
    print(f"    Improvement: {(1 - loo_mae_eff / loo_mae_raw) * 100:.1f}%")

    loo_results = {
        "n_experiments": len(experiments),
        "raw_norm_loo_mae": loo_mae_raw,
        "eff_perturb_loo_mae": loo_mae_eff,
        "improvement_pct": float((1 - loo_mae_eff / loo_mae_raw) * 100),
    }

    # === Analysis 4: Direction stability ===
    print("\n" + "=" * 70)
    print("Analysis 4: Direction stability within source")
    print("=" * 70)

    stability_results = {}
    for source in ["numinamath", "numinamath_hard", "openr1"]:
        source_exps = {k: v for k, v in experiments.items() if v["data_source"] == source}
        sorted_exps = sorted(source_exps.items(), key=lambda x: x[1]["num_samples"])

        if len(sorted_exps) < 2:
            continue

        # Cosine between smallest and largest N
        smallest = sorted_exps[0]
        largest = sorted_exps[-1]
        cos_min_max = cosine_similarity(smallest[1]["full_vector"], largest[1]["full_vector"])

        # Pairwise cosines between consecutive N values
        consecutive_cosines = []
        for i in range(len(sorted_exps) - 1):
            cos = cosine_similarity(sorted_exps[i][1]["full_vector"],
                                    sorted_exps[i + 1][1]["full_vector"])
            consecutive_cosines.append({
                "from": sorted_exps[i][1]["num_samples"],
                "to": sorted_exps[i + 1][1]["num_samples"],
                "cosine": float(cos),
            })

        print(f"\n  {source}:")
        print(f"    N={smallest[1]['num_samples']} vs N={largest[1]['num_samples']}: cos = {cos_min_max:.4f}")
        for c in consecutive_cosines:
            print(f"    N={c['from']} -> N={c['to']}: cos = {c['cosine']:.4f}")

        stability_results[source] = {
            "min_N": smallest[1]["num_samples"],
            "max_N": largest[1]["num_samples"],
            "cos_min_max": float(cos_min_max),
            "consecutive_cosines": consecutive_cosines,
        }

    # === Analysis 5: Predicted vs actual scatter data ===
    print("\n" + "=" * 70)
    print("Analysis 5: Full predicted vs actual data for plotting")
    print("=" * 70)

    # Fit on all data for scatter plot
    a_raw_all, b_raw_all, r2_raw_all, _ = fit_linear(all_norms, all_accs)
    a_eff_all, b_eff_all, r2_eff_all, _ = fit_linear(all_eff, all_accs)

    scatter_data = []
    for exp_name, exp_data in sorted(experiments.items()):
        scatter_data.append({
            "name": exp_name,
            "data_source": exp_data["data_source"],
            "num_samples": exp_data["num_samples"],
            "frob_norm": exp_data["frob_norm"],
            "effective_perturbation": exp_data["effective_perturbation"],
            "cos_with_ref": exp_data["cos_with_ref"],
            "actual_orz": exp_data["orz_accuracy"],
            "pred_raw": float(a_raw_all * exp_data["frob_norm"] + b_raw_all),
            "pred_eff": float(a_eff_all * exp_data["effective_perturbation"] + b_eff_all),
        })

    print(f"  Raw norm model (all): R2 = {r2_raw_all:.4f}")
    print(f"  Eff perturb model (all): R2 = {r2_eff_all:.4f}")

    # === Save results ===
    output = {
        "description": "Predictive direction probe analysis (Experiment A)",
        "reference_direction": ref_name,
        "per_source_train_test": per_source_results,
        "cross_source_prediction": cross_source_results,
        "leave_one_out_cv": loo_results,
        "direction_stability": stability_results,
        "scatter_data": scatter_data,
        "global_fit": {
            "raw_norm": {"a": a_raw_all, "b": b_raw_all, "r2": r2_raw_all},
            "eff_perturb": {"a": a_eff_all, "b": b_eff_all, "r2": r2_eff_all},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "predictive_probe_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print key summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  LOO-MAE raw norm: {loo_mae_raw:.4f}")
    print(f"  LOO-MAE eff perturb: {loo_mae_eff:.4f}")
    if cross_source_results:
        print(f"  Cross-source (NM->OR1) raw MAE: {cross_source_results['raw_norm_mae']:.4f}")
        print(f"  Cross-source (NM->OR1) eff MAE: {cross_source_results['eff_perturb_mae']:.4f}")


if __name__ == "__main__":
    main()
