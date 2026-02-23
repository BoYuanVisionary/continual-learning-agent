#!/usr/bin/env python3
"""
Bootstrap and statistical analysis of SFT experiment results.

Computes:
1. Bootstrap 95% CIs on ORZ accuracy for each experiment (parametric bootstrap
   from Bernoulli(p_hat), 10000 resamples)
2. Wilson score confidence intervals as a closed-form alternative
3. Two-proportion z-tests for pairwise comparisons at the same sample count N
4. Statistical significance test: best N=100 result vs. baseline

Since we only have aggregate counts (correct/total), we simulate bootstrap
samples from Bernoulli(p_hat) and compute percentile CIs.

Runs on CPU only -- no GPU required.
"""

import json
import glob
import os
import sys
import math
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = "/home/hice1/byuan48/research/research_agent/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "bootstrap_analysis.json")
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
ALPHA = 0.05  # for 95% CI

# Baseline values from CLAUDE.md
BASELINE = {
    "orz": {"accuracy": 0.2891, "correct": 296, "total": 1024},
    "sciknoweval": {"accuracy": 0.3434, "correct": 650, "total": 1893},
    "toolalpaca_sim_func": {"accuracy": 0.7889, "correct": None, "total": None},
    "toolalpaca_sim_pass": {"rate": 0.7222, "correct": None, "total": None},
    "toolalpaca_real_func": {"accuracy": 0.8922, "correct": None, "total": None},
    "toolalpaca_real_pass": {"rate": 0.8725, "correct": None, "total": None},
}


# ---------------------------------------------------------------------------
# Custom JSON encoder for numpy types
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that the default encoder cannot serialize."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------
def bootstrap_ci_bernoulli(n_correct, n_total, n_bootstrap=N_BOOTSTRAP,
                           alpha=ALPHA, rng=None):
    """
    Parametric bootstrap CI for a binomial proportion.

    Simulate n_bootstrap samples of size n_total from Bernoulli(p_hat),
    compute the sample proportion for each, and return the alpha/2 and
    1-alpha/2 percentiles.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    p_hat = n_correct / n_total
    # Each element is one bootstrap replicate: count of successes in n_total trials
    boot_successes = rng.binomial(n_total, p_hat, size=n_bootstrap)
    boot_proportions = boot_successes / n_total

    ci_lower = float(np.percentile(boot_proportions, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_proportions, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_proportions, ddof=1))

    return {
        "p_hat": float(p_hat),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "bootstrap_se": se,
        "n_bootstrap": n_bootstrap,
    }


def wilson_score_interval(n_correct, n_total, alpha=ALPHA):
    """
    Wilson score confidence interval for a binomial proportion.
    Closed-form -- no simulation needed.
    """
    n = n_total
    p_hat = n_correct / n
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    return {
        "p_hat": float(p_hat),
        "ci_lower": float(centre - margin),
        "ci_upper": float(centre + margin),
        "ci_width": float(2 * margin),
    }


def two_proportion_z_test(n1_correct, n1_total, n2_correct, n2_total):
    """
    Two-proportion z-test (pooled).
    H0: p1 == p2
    Returns z-statistic and two-sided p-value.
    """
    p1 = n1_correct / n1_total
    p2 = n2_correct / n2_total
    p_pool = (n1_correct + n2_correct) / (n1_total + n2_total)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1_total + 1 / n2_total))

    if se == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "significant": False}

    z = (p1 - p2) / se
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    return {
        "z_stat": float(z),
        "p_value": p_value,
        "significant": bool(p_value < ALPHA),
    }


def parse_experiment_name(name):
    """
    Extract metadata from experiment name string.
    Example: sft_numinamath_n100_r8_lr5e-5_ep1
    Returns dict with data_source, sample_count, rank, lr, epochs.
    """
    parts = name.split("_")
    info = {"raw_name": name}

    # Find sample count (nXXX), rank (rXX), lr, epochs
    for p in parts:
        if p.startswith("n") and p[1:].isdigit():
            info["sample_count"] = int(p[1:])
        elif p.startswith("r") and p[1:].isdigit():
            info["rank"] = int(p[1:])
        elif p.startswith("lr"):
            info["lr"] = p[2:]
        elif p.startswith("ep") and p[2:].isdigit():
            info["epochs"] = int(p[2:])

    # Data source: everything between sft_ and _nXXX
    try:
        after_sft = name[4:]  # strip "sft_"
        for i, p in enumerate(after_sft.split("_")):
            if p.startswith("n") and p[1:].isdigit():
                info["data_source"] = "_".join(after_sft.split("_")[:i])
                break
    except Exception:
        info["data_source"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load all eval JSONs
    eval_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_eval.json")))
    if not eval_files:
        print("ERROR: No eval JSON files found in", RESULTS_DIR)
        sys.exit(1)

    print(f"Found {len(eval_files)} eval result files.\n")

    experiments = []
    for fpath in eval_files:
        with open(fpath) as f:
            data = json.load(f)
        exp_name = data.get("experiment_name", os.path.basename(fpath).replace("_eval.json", ""))
        meta = parse_experiment_name(exp_name)
        meta["file"] = os.path.basename(fpath)
        meta["data"] = data
        experiments.append(meta)

    # ------------------------------------------------------------------
    # 1. Bootstrap CIs and Wilson CIs for every experiment (ORZ)
    # ------------------------------------------------------------------
    print("=" * 90)
    print("BOOTSTRAP & WILSON SCORE 95% CONFIDENCE INTERVALS -- ORZ ACCURACY")
    print("=" * 90)
    print(f"{'Experiment':<52} {'Acc':>6} {'Boot CI':>18} {'Wilson CI':>18}")
    print("-" * 90)

    analysis_results = {}

    # Baseline first
    bl_orz = BASELINE["orz"]
    bl_boot = bootstrap_ci_bernoulli(bl_orz["correct"], bl_orz["total"], rng=rng)
    bl_wilson = wilson_score_interval(bl_orz["correct"], bl_orz["total"])
    analysis_results["baseline"] = {
        "orz_accuracy": bl_orz["accuracy"],
        "orz_correct": bl_orz["correct"],
        "orz_total": bl_orz["total"],
        "bootstrap_ci": bl_boot,
        "wilson_ci": bl_wilson,
    }
    print(f"{'BASELINE (Qwen2.5-3B-Instruct)':<52} {bl_orz['accuracy']:>6.2%} "
          f"[{bl_boot['ci_lower']:.4f}, {bl_boot['ci_upper']:.4f}] "
          f"[{bl_wilson['ci_lower']:.4f}, {bl_wilson['ci_upper']:.4f}]")
    print("-" * 90)

    for meta in experiments:
        data = meta["data"]
        orz = data.get("orz", {})
        n_correct = orz.get("correct", 0)
        n_total = orz.get("total", 1)

        boot = bootstrap_ci_bernoulli(n_correct, n_total, rng=rng)
        wilson = wilson_score_interval(n_correct, n_total)

        exp_name = meta["raw_name"]
        analysis_results[exp_name] = {
            "orz_accuracy": n_correct / n_total,
            "orz_correct": n_correct,
            "orz_total": n_total,
            "sample_count": meta.get("sample_count"),
            "data_source": meta.get("data_source"),
            "rank": meta.get("rank"),
            "lr": meta.get("lr"),
            "epochs": meta.get("epochs"),
            "valid": data.get("valid"),
            "bootstrap_ci": boot,
            "wilson_ci": wilson,
        }

        print(f"{exp_name:<52} {n_correct/n_total:>6.2%} "
              f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] "
              f"[{wilson['ci_lower']:.4f}, {wilson['ci_upper']:.4f}]")

    # ------------------------------------------------------------------
    # 2. Key comparison: best N=100 vs. baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("KEY COMPARISON: Best N=100 (sft_numinamath_n100_r8_lr5e-5_ep1) vs. BASELINE")
    print("=" * 90)

    best_key = "sft_numinamath_n100_r8_lr5e-5_ep1"
    if best_key in analysis_results:
        best = analysis_results[best_key]
        z_test = two_proportion_z_test(
            best["orz_correct"], best["orz_total"],
            bl_orz["correct"], bl_orz["total"],
        )

        ci_overlap = bool(
            best["bootstrap_ci"]["ci_lower"] <= bl_boot["ci_upper"] and
            bl_boot["ci_lower"] <= best["bootstrap_ci"]["ci_upper"]
        )

        print(f"  Best N=100 accuracy:  {best['orz_accuracy']:.4f}  "
              f"CI: [{best['bootstrap_ci']['ci_lower']:.4f}, {best['bootstrap_ci']['ci_upper']:.4f}]")
        print(f"  Baseline accuracy:    {bl_orz['accuracy']:.4f}  "
              f"CI: [{bl_boot['ci_lower']:.4f}, {bl_boot['ci_upper']:.4f}]")
        print(f"  Difference:           {best['orz_accuracy'] - bl_orz['accuracy']:+.4f} "
              f"({best['orz_accuracy'] - bl_orz['accuracy']:+.2%} absolute)")
        print(f"  Z-statistic:          {z_test['z_stat']:.4f}")
        print(f"  P-value (two-sided):  {z_test['p_value']:.6f}")
        print(f"  Significant (a=0.05): {'YES' if z_test['significant'] else 'NO'}")
        print(f"  CIs overlap:          {'YES' if ci_overlap else 'NO'}")

        analysis_results["key_comparison_best_vs_baseline"] = {
            "best_experiment": best_key,
            "best_accuracy": best["orz_accuracy"],
            "baseline_accuracy": bl_orz["accuracy"],
            "difference": best["orz_accuracy"] - bl_orz["accuracy"],
            "z_test": z_test,
            "cis_overlap": ci_overlap,
        }
    else:
        print(f"  WARNING: {best_key} not found in results.")

    # ------------------------------------------------------------------
    # 3. Pairwise comparisons within each sample count N
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PAIRWISE Z-TESTS WITHIN EACH SAMPLE COUNT N")
    print("=" * 90)

    # Group experiments by sample count
    by_n = defaultdict(list)
    for name, res in analysis_results.items():
        if name in ("baseline", "key_comparison_best_vs_baseline"):
            continue
        n = res.get("sample_count")
        if n is not None:
            by_n[n].append((name, res))

    pairwise_results = {}

    for n in sorted(by_n.keys()):
        exps = by_n[n]
        if len(exps) < 2:
            continue

        print(f"\n--- N = {n} ({len(exps)} experiments) ---")
        print(f"  {'Experiment A':<45} {'Experiment B':<45} {'Diff':>7} {'z':>7} {'p':>9} {'Sig?':>5}")

        pairwise_results[str(n)] = []

        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                name_a, res_a = exps[i]
                name_b, res_b = exps[j]

                z_test = two_proportion_z_test(
                    res_a["orz_correct"], res_a["orz_total"],
                    res_b["orz_correct"], res_b["orz_total"],
                )

                diff = res_a["orz_accuracy"] - res_b["orz_accuracy"]
                sig_str = "YES" if z_test["significant"] else "no"

                # Abbreviate names for display
                short_a = name_a.replace("sft_numinamath_", "nm_").replace("sft_openr1_", "or1_")
                short_b = name_b.replace("sft_numinamath_", "nm_").replace("sft_openr1_", "or1_")

                print(f"  {short_a:<45} {short_b:<45} {diff:>+7.4f} {z_test['z_stat']:>7.3f} "
                      f"{z_test['p_value']:>9.6f} {sig_str:>5}")

                pairwise_results[str(n)].append({
                    "experiment_a": name_a,
                    "experiment_b": name_b,
                    "accuracy_a": res_a["orz_accuracy"],
                    "accuracy_b": res_b["orz_accuracy"],
                    "difference": diff,
                    "z_test": z_test,
                })

    analysis_results["pairwise_comparisons"] = pairwise_results

    # ------------------------------------------------------------------
    # 4. All experiments vs. baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS vs. BASELINE (two-proportion z-test)")
    print("=" * 90)
    print(f"  {'Experiment':<52} {'Acc':>6} {'Diff':>7} {'z':>7} {'p':>9} {'Sig?':>5}")
    print("  " + "-" * 86)

    vs_baseline = {}
    sorted_exps = sorted(
        [(name, res) for name, res in analysis_results.items()
         if name not in ("baseline", "key_comparison_best_vs_baseline", "pairwise_comparisons")
         and "orz_correct" in res],
        key=lambda x: -x[1]["orz_accuracy"]
    )

    for name, res in sorted_exps:
        z_test = two_proportion_z_test(
            res["orz_correct"], res["orz_total"],
            bl_orz["correct"], bl_orz["total"],
        )

        diff = res["orz_accuracy"] - bl_orz["accuracy"]
        if z_test["significant"] and diff > 0:
            sig_str = "YES*"
            direction = "BETTER"
        elif z_test["significant"] and diff < 0:
            sig_str = "YES-"
            direction = "WORSE"
        else:
            sig_str = "no"
            direction = "~same"

        print(f"  {name:<52} {res['orz_accuracy']:>6.2%} {diff:>+7.4f} "
              f"{z_test['z_stat']:>7.3f} {z_test['p_value']:>9.6f} {sig_str:>5}")

        vs_baseline[name] = {
            "accuracy": res["orz_accuracy"],
            "difference_from_baseline": diff,
            "z_test": z_test,
            "direction": direction,
        }

    analysis_results["vs_baseline_all"] = vs_baseline

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    n_significantly_better = sum(1 for v in vs_baseline.values()
                                  if v["direction"] == "BETTER")
    n_significantly_worse = sum(1 for v in vs_baseline.values()
                                 if v["direction"] == "WORSE")
    n_no_difference = sum(1 for v in vs_baseline.values()
                           if v["direction"] == "~same")

    print(f"  Total experiments:            {len(vs_baseline)}")
    print(f"  Significantly better (p<.05): {n_significantly_better}")
    print(f"  Significantly worse (p<.05):  {n_significantly_worse}")
    print(f"  No significant difference:    {n_no_difference}")

    # Best by sample count
    print(f"\n  {'N':>6} {'Best Experiment':<50} {'Acc':>6} {'vs BL':>7} {'Sig?':>5}")
    print("  " + "-" * 76)

    best_by_n = {}
    for n in sorted(by_n.keys()):
        exps = by_n[n]
        best_name, best_res = max(exps, key=lambda x: x[1]["orz_accuracy"])
        vbl = vs_baseline.get(best_name, {})
        if vbl.get("direction") == "BETTER":
            sig = "YES"
        elif vbl.get("direction") == "WORSE":
            sig = "WORSE"
        else:
            sig = "no"
        diff = best_res["orz_accuracy"] - bl_orz["accuracy"]
        print(f"  {n:>6} {best_name:<50} {best_res['orz_accuracy']:>6.2%} {diff:>+7.4f} {sig:>5}")
        best_by_n[str(n)] = {
            "experiment": best_name,
            "accuracy": best_res["orz_accuracy"],
            "difference": diff,
            "significant_vs_baseline": bool(vbl.get("z_test", {}).get("significant", False)),
        }

    analysis_results["summary"] = {
        "total_experiments": len(vs_baseline),
        "significantly_better_than_baseline": n_significantly_better,
        "significantly_worse_than_baseline": n_significantly_worse,
        "no_significant_difference": n_no_difference,
        "best_by_sample_count": best_by_n,
        "baseline_accuracy": bl_orz["accuracy"],
        "baseline_correct": bl_orz["correct"],
        "baseline_total": bl_orz["total"],
        "alpha": ALPHA,
        "n_bootstrap": N_BOOTSTRAP,
    }

    # ------------------------------------------------------------------
    # 6. Confidence interval width analysis
    # ------------------------------------------------------------------
    bl_wilson_width = wilson_score_interval(296, 1024)["ci_width"]
    print(f"\n  Note: With n=1024, Wilson 95% CI half-width ~ +/-{bl_wilson_width/2:.4f}")
    print(f"  This means differences < ~{bl_wilson_width:.1%} are hard to distinguish.")

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    # Clean up: remove the raw 'data' field that contains the full eval JSON
    output = {}
    for key, val in analysis_results.items():
        if isinstance(val, dict) and key not in ("pairwise_comparisons", "vs_baseline_all",
                                                   "key_comparison_best_vs_baseline", "summary"):
            clean = {k: v for k, v in val.items() if k != "data"}
            output[key] = clean
        else:
            output[key] = val

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print(f"  Total experiments analyzed: {len(eval_files)}")


if __name__ == "__main__":
    main()
