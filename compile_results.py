#!/usr/bin/env python3
"""Compile all experiment results into a structured log."""
import json
import os
import glob
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

BASELINES = {
    "orz_accuracy": 0.2891,
    "sciknoweval_accuracy": 0.3434,
    "toolalpaca_sim_func_acc": 0.7889,
    "toolalpaca_sim_pass_rate": 0.7222,
    "toolalpaca_real_func_acc": 0.8922,
    "toolalpaca_real_pass_rate": 0.8725,
}

def parse_experiment_name(name):
    """Extract parameters from experiment name."""
    info = {"name": name}

    # Data source
    for src in ["numinamath_comp", "numinamath_hard", "numinamath", "openr1", "orz_self"]:
        if f"sft_{src}" in name:
            info["data_source"] = src
            break

    # Sample count
    m = re.search(r"_n(\d+)_", name)
    if m:
        info["num_samples"] = int(m.group(1))

    # LoRA rank
    m = re.search(r"_r(\d+)_", name)
    if m:
        info["lora_rank"] = int(m.group(1))

    # Learning rate
    m = re.search(r"_lr([\de\-\.]+)_", name)
    if m:
        info["lr"] = m.group(1)

    # Epochs
    m = re.search(r"_ep(\d+)$", name)
    if m:
        info["epochs"] = int(m.group(1))

    return info

def main():
    eval_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_eval.json")))

    log = []
    for fpath in eval_files:
        with open(fpath) as f:
            data = json.load(f)

        name = data.get("experiment_name", os.path.basename(fpath).replace("_eval.json", ""))
        params = parse_experiment_name(name)

        entry = {
            "experiment_name": name,
            **params,
            "valid": data.get("valid", False),
            "forgetting_issues": data.get("forgetting_issues", []),
        }

        if "orz" in data:
            entry["orz_accuracy"] = data["orz"]["accuracy"]
            entry["orz_delta"] = data["orz"]["accuracy"] - BASELINES["orz_accuracy"]

        if "sciknoweval" in data:
            entry["sciknoweval_accuracy"] = data["sciknoweval"]["accuracy"]
            entry["sciknoweval_delta"] = data["sciknoweval"]["accuracy"] - BASELINES["sciknoweval_accuracy"]

        if "toolalpaca" in data:
            ta = data["toolalpaca"]
            if "simulated" in ta and ta["simulated"]:
                entry["toolalpaca_sim_func_acc"] = ta["simulated"]["func_accuracy"]
                entry["toolalpaca_sim_pass_rate"] = ta["simulated"]["pass_rate"]
            if "real" in ta and ta["real"]:
                entry["toolalpaca_real_func_acc"] = ta["real"]["func_accuracy"]
                entry["toolalpaca_real_pass_rate"] = ta["real"]["pass_rate"]

        log.append(entry)

    # Sort by data source, then num_samples
    log.sort(key=lambda x: (x.get("data_source", ""), x.get("num_samples", 0), x.get("lr", ""), x.get("epochs", 0)))

    # Save log
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved {len(log)} experiments to {log_path}")

    # Print summary table
    print(f"\n{'=' * 120}")
    print(f"{'Experiment':<55} {'N':>6} {'ORZ%':>7} {'Δ':>7} {'SKE%':>7} {'TA-S%':>6} {'TA-R%':>6} {'Valid':>5}")
    print(f"{'-' * 120}")

    # Group by data source
    current_source = None
    for entry in log:
        src = entry.get("data_source", "?")
        if src != current_source:
            print(f"\n  [{src}]")
            current_source = src

        n = entry.get("num_samples", "?")
        orz = entry.get("orz_accuracy", 0)
        orz_delta = entry.get("orz_delta", 0)
        ske = entry.get("sciknoweval_accuracy", 0)
        ta_s = entry.get("toolalpaca_sim_func_acc", 0)
        ta_r = entry.get("toolalpaca_real_func_acc", 0)
        valid = "YES" if entry.get("valid") else "NO"

        delta_str = f"+{orz_delta:.2%}" if orz_delta >= 0 else f"{orz_delta:.2%}"

        print(f"  {entry['experiment_name']:<53} {n:>6} {orz:>7.2%} {delta_str:>7} {ske:>7.2%} {ta_s:>6.2%} {ta_r:>6.2%} {valid:>5}")

    print(f"\n{'=' * 120}")
    print(f"\nBaseline:{'':>46} {'':>6} {BASELINES['orz_accuracy']:>7.2%} {'':>7} {BASELINES['sciknoweval_accuracy']:>7.2%} {BASELINES['toolalpaca_sim_func_acc']:>6.2%} {BASELINES['toolalpaca_real_func_acc']:>6.2%}")

    # Find best valid experiment
    valid_exps = [e for e in log if e.get("valid")]
    if valid_exps:
        best = max(valid_exps, key=lambda x: x.get("orz_accuracy", 0))
        print(f"\nBest valid: {best['experiment_name']} → ORZ={best.get('orz_accuracy', 0):.2%}")

    # Find best per N
    print("\n=== Best Valid ORZ by Sample Count ===")
    for n in [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]:
        candidates = [e for e in log if e.get("num_samples") == n and e.get("valid")]
        if candidates:
            best = max(candidates, key=lambda x: x.get("orz_accuracy", 0))
            delta = best.get("orz_delta", 0)
            delta_str = f"+{delta:.2%}" if delta >= 0 else f"{delta:.2%}"
            print(f"  N={n:>5}: {best.get('orz_accuracy', 0):.2%} ({delta_str}) - {best['experiment_name']}")

if __name__ == "__main__":
    main()
