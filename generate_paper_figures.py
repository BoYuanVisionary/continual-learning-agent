#!/usr/bin/env python3
"""Generate publication-quality figures for the perturbation geometry paper.

Figures:
1. ||dW||_F vs ORZ accuracy (scatter, colored by data source) — shows norm alone fails
2. Pairwise cosine similarity heatmap — shows block structure by data source
3. Effective perturbation vs ORZ accuracy — universal curve collapse (hero result)
4. Per-layer norm profiles — NuminaMath vs OpenR1 comparison
5. Format decomposition — strict vs tolerant GSM8K for both sources
6. OpenR1 cliff characterization — ORZ, GSM8K boxed rate, response length vs N

Usage:
    python generate_paper_figures.py

Output:
    results/figures/fig1_norm_vs_orz.png
    results/figures/fig2_cosine_heatmap.png
    results/figures/fig3_universal_curve.png
    results/figures/fig4_layer_profiles.png
    results/figures/fig5_format_decomposition.png
    results/figures/fig6_openr1_cliff.png
"""

import json
import os
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Color scheme for data sources
SOURCE_COLORS = {
    "numinamath": "#2196F3",       # blue
    "numinamath_hard": "#4CAF50",  # green
    "numinamath_comp": "#9C27B0",  # purple
    "openr1": "#F44336",           # red
}
SOURCE_MARKERS = {
    "numinamath": "o",
    "numinamath_hard": "s",
    "numinamath_comp": "D",
    "openr1": "^",
}
SOURCE_LABELS = {
    "numinamath": "NuminaMath",
    "numinamath_hard": "NuminaMath-Hard",
    "numinamath_comp": "NuminaMath-Comp",
    "openr1": "OpenR1",
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def load_experiment_data():
    """Load experiment log and LoRA weight analysis."""
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path) as f:
        exp_log = json.load(f)

    weight_path = os.path.join(RESULTS_DIR, "lora_weight_analysis.json")
    with open(weight_path) as f:
        weight_data = json.load(f)

    direction_path = os.path.join(RESULTS_DIR, "lora_direction_analysis.json")
    if os.path.exists(direction_path):
        with open(direction_path) as f:
            direction_data = json.load(f)
    else:
        direction_data = None

    return exp_log, weight_data, direction_data


def get_norm_map(weight_data):
    """Build experiment_name -> total_frobenius_norm mapping."""
    norm_map = {}
    for exp in weight_data.get("experiments", []):
        norm_map[exp["adapter_name"]] = exp["total_frobenius_norm"]
    # Also check full_results
    for exp in weight_data.get("full_results", []):
        if exp.get("adapter_name") and "total_frobenius_norm" in exp:
            norm_map[exp["adapter_name"]] = exp["total_frobenius_norm"]
    return norm_map


def parse_source(name):
    """Extract data source from experiment name."""
    m = re.search(r'sft_(\w+?)_n', name)
    return m.group(1) if m else ""


def parse_n(name):
    """Extract N from experiment name."""
    m = re.search(r'_n(\d+)_', name)
    return int(m.group(1)) if m else 0


def is_standard_config(name):
    """Check if experiment uses standard r8/lr5e-5/ep1 config."""
    parts = name.split("_")
    return ("r8" in parts and "lr5e-5" in parts and "ep1" in parts
            and not any(p.startswith("seed") or p.startswith("kl") for p in parts)
            and "0p1" not in name and "0p5" not in name)


def fig1_norm_vs_orz(exp_log, norm_map):
    """Fig 1: ||dW||_F vs ORZ accuracy, colored by data source."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for exp in exp_log:
        name = exp.get("experiment_name", exp.get("name", ""))
        if not is_standard_config(name):
            continue
        source = parse_source(name)
        if source not in SOURCE_COLORS:
            continue
        if name not in norm_map:
            continue
        orz = exp.get("orz_accuracy")
        if orz is None:
            continue

        norm = norm_map[name]
        n_val = parse_n(name)

        ax.scatter(norm, orz, c=SOURCE_COLORS[source], marker=SOURCE_MARKERS[source],
                   s=80, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)

        # Annotate with N value
        ax.annotate(f"N={n_val}", (norm, orz), fontsize=7,
                    xytext=(4, 4), textcoords='offset points', alpha=0.7)

    # Add within-source regression lines
    for source in SOURCE_COLORS:
        norms = []
        accs = []
        for exp in exp_log:
            name = exp.get("experiment_name", exp.get("name", ""))
            if not is_standard_config(name):
                continue
            if parse_source(name) != source:
                continue
            if name not in norm_map:
                continue
            orz = exp.get("orz_accuracy")
            if orz is None:
                continue
            norms.append(norm_map[name])
            accs.append(orz)

        if len(norms) >= 3:
            norms_arr = np.array(norms)
            accs_arr = np.array(accs)
            # Fit line
            z = np.polyfit(norms_arr, accs_arr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(norms_arr), max(norms_arr), 100)
            r = np.corrcoef(norms_arr, accs_arr)[0, 1]
            ax.plot(x_line, p(x_line), '--', color=SOURCE_COLORS[source], alpha=0.5,
                    label=f"{SOURCE_LABELS[source]} (r={r:.2f})")

    # Add legend
    handles = [plt.Line2D([0], [0], marker=SOURCE_MARKERS[s], color='w',
               markerfacecolor=SOURCE_COLORS[s], markersize=8,
               markeredgecolor='black', markeredgewidth=0.5, label=SOURCE_LABELS[s])
               for s in SOURCE_COLORS if any(parse_source(e.get("experiment_name", "")) == s for e in exp_log)]
    ax.legend(handles=handles, loc='upper right')

    ax.set_xlabel('LoRA Perturbation Norm ||dW||_F')
    ax.set_ylabel('ORZ Math Accuracy')
    ax.set_title('The Norm Paradox: Same Norm, Different Outcomes')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.2891, color='gray', linestyle=':', alpha=0.5, label='Baseline')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_norm_vs_orz.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig1_norm_vs_orz.png")


def fig2_cosine_heatmap(direction_data):
    """Fig 2: Pairwise cosine similarity heatmap."""
    if direction_data is None:
        print("  Skipping fig2 — no direction data")
        return

    matrix_data = direction_data.get("cosine_matrix", {})
    exp_order = matrix_data.get("experiment_order", [])
    matrix = matrix_data.get("matrix", {})

    if not exp_order:
        print("  Skipping fig2 — no cosine matrix data")
        return

    # Build numpy matrix
    n = len(exp_order)
    cos_mat = np.zeros((n, n))
    for i, n1 in enumerate(exp_order):
        for j, n2 in enumerate(exp_order):
            cos_mat[i, j] = matrix.get(n1, {}).get(n2, 0)

    # Create short labels
    short_labels = []
    for name in exp_order:
        source = parse_source(name)
        n_val = parse_n(name)
        abbrev = {"numinamath": "NM", "numinamath_hard": "NMH",
                  "numinamath_comp": "NMC", "openr1": "OR1"}
        short_labels.append(f"{abbrev.get(source, source)}-{n_val}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(cos_mat, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity')

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=7)

    # Add block borders for different sources
    sources_in_order = [parse_source(name) for name in exp_order]
    prev_source = sources_in_order[0]
    for i in range(1, n):
        if sources_in_order[i] != prev_source:
            ax.axhline(y=i - 0.5, color='black', linewidth=2)
            ax.axvline(x=i - 0.5, color='black', linewidth=2)
            prev_source = sources_in_order[i]

    # Add values in cells (for smaller matrices)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                color = 'white' if cos_mat[i, j] > 0.7 or cos_mat[i, j] < 0.3 else 'black'
                ax.text(j, i, f'{cos_mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=5, color=color)

    ax.set_title('Pairwise Cosine Similarity Between LoRA Weight Updates')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_cosine_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig2_cosine_heatmap.png")


def fig3_source_dependent_regression(direction_data, exp_log):
    """Fig 3: Source-dependent regression — norm predicts accuracy within but not across sources."""
    if direction_data is None:
        print("  Skipping fig3 — no direction data")
        return

    eff_perturbs = direction_data.get("effective_perturbations", [])
    if not eff_perturbs:
        print("  Skipping fig3 — no effective perturbation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Raw norm with per-source regression lines
    ax = axes[0]
    source_data = defaultdict(lambda: {"norms": [], "accs": [], "ns": []})
    for entry in eff_perturbs:
        if entry["orz_accuracy"] is None:
            continue
        source = entry["data_source"]
        if source not in SOURCE_COLORS:
            continue
        source_data[source]["norms"].append(entry["frob_norm"])
        source_data[source]["accs"].append(entry["orz_accuracy"])
        source_data[source]["ns"].append(entry["num_samples"])
        ax.scatter(entry["frob_norm"], entry["orz_accuracy"],
                   c=SOURCE_COLORS[source], marker=SOURCE_MARKERS[source],
                   s=80, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)

    # Per-source regression lines
    for source, data in source_data.items():
        norms = np.array(data["norms"])
        accs = np.array(data["accs"])
        if len(norms) >= 3:
            z = np.polyfit(norms, accs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(norms) * 0.9, max(norms) * 1.1, 100)
            r = np.corrcoef(norms, accs)[0, 1]
            ax.plot(x_line, p(x_line), '--', color=SOURCE_COLORS[source], alpha=0.6, linewidth=2,
                    label=f"{SOURCE_LABELS[source]} (r={r:.2f})")

    ax.set_xlabel('LoRA Perturbation Norm ||dW||$_F$')
    ax.set_ylabel('ORZ Math Accuracy')
    ax.set_title('(a) Within-Source: Norm Predicts Accuracy\n(but slopes differ across sources)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    r2_data = direction_data.get("r2_analysis", {})
    raw_r2 = r2_data.get("across_all_raw_norm", {}).get("r2", "N/A")
    if isinstance(raw_r2, float):
        ax.text(0.05, 0.05, f"Overall R$^2$ = {raw_r2:.3f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right panel: Degradation efficiency (damage per unit norm) vs N
    ax = axes[1]
    baseline_acc = 0.2891  # Baseline ORZ accuracy

    for source, data in source_data.items():
        norms = np.array(data["norms"])
        accs = np.array(data["accs"])
        ns = np.array(data["ns"])
        # Compute degradation efficiency = (baseline - acc) / norm
        efficiency = (baseline_acc - accs) / np.maximum(norms, 0.01)

        # Sort by N for line plot
        idx = np.argsort(ns)
        ax.plot(ns[idx], efficiency[idx], '-', color=SOURCE_COLORS[source],
                marker=SOURCE_MARKERS[source], markersize=7, alpha=0.8,
                label=SOURCE_LABELS[source])

    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('Degradation Efficiency\n(accuracy loss / ||dW||$_F$)')
    ax.set_title('(b) Per-Unit-Norm Damage\n(OpenR1 direction is more harmful)')
    ax.set_xscale('log')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_source_regression.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3_source_regression.png")


def fig4_layer_profiles(direction_data):
    """Fig 4: Per-layer norm profiles comparing NuminaMath vs OpenR1."""
    if direction_data is None:
        print("  Skipping fig4 — no direction data")
        return

    profiles = direction_data.get("layer_norm_profiles", {})
    if not profiles:
        print("  Skipping fig4 — no layer norm profiles")
        return

    # Select representative experiments
    comparisons = [
        ("sft_numinamath_n2000_r8_lr5e-5_ep1", "sft_openr1_n2000_r8_lr5e-5_ep1"),
        ("sft_numinamath_n10000_r8_lr5e-5_ep1", "sft_openr1_n10000_r8_lr5e-5_ep1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (nm_name, or_name) in enumerate(comparisons):
        ax = axes[idx]
        nm_profile = profiles.get(nm_name, {})
        or_profile = profiles.get(or_name, {})

        if not nm_profile or not or_profile:
            ax.text(0.5, 0.5, "Data not available", transform=ax.transAxes, ha='center')
            continue

        # Group by layer number, sum norms across module types
        def aggregate_by_layer(profile):
            layer_norms = defaultdict(float)
            for key, norm in profile.items():
                m = re.match(r'L(\d+)_', key)
                if m:
                    layer_norms[int(m.group(1))] += norm ** 2
            return {k: v ** 0.5 for k, v in sorted(layer_norms.items())}

        nm_layers = aggregate_by_layer(nm_profile)
        or_layers = aggregate_by_layer(or_profile)

        layers = sorted(set(nm_layers.keys()) | set(or_layers.keys()))
        nm_vals = [nm_layers.get(l, 0) for l in layers]
        or_vals = [or_layers.get(l, 0) for l in layers]

        ax.bar(np.array(layers) - 0.2, nm_vals, 0.4, label='NuminaMath',
               color=SOURCE_COLORS["numinamath"], alpha=0.7)
        ax.bar(np.array(layers) + 0.2, or_vals, 0.4, label='OpenR1',
               color=SOURCE_COLORS["openr1"], alpha=0.7)

        n_val = parse_n(nm_name)
        ax.set_xlabel('Transformer Layer')
        ax.set_ylabel('||dW||_F (per layer)')
        ax.set_title(f'Layer-wise Norm Profile (N={n_val})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_layer_profiles.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_layer_profiles.png")


def fig5_format_decomposition(exp_log):
    """Fig 5: Format decomposition — strict vs tolerant GSM8K accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Load all GSM8K results
    gsm8k_results = {}
    for f in os.listdir(RESULTS_DIR):
        if f.endswith("_gsm8k.json"):
            path = os.path.join(RESULTS_DIR, f)
            with open(path) as fp:
                data = json.load(fp)
            exp_name = data.get("experiment_name", "")
            gsm8k_results[exp_name] = data

    # Panel (a): NuminaMath strict vs tolerant
    ax = axes[0]
    for source, label in [("numinamath", "NuminaMath"), ("numinamath_hard", "NuminaMath-Hard")]:
        strict_data = []
        tolerant_data = []

        for exp in exp_log:
            name = exp.get("experiment_name", exp.get("name", ""))
            if not is_standard_config(name):
                continue
            if parse_source(name) != source:
                continue
            n_val = parse_n(name)

            # Find strict and tolerant GSM8K
            strict_key = f"{name}"  # strict results stored without suffix
            tolerant_key = f"{name}_tolerant"

            if strict_key in gsm8k_results:
                strict_data.append((n_val, gsm8k_results[strict_key]["gsm8k"]["accuracy"]))
            if tolerant_key in gsm8k_results:
                tolerant_data.append((n_val, gsm8k_results[tolerant_key]["gsm8k"]["accuracy"]))

        if strict_data:
            strict_data.sort()
            ax.plot([d[0] for d in strict_data], [d[1] for d in strict_data],
                    '-o', label=f'{label} (strict)', color=SOURCE_COLORS[source], alpha=0.8)
        if tolerant_data:
            tolerant_data.sort()
            ax.plot([d[0] for d in tolerant_data], [d[1] for d in tolerant_data],
                    '--s', label=f'{label} (tolerant)', color=SOURCE_COLORS[source], alpha=0.5)

    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('GSM8K Accuracy')
    ax.set_title('(a) NuminaMath: Format Decomposition')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): OpenR1 strict vs tolerant — load directly from result files
    ax = axes[1]
    strict_data = []
    tolerant_data = []

    for n_val in [100, 500, 1000, 1250, 1500, 2000, 5000, 10000]:
        name = f"sft_openr1_n{n_val}_r8_lr5e-5_ep1"

        # Strict: stored as {name}_strict_gsm8k.json
        strict_path = os.path.join(RESULTS_DIR, f"{name}_strict_gsm8k.json")
        if os.path.exists(strict_path):
            with open(strict_path) as fp:
                data = json.load(fp)
            gsm = data["gsm8k"]
            strict_data.append((n_val, gsm["accuracy"], gsm["boxed_found"] / gsm["total"]))

        # Tolerant: stored as {name}_tolerant_gsm8k.json or {name}_gsm8k.json (older format)
        for tol_path in [os.path.join(RESULTS_DIR, f"{name}_tolerant_gsm8k.json"),
                         os.path.join(RESULTS_DIR, f"{name}_gsm8k.json")]:
            if os.path.exists(tol_path):
                with open(tol_path) as fp:
                    data = json.load(fp)
                if data.get("tolerant", False):
                    tolerant_data.append((n_val, data["gsm8k"]["accuracy"]))
                    break

    if strict_data:
        strict_data.sort()
        ns = [d[0] for d in strict_data]
        accs = [d[1] for d in strict_data]
        boxed = [d[2] for d in strict_data]
        ax.plot(ns, accs, '-o', label='OpenR1 (strict)', color=SOURCE_COLORS["openr1"])
        ax.plot(ns, boxed, '--^', label='OpenR1 (boxed rate)', color=SOURCE_COLORS["openr1"], alpha=0.5)

    if tolerant_data:
        tolerant_data.sort()
        ax.plot([d[0] for d in tolerant_data], [d[1] for d in tolerant_data],
                '-s', label='OpenR1 (tolerant)', color='#FF8A80')

    # Add cliff annotation
    ax.axvspan(1500, 2000, alpha=0.1, color='red', label='Cliff zone')

    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('GSM8K Accuracy / Boxed Rate')
    ax.set_title('(b) OpenR1: Format Decomposition')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_format_decomposition.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_format_decomposition.png")


def fig6_openr1_cliff(exp_log):
    """Fig 6: OpenR1 cliff characterization — ORZ accuracy, GSM8K, output analysis vs N."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Get OpenR1 standard experiments from exp_log + check for finegrained
    openr1_exps = []
    seen_ns = set()
    for exp in exp_log:
        name = exp.get("experiment_name", exp.get("name", ""))
        if parse_source(name) != "openr1" or not is_standard_config(name):
            continue
        n_val = parse_n(name)
        if n_val not in seen_ns:
            openr1_exps.append((n_val, exp))
            seen_ns.add(n_val)

    # Also check for fine-grained eval results
    for n_val in [1250, 1500]:
        if n_val in seen_ns:
            continue
        eval_path = os.path.join(RESULTS_DIR, f"sft_openr1_n{n_val}_r8_lr5e-5_ep1_eval.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                data = json.load(f)
            openr1_exps.append((n_val, {"orz_accuracy": data["orz"]["accuracy"],
                                         "experiment_name": f"sft_openr1_n{n_val}_r8_lr5e-5_ep1"}))
            seen_ns.add(n_val)

    openr1_exps.sort()

    # Panel (a): ORZ accuracy vs N
    ax = axes[0]
    ns = [e[0] for e in openr1_exps]
    orz_accs = [e[1].get("orz_accuracy", 0) for e in openr1_exps]
    ax.plot(ns, orz_accs, '-o', color=SOURCE_COLORS["openr1"], linewidth=2, markersize=8)
    ax.axhline(y=0.2891, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('ORZ Math Accuracy')
    ax.set_title('(a) ORZ Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Also plot NuminaMath for comparison
    nm_exps = []
    for exp in exp_log:
        name = exp.get("experiment_name", exp.get("name", ""))
        if parse_source(name) != "numinamath" or not is_standard_config(name):
            continue
        nm_exps.append((parse_n(name), exp))
    nm_exps.sort()
    if nm_exps:
        nm_ns = [e[0] for e in nm_exps]
        nm_accs = [e[1].get("orz_accuracy", 0) for e in nm_exps]
        ax.plot(nm_ns, nm_accs, '-s', color=SOURCE_COLORS["numinamath"], linewidth=1.5,
                markersize=6, alpha=0.6, label='NuminaMath')
        ax.legend()

    # Panel (b): GSM8K results — strict + tolerant
    ax = axes[1]
    gsm8k_strict = []
    gsm8k_tolerant = []

    for n_val in [100, 500, 1000, 1250, 1500, 2000, 5000, 10000]:
        name = f"sft_openr1_n{n_val}_r8_lr5e-5_ep1"

        # Strict
        strict_path = os.path.join(RESULTS_DIR, f"{name}_strict_gsm8k.json")
        if os.path.exists(strict_path):
            with open(strict_path) as f:
                data = json.load(f)
            gsm8k_strict.append((n_val, data["gsm8k"]["accuracy"],
                                 data["gsm8k"]["boxed_found"] / data["gsm8k"]["total"]))

        # Tolerant
        for tol_path in [os.path.join(RESULTS_DIR, f"{name}_tolerant_gsm8k.json"),
                         os.path.join(RESULTS_DIR, f"{name}_gsm8k.json")]:
            if os.path.exists(tol_path):
                with open(tol_path) as f:
                    data = json.load(f)
                if data.get("tolerant", False):
                    gsm8k_tolerant.append((n_val, data["gsm8k"]["accuracy"]))
                    break

    if gsm8k_strict:
        gsm8k_strict.sort()
        ns_s = [d[0] for d in gsm8k_strict]
        accs_s = [d[1] for d in gsm8k_strict]
        boxed_s = [d[2] for d in gsm8k_strict]
        ax.plot(ns_s, accs_s, '-o', color=SOURCE_COLORS["openr1"], label='Strict accuracy')
        ax.plot(ns_s, boxed_s, '--^', color='#FF8A80', label='Boxed rate', alpha=0.7)

    if gsm8k_tolerant:
        gsm8k_tolerant.sort()
        ax.plot([d[0] for d in gsm8k_tolerant], [d[1] for d in gsm8k_tolerant],
                '-s', color='#E57373', label='Tolerant accuracy', alpha=0.7)

    ax.axhline(y=0.84, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.axvspan(1500, 2000, alpha=0.1, color='red')
    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('Accuracy / Rate')
    ax.set_title('(b) GSM8K Performance')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel (c): Output analysis (response length, boxed rate on ORZ)
    ax = axes[2]
    output_data = []
    for n_val, _ in openr1_exps:
        name = f"sft_openr1_n{n_val}_r8_lr5e-5_ep1"
        out_path = os.path.join(RESULTS_DIR, f"{name}_output_analysis.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            output_data.append((n_val, data.get("avg_response_length", 0),
                                data.get("boxed_rate", 0)))

    if output_data:
        output_data.sort()
        ns_o = [d[0] for d in output_data]
        lengths = [d[1] for d in output_data]
        boxed_rates = [d[2] for d in output_data]

        ax2 = ax.twinx()
        l1 = ax.plot(ns_o, lengths, '-o', color='#FF6F00', label='Avg response length')
        l2 = ax2.plot(ns_o, boxed_rates, '--s', color='#7B1FA2', label='Boxed rate (ORZ)')
        ax.set_ylabel('Avg Response Length (chars)', color='#FF6F00')
        ax2.set_ylabel('Boxed Rate', color='#7B1FA2')

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
    else:
        ax.text(0.5, 0.5, "Output analysis data\nnot yet available",
                transform=ax.transAxes, ha='center', fontsize=12)

    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_title('(c) Output Characteristics')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_openr1_cliff.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig6_openr1_cliff.png")


def fig7_cosine_divergence(direction_data):
    """Fig 7: Cosine similarity between sources DECREASES with N."""
    if direction_data is None:
        print("  Skipping fig7 — no direction data")
        return

    matching_n = direction_data.get("matching_n_cosines", {})
    if not matching_n:
        print("  Skipping fig7 — no matching N cosine data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Parse data
    pairs_data = defaultdict(lambda: {"ns": [], "cosines": []})
    for key, cos_val in matching_n.items():
        # Parse "N=100: numinamath_hard vs openr1"
        m = re.match(r'N=(\d+): (.+) vs (.+)', key)
        if m:
            n_val = int(m.group(1))
            s1 = m.group(2).strip()
            s2 = m.group(3).strip()
            pair_key = f"{s1} vs {s2}"
            pairs_data[pair_key]["ns"].append(n_val)
            pairs_data[pair_key]["cosines"].append(min(cos_val, 1.0))  # Clamp

    pair_colors = {
        "numinamath vs openr1": SOURCE_COLORS["openr1"],
        "numinamath_hard vs openr1": "#FF8A80",
        "numinamath_hard vs numinamath": SOURCE_COLORS["numinamath_hard"],
    }
    pair_labels = {
        "numinamath vs openr1": "NuminaMath vs OpenR1",
        "numinamath_hard vs openr1": "NuminaMath-Hard vs OpenR1",
        "numinamath_hard vs numinamath": "NuminaMath vs NuminaMath-Hard",
    }

    for pair, data in pairs_data.items():
        ns = np.array(data["ns"])
        cosines = np.array(data["cosines"])
        idx = np.argsort(ns)
        color = pair_colors.get(pair, 'gray')
        label = pair_labels.get(pair, pair)
        ax.plot(ns[idx], cosines[idx], '-o', color=color, label=label,
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of SFT Samples (N)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Perturbation Directions Diverge with Increasing N')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_cosine_divergence.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig7_cosine_divergence.png")


def fig_source_similarity_summary(direction_data):
    """Bonus: Summary bar chart of average cosine similarity by source pair."""
    if direction_data is None:
        return

    summary = direction_data.get("source_pair_summary", {})
    if not summary:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    pairs = sorted(summary.keys())
    means = [summary[p]["mean"] for p in pairs]
    stds = [summary[p]["std"] for p in pairs]

    # Color based on intra vs inter
    colors = []
    for p in pairs:
        parts = p.split(" vs ")
        if len(parts) == 2 and parts[0] == parts[1]:
            colors.append('#4CAF50')  # intra-source: green
        elif "openr1" in p and ("numinamath" in p.split(" vs ")[0] or "numinamath" in p.split(" vs ")[1]):
            colors.append('#F44336')  # cross-paradigm: red
        else:
            colors.append('#FFC107')  # related sources: yellow

    bars = ax.bar(range(len(pairs)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("('", "").replace("')", "").replace("', '", " vs ") for p in pairs],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('Cosine Similarity Between Data Source Directions')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_source_similarity.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_source_similarity.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    exp_log, weight_data, direction_data = load_experiment_data()
    norm_map = get_norm_map(weight_data)

    print(f"Loaded {len(exp_log)} experiments, {len(norm_map)} with norms")
    if direction_data:
        print(f"Direction data: {direction_data.get('num_experiments_loaded', 0)} experiments")

    print("\n--- Generating figures ---")
    fig1_norm_vs_orz(exp_log, norm_map)
    fig2_cosine_heatmap(direction_data)
    fig3_source_dependent_regression(direction_data, exp_log)
    fig4_layer_profiles(direction_data)
    fig5_format_decomposition(exp_log)
    fig6_openr1_cliff(exp_log)
    fig7_cosine_divergence(direction_data)
    fig_source_similarity_summary(direction_data)

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
