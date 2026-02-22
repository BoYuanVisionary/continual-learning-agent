#!/usr/bin/env python3
"""Generate a NeurIPS-style two-column research paper PDF."""

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from fpdf import FPDF


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
PAGE_W = 210        # A4 width mm
PAGE_H = 297        # A4 height mm
MARGIN_L = 18
MARGIN_R = 18
MARGIN_T = 20
MARGIN_B = 22
COL_GAP = 6
BODY_W = PAGE_W - MARGIN_L - MARGIN_R
COL_W = (BODY_W - COL_GAP) / 2  # ~84 mm each
FONT_BODY = 9
FONT_SMALL = 8
LINE_H = 4.0        # body line height
TABLE_LINE_H = 3.8

BASELINES = {
    "orz": 0.2891,
    "sci": 0.3434,
    "ta_sim_func": 0.7889,
    "ta_real_func": 0.8922,
}


def sanitize(text):
    reps = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
        "\u2248": "~=", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x",
        "\u0394": "Delta", "\u03b1": "alpha", "\u03b2": "beta",
        "\u2022": "-", "\u00b1": "+/-",
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ===========================================================================
# Figure generation
# ===========================================================================
def make_figures(experiments):
    os.makedirs("results/figures", exist_ok=True)

    # ---- helpers ----
    def best_valid_per_n(exps, src=None):
        b = {}
        for e in exps:
            if not e.get("valid"):
                continue
            if src and e["data_source"] != src:
                continue
            n = e["num_samples"]
            if n not in b or e["orz_accuracy"] > b[n]["orz_accuracy"]:
                b[n] = e
        return dict(sorted(b.items()))

    def fixed_hp(exps, src, r=8, lr="5e-5", ep=1):
        out = {}
        for e in exps:
            if (e["data_source"] == src and e["lora_rank"] == r
                    and e["lr"] == lr and e["epochs"] == ep):
                out[e["num_samples"]] = e
        return dict(sorted(out.items()))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "lines.markersize": 5, "figure.dpi": 200,
    })

    # ================================================================
    # Fig 1: ORZ accuracy vs N, multiple data sources (fixed HP)
    # ================================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for src, marker, label in [
        ("numinamath", "o", "NuminaMath"),
        ("numinamath_hard", "s", "NuminaMath-Hard"),
        ("numinamath_comp", "^", "NuminaMath-Comp"),
        ("openr1", "D", "OpenR1"),
    ]:
        if src in ("numinamath", "numinamath_hard", "openr1"):
            data = fixed_hp(experiments, src, r=8, lr="5e-5", ep=1)
        else:
            data = fixed_hp(experiments, src, r=16, lr="1e-4", ep=2)
        if not data:
            continue
        ns = sorted(data.keys())
        accs = [data[n]["orz_accuracy"] * 100 for n in ns]
        ax.plot(ns, accs, marker=marker, label=label)

    ax.axhline(y=28.91, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("SFT Sample Count (N)")
    ax.set_ylabel("ORZ Math Accuracy (%)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("(a) ORZ Accuracy vs. Sample Count", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/fig1_orz_vs_n.png", bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # Fig 2: SciKnowEval vs N
    # ================================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for src, marker, label in [
        ("numinamath", "o", "NuminaMath"),
        ("numinamath_hard", "s", "NuminaMath-Hard"),
        ("openr1", "D", "OpenR1"),
    ]:
        data = fixed_hp(experiments, src, r=8, lr="5e-5", ep=1)
        if not data:
            continue
        ns = sorted(data.keys())
        accs = [data[n]["sciknoweval_accuracy"] * 100 for n in ns]
        ax.plot(ns, accs, marker=marker, label=label)

    ax.axhline(y=34.34, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.axhline(y=34.34 - 3.0, color="red", ls=":", lw=0.8, label="3% Threshold")
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 1000, 2000, 5000, 10000])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("SFT Sample Count (N)")
    ax.set_ylabel("SciKnowEval Accuracy (%)")
    ax.legend(fontsize=6.5)
    ax.set_title("(b) Cross-Domain: Chemistry MCQ", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/fig2_sci_vs_n.png", bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # Fig 3: ToolAlpaca vs N
    # ================================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for src, marker, label in [
        ("numinamath", "o", "NuminaMath"),
        ("numinamath_hard", "s", "NuminaMath-Hard"),
        ("openr1", "D", "OpenR1"),
    ]:
        data = fixed_hp(experiments, src, r=8, lr="5e-5", ep=1)
        if not data:
            continue
        ns = sorted(data.keys())
        accs = [data[n]["toolalpaca_real_func_acc"] * 100 for n in ns]
        ax.plot(ns, accs, marker=marker, label=label)

    ax.axhline(y=89.22, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.axhline(y=89.22 - 3.0, color="red", ls=":", lw=0.8, label="3% Threshold")
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 1000, 2000, 5000, 10000])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("SFT Sample Count (N)")
    ax.set_ylabel("ToolAlpaca Real Func Acc (%)")
    ax.legend(fontsize=6.5)
    ax.set_title("(c) Cross-Domain: Tool-Use", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/fig3_tool_vs_n.png", bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # Fig 4: HP sensitivity at N=100
    # ================================================================
    n100 = [e for e in experiments
            if e["num_samples"] == 100 and e["data_source"] == "numinamath"]
    if n100:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        labels = []
        accs = []
        for e in sorted(n100, key=lambda x: x["orz_accuracy"], reverse=True):
            lbl = f"r={e['lora_rank']},lr={e['lr']},ep={e['epochs']}"
            labels.append(lbl)
            accs.append(e["orz_accuracy"] * 100)
        colors = ["#2ecc71" if a > 28.91 else "#e74c3c" for a in accs]
        bars = ax.barh(range(len(labels)), accs, color=colors, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.axvline(x=28.91, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("ORZ Accuracy (%)")
        ax.set_title("(d) HP Sensitivity at N=100", fontsize=9)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig("results/figures/fig4_hp_n100.png", bbox_inches="tight")
        plt.close(fig)

    print("Figures generated in results/figures/")


# ===========================================================================
# Two-column PDF class
# ===========================================================================
class NeurIPSPaper(FPDF):
    """Two-column PDF mimicking NeurIPS style."""

    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(MARGIN_L, MARGIN_T, MARGIN_R)
        self.set_auto_page_break(auto=False)
        self._col = 0          # 0=left, 1=right
        self._col_y_start = 0  # y where current column region starts
        self._full_width_mode = False
        self._footnotes = []

    # ---- column helpers ----
    def _col_left_x(self):
        return MARGIN_L if self._col == 0 else MARGIN_L + COL_W + COL_GAP

    def _set_col_margins(self):
        if self._full_width_mode:
            self.set_left_margin(MARGIN_L)
            self.set_right_margin(MARGIN_R)
        elif self._col == 0:
            self.set_left_margin(MARGIN_L)
            self.set_right_margin(PAGE_W - MARGIN_L - COL_W)
        else:
            self.set_left_margin(MARGIN_L + COL_W + COL_GAP)
            self.set_right_margin(MARGIN_R)
        self.set_x(self.l_margin)

    def _remaining_in_col(self):
        return PAGE_H - MARGIN_B - self.get_y()

    def _next_col_or_page(self):
        if self._col == 0:
            self._col = 1
            self._set_col_margins()
            self.set_y(self._col_y_start)
        else:
            self.add_page()
            self._col = 0
            self._col_y_start = self.get_y()
            self._set_col_margins()

    def _ensure_space(self, h):
        if self._remaining_in_col() < h:
            self._next_col_or_page()

    # ---- high-level writing ----
    def start_two_col(self):
        self._full_width_mode = False
        self._col = 0
        self._col_y_start = self.get_y()
        self._set_col_margins()

    def start_full_width(self):
        self._full_width_mode = True
        self._col = 0
        self.set_left_margin(MARGIN_L)
        self.set_right_margin(MARGIN_R)
        self.set_x(MARGIN_L)

    # ---- overrides ----
    def header(self):
        pass  # no running header on first page

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 8)
        self.cell(0, 10, str(self.page_no()), align="C")

    # ---- content methods ----
    def write_title(self, title, authors, affiliation):
        self.set_font("Times", "B", 15)
        self.multi_cell(0, 7, sanitize(title), align="C")
        self.ln(5)
        self.set_font("Times", "", 11)
        self.cell(0, 5, sanitize(authors), align="C")
        self.ln(5)
        self.set_font("Times", "I", 10)
        self.cell(0, 5, sanitize(affiliation), align="C")
        self.ln(8)

    def write_abstract(self, text):
        self.set_font("Times", "B", 10)
        self.cell(0, 5, "Abstract", align="C")
        self.ln(5)
        x0 = MARGIN_L + 10
        w0 = BODY_W - 20
        self.set_font("Times", "I", 9)
        self.set_x(x0)
        self.set_left_margin(x0)
        self.set_right_margin(PAGE_W - x0 - w0)
        self.multi_cell(w0, 4, sanitize(text), align="J")
        self.set_left_margin(MARGIN_L)
        self.set_right_margin(MARGIN_R)
        self.ln(4)

    def section(self, num, title):
        self._ensure_space(12)
        self.ln(2)
        self.set_font("Times", "B", 11)
        if num:
            self.cell(0, 5, sanitize(f"{num}  {title}"))
        else:
            self.cell(0, 5, sanitize(title))
        self.ln(6)

    def subsection(self, num, title):
        self._ensure_space(10)
        self.ln(1)
        self.set_font("Times", "B", 9.5)
        self.cell(0, 5, sanitize(f"{num}  {title}"))
        self.ln(5.5)

    def para(self, text, bold=False, italic=False):
        self._ensure_space(LINE_H * 2)
        style = ""
        if bold:
            style = "B"
        if italic:
            style = "I"
        self.set_font("Times", style, FONT_BODY)
        self.multi_cell(0, LINE_H, sanitize(text), align="J")
        self.ln(1.5)

    def bullet(self, text, indent=3):
        self._ensure_space(LINE_H * 2)
        self.set_font("Times", "", FONT_BODY)
        x0 = self.get_x()
        self.set_x(x0 + indent)
        old_l = self.l_margin
        self.set_left_margin(old_l + indent)
        self.multi_cell(0, LINE_H, sanitize("- " + text), align="J")
        self.set_left_margin(old_l)
        self.ln(0.5)

    def table(self, caption, headers, rows, col_widths=None, font_size=7.5):
        """Write a table that fits within the current column."""
        cw = COL_W if not self._full_width_mode else BODY_W
        if col_widths is None:
            col_widths = [cw / len(headers)] * len(headers)
        else:
            # Scale col_widths to fit
            total = sum(col_widths)
            col_widths = [w / total * cw for w in col_widths]

        row_h = TABLE_LINE_H
        needed = 6 + row_h * (len(rows) + 1) + 8
        self._ensure_space(needed)

        # Caption
        self.set_font("Times", "B", 8)
        self.multi_cell(0, 3.5, sanitize(caption), align="C")
        self.ln(1.5)

        x0 = self.get_x()

        # Header
        self.set_font("Times", "B", font_size)
        self.set_fill_color(235, 235, 235)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h + 0.5, sanitize(h), border="TB",
                      fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Times", "", font_size)
        for ri, row in enumerate(rows):
            self.set_x(x0)
            border = "B" if ri == len(rows) - 1 else ""
            for i, val in enumerate(row):
                self.cell(col_widths[i], row_h, sanitize(str(val)),
                          border=border, align="C")
            self.ln()
        self.ln(3)

    def insert_image(self, path, w=None, caption=None):
        if w is None:
            w = COL_W if not self._full_width_mode else BODY_W
        # Estimate height (assume 0.7 aspect ratio)
        h_est = w * 0.72
        self._ensure_space(h_est + 12)
        x0 = self.get_x()
        self.image(path, x=x0, w=w)
        self.ln(1)
        if caption:
            self.set_font("Times", "I", 7.5)
            self.multi_cell(0 if self._full_width_mode else COL_W,
                            3.5, sanitize(caption), align="C")
            self.ln(3)


# ===========================================================================
# Paper content
# ===========================================================================
def generate_paper():
    with open("results/experiment_log.json") as f:
        experiments = json.load(f)

    # Generate figures
    make_figures(experiments)

    # Compute helpers
    def best_valid(exps, src=None):
        b = {}
        for e in exps:
            if not e.get("valid"):
                continue
            if src and e["data_source"] != src:
                continue
            n = e["num_samples"]
            if n not in b or e["orz_accuracy"] > b[n]["orz_accuracy"]:
                b[n] = e
        return dict(sorted(b.items()))

    best = best_valid(experiments)
    n_valid = sum(1 for e in experiments if e.get("valid"))
    n_invalid = len(experiments) - n_valid

    pdf = NeurIPSPaper()
    pdf.add_page()

    # =====================================================================
    # TITLE (full width)
    # =====================================================================
    pdf.start_full_width()
    pdf.write_title(
        "When Less is More for Fine-Tuning: Distribution Mismatch\n"
        "Inverts the Data Scaling Curve in LoRA SFT",
        "Anonymous Authors",
        "Georgia Institute of Technology"
    )

    pdf.write_abstract(
        "We present a systematic empirical study of how supervised fine-tuning "
        "(SFT) data volume interacts with distribution mismatch and cross-domain "
        "forgetting. Fine-tuning Qwen2.5-3B-Instruct on math chain-of-thought "
        "data across 43 experiments (4 data sources, 5 hyperparameter "
        "configurations, N=50 to 10,000 samples), we observe that ORZ math "
        "accuracy peaks at just N=100 (+1.75%) and monotonically declines with "
        "more data--a direct consequence of distribution mismatch between "
        "available SFT corpora and the evaluation benchmark. Meanwhile, "
        "cross-domain capabilities (chemistry MCQ, tool-use) remain remarkably "
        "robust, rarely exceeding 3% degradation even under aggressive training. "
        "Rather than proposing a new method, we provide a detailed empirical "
        "characterization of the SFT data scaling landscape, demonstrating that "
        "(1) distribution match dominates data quantity as the primary factor in "
        "SFT success, (2) LoRA effectively compartmentalizes cross-domain "
        "knowledge even when in-domain performance degrades, and (3) practitioners "
        "should prefer micro-tuning (<200 samples, 1 epoch) over large-scale SFT "
        "when training data distribution does not closely match evaluation. "
        "We release all 43 experiment configurations and results to support "
        "reproducibility."
    )

    # divider
    y = pdf.get_y()
    pdf.line(MARGIN_L, y, PAGE_W - MARGIN_R, y)
    pdf.ln(3)

    # =====================================================================
    # BEGIN TWO-COLUMN
    # =====================================================================
    pdf.start_two_col()

    # ----------- 1. INTRODUCTION -----------
    pdf.section("1", "Introduction")

    pdf.para(
        "Supervised fine-tuning (SFT) is the dominant paradigm for specializing "
        "large language models (LLMs) to downstream tasks. A natural assumption "
        "is that more high-quality training data yields better performance. "
        "While this holds during pre-training, the dynamics of SFT on an "
        "already-capable instruction-tuned model are less understood."
    )
    pdf.para(
        "We study this question through a continual learning lens: we fine-tune "
        "Qwen2.5-3B-Instruct on math chain-of-thought (CoT) data using LoRA "
        "and measure both (a) math reasoning accuracy on the ORZ benchmark, "
        "and (b) catastrophic forgetting on two unrelated domains--chemistry "
        "MCQ (SciKnowEval) and tool-use (ToolAlpaca)."
    )
    pdf.para(
        "Our 43-experiment campaign yields three main findings:"
    )
    pdf.para(
        "(1) When SFT data distribution does not match the evaluation "
        "distribution, more data actively degrades performance. This is not "
        "a subtle effect: ORZ accuracy drops from 30.7% (N=100) to 20.0% "
        "(N=10,000)--a 10 percentage point decline."
    )
    pdf.para(
        "(2) Cross-domain forgetting is far more robust than in-domain accuracy. "
        "SciKnowEval and ToolAlpaca scores remain within 3% of baseline across "
        "39 of 43 experiments, even when ORZ accuracy has collapsed."
    )
    pdf.para(
        "(3) The interaction between data source, sample count, and "
        "hyperparameters creates a complex landscape where data quality and "
        "distribution match dominate data quantity at every scale."
    )
    pdf.para(
        "We frame these results as a practical empirical study for practitioners "
        "rather than a claim of methodological novelty. The inverted scaling "
        "curve is predictable given distribution mismatch--but the magnitude "
        "of the effect, the robustness of cross-domain knowledge, and the "
        "detailed characterization across 43 configurations provide actionable "
        "guidelines for real-world SFT practice."
    )

    # ----------- 2. RELATED WORK -----------
    pdf.section("2", "Related Work")

    pdf.subsection("2.1", "Data Scaling in SFT")
    pdf.para(
        "Zhou et al. (2023) demonstrated in LIMA that as few as 1,000 carefully "
        "curated examples can produce strong alignment, challenging the "
        "assumption that SFT requires large datasets. Our work extends this "
        "finding by mapping the full sample-count curve and showing that "
        "performance can actively decrease with more data when distributions "
        "are mismatched."
    )
    pdf.para(
        "Recent data selection work (Chen et al., 2023; Liu et al., 2024) "
        "has shown that quality filtering can improve SFT outcomes. Our "
        "comparison of four data sources (random, competition-filtered, "
        "hard-filtered, R1-distilled) provides complementary evidence that "
        "distribution match matters more than difficulty or solution quality."
    )

    pdf.subsection("2.2", "Catastrophic Forgetting")
    pdf.para(
        "Catastrophic forgetting in LLMs during continual fine-tuning has "
        "been studied by Luo et al. (2023) and others. Methods like EWC "
        "(Kirkpatrick et al., 2017), experience replay, and LoRA (Hu et al., "
        "2022) aim to mitigate forgetting. Our results show that LoRA "
        "provides strong cross-domain protection but does not prevent "
        "in-domain degradation from distribution shift--a distinction "
        "not well characterized in prior work."
    )

    pdf.subsection("2.3", "Math Reasoning via SFT")
    pdf.para(
        "NuminaMath-CoT (Li et al., 2024) and OpenR1-Math provide large-scale "
        "CoT annotations for math training. Most work assumes more CoT data "
        "improves math reasoning. We show this assumption fails when the base "
        "model already has math capability and the SFT distribution diverges "
        "from evaluation."
    )

    # ----------- 3. EXPERIMENTAL SETUP -----------
    pdf.section("3", "Experimental Setup")

    pdf.subsection("3.1", "Model and Training")
    pdf.para(
        "Base model: Qwen2.5-3B-Instruct (3B parameters, bfloat16, SDPA "
        "attention). Fine-tuning: LoRA applied to all linear layers via TRL "
        "SFTTrainer. Effective batch size 16 (bs=2, grad_accum=8), cosine "
        "LR schedule with 10% warmup, gradient checkpointing."
    )

    pdf.subsection("3.2", "Data Sources")
    pdf.para(
        "We test four SFT data configurations:"
    )
    pdf.bullet(
        "NuminaMath (random): Random subset from 860K NuminaMath-CoT examples."
    )
    pdf.bullet(
        "NuminaMath-Comp: Filtered to competition math (olympiads, AMC/AIME, AOPS)."
    )
    pdf.bullet(
        "NuminaMath-Hard: Competition sources sorted by solution length, "
        "selecting the most detailed solutions."
    )
    pdf.bullet(
        "OpenR1-Math: DeepSeek R1 distillation traces with long-form reasoning."
    )

    pdf.subsection("3.3", "Hyperparameter Grid")
    pdf.para(
        "We explore five configurations spanning conservative to aggressive "
        "training. See Table 1."
    )

    pdf.table(
        "Table 1: Hyperparameter configurations",
        ["Config", "LR", "Ep", "Rank"],
        [
            ["Aggressive", "2e-4", "3", "16"],
            ["Conservative", "5e-5", "1", "8"],
            ["Competition", "1e-4", "2", "16"],
            ["Ultra-Cons.", "2e-5", "1", "4"],
            ["Sweep", "var.", "1-5", "4-32"],
        ],
        [3, 1, 1, 1],
    )

    pdf.subsection("3.4", "Evaluation")
    pdf.para(
        "We evaluate on three benchmarks: (1) ORZ Math (1024 problems): "
        "math accuracy via boxed-answer extraction with symbolic comparison; "
        "(2) SciKnowEval (1893 MCQ): chemistry knowledge retention; "
        "(3) ToolAlpaca (192 queries): tool-use function accuracy. "
        "A configuration is 'valid' if SciKnowEval and ToolAlpaca "
        "degrade by <=3% from baseline."
    )

    pdf.table(
        "Table 2: Baseline scores (Qwen2.5-3B-Instruct)",
        ["Benchmark", "Metric", "Baseline"],
        [
            ["ORZ Math", "Accuracy", "28.91%"],
            ["SciKnowEval", "MCQ Acc", "34.34%"],
            ["ToolAlpaca Sim", "Func Acc", "78.89%"],
            ["ToolAlpaca Real", "Func Acc", "89.22%"],
        ],
        [2.5, 2, 1.5],
    )

    # ----------- 4. RESULTS -----------
    pdf.section("4", "Results")

    pdf.subsection("4.1", "The Inverted Scaling Curve")
    pdf.para(
        "Table 3 shows the best valid ORZ accuracy at each sample count. "
        "Performance peaks at N=100 (+1.75%) and declines monotonically, "
        "reaching -7.33% at N=10,000."
    )

    curve_rows = []
    for n, exp in best.items():
        ds = exp["data_source"]
        if ds == "numinamath":
            ds = "numi"
        elif ds == "numinamath_hard":
            ds = "hard"
        elif ds == "numinamath_comp":
            ds = "comp"
        curve_rows.append([
            str(n),
            f"{exp['orz_accuracy']*100:.1f}",
            f"{exp['orz_delta']*100:+.1f}",
            f"{exp['sciknoweval_accuracy']*100:.1f}",
            f"{exp['toolalpaca_real_func_acc']*100:.1f}",
            ds,
        ])

    pdf.table(
        "Table 3: Best valid ORZ accuracy per sample count",
        ["N", "ORZ%", "Delta", "Sci%", "TAR%", "Src"],
        curve_rows,
        [1.2, 1, 1, 1, 1, 1],
    )

    # Insert figure 1
    if os.path.exists("results/figures/fig1_orz_vs_n.png"):
        pdf.insert_image(
            "results/figures/fig1_orz_vs_n.png",
            w=COL_W,
            caption="Figure 1: ORZ accuracy vs. sample count by data source "
                    "(fixed HP: r=8, LR=5e-5, 1 epoch). All sources show "
                    "monotonic decline. Baseline = 28.91%."
        )

    pdf.subsection("4.2", "Cross-Domain Robustness")
    pdf.para(
        "Despite the severe in-domain degradation, cross-domain metrics are "
        "remarkably stable. Of 43 experiments, 39 remain valid (SciKnowEval "
        "and ToolAlpaca within 3% of baseline). The 4 invalid runs all "
        "involve either aggressive hyperparameters (LR=2e-4, r=16, 3 epochs) "
        "or OpenR1 data at N>=1000."
    )

    # Insert figures 2 and 3
    if os.path.exists("results/figures/fig2_sci_vs_n.png"):
        pdf.insert_image(
            "results/figures/fig2_sci_vs_n.png",
            w=COL_W,
            caption="Figure 2: SciKnowEval (chemistry) remains above the 3% "
                    "threshold across most configurations."
        )

    if os.path.exists("results/figures/fig3_tool_vs_n.png"):
        pdf.insert_image(
            "results/figures/fig3_tool_vs_n.png",
            w=COL_W,
            caption="Figure 3: ToolAlpaca (tool-use) is highly robust. Only "
                    "OpenR1 at N>=1000 causes marginal degradation."
        )

    pdf.subsection("4.3", "Data Source Comparison")
    pdf.para(
        "At fixed hyperparameters (r=8, LR=5e-5, 1 epoch), we compare data "
        "sources across sample counts (Table 4). NuminaMath-Hard degrades "
        "most slowly, consistent with better distribution match to the "
        "competition-style ORZ benchmark. OpenR1 is catastrophic at N>1000 "
        "due to its extremely long reasoning format."
    )

    # Build Table 4
    sources = ["numinamath", "numinamath_hard", "openr1"]
    ns_list = [100, 500, 1000, 2000, 5000, 10000]
    ds_rows = []
    for n in ns_list:
        row = [str(n)]
        for src in sources:
            match = [e for e in experiments
                     if e["data_source"] == src and e["num_samples"] == n
                     and e["lora_rank"] == 8 and e["lr"] == "5e-5"
                     and e["epochs"] == 1]
            if match:
                row.append(f"{match[0]['orz_accuracy']*100:.1f}")
            else:
                row.append("-")
        ds_rows.append(row)

    pdf.table(
        "Table 4: ORZ % by data source (r=8, LR=5e-5, ep=1)",
        ["N", "Numi", "Hard", "OpenR1"],
        ds_rows,
        [1.2, 1, 1, 1],
    )

    pdf.subsection("4.4", "Hyperparameter Sensitivity")
    pdf.para(
        "Figure 4 shows ORZ accuracy for all configurations at N=100. "
        "The best config (r=8, LR=5e-5, 1 epoch) achieves 30.66%, while "
        "aggressive training (r=16, LR=2e-4, 3 epochs) drops to 21.0%. "
        "We rank factor importance: (1) sample count, (2) learning rate, "
        "(3) LoRA rank, (4) epochs."
    )

    if os.path.exists("results/figures/fig4_hp_n100.png"):
        pdf.insert_image(
            "results/figures/fig4_hp_n100.png",
            w=COL_W,
            caption="Figure 4: HP sensitivity at N=100 (NuminaMath). "
                    "Green = above baseline; red = below."
        )

    # ----------- 5. ANALYSIS -----------
    pdf.section("5", "Analysis")

    pdf.subsection("5.1", "Distribution Mismatch Explains the Curve")
    pdf.para(
        "The inverted scaling curve is a predictable consequence of "
        "distribution mismatch. NuminaMath-CoT contains many synthetic, "
        "formulaic problems that differ from ORZ's competition-style "
        "reasoning. SFT on this data overwrites the model's pre-trained "
        "math patterns rather than supplementing them."
    )
    pdf.para(
        "Three lines of evidence support this interpretation:"
    )
    pdf.para(
        "(1) Hard problem filtering (NuminaMath-Hard) slows the decline. "
        "At N=2000, hard-filtered achieves 26.1% vs. 23.3% for random, "
        "a 2.8pp gap that grows with N. Better distribution match = slower "
        "degradation."
    )
    pdf.para(
        "(2) OpenR1 data, with its extremely long reasoning traces, is "
        "catastrophically harmful (5.4% at N=2000). The format mismatch is "
        "so severe that the model stops producing extractable boxed answers."
    )
    pdf.para(
        "(3) At N=100, the perturbation is ~6 gradient steps (100/16 batch). "
        "This micro-tuning nudges generation style without overwriting "
        "reasoning patterns, yielding a net positive effect."
    )

    pdf.subsection("5.2", "Why Cross-Domain Knowledge Survives")
    pdf.para(
        "The asymmetry between in-domain degradation and cross-domain "
        "robustness suggests that LoRA updates concentrate on math-relevant "
        "parameters while leaving chemistry and tool-use subspaces largely "
        "untouched. Even at N=10,000 with LR=2e-4, SciKnowEval drops only "
        "2.6% and ToolAlpaca stays within 2%."
    )
    pdf.para(
        "We hypothesize this reflects functional modularity in the 3B model: "
        "math reasoning, scientific knowledge, and structured tool-use occupy "
        "sufficiently disjoint parameter subspaces that low-rank perturbation "
        "in one domain does not significantly affect others. Verifying this "
        "hypothesis requires gradient overlap or weight-space geometry "
        "analysis, which we leave to future work."
    )

    pdf.subsection("5.3", "Practical Implications")
    pdf.para(
        "Our results yield concrete guidelines for practitioners:"
    )
    pdf.bullet(
        "Verify distribution match before scaling data. If your SFT data "
        "distribution diverges from evaluation, adding more data will hurt."
    )
    pdf.bullet(
        "Prefer micro-tuning when distribution match is uncertain. N<200 "
        "with 1 epoch and moderate LR (5e-5) is a safe default."
    )
    pdf.bullet(
        "Data quality > quantity at all scales. Hard-filtered data "
        "consistently outperforms random sampling."
    )
    pdf.bullet(
        "LoRA provides robust cross-domain protection. Forgetting non-target "
        "capabilities is unlikely to be the binding constraint."
    )

    # ----------- 6. LIMITATIONS -----------
    pdf.section("6", "Limitations and Future Work")

    pdf.para(
        "We acknowledge several limitations that bound the strength of our "
        "conclusions:"
    )
    pdf.para(
        "Single model, single scale. All experiments use Qwen2.5-3B-Instruct. "
        "The inverted curve may differ at 7B+ scales or with different model "
        "families (Llama, Mistral). Generalization requires cross-model "
        "validation."
    )
    pdf.para(
        "No confidence intervals. Each experiment is a single run. The best "
        "result (+1.75%) is within plausible variance of a 1024-sample eval. "
        "Multiple seeds per configuration are needed for statistical rigor."
    )
    pdf.para(
        "Distribution mismatch is asserted, not measured. We provide indirect "
        "evidence (hard filtering helps, OpenR1 format hurts) but no direct "
        "distributional distance metrics between training and evaluation data."
    )
    pdf.para(
        "Missing in-distribution control. The strongest test of our hypothesis "
        "would be SFT on ORZ-distribution data (e.g., rejection-sampled "
        "solutions from the base model). If the curve inverts even with "
        "matched distributions, our explanation is wrong."
    )
    pdf.para(
        "Tiny forgetting eval sets. ToolAlpaca (90+102 samples) is so small "
        "that a single changed response shifts accuracy by ~1%, making the "
        "3% threshold noisy."
    )
    pdf.para(
        "Future work should address these gaps: multi-model validation, "
        "confidence intervals via repeated runs, in-distribution SFT controls, "
        "and mechanistic analysis of the cross-domain robustness phenomenon "
        "(e.g., gradient overlap, weight-space geometry)."
    )

    # ----------- 7. CONCLUSION -----------
    pdf.section("7", "Conclusion")

    pdf.para(
        "We have presented a 43-experiment empirical study of SFT data "
        "scaling in Qwen2.5-3B-Instruct. Our results demonstrate that "
        "distribution mismatch between SFT data and evaluation can invert "
        "the data scaling curve: more training data degrades performance. "
        "This effect is large (10pp decline from N=100 to N=10,000) and "
        "consistent across four data sources and five HP configurations."
    )
    pdf.para(
        "Simultaneously, cross-domain capabilities prove remarkably robust "
        "under LoRA fine-tuning, with chemistry and tool-use scores rarely "
        "exceeding 3% degradation. The forgetting frontier lies well beyond "
        "the distribution-mismatch frontier."
    )
    pdf.para(
        "We hope this detailed empirical characterization provides useful "
        "guidance for practitioners considering SFT data scaling decisions, "
        "and motivates further investigation of distribution-aware fine-tuning "
        "strategies."
    )

    # ----------- REFERENCES -----------
    pdf.section("", "References")
    refs = [
        "[1] Chen, L., et al. (2023). AlpaGasus: Training a Better Alpaca with Fewer Data. arXiv:2307.08701.",
        "[2] Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.",
        "[3] Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS 114(13).",
        "[4] Li, Y., et al. (2024). NuminaMath: A Large-Scale Curated Dataset for Mathematical Reasoning.",
        "[5] Li, Z. & Hoiem, D. (2017). Learning without Forgetting. IEEE TPAMI 40(12).",
        "[6] Liu, W., et al. (2024). What Makes Good Data for Alignment? A Comprehensive Study. arXiv:2311.15430.",
        "[7] Luo, Y., et al. (2023). An Empirical Study of Catastrophic Forgetting in LLMs. arXiv:2308.08747.",
        "[8] Open-Reasoner (2024). OpenR1-Math: High-Quality Math Reasoning Data from R1 Distillation.",
        "[9] Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS.",
        "[10] Yang, A., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.",
        "[11] Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. NeurIPS.",
    ]

    self = pdf
    self.set_font("Times", "", 7.5)
    for ref in refs:
        self._ensure_space(8)
        self.multi_cell(0, 3.5, sanitize(ref), align="L")
        self.ln(1)

    # ----------- APPENDIX -----------
    pdf._next_col_or_page()
    pdf.section("A", "Appendix: Full Results (43 Experiments)")

    pdf.para(
        "Table 5 lists all 43 experiments ranked by ORZ accuracy. "
        "V = valid (forgetting < 3%)."
    )

    sorted_exps = sorted(experiments, key=lambda x: x["orz_accuracy"],
                         reverse=True)

    full_headers = ["#", "Src", "N", "r", "LR", "Ep", "ORZ", "Sci",
                    "TA-R", "V"]

    # Split into chunks that fit in a column
    chunk_size = 22
    for chunk_start in range(0, len(sorted_exps), chunk_size):
        chunk = sorted_exps[chunk_start:chunk_start + chunk_size]
        rows = []
        for i, exp in enumerate(chunk, chunk_start + 1):
            src = exp["data_source"]
            if src == "numinamath":
                src = "numi"
            elif src == "numinamath_hard":
                src = "hard"
            elif src == "numinamath_comp":
                src = "comp"
            elif src == "openr1":
                src = "or1"
            rows.append([
                str(i),
                src,
                str(exp["num_samples"]),
                str(exp["lora_rank"]),
                exp["lr"],
                str(exp["epochs"]),
                f"{exp['orz_accuracy']*100:.1f}",
                f"{exp['sciknoweval_accuracy']*100:.1f}",
                f"{exp['toolalpaca_real_func_acc']*100:.1f}",
                "Y" if exp.get("valid") else "N",
            ])

        cap = "Table 5: All experiments (ranked by ORZ accuracy)"
        if chunk_start > 0:
            cap = "Table 5 (continued)"
        pdf.table(cap, full_headers, rows,
                  [0.6, 0.8, 1.0, 0.5, 0.9, 0.5, 0.8, 0.8, 0.8, 0.5],
                  font_size=7)

    # Save
    os.makedirs("results", exist_ok=True)
    pdf.output("results/approach_summary.pdf")
    print(f"PDF generated: results/approach_summary.pdf ({pdf.page_no()} pages)")


if __name__ == "__main__":
    generate_paper()
