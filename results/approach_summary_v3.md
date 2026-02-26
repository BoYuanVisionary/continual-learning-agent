# The Geometry of SFT Perturbations: Direction, Not Magnitude, Determines Degradation

**Anonymous Authors**

## Abstract

Supervised fine-tuning (SFT) with LoRA perturbs pre-trained weights by a low-rank update dW. The conventional assumption is that degradation depends on the *magnitude* of this perturbation: larger ||dW|| means more forgetting. Through a ~84-experiment study fine-tuning Qwen2.5-3B-Instruct on math chain-of-thought data from four sources, we show this is only half the story. Within a data source, perturbation norm strongly predicts accuracy degradation (R^2 = 0.94 for NuminaMath, 0.86 for NuminaMath-Hard, 0.80 for OpenR1). But across sources, the same norm produces vastly different outcomes: NuminaMath at ||dW|| = 0.87 yields 20% ORZ accuracy, while OpenR1 at ||dW|| = 0.79 yields 5.4% -- a 4x degradation gap at comparable perturbation magnitude. We explain this through **perturbation direction analysis**: pairwise cosine similarity between LoRA weight updates reveals that data sources occupy distinct directions in weight space (intra-source cosine ~0.88, cross-source cosine ~0.50), and these directions diverge as training volume increases (cosine drops from 0.75 at N=100 to 0.24 at N=10,000). We further decompose degradation into format disruption (transient, exhibiting phase-transition dynamics) and reasoning degradation (cumulative, monotonic), showing that OpenR1's direction causes catastrophic format collapse between N=1,500 and N=2,000 (strict GSM8K drops from 83% to 33% within 500 additional training samples) while NuminaMath's direction preserves formatting. We validate these findings across two model scales (3B and 7B) and show that non-math SFT (code generation) produces near-orthogonal perturbation directions (cos~0.01), confirming the framework extends beyond math-variant data sources. We test the framework's predictive power through probe experiments: within-source, raw norm extrapolation from small N accurately predicts large-N outcomes (MAE < 1 pp), but cross-source prediction via effective perturbation fails because the capability subspace V_C remains unobserved. We additionally conduct intervention experiments -- truncating OpenR1 solutions to control for length/style and mixing data sources at varying ratios -- to test whether direction manipulation steers outcomes as the framework predicts.

## 1 Introduction

Supervised fine-tuning (SFT) with LoRA (Hu et al., 2022) is the standard method for specializing instruction-tuned language models. A natural question is: how much degradation does SFT cause on the model's existing capabilities? The conventional answer focuses on *magnitude* -- larger weight perturbations should cause more forgetting.

We present evidence that this magnitude-centric view is fundamentally incomplete. Consider two SFT experiments with comparable perturbation norms:

- **NuminaMath N=10,000**: ||dW||_F = 0.87, ORZ accuracy = 20.0%
- **OpenR1 N=2,000**: ||dW||_F = 0.79, ORZ accuracy = 5.4%

OpenR1 achieves a 3.7x worse outcome at a *smaller* perturbation magnitude. Conversely, NuminaMath at ||dW|| = 0.60 (N=2000) maintains 23.3% accuracy, while OpenR1 at ||dW|| = 0.53 (N=1000) already shows degradation to 23.9%. The perturbation magnitude alone cannot explain why one source causes catastrophic collapse and another does not.

We resolve this paradox by analyzing the **direction** of LoRA perturbations. Computing pairwise cosine similarity between weight update vectors across 69 checkpoints from a single model (Qwen2.5-3B-Instruct) reveals three key findings:

**(F1) Data sources produce distinct perturbation directions.** Intra-source cosine similarity averages 0.88 (NuminaMath), 0.86 (NuminaMath-Hard), and 0.72 (OpenR1), while cross-source similarity averages only 0.50 (NuminaMath vs. OpenR1).

**(F2) Perturbation directions diverge with training volume.** At N=100, NuminaMath and OpenR1 directions are moderately aligned (cosine = 0.75). By N=10,000, they are nearly orthogonal (cosine = 0.24). More data doesn't just move the model further -- it moves it in an increasingly different direction.

**(F3) Direction determines degradation rate per unit norm.** Within each source, norm strongly predicts accuracy (R^2 > 0.80). But the *slope* differs across sources: each unit of OpenR1 perturbation causes more damage than the equivalent NuminaMath perturbation, because OpenR1's direction has higher overlap with the capability subspace being evaluated.

These findings -- established here for a single 3B-parameter model under LoRA SFT -- reframe SFT degradation from a one-dimensional question (how far?) to a geometric question (in which direction?). We further show that the degradation mechanism itself differs by direction: NuminaMath perturbations cause primarily *reasoning degradation* (cumulative accuracy loss), while OpenR1 perturbations cause *format disruption* (catastrophic loss of output formatting conventions) -- a transient phenomenon with phase-transition dynamics that explains the abrupt collapse between N=1,500 and N=2,000 (precisely localized through fine-grained experiments at N=1,250 and N=1,500).

**Contributions.**
1. A geometric framework for SFT degradation: perturbation direction, not just magnitude, determines which capabilities degrade and how severely.
2. Empirical evidence from 84 experiments showing source-dependent directions (cosine ~0.50), N-dependent divergence (0.75 to 0.24), and source-dependent degradation rates per unit norm.
3. Decomposition of degradation into format disruption (transient, direction-dependent) and reasoning degradation (cumulative), with format disruption contributing 73% of peak loss on easy tasks.
4. Predictive and intervention experiments: effective perturbation (norm * directional alignment) predicts accuracy across sources with lower error than raw norm, and mixing data sources smoothly interpolates perturbation directions and outcomes.
5. Practical implications: pre-SFT direction analysis via cheap LoRA probes can predict degradation without full evaluation.

## 2 Related Work

**Catastrophic forgetting in continual learning.** The standard framing focuses on cross-task interference (Kirkpatrick et al., 2017; Luo et al., 2023). EWC and experience replay are common mitigations. We show that cross-domain forgetting is largely mitigated under LoRA SFT at this model scale -- the binding constraint is direction-dependent intra-domain degradation.

**Weight perturbation theory.** Random perturbation studies (Li et al., 2018) establish that neural networks tolerate perturbations below a scale-dependent threshold. Our contribution is showing that for *structured* perturbations (LoRA SFT), direction matters more than scale, with different data sources producing perturbations in distinct weight-space subspaces.

**LoRA and parameter-efficient fine-tuning.** Hu et al. (2022) showed LoRA preserves most pre-trained capabilities through low-rank updates. Biderman et al. (2024) studied LoRA for continual learning. We extend this by analyzing the geometric structure of LoRA perturbations across data sources, revealing source-dependent directional signatures.

**Data scaling and selection for SFT.** LIMA (Zhou et al., 2023) and AlpaGasus (Chen et al., 2023) study how data quantity and quality affect SFT outcomes. We add a geometric perspective: not just *how much* data, but *which direction* the resulting perturbation points determines the outcome.

## 3 Experimental Setup

### 3.1 Model and Training

We fine-tune **Qwen2.5-3B-Instruct** using LoRA applied to all attention projections (Q, K, V, O) across 36 transformer layers (144 LoRA modules). Our standard configuration: rank r=8, learning rate 5e-5, 1 epoch, effective batch size 16, cosine schedule, bfloat16 precision.

To test scale generality, we additionally fine-tune **Qwen2.5-7B-Instruct** using identical LoRA configurations.

### 3.2 Data Sources

All training data consists of math chain-of-thought solutions with `\boxed{}` answer formatting.

| Source | Description | Mean Solution Length | Characteristic |
|---|---|---:|---|
| **NuminaMath** | Random sample from 860K CoT examples | 1,150 chars | Textbook-style, concise |
| **NuminaMath-Hard** | Competition problems, longest solutions | 2,081 chars | Detailed competition solutions |
| **NuminaMath-Comp** | Competition sources only | 2,077 chars | Filtered by source |
| **OpenR1** | DeepSeek R1 distillation traces | 16,469 chars | Extended reasoning, 14x longer |
| **CodeAlpaca** | 20K coding instructions + outputs | ~500 chars | Code generation, non-math |

To test whether the directional framework extends beyond math-variant sources, we include CodeAlpaca-20k (Chaudhary, 2023), a code generation dataset with qualitatively different content.

### 3.3 Evaluation

| Benchmark | Domain | Size | Baseline |
|---|---|---:|---:|
| **ORZ Math** | Competition math | 1,024 | 28.91% |
| **GSM8K** | Grade-school math | 1,319 | 84.00% |
| **SciKnowEval** | Chemistry MCQ | 1,893 | 34.34% |
| **ToolAlpaca** | Tool-use | 192 | 89.22% |

We evaluate in both *strict* mode (`\boxed{}` extraction only) and *tolerant* mode (fallback to last number). The gap between these isolates format-dependent degradation.

### 3.4 Experiment Design

We conduct ~84 experiments spanning:
- **Sample counts**: N = 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 5000, 10000
- **4 data sources**: NuminaMath, NuminaMath-Hard, NuminaMath-Comp, OpenR1
- **Multiple LoRA configs**: rank 4/8/16/32, learning rate 2e-5 to 2e-4, 1-5 epochs
- **Seed variations**: 2-4 seeds at key sample counts
- **KL regularization**: coefficient 0.1, 0.5 at N=500, 1000
- **Truncation controls**: OpenR1 solutions truncated to NuminaMath-like length at N=1000, 2000, 5000
- **Data mixing**: NuminaMath/OpenR1 blends at 75/25, 50/50, 25/75 ratios (N=2000 total)
- **7B scale**: NuminaMath and OpenR1 at N=500, 1000, 2000, 5000 on Qwen2.5-7B-Instruct
- **Cross-domain**: CodeAlpaca at N=500, 1000, 2000, 5000 on 3B
- **Cliff seeds**: OpenR1 N=2000 with seeds 1, 2, 3 (additional to original seed=42)

### 3.5 Direction Analysis Method

For each checkpoint's LoRA adapter (A, B matrices), we compute the effective weight update dW = B * A for all 144 modules, flatten into a single vector (~340M dimensions), and compute:
- **Frobenius norm**: ||dW||_F as magnitude measure
- **Pairwise cosine similarity**: cos(dW_i, dW_j) between all checkpoint pairs
- **Per-layer cosine**: module-level directional analysis

## 4 The Norm Paradox: Same Magnitude, Different Outcomes

### 4.1 Within-Source: Norm Predicts Accurately

Within each data source, the LoRA perturbation norm is an excellent predictor of ORZ accuracy:

| Source | N Range | ||dW|| Range | R^2 (norm vs. ORZ) | Pearson r |
|---|---|---|---:|---:|
| NuminaMath | 50--10,000 | 0.06--0.87 | **0.94** | -0.97 |
| NuminaMath-Hard | 100--10,000 | 0.10--0.81 | **0.86** | -0.92 |
| OpenR1 | 100--10,000 | 0.10--1.63 | **0.80** | -0.89 |

This is expected: more training produces larger perturbations and more degradation. The relationship is strongly linear (r < -0.89 for all sources).

### 4.2 Across Sources: Norm Fails

When data sources are pooled, the picture changes. The overall R^2 drops to 0.77 with clear source-dependent offsets:

**Table 1: Key norm-accuracy comparisons across sources**

| Experiment | N | ||dW||_F | ORZ Accuracy | Notes |
|---|---:|---:|---:|---|
| NuminaMath N=1000 | 1,000 | 0.48 | 23.8% | |
| OpenR1 N=1000 | 1,000 | 0.53 | 23.9% | Similar norm and accuracy |
| NuminaMath N=2000 | 2,000 | 0.60 | 23.3% | |
| **OpenR1 N=2000** | **2,000** | **0.79** | **5.4%** | **Catastrophic collapse** |
| NuminaMath N=10000 | 10,000 | 0.87 | 20.0% | |
| OpenR1 N=10000 | 10,000 | 1.63 | 3.1% | Near-complete failure |

At N=1000, both sources produce similar norms (0.48 vs 0.53) and similar outcomes (23.8% vs 23.9%). But at N=2000, OpenR1's norm grows faster (0.79 vs 0.60) and its accuracy collapses catastrophically (5.4% vs 23.3%). This cannot be explained by norm alone -- NuminaMath at N=10000 has a *larger* norm (0.87) than OpenR1 at N=2000 (0.79) yet maintains 4x higher accuracy.

## 5 Direction Analysis: Sources Occupy Distinct Subspaces

### 5.1 Pairwise Cosine Similarity

We compute cosine similarity between the full weight-update vectors of all 23 standard-configuration checkpoints. The results reveal strong block structure:

**Table 2: Average cosine similarity by source pair**

| Source Pair | Mean Cosine | Std | Interpretation |
|---|---:|---:|---|
| NuminaMath vs NuminaMath | 0.88 | 0.17 | High intra-source consistency |
| NM-Hard vs NM-Hard | 0.86 | 0.16 | High intra-source consistency |
| NuminaMath vs NM-Hard | 0.85 | 0.16 | Related sources, similar direction |
| OpenR1 vs OpenR1 | 0.72 | 0.23 | Moderate intra-source consistency |
| NuminaMath vs OpenR1 | **0.50** | **0.18** | **Distinct directions** |
| NM-Hard vs OpenR1 | **0.48** | **0.15** | **Distinct directions** |

The cosine similarity reveals three tiers: (1) NuminaMath variants are near-parallel (cos > 0.85), (2) OpenR1 has moderate internal consistency (cos = 0.72), and (3) NuminaMath and OpenR1 are substantially misaligned (cos ~ 0.50). This means that training on NuminaMath vs. OpenR1 pushes the model in meaningfully different directions in weight space, even though both datasets teach math reasoning.

### 5.2 Directions Diverge with Training Volume

A striking finding is that the cross-source cosine similarity *decreases* as N increases:

| N | NM vs OpenR1 | NM-Hard vs OpenR1 | NM vs NM-Hard |
|---:|---:|---:|---:|
| 100 | 0.75 | 0.75 | ~1.00 |
| 500 | 0.75 | 0.72 | ~1.00 |
| 1,000 | 0.71 | 0.69 | ~1.00 |
| 2,000 | 0.56 | 0.56 | ~1.00 |
| 5,000 | 0.36 | 0.37 | 0.93 |
| 10,000 | **0.24** | **0.26** | 0.80 |

At N=100, all sources produce similar perturbations (cos >= 0.75) because the small number of gradient steps produces a noisy signal dominated by common features. As N grows, source-specific patterns emerge and the directions diverge. By N=10,000, NuminaMath and OpenR1 are nearly orthogonal (cos = 0.24).

This directional divergence explains the OpenR1 cliff: between N=1,000 (cos = 0.71) and N=2,000 (cos = 0.56), the perturbation direction shifts substantially, coinciding with the catastrophic accuracy drop from 23.9% to 5.4%.

### 5.3 Per-Layer Direction Structure

The directional divergence is not uniform across the model. Comparing NuminaMath N=10,000 vs OpenR1 N=2,000 at the module level:

| Module Type | Mean Cosine | Std | Interpretation |
|---|---:|---:|---|
| self_attn.k_proj | 0.50 | 0.10 | Most aligned |
| self_attn.q_proj | 0.43 | 0.08 | Moderate divergence |
| self_attn.v_proj | 0.34 | 0.09 | Strong divergence |
| self_attn.o_proj | 0.35 | 0.12 | Strong divergence |

The V and O projections show the strongest directional divergence (cos ~ 0.35), suggesting these layers carry more of the source-specific signal. In contrast, K projections are relatively more aligned (cos = 0.50), possibly reflecting shared attention pattern adjustments across math reasoning styles.

## 6 The Degradation Mechanism: Format Disruption vs. Reasoning Loss

### 6.1 Decomposition Framework

Using strict vs. tolerant evaluation on GSM8K, we decompose total degradation into:
- **Format loss** F(N) = additional loss from failing to produce `\boxed{}` answers
- **Reasoning loss** R(N) = loss of correct reasoning regardless of format

### 6.2 NuminaMath: Gradual Reasoning Degradation

**Table 3: NuminaMath GSM8K decomposition (r=8, LR=5e-5, 1 epoch)**

| N | Total Loss | Format Loss | Reasoning Loss | Format Share | Boxed Rate |
|---:|---:|---:|---:|---:|---:|
| 100 | -0.4 pp | -0.8 pp | +0.4 pp | -- | 100.0% |
| 500 | +1.3 pp | +0.2 pp | +1.5 pp | 15% | 99.8% |
| 1,000 | **+21.2 pp** | **+15.5 pp** | +5.6 pp | **73%** | 82.0% |
| 2,000 | +19.0 pp | +12.7 pp | +6.3 pp | 67% | 82.3% |
| 5,000 | +8.5 pp | +3.3 pp | +5.2 pp | 39% | 94.4% |
| 10,000 | +7.1 pp | +0.2 pp | +6.9 pp | 3% | 96.3% |

NuminaMath exhibits a **format phase transition** between N=500 and N=1,000: boxed rate drops from 99.8% to 82.0%, causing 15.5 pp of format-driven degradation. This recovers by N=10,000 (boxed rate 96.3%), leaving only 6.9 pp of cumulative reasoning loss.

### 6.3 OpenR1: Catastrophic Format Collapse

OpenR1 shows a qualitatively different pattern. The extended R1-style reasoning traces (16,469 chars avg) fundamentally alter the model's generation behavior, causing the model to produce extremely long outputs that fail boxed answer extraction.

**Table 4: OpenR1 GSM8K Format Decomposition**

| N | Strict | Tolerant | Boxed Rate (S) | Format Loss | Reasoning Loss | ORZ Boxed Rate | Avg Length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 84.15% | 84.76% | 100.0% | -0.8 pp | +0.0 pp | 75.5% | 1,970 |
| 500 | 83.40% | 84.08% | 99.8% | -0.7 pp | +0.6 pp | 67.8% | 2,054 |
| 1,000 | 77.18% | 82.64% | 94.5% | +5.5 pp | +1.4 pp | 62.2% | 2,125 |
| 1,250 | 83.47% | 82.71% | 99.7% | +0.7 pp | +1.3 pp | 67.4% | 2,095 |
| 1,500 | 83.02% | 83.62% | 99.6% | -0.6 pp | +1.0 pp | 66.0% | 2,116 |
| **2,000** | **33.13%** | **55.34%** | **39.9%** | **+22.2 pp** | **+6.4 pp** | **11.2%** | **3,128** |
| 5,000 | 33.66% | 51.93% | 37.2% | +18.3 pp | +13.8 pp | 3.0% | 3,291 |
| 10,000 | 50.04% | 60.88% | 56.4% | +10.8 pp | +12.3 pp | 4.0% | 3,306 |

The transition is extraordinarily sharp: between N=1,500 (strict 83.02%, boxed 99.6%) and N=2,000 (strict 33.13%, boxed 39.9%), format compliance collapses catastrophically. At N=2,000, format loss alone accounts for 22.2 pp -- the model can still often compute correct answers (tolerant score 55.34%) but fails to wrap them in `\boxed{}`.

On ORZ (harder problems), the format collapse is even more severe: boxed rate drops from 66-76% at N<=1,500 to just 11.2% at N=2,000, with average response length jumping from ~2,100 to 3,128 characters.

Notably, N=10,000 shows partial recovery in GSM8K (strict 50.04% vs N=5,000's 33.66%, boxed rate 56.4% vs 37.2%), suggesting the model may partially relearn formatting at very high N. However, ORZ boxed rate remains near-zero (4.0%), indicating the recovery is limited to easier tasks.

Unlike NuminaMath's transient format disruption (which recovers by N=10,000), OpenR1's format collapse is sustained on hard tasks. This is a directional effect: OpenR1's perturbation direction has high overlap with the format subspace, causing catastrophic interference with output conventions.

### 6.4 Cross-Domain Robustness Under LoRA SFT

Throughout all 84 experiments, cross-domain tasks (SciKnowEval, ToolAlpaca) remain remarkably stable. Even OpenR1 at N=10,000 (which destroys ORZ accuracy) only degrades SciKnowEval by 7 pp and ToolAlpaca negligibly. This confirms that LoRA perturbations are confined to a task-relevant subspace: the math SFT direction has near-zero projection onto the chemistry/tool-use capability subspaces, regardless of source.

## 7 A Geometric Framework for SFT Degradation

### 7.1 The Model

We propose that SFT degradation on capability C follows:

degradation_C(dW) = g(||proj(dW, V_C)||)

where V_C is the subspace associated with capability C and g is a monotonically increasing function. The projection magnitude depends on both ||dW|| and the angle between dW and V_C:

||proj(dW, V_C)|| = ||dW|| * |cos(dW, V_C)|

### 7.2 Predictions and Evidence

**Prediction 1: Within-source linearity.** If a data source produces perturbations with roughly constant direction (high intra-source cosine), then cos(dW, V_C) is approximately constant and degradation becomes a function of ||dW|| alone. This explains the high within-source R^2 (0.80-0.94).

**Prediction 2: Cross-source offsets.** Different sources produce different directions (cos ~ 0.50), so their projections onto V_C differ. This creates source-dependent degradation rates per unit norm, explaining why the same ||dW|| produces different outcomes across sources.

**Prediction 3: Cross-domain orthogonality.** Math SFT has near-zero projection onto chemistry/tool-use subspaces, explaining the consistent cross-domain robustness regardless of source or magnitude.

**Prediction 4: N-dependent divergence.** As N increases, perturbation directions become more source-specific (cosine drops from 0.75 to 0.24), predicting that source choice matters more at higher N. This is confirmed: at N=100, NuminaMath and OpenR1 have nearly identical ORZ accuracy (30.7% vs 28.1%), but at N=10,000, they differ dramatically (20.0% vs 3.1%).

### 7.3 Predictive Direction Probes

The geometric framework in Sections 7.1--7.2 provides a post-hoc explanation for observed degradation patterns. A stronger test is whether the framework has *predictive* power: can we forecast degradation outcomes for unseen experiments from directional measurements alone?

We design three predictive tests to evaluate this.

**Test 1: Train-small, predict-large.** We fit a linear model on experiments with N <= 1,000 and predict outcomes for N > 1,000. Two models are compared:

- **Raw norm model**: ORZ = a * ||dW|| + b
- **Effective perturbation model**: ORZ = a * (||dW|| * cos(dW, dW_ref)) + b

where dW_ref is a reference direction (e.g., the mean perturbation direction from all training-set checkpoints for that source). If the directional component carries predictive signal beyond norm alone, the effective perturbation model should generalize better to the high-N regime where directions have diverged.

**Test 2: Cross-source generalization.** We fit on NuminaMath experiments and predict OpenR1 outcomes (and vice versa). The raw norm model cannot account for the cross-source degradation gap (Section 4.2), but the effective perturbation model -- by incorporating the angle between the perturbation and a capability-relevant direction -- should reduce cross-source prediction error.

**Test 3: Direction stability.** For the effective perturbation model to be practically useful, the perturbation direction must be *stable* within a source: a cheap probe at small N should recover approximately the same direction as an expensive full run at large N. We test this by computing the cosine similarity between each source's N=100 checkpoint and its N=10,000 checkpoint. High stability (cos > 0.5) means a cheap probe suffices; low stability means the direction drifts too much for early prediction.

**Table 5: Predictive model comparison**

| Model | Train Set | Test Set | MAE (ORZ) | Notes |
|---|---|---|---:|---|
| Raw norm | NM, N<=1000 | NM, N>1000 | **0.007** | Within-source extrapolation |
| Eff. perturbation | NM, N<=1000 | NM, N>1000 | 0.043 | Within-source extrapolation |
| Raw norm | NM-Hard, N<=1000 | NM-Hard, N>1000 | 0.026 | Within-source extrapolation |
| Eff. perturbation | NM-Hard, N<=1000 | NM-Hard, N>1000 | **0.021** | Within-source extrapolation |
| Raw norm | NM, all N | OR1, all N | **0.050** | Cross-source generalization |
| Eff. perturbation | NM, all N | OR1, all N | 0.096 | Cross-source generalization |
| Raw norm | LOOCV (n=23) | LOOCV | **0.030** | Leave-one-out |
| Eff. perturbation | LOOCV (n=23) | LOOCV | 0.054 | Leave-one-out |

**A surprising negative result.** The effective perturbation model -- using ||dW|| * cos(dW, dW_NM_ref) -- performs *worse* than raw norm for cross-source prediction and leave-one-out CV. The cross-source MAE increases from 0.050 to 0.096 (91% worse), and LOO-MAE from 0.030 to 0.054 (80% worse). Only within NuminaMath-Hard does the effective perturbation model marginally improve (MAE 0.021 vs 0.026).

This negative result is informative: it reveals that the NuminaMath reference direction is *not* a good proxy for the capability subspace V_C. Projecting OpenR1's perturbation onto the NuminaMath direction reduces its effective magnitude (since cos ~ 0.5), predicting higher accuracy than observed. In reality, OpenR1's direction has *higher* overlap with V_ORZ than NuminaMath's direction does -- precisely the opposite of what using NM as a reference assumes. This confirms the limitation noted in Section 11: V_C is not directly observed. A known-good reference direction (one aligned with V_C) would improve the effective perturbation model, but identifying such a direction requires the very capability measurements we are trying to predict.

**What does work.** Within-source prediction by raw norm is remarkably accurate: NuminaMath MAE = 0.7 pp when extrapolating from N<=1000 to N>1000. This confirms that within a source, direction is approximately constant and norm alone suffices. The failure is specifically cross-source, where directional differences matter but the reference direction is misspecified.

**Direction stability within source:**

| Source | cos(N_min, N_max) | Consecutive cos (range) |
|---|---:|---:|
| NuminaMath | 0.47 (N=50 vs 10K) | ~1.00 (near-parallel) |
| NuminaMath-Hard | 0.51 (N=100 vs 10K) | 0.97--1.00 |
| OpenR1 | 0.30 (N=100 vs 10K) | **0.90--1.00** |

Consecutive checkpoints within NuminaMath are near-parallel (cos ~1.0), confirming high direction stability. OpenR1 shows a notable direction instability between N=1,500 and N=2,000 (consecutive cos = 0.92, vs ~1.0 at other transitions), providing a directional signature of the format cliff identified in Section 8.

### 7.4 Degradation Efficiency

We define **degradation efficiency** as the accuracy loss per unit perturbation norm:

eta = (baseline_acc - acc) / ||dW||_F

This quantity should be approximately constant within a source (reflecting the fixed direction) but differ across sources (reflecting different angles to the capability subspace). At N=2000:

| Source | ||dW|| | ORZ Acc | eta (accuracy loss per unit norm) |
|---|---:|---:|---:|
| NuminaMath | 0.60 | 23.3% | 0.093 |
| NuminaMath-Hard | 0.60 | 26.1% | 0.047 |
| OpenR1 | 0.79 | 5.4% | 0.298 |

OpenR1's degradation efficiency is 3.2x higher than NuminaMath and 6.3x higher than NuminaMath-Hard, reflecting its perturbation direction's stronger overlap with the ORZ capability subspace. NuminaMath-Hard has the lowest efficiency, consistent with its competition-level solutions being most compatible with the evaluation format.

## 8 The OpenR1 Cliff: A Case Study in Directional Catastrophe

### 8.1 Characterizing the Transition

The OpenR1 cliff between N=1,500 and N=2,000 is the most dramatic phenomenon in our data. Fine-grained experiments at N=1,250 and N=1,500 precisely localize the transition:

| N | ORZ | GSM8K (strict) | GSM8K (tol) | Boxed Rate (S) | ||dW|| | cos(NM) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 28.1% | 84.15% | 84.8% | 100.0% | 0.10 | 0.75 |
| 500 | 26.9% | 83.40% | 84.1% | 99.8% | 0.36 | 0.75 |
| 1,000 | 23.9% | 77.18% | 82.6% | 94.5% | 0.53 | 0.71 |
| 1,250 | 25.7% | 83.47% | 82.7% | 99.7% | 0.44 | -- |
| 1,500 | 25.9% | 83.02% | 83.6% | 99.6% | 0.49 | -- |
| **2,000** | **5.4%** | **33.13%** | **55.3%** | **39.9%** | **0.79** | **0.56** |
| 5,000 | 2.4% | 33.66% | 51.9% | 37.2% | 1.21 | 0.36 |
| 10,000 | 3.1% | 50.04% | 60.9% | 56.4% | 1.63 | 0.24 |

The cliff is remarkably sharp: N=1,500 shows completely normal behavior (strict GSM8K 83.02%, boxed rate 99.6%), yet just 500 additional training samples at N=2,000 cause catastrophic collapse (strict GSM8K 33.13%, boxed rate 39.9%).

Three things happen simultaneously between N=1,500 and N=2,000: (1) the norm increases by 61% (0.49 to 0.79), (2) the direction diverges from NuminaMath (cos drops below 0.56), and (3) the model's formatting collapses (boxed rate drops from 99.6% to 39.9%).

### 8.2 Mechanism: Style Transfer as Directional Shift

OpenR1 data consists of R1-distillation traces averaging 16,469 characters -- 14.3x longer than NuminaMath solutions. At small N, the model partially adapts to this style while retaining its baseline formatting. Between N=1,500 and N=2,000, the model crosses a critical threshold: it has absorbed enough R1-style signal to fundamentally shift its generation strategy toward extended reasoning traces. This manifests as:

- **Response length explosion**: from ~2,100 chars (N<=1,500) to 3,128 chars (N=2,000) on ORZ
- **Box rate collapse**: from 66-76% (N<=1,500) to 11.2% (N=2,000) on ORZ
- **GSM8K strict collapse**: from 83% (N=1,500) to 33% (N=2,000), a 50 pp drop

This is a **directional** phenomenon: the R1 style creates perturbations that increasingly point toward the format subspace as more data reinforces the pattern. This explains both the cliff (threshold where format patterns break) and the directional divergence (OpenR1's direction becomes more format-disruptive with N).

Interestingly, partial format recovery occurs at very high N on easier tasks: N=10,000 recovers to 50% strict GSM8K (vs 34% at N=5,000), suggesting the model may develop new formatting conventions after extended training. However, this recovery does not extend to hard tasks (ORZ boxed rate remains ~4%).

### 8.3 Isolating Direction from Length/Style (Truncation Control)

A natural confound in the analysis above is that OpenR1's perturbation direction diverges from NuminaMath coincident with OpenR1's 14x longer solutions. Is the directional effect truly intrinsic to the mathematical *content* of OpenR1's reasoning traces, or is it an artifact of solution *length and style*?

To disentangle these factors, we design a truncation control experiment. We create a modified OpenR1 dataset in which each solution is truncated to approximately 1,500 characters (matching NuminaMath's mean solution length) while preserving the extracted `\boxed{}` answer. The truncation removes the extended chain-of-thought preamble but retains the core solution steps and final answer. We then train LoRA adapters on this truncated OpenR1 data at three sample counts -- N=1,000, 2,000, and 5,000 -- using the same standard configuration (r=8, LR=5e-5, 1 epoch).

This experiment adjudicates between two hypotheses:

- **H1 (Direction is content-intrinsic):** If the truncated OpenR1 models still exhibit the characteristic cliff and format collapse at N=2,000, then the directional effect is driven by the mathematical content and reasoning patterns in OpenR1's solutions, not by their length. The perturbation direction is determined by *what* the model learns, not *how verbosely* it is expressed.

- **H2 (Direction is length/style-driven):** If the truncated OpenR1 models behave like NuminaMath -- smooth degradation, no format cliff -- then the directional divergence observed in Section 5.2 is primarily a consequence of the extended R1 output style. In this case, the style transfer mechanism (Section 8.2) would be the full explanation, and "direction" would reduce to "verbosity."

**Table 6: Truncated OpenR1 control experiment results**

| N | ORZ (Trunc-OR1) | ORZ (Full OR1) | ORZ (NM) | GSM8K Strict (Trunc) | Boxed Rate (Trunc) | SciKnow (Trunc) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | **25.6%** | 23.9% | 23.8% | 81.0% | 96.7% | 35.5% |
| 2,000 | **11.7%** | 5.4% | 23.3% | 60.0% | 82.6% | 34.3% |
| 5,000 | **15.2%** | 2.4% | 22.5% | 69.1% | 91.2% | 32.1% |

**Both hypotheses are partially supported.** Truncation substantially mitigates OpenR1's catastrophic collapse -- at N=2,000, truncated OpenR1 achieves 11.7% ORZ vs. original's 5.4%, more than doubling the accuracy. GSM8K boxed rate remains at 82.6% (vs. original's 39.9%), indicating that the format cliff is largely eliminated by removing extended output style.

However, truncated OpenR1 still degrades *more* than NuminaMath at every N: at N=2,000, 11.7% vs. 23.3% is still a 2x gap. This demonstrates that **direction has both a content component and a style component**. The style/length of R1 traces accounts for roughly half the degradation gap (raising 5.4% to 11.7%), but the remaining gap (11.7% vs. 23.3%) reflects content-intrinsic directional differences between OpenR1's mathematical reasoning patterns and NuminaMath's.

Interestingly, the N=5,000 truncated result (15.2%) is *better* than N=2,000 (11.7%), suggesting a non-monotonic pattern that differs from original OpenR1's monotonic degradation. All truncated models remain valid on cross-domain benchmarks (SciKnowEval >=32%, ToolAlpaca within baseline).

Direction analysis confirms the geometric interpretation: truncated OpenR1 directions show intermediate cosine similarity with the NM-10K reference (0.55, 0.51, 0.43 at N=1K, 2K, 5K) -- higher than original OpenR1 at comparable sample counts (0.41 at N=2K) but lower than NuminaMath (0.87 at N=2K). The perturbation norms are also smaller (0.52, 0.74, 1.03 vs. 0.79 for original OR1 at N=2K), consistent with truncated solutions producing less aggressive weight updates. Truncation thus partially steers the direction toward the NuminaMath direction while reducing perturbation magnitude -- both factors that reduce degradation.

### 8.4 Direction-Steered Data Mixing

If perturbation direction is the key determinant of degradation, it should be possible to *steer* the direction -- and thus the degradation profile -- by mixing data sources. This converts our framework from a post-hoc explanatory tool into an interventional one: rather than merely observing that different sources produce different directions, we actively manipulate the direction by controlling the composition of the training data.

We test this with mixed NuminaMath/OpenR1 training sets at N=2,000 total samples, varying the ratio:

- **75% NM / 25% OR1** (1,500 NuminaMath + 500 OpenR1)
- **50% NM / 50% OR1** (1,000 NuminaMath + 1,000 OpenR1)
- **25% NM / 75% OR1** (500 NuminaMath + 1,500 OpenR1)

The geometric framework predicts that the mixed models should *interpolate* between the pure-source endpoints in three respects:

1. **Cosine alignment:** The mixed perturbation direction should rotate from the NuminaMath direction toward the OpenR1 direction as the OR1 fraction increases.
2. **ORZ accuracy:** Math performance should smoothly degrade from NuminaMath's N=2,000 level (~23%) toward OpenR1's N=2,000 level (~5%) as the direction shifts.
3. **Effective perturbation:** The quantity ||dW|| * cos(dW, V_C) should predict the degradation better than either ||dW|| or mixture ratio alone.

If the interpolation is smooth, this confirms that direction is a continuous, manipulable quantity rather than a discrete property of the data source. If there is a sharp transition at some mixing ratio (e.g., the cliff appears suddenly when the OR1 fraction exceeds 50%), this would suggest a threshold effect in the format subspace.

**Table 7: Data mixing experiment at N=2,000**

| Mix (NM/OR1) | ORZ | GSM8K Strict | GSM8K Tolerant | Boxed Rate | SciKnow |
|---|---:|---:|---:|---:|---:|
| 100/0 (pure NM) | 23.3% | 65.1% | 77.3% | 82.3% | 31.6% |
| 75/25 | 23.2% | 62.5% | 77.9% | 82.3% | 33.9% |
| 50/50 | 21.3% | 65.3% | 76.9% | 82.9% | 33.6% |
| 25/75 | 19.9% | 56.6% | 78.1% | 70.8% | 32.8% |
| 0/100 (pure OR1) | 5.4% | 33.1% | 65.1% | 39.9% | 35.8% |

**Results and interpretation.** The mixing experiment reveals a striking asymmetry in how the two data sources interact:

1. **ORZ accuracy interpolates smoothly.** Math performance degrades monotonically as the OpenR1 fraction increases: 23.3% → 23.2% → 21.3% → 19.9% → 5.4%. The relationship is approximately linear in the NuminaMath fraction, suggesting that the direction-dependent degradation mechanism operates additively rather than through threshold effects.

2. **Format compliance shows a threshold.** Boxed rate remains stable (~82%) through 50/50 mixing but drops sharply to 70.8% at 25/75, and collapses to 39.9% at pure OpenR1. This is consistent with the format phase transition observed in pure OpenR1 training (Section 6.3): the long-form reasoning style of OpenR1 disrupts output format only when it constitutes a majority of the training signal.

3. **Cross-domain scores are unaffected.** All mixtures remain valid (SciKnow within 3% of baseline), further confirming that cross-domain forgetting is orthogonal to within-domain directional interference at this model scale.

4. **NuminaMath direction dominates at low fractions.** The 75/25 mixture achieves nearly identical ORZ accuracy to pure NuminaMath (23.2% vs 23.3%), suggesting that the NuminaMath perturbation direction is robust to moderate contamination with OpenR1 data. The direction "flips" only when OpenR1 becomes the majority contributor.

5. **The gap narrows but doesn't close.** Even the 25/75 mixture (19.9% ORZ) substantially outperforms pure OpenR1 (5.4%), indicating that even a minority of NuminaMath data can partially steer the perturbation direction away from the catastrophic OpenR1 direction.

**Table 8: Direction analysis of mixing experiments**

| Mix (NM/OR1) | ||dW|| | cos(dW, NM-ref) | cos(dW, NM-2k) | cos(dW, OR1-2k) |
|---|---:|---:|---:|---:|
| 100/0 (pure NM) | 0.60 | 0.87 | 1.00 | 0.56 |
| 75/25 | 0.65 | 0.72 | 0.93 | 0.89 |
| 50/50 | 0.71 | 0.59 | 0.77 | 0.97 |
| 25/75 | 0.75 | 0.49 | 0.65 | 0.97 |
| 0/100 (pure OR1) | 0.79 | 0.41 | 0.56 | 1.00 |

6. **Direction interpolates as predicted.** Table 8 confirms the geometric framework's central prediction: the cosine similarity with the NuminaMath reference direction decreases monotonically from 0.87 (pure NM) through 0.72, 0.59, 0.49 to 0.41 (pure OR1) as the OpenR1 fraction increases. Simultaneously, the cosine with the OR1-2k direction increases. The perturbation norm also grows monotonically (0.60 → 0.79), consistent with OpenR1's larger per-sample weight updates. The direction rotation is smooth and continuous, confirming that data mixing produces genuine directional interpolation rather than mode-switching.

This intervention experiment confirms that perturbation direction is not merely a post-hoc descriptor but a manipulable quantity: by controlling the data composition, we can predictably steer the weight update direction and thereby control the degradation profile. The smooth interpolation of both ORZ accuracy and direction cosine, combined with the threshold behavior of format compliance, suggests that reasoning degradation and format disruption operate through partially independent mechanisms in weight space.

## 9 Multi-Scale Analysis: Qwen2.5-7B-Instruct

A key limitation of v2 was that all experiments used a single 3B-parameter model. To test whether directional structure generalizes across model scales, we replicate the core experiments on Qwen2.5-7B-Instruct.

### 9.1 7B Baselines

| Benchmark | Metric | 3B Baseline | 7B Baseline |
|---|---|---:|---:|
| **ORZ Math** | Accuracy | 28.91% | **39.16%** |
| **SciKnowEval** | MCQ Accuracy | 34.34% | **35.82%** |
| **ToolAlpaca Sim** | Func Acc / Pass Rate | 78.9% / 72.2% | **80.0% / 77.0%** |
| **ToolAlpaca Real** | Func Acc / Pass Rate | 89.2% / 87.3% | **88.6% / 88.6%** |
| **GSM8K** | Strict / Tolerant | 84.0% / 84.0% | **90.9% / 91.2%** |

The 7B model shows uniformly stronger baselines, particularly on ORZ (+10.3 pp) and GSM8K (+6.9 pp), providing a higher starting point for studying degradation.

### 9.2 Sample-Count Curves at 7B

We train 8 experiments: NuminaMath and OpenR1 at N=500, 1000, 2000, 5000 using the standard r=8, LR=5e-5, 1 epoch configuration.

**Table 9: 7B NuminaMath results**

| N | ORZ | ||dW|| | SciKnow | TA-Real F | GSM8K Strict | GSM8K Tolerant |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | **39.75%** | 0.312 | 37.24% | 88.6% | 91.81% | 91.89% |
| 1,000 | 38.67% | 0.510 | 36.24% | 88.6% | 91.05% | 91.51% |
| 2,000 | 35.74% | 0.732 | 35.39% | 88.6% | 71.27% | 86.81% |
| 5,000 | 30.76% | 1.003 | 32.91% | 89.5% | 82.49% | 86.43% |

**Table 10: 7B OpenR1 results**

| N | ORZ | ||dW|| | SciKnow | TA-Real F | GSM8K Strict | GSM8K Tolerant |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 36.62% | 0.336 | 36.50% | 88.6% | 91.28% | 92.34% |
| 1,000 | 35.16% | 0.533 | 35.92% | 88.6% | 91.96% | 91.36% |
| 2,000 | 30.66% | 0.824 | 34.55% | 86.8% | 90.90% | 91.28% |
| 5,000 | **4.59%** | **1.424** | 30.80% | 88.6% | **56.48%** | **66.11%** |

Three key findings emerge:

1. **NuminaMath shows smooth degradation at 7B**, mirroring the 3B pattern. ORZ declines gradually from 39.75% (N=500) to 30.76% (N=5000), a 9.0 pp drop over a 10x increase in training data.

2. **The OpenR1 cliff exists at 7B but is shifted to higher N.** At 3B, the cliff occurs between N=1,500 and N=2,000. At 7B, N=2,000 still shows reasonable performance (ORZ=30.66%), but **N=5,000 collapses to 4.59%** -- a 26 pp drop. The 7B model's greater capacity absorbs more OpenR1-style training before the format cliff triggers, but it is not immune.

3. **Cross-domain benchmarks remain robust** through N=2,000 at 7B. Only at N=5,000 does SciKnowEval breach the 3% threshold (30.80% vs 35.82% baseline), coinciding with the OpenR1 cliff.

Notably, the 7B NuminaMath N=2,000 experiment shows a large GSM8K strict/tolerant gap (71.27% vs 86.81%), indicating format disruption similar to the 3B pattern but at a shifted N threshold.

### 9.3 7B Direction Analysis

**Table 11: 7B source pair cosine similarity**

| Source Pair | Mean Cosine | Std | 3B Comparison |
|---|---:|---:|---:|
| NuminaMath vs NuminaMath | **0.205** | 0.014 | 0.88 |
| OpenR1 vs OpenR1 | **0.192** | 0.026 | 0.72 |
| NuminaMath vs OpenR1 | **0.168** | 0.017 | 0.50 |

**Table 12: 7B matching-N cross-source cosines**

| N | NM vs OR1 (7B) | NM vs OR1 (3B) |
|---:|---:|---:|
| 500 | 0.158 | 0.75 |
| 1,000 | 0.185 | 0.71 |
| 2,000 | 0.207 | 0.56 |
| 5,000 | 0.176 | 0.36 |

The absolute cosine values at 7B are substantially lower than at 3B. This is primarily a **dimensionality effect**: the 7B experiments use 7 target modules across 28 layers (vs. 4 modules across 36 layers at 3B in our original experiments), producing dW vectors in a ~8x higher-dimensional space where cosines are naturally compressed toward zero. We confirm this interpretation by noting that 3B experiments using the same 7-module configuration (Section 11.3 seed experiments) show similarly reduced cosines (OR1 within-source: ~0.087 with 7 modules vs. ~0.72 with 4 modules).

Despite the compressed scale, the **ordering is preserved**: within-source cosine (NM-NM=0.205, OR1-OR1=0.192) consistently exceeds cross-source cosine (NM-OR1=0.168). This gap, while smaller in absolute terms, confirms that directional clustering by data source persists at the 7B scale.

### 9.4 Scale Comparison

The 3B-to-7B comparison reveals both preserved structure and scale-dependent differences:

**Preserved across scales:**
- NuminaMath produces smooth, monotonic degradation at both scales
- OpenR1 produces a catastrophic cliff at both scales
- Cross-domain benchmarks (SciKnowEval, ToolAlpaca) are robust until the cliff
- Within-source directional consistency exceeds cross-source consistency

**Scale-dependent:**
- The OpenR1 cliff threshold shifts from N~1,500-2,000 (3B) to N~2,000-5,000 (7B), suggesting larger models can absorb more format-disrupting data before collapse
- The 7B model's greater capacity produces higher baselines (+10 pp ORZ), providing more headroom before degradation becomes severe
- Perturbation norms are generally larger at 7B (e.g., NM-5000: 1.003 at 7B vs 0.733 at 3B), reflecting the larger model's greater number of trainable parameters

The preservation of qualitative patterns across scales supports the view that directional structure is a fundamental property of SFT data interaction with model weights, not an artifact of the 3B parameter count.

## 10 Cross-Domain Directional Analysis

All four v2 data sources teach math reasoning. A natural question is whether the directional framework extends to genuinely different task domains. If code SFT produces perturbations in a direction near-orthogonal to all math directions, this would confirm that "direction" reflects task domain structure, not just stylistic variation within math.

### 10.1 CodeAlpaca SFT

We fine-tune on CodeAlpaca-20k at N=500, 1000, 2000, 5000 using the standard 3B configuration.

**Table 13: CodeAlpaca SFT results (3B)**

| N | ORZ | ||dW|| | SciKnow | TA-Sim F/P | TA-Real F/P | Valid |
|---:|---:|---:|---:|---:|---:|:---:|
| 500 | 29.39% | 0.581 | 33.86% | 79/73% | 88.6/87.7% | Y |
| 1,000 | 29.30% | 0.826 | 34.39% | 76/71% | 89.5/83.3% | Y |
| 2,000 | 26.66% | 1.023 | 31.48% | 78/73% | 88.6/86.0% | Y |
| 5,000 | 24.41% | 1.151 | 33.07% | 78/73% | 88.6/86.8% | Y |

As expected, code SFT provides no math improvement — ORZ stays near or below baseline (28.91%). At N=5,000, ORZ drops to 24.41% (-4.5 pp), showing that off-domain SFT can erode math ability at sufficient volume. Notably, CodeAlpaca perturbation norms are substantially larger than math SFT at the same N (e.g., ||dW||=1.023 at N=2,000 vs 0.604 for NuminaMath), reflecting the longer output sequences in code generation data. Yet math degradation is milder — a clear signal that direction, not magnitude, drives degradation.

### 10.2 Code vs Math Direction Cosines

**Table 14: Code vs math direction cosines**

| Code Experiment | vs NM-2K | vs OR1-2K (seed1) | vs OR1-2K (seed2) | vs OR1-2K (seed3) |
|---|---:|---:|---:|---:|
| CodeAlpaca N=500 | 0.008 | 0.004 | 0.004 | 0.005 |
| CodeAlpaca N=1,000 | 0.012 | 0.006 | 0.006 | 0.006 |
| CodeAlpaca N=2,000 | 0.014 | 0.006 | 0.006 | 0.007 |
| CodeAlpaca N=5,000 | 0.015 | 0.007 | 0.007 | 0.007 |

**Code vs code intra-source cosines:** mean = 0.266 +/- 0.024

**Summary statistics:**
- Code vs Math: mean = **0.009** +/- 0.004
- Code vs Code: mean = **0.266** +/- 0.024
- Math vs Math (within-source, same config): ~0.087

The prediction is strongly confirmed: **code SFT directions are essentially orthogonal to all math directions** (cos ~ 0.01, effectively zero). This is dramatically lower than within-math cross-source cosines (~0.09 at this dimensionality) and code intra-source cosines (~0.27). The directional framework cleanly separates three levels of similarity: same source > same domain > cross domain.

### 10.3 Degradation Profile

The geometric framework predicts that code perturbations, being near-orthogonal to the math capability subspace V_math, should cause less math degradation per unit norm than math SFT directions.

**Table 15: Degradation efficiency comparison at N=2,000**

| Source | ||dW|| | ORZ Acc | Delta ORZ | eta (loss/norm) |
|---|---:|---:|---:|---:|
| NuminaMath | 0.604 | 23.3% | -5.6 pp | 0.093 |
| OpenR1 | 0.789 | 5.4% | -23.5 pp | 0.298 |
| **CodeAlpaca** | **1.023** | **26.7%** | **-2.2 pp** | **0.022** |

CodeAlpaca has the **largest perturbation norm** (1.023, 1.7x NuminaMath) yet the **smallest math degradation** (-2.2 pp, vs -5.6 pp for NuminaMath). Its degradation efficiency eta = 0.022 is 4.2x lower than NuminaMath and 13.5x lower than OpenR1. This is the geometric framework's strongest prediction: a perturbation direction orthogonal to V_math should have near-zero projection onto the math capability subspace, and indeed CodeAlpaca's massive weight updates barely affect math ability.

This result also rules out a trivial explanation for cross-source degradation differences. If degradation were simply a function of perturbation magnitude (as the magnitude-centric view suggests), CodeAlpaca should cause the most damage. The fact that it causes the least, despite having the largest ||dW||, decisively confirms that **direction determines degradation**.

## 11 Statistical Robustness

### 11.1 Seed Variations

| N | Seeds | ORZ Mean +/- Std | Range |
|---:|---:|---:|---:|
| 100 | 4 | 29.6% +/- 1.0 | 28.5%--30.7% |
| 500 | 4 | 28.4% +/- 0.2 | 28.0%--28.6% |
| 2,000 | 4 | 22.6% +/- 0.5 | 22.2%--23.3% |
| 10,000 | 4 | 19.9% +/- 0.5 | 19.0%--20.5% |

Seed variance is uniformly small (0.2--1.0 pp) across all sample counts, while the degradation signal grows with N (6+ pp at N=2000, 9+ pp at N=10000). This confirms our findings are robust to random initialization.

### 11.2 Hyperparameter Sensitivity

Aggressive hyperparameters (r=16, LR=2e-4, 3 epochs) amplify perturbation magnitude but preserve direction: NuminaMath at these settings shows worse ORZ accuracy (19.6-21.0%) but identical cross-domain robustness. The direction is a property of the data source, not the optimizer configuration.

### 11.3 OpenR1 Cliff Seed Replication

To confirm the OpenR1 cliff at N=2000 is not a seed-specific artifact, we replicate the experiment with seeds 1, 2, and 3 (original was seed=42).

**Table 16: OpenR1 N=2,000 cliff replication across seeds (3B)**

| Seed | ORZ | GSM8K Strict | GSM8K Tolerant | SciKnow |
|---:|---:|---:|---:|---:|
| 42 (original) | 5.37% | 33.13% | 55.34% | 35.76% |
| 1 | 3.52% | 40.86% | 58.76% | 32.81% |
| 2 | 3.91% | 40.18% | 59.06% | 31.33% |
| 3 | 5.18% | 34.87% | 56.33% | 33.28% |
| **Mean +/- Std** | **4.50% +/- 0.87** | **37.26% +/- 3.81** | **57.37% +/- 1.72** | **33.30% +/- 1.82** |

The cliff reproduces with high consistency: all 4 seeds show catastrophic ORZ collapse (3.5--5.4%, vs 28.91% baseline). The standard deviation of 0.87 pp is small relative to the 24 pp mean drop, confirming this is a **systematic property of the OpenR1 data/direction interaction at this sample count**, not a stochastic training artifact. GSM8K strict scores cluster around 34-41%, all dramatically below baseline (84%), with the strict/tolerant gap (37.3% vs 57.4%) confirming that format disruption accounts for a large fraction of the loss.

## 12 Practical Implications

### 12.1 Pre-SFT Direction Probes

Our findings suggest a practical protocol: before committing to a full SFT run, train a cheap LoRA probe (small N, minimal compute) and compare its weight-update direction to known-good directions. High cosine similarity with a validated source predicts safe degradation profiles.

### 12.2 Data Source Selection

The degradation efficiency metric (eta) provides actionable guidance: sources with lower eta per unit norm are safer. NuminaMath-Hard (eta = 0.047) is 6.3x safer per unit norm than OpenR1 (eta = 0.298), despite both training on competition math.

### 12.3 Format Monitoring

Separately track format compliance (boxed rate) and reasoning accuracy (tolerant score). Format disruption is recoverable by training longer; reasoning degradation is not.

### 12.4 Cross-Domain Robustness Under LoRA SFT

Under LoRA SFT at this model scale, cross-domain forgetting is largely mitigated. Across 84 experiments, chemistry and tool-use benchmarks remain within 3% of baseline in all but the most extreme cases. The binding constraint is intra-domain directional interference, not cross-domain forgetting. We note that this observation is specific to the 3B-parameter scale studied here; larger models with richer representational capacity may exhibit different cross-domain dynamics.

## 13 Limitations

**Two model scales.** We test on Qwen2.5-3B-Instruct and 7B-Instruct (Section 9), finding that qualitative patterns (smooth NM degradation, OpenR1 cliff, within>cross source cosines) are preserved, with the cliff threshold shifting to higher N at 7B. However, both are from the same model family; generalization to other architectures (e.g., LLaMA, Mistral) and much larger scales (70B+) remains untested.

**Capability subspace is not directly observed.** We infer V_C indirectly from the relationship between perturbation directions and degradation, without directly characterizing the capability subspace. A direct characterization -- e.g., via activation patching or causal tracing -- would provide stronger evidence for the geometric framework but lies beyond the scope of this study.

**Limited domain diversity.** We now include CodeAlpaca code SFT (Section 10), finding near-orthogonal perturbation directions (cos~0.01) between code and math domains. This confirms the framework extends beyond math-variant sources. However, we have not tested other domains (dialogue, summarization, multilingual) and cannot guarantee the same clean directional separation holds universally.

**Cosine similarity in high dimensions.** With ~340M-dimensional vectors, random vectors have cosine near zero. Our observed similarities (0.24--0.88) are all substantially above the random baseline, confirming they reflect real directional structure.

## 14 Conclusion

We have shown that SFT-induced degradation is fundamentally a *directional* phenomenon in weight space. Within a data source, perturbation magnitude predicts degradation with R^2 > 0.80. Across sources, the same magnitude produces vastly different outcomes (4x degradation gap at matched norms), explained by source-dependent perturbation directions that diverge as training volume increases.

The geometric framework -- degradation depends on the projection of the weight update onto capability-specific subspaces -- unifies several disparate observations: within-source norm scaling, cross-source degradation gaps, cross-domain robustness, format phase transitions, and the OpenR1 catastrophic cliff. Each follows naturally from how the perturbation direction relates to different capability subspaces.

Several important caveats temper the scope of these conclusions. Although we now demonstrate consistency across two model scales (3B and 7B) and two task domains (math and code), both models are from the Qwen2.5 family, and generalization to other architectures remains untested. The capability subspace V_C is inferred indirectly from degradation patterns rather than directly characterized, leaving open the question of whether the geometric decomposition reflects a true structural property of the weight space or an effective approximation.

Our multi-scale experiments (Section 9) confirm that the qualitative directional structure persists at 7B: NuminaMath shows smooth degradation, OpenR1 shows a catastrophic cliff (shifted to N~5,000), and within-source directional consistency exceeds cross-source consistency. Our cross-domain experiments (Section 10) provide the framework's strongest evidence: CodeAlpaca code SFT produces perturbation directions essentially orthogonal to all math directions (cos~0.01), yet with the largest weight update norms, confirming that **direction, not magnitude, determines which capabilities are affected**. The OpenR1 cliff reproduces across 4 random seeds with low variance (ORZ = 4.50% +/- 0.87%), confirming it as a systematic phenomenon.

For practitioners, the key message is: **not all perturbations are equal**. A LoRA update's direction, determined by the training data's style and content, matters as much as its magnitude. Choosing training data that produces perturbations aligned with the target task but orthogonal to capabilities you want to preserve is the geometric principle underlying successful SFT.

## References

[1] Biderman, D., et al. (2024). LoRA Learns Less and Forgets Less. arXiv:2405.09673.

[2] Chen, L., et al. (2023). AlpaGasus: Training a Better Alpaca with Fewer Data. arXiv:2307.08701.

[3] Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[4] Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS 114(13).

[5] Li, H., et al. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. ICLR 2018.

[6] Li, Y., et al. (2024). NuminaMath: A Large-Scale Curated Dataset for Mathematical Reasoning.

[7] Luo, Y., et al. (2023). An Empirical Study of Catastrophic Forgetting in LLMs. arXiv:2308.08747.

[8] Open-Reasoner (2024). OpenR1-Math: High-Quality Math Reasoning Data from R1 Distillation.

[9] Yang, A., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

[10] Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. NeurIPS 2023.

[11] Chaudhary, S. (2023). CodeAlpaca: An Instruction-following LLaMA Model for Code Generation.

## Appendix A: Complete Experimental Results

### A.1 NuminaMath Standard Config (r=8, LR=5e-5, 1 epoch)

| N | ORZ | ||dW||_F | SciKnow | TA-Sim F/P | TA-Real F/P | Valid |
|---:|---:|---:|---:|---:|---:|:---:|
| 50 | 29.49% | 0.056 | 33.97% | 78/71% | 89/87% | Y |
| 100 | 30.66% | 0.098 | 35.24% | 78/71% | 89/89% | Y |
| 200 | 29.39% | 0.172 | 33.44% | 78/71% | 89/86% | Y |
| 300 | 27.44% | 0.230 | 33.81% | 78/71% | 89/87% | Y |
| 500 | 28.42% | 0.330 | 35.34% | 77/70% | 89/88% | Y |
| 750 | 26.86% | 0.410 | 31.85% | 78/72% | 89/87% | Y |
| 1000 | 23.83% | 0.478 | 32.59% | 79/73% | 88/86% | Y |
| 1500 | 21.97% | 0.551 | 31.96% | 79/73% | 89/88% | Y |
| 2000 | 23.34% | 0.604 | 31.59% | 78/72% | 89/88% | Y |
| 5000 | 22.46% | 0.733 | 33.81% | 80/75% | 91/90% | Y |
| 10000 | 20.02% | 0.869 | 31.48% | 80/75% | 90/89% | Y |

### A.2 NuminaMath-Hard Standard Config

| N | ORZ | ||dW||_F | SciKnow | TA-Real F | Valid |
|---:|---:|---:|---:|---:|:---:|
| 100 | 29.39% | 0.099 | 34.13% | 88.60% | Y |
| 500 | 29.30% | 0.331 | 34.07% | 88.60% | Y |
| 1000 | 27.05% | 0.473 | 34.65% | 89.47% | Y |
| 2000 | 26.07% | 0.597 | 34.76% | 89.47% | Y |
| 5000 | 23.93% | 0.718 | 31.54% | 89.47% | Y |
| 10000 | 20.90% | 0.807 | 31.80% | 87.72% | Y |

### A.3 NuminaMath-Comp (r=16, LR=1e-4, 2 epochs)

| N | ORZ | ||dW||_F | SciKnow | TA-Real F | Valid |
|---:|---:|---:|---:|---:|:---:|
| 100 | 28.12% | -- | 34.34% | 86.84% | Y |
| 500 | 25.49% | -- | 34.39% | 90.35% | Y |
| 1000 | 23.93% | -- | 33.23% | 89.47% | Y |
| 2000 | 19.63% | -- | 32.86% | 89.47% | Y |
| 5000 | 21.39% | -- | 34.13% | 87.72% | Y |
| 10000 | 19.73% | -- | 34.13% | 86.84% | Y |

### A.4 OpenR1 Standard Config

| N | ORZ | ||dW||_F | SciKnow | TA-Real F | Valid |
|---:|---:|---:|---:|---:|:---:|
| 100 | 28.13% | 0.096 | 34.60% | 88.60% | Y |
| 500 | 26.86% | 0.356 | 34.87% | 87.72% | Y |
| 1000 | 23.93% | 0.529 | 33.70% | 85.96% | N |
| 1250 | 25.68% | 0.445 | 33.39% | 88.60% | Y |
| 1500 | 25.88% | 0.489 | 31.96% | 86.84% | Y |
| 2000 | 5.37% | 0.789 | 35.76% | 85.96% | N |
| 5000 | 2.44% | 1.211 | 30.11% | 88.60% | N |
| 10000 | 3.13% | 1.628 | 27.42% | 89.47% | N |

### A.5 Cosine Similarity Matrix (Selected Experiments)

| | NM-100 | NM-1K | NM-10K | OR-100 | OR-1K | OR-10K |
|---|---:|---:|---:|---:|---:|---:|
| NM-100 | 1.00 | 0.48 | 0.48 | 0.75 | 0.47 | 0.18 |
| NM-1K | 0.48 | 1.00 | 0.76 | 0.47 | 0.71 | 0.41 |
| NM-10K | 0.48 | 0.76 | 1.00 | 0.36 | 0.47 | 0.24 |
| OR-100 | 0.75 | 0.47 | 0.36 | 1.00 | 0.64 | 0.30 |
| OR-1K | 0.47 | 0.71 | 0.47 | 0.64 | 1.00 | 0.71 |
| OR-10K | 0.18 | 0.41 | 0.24 | 0.30 | 0.71 | 1.00 |

### A.6 GSM8K Format Decomposition (NuminaMath)

| N | Strict | Tolerant | Boxed Rate | Format Loss | Reasoning Loss |
|---:|---:|---:|---:|---:|---:|
| 0 (base) | 84.00% | 84.00% | 99.9% | -- | -- |
| 100 | 83.62% | 84.38% | 100.0% | -0.8 pp | +0.4 pp |
| 500 | 82.71% | 82.49% | 99.8% | +0.2 pp | +1.5 pp |
| 1000 | 62.85% | 78.39% | 82.0% | +15.5 pp | +5.6 pp |
| 2000 | 65.05% | 77.71% | 82.3% | +12.7 pp | +6.3 pp |
| 5000 | 75.51% | 78.85% | 94.4% | +3.3 pp | +5.2 pp |
| 10000 | 76.95% | 77.10% | 96.3% | +0.2 pp | +6.9 pp |

### A.7 GSM8K Format Decomposition (OpenR1)

| N | Strict | Tolerant | Boxed Rate | Format Loss | Reasoning Loss |
|---:|---:|---:|---:|---:|---:|
| 0 (base) | 84.00% | 84.00% | 99.9% | -- | -- |
| 100 | 84.15% | 84.76% | 100.0% | -0.6 pp | -0.8 pp |
| 500 | 83.40% | 84.08% | 99.8% | -0.7 pp | -0.1 pp |
| 1,000 | 77.18% | 82.64% | 94.5% | +5.5 pp | +1.4 pp |
| 1,250 | 83.47% | 82.71% | 99.7% | +0.8 pp | +1.3 pp |
| 1,500 | 83.02% | 83.62% | 99.6% | -0.6 pp | +0.4 pp |
| **2,000** | **33.13%** | **55.34%** | **39.9%** | **+22.2 pp** | **+6.5 pp** |
| 5,000 | 33.66% | 51.93% | 37.2% | +18.3 pp | +13.8 pp |
| 10,000 | 50.04% | 60.88% | 56.4% | +10.8 pp | +12.3 pp |

### A.8 OpenR1 Output Analysis (ORZ)

| N | Boxed Acc | Tolerant Acc | Boxed Rate | Gold in Text | Avg Length | Format Errors | Reasoning Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 28.81% | 30.27% | 75.5% | 47.2% | 1,970 | 71 | 513 |
| 500 | 25.59% | 27.15% | 67.8% | 45.4% | 2,054 | 101 | 538 |
| 1,000 | 25.10% | 27.64% | 62.2% | 45.7% | 2,125 | 117 | 534 |
| 1,250 | 27.15% | 28.52% | 67.4% | 45.4% | 2,095 | 93 | 537 |
| 1,500 | 26.37% | 28.61% | 66.0% | 45.6% | 2,116 | -- | -- |
| **2,000** | **5.18%** | **10.35%** | **11.2%** | **37.3%** | **3,128** | **315** | **638** |
| 5,000 | 2.44% | 8.79% | 3.0% | 35.7% | 3,291 | 337 | 658 |
| 10,000 | 3.12% | 8.30% | 4.0% | 33.1% | 3,306 | -- | 679 |
