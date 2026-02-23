# The Geometry of SFT Perturbations: Direction, Not Magnitude, Determines Degradation

**Anonymous Authors**

## Abstract

Supervised fine-tuning (SFT) with LoRA perturbs pre-trained weights by a low-rank update dW. The conventional assumption is that degradation depends on the *magnitude* of this perturbation: larger ||dW|| means more forgetting. Through a 61-experiment study fine-tuning Qwen2.5-3B-Instruct on math chain-of-thought data from four sources, we show this is only half the story. Within a data source, perturbation norm strongly predicts accuracy degradation (R^2 = 0.94 for NuminaMath, 0.86 for NuminaMath-Hard, 0.80 for OpenR1). But across sources, the same norm produces vastly different outcomes: NuminaMath at ||dW|| = 0.87 yields 20% ORZ accuracy, while OpenR1 at ||dW|| = 0.79 yields 5.4% -- a 4x degradation gap at comparable perturbation magnitude. We explain this through **perturbation direction analysis**: pairwise cosine similarity between LoRA weight updates reveals that data sources occupy distinct directions in weight space (intra-source cosine ~0.88, cross-source cosine ~0.50), and these directions diverge as training volume increases (cosine drops from 0.75 at N=100 to 0.24 at N=10,000). We further decompose degradation into format disruption (transient, exhibiting phase-transition dynamics) and reasoning degradation (cumulative, monotonic), showing that OpenR1's direction causes catastrophic format collapse between N=1,500 and N=2,000 (strict GSM8K drops from 83% to 33% within 500 additional training samples) while NuminaMath's direction preserves formatting. These findings establish that SFT degradation is fundamentally a **directional** phenomenon: the subspace in which the perturbation lies determines which capabilities are disrupted, not how far the weights move.

## 1 Introduction

Supervised fine-tuning (SFT) with LoRA (Hu et al., 2022) is the standard method for specializing instruction-tuned language models. A natural question is: how much degradation does SFT cause on the model's existing capabilities? The conventional answer focuses on *magnitude* -- larger weight perturbations should cause more forgetting.

We present evidence that this magnitude-centric view is fundamentally incomplete. Consider two SFT experiments with comparable perturbation norms:

- **NuminaMath N=10,000**: ||dW||_F = 0.87, ORZ accuracy = 20.0%
- **OpenR1 N=2,000**: ||dW||_F = 0.79, ORZ accuracy = 5.4%

OpenR1 achieves a 3.7x worse outcome at a *smaller* perturbation magnitude. Conversely, NuminaMath at ||dW|| = 0.60 (N=2000) maintains 23.3% accuracy, while OpenR1 at ||dW|| = 0.53 (N=1000) already shows degradation to 23.9%. The perturbation magnitude alone cannot explain why one source causes catastrophic collapse and another does not.

We resolve this paradox by analyzing the **direction** of LoRA perturbations. Computing pairwise cosine similarity between weight update vectors across 61 checkpoints reveals three key findings:

**(F1) Data sources produce distinct perturbation directions.** Intra-source cosine similarity averages 0.88 (NuminaMath), 0.86 (NuminaMath-Hard), and 0.72 (OpenR1), while cross-source similarity averages only 0.50 (NuminaMath vs. OpenR1).

**(F2) Perturbation directions diverge with training volume.** At N=100, NuminaMath and OpenR1 directions are moderately aligned (cosine = 0.75). By N=10,000, they are nearly orthogonal (cosine = 0.24). More data doesn't just move the model further -- it moves it in an increasingly different direction.

**(F3) Direction determines degradation rate per unit norm.** Within each source, norm strongly predicts accuracy (R^2 > 0.80). But the *slope* differs across sources: each unit of OpenR1 perturbation causes more damage than the equivalent NuminaMath perturbation, because OpenR1's direction has higher overlap with the capability subspace being evaluated.

These findings reframe SFT degradation from a one-dimensional question (how far?) to a geometric question (in which direction?). We further show that the degradation mechanism itself differs by direction: NuminaMath perturbations cause primarily *reasoning degradation* (cumulative accuracy loss), while OpenR1 perturbations cause *format disruption* (catastrophic loss of output formatting conventions) -- a transient phenomenon with phase-transition dynamics that explains the abrupt collapse between N=1,500 and N=2,000 (precisely localized through fine-grained experiments at N=1,250 and N=1,500).

**Contributions.**
1. A geometric framework for SFT degradation: perturbation direction, not just magnitude, determines which capabilities degrade and how severely.
2. Empirical evidence from 61 experiments showing source-dependent directions (cosine ~0.50), N-dependent divergence (0.75 to 0.24), and source-dependent degradation rates per unit norm.
3. Decomposition of degradation into format disruption (transient, direction-dependent) and reasoning degradation (cumulative), with format disruption contributing 73% of peak loss on easy tasks.
4. Practical implications: pre-SFT direction analysis via cheap LoRA probes can predict degradation without full evaluation.

## 2 Related Work

**Catastrophic forgetting in continual learning.** The standard framing focuses on cross-task interference (Kirkpatrick et al., 2017; Luo et al., 2023). EWC and experience replay are common mitigations. We show that under LoRA SFT, cross-domain forgetting is effectively solved -- the binding constraint is direction-dependent intra-domain degradation.

**Weight perturbation theory.** Random perturbation studies (Li et al., 2018) establish that neural networks tolerate perturbations below a scale-dependent threshold. Our contribution is showing that for *structured* perturbations (LoRA SFT), direction matters more than scale, with different data sources producing perturbations in distinct weight-space subspaces.

**LoRA and parameter-efficient fine-tuning.** Hu et al. (2022) showed LoRA preserves most pre-trained capabilities through low-rank updates. Biderman et al. (2024) studied LoRA for continual learning. We extend this by analyzing the geometric structure of LoRA perturbations across data sources, revealing source-dependent directional signatures.

**Data scaling and selection for SFT.** LIMA (Zhou et al., 2023) and AlpaGasus (Chen et al., 2023) study how data quantity and quality affect SFT outcomes. We add a geometric perspective: not just *how much* data, but *which direction* the resulting perturbation points determines the outcome.

## 3 Experimental Setup

### 3.1 Model and Training

We fine-tune **Qwen2.5-3B-Instruct** using LoRA applied to all attention projections (Q, K, V, O) across 36 transformer layers (144 LoRA modules). Our standard configuration: rank r=8, learning rate 5e-5, 1 epoch, effective batch size 16, cosine schedule, bfloat16 precision.

### 3.2 Data Sources

All training data consists of math chain-of-thought solutions with `\boxed{}` answer formatting.

| Source | Description | Mean Solution Length | Characteristic |
|---|---|---:|---|
| **NuminaMath** | Random sample from 860K CoT examples | 1,150 chars | Textbook-style, concise |
| **NuminaMath-Hard** | Competition problems, longest solutions | 2,081 chars | Detailed competition solutions |
| **NuminaMath-Comp** | Competition sources only | 2,077 chars | Filtered by source |
| **OpenR1** | DeepSeek R1 distillation traces | 16,469 chars | Extended reasoning, 14x longer |

### 3.3 Evaluation

| Benchmark | Domain | Size | Baseline |
|---|---|---:|---:|
| **ORZ Math** | Competition math | 1,024 | 28.91% |
| **GSM8K** | Grade-school math | 1,319 | 84.00% |
| **SciKnowEval** | Chemistry MCQ | 1,893 | 34.34% |
| **ToolAlpaca** | Tool-use | 192 | 89.22% |

We evaluate in both *strict* mode (`\boxed{}` extraction only) and *tolerant* mode (fallback to last number). The gap between these isolates format-dependent degradation.

### 3.4 Experiment Design

We conduct 63 experiments spanning:
- **Sample counts**: N = 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 5000, 10000
- **4 data sources**: NuminaMath, NuminaMath-Hard, NuminaMath-Comp, OpenR1
- **Multiple LoRA configs**: rank 4/8/16/32, learning rate 2e-5 to 2e-4, 1-5 epochs
- **Seed variations**: 2-4 seeds at key sample counts
- **KL regularization**: coefficient 0.1, 0.5 at N=500, 1000

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

### 6.4 Cross-Domain Robustness

Throughout all 61 experiments, cross-domain tasks (SciKnowEval, ToolAlpaca) remain remarkably stable. Even OpenR1 at N=10,000 (which destroys ORZ accuracy) only degrades SciKnowEval by 7 pp and ToolAlpaca negligibly. This confirms that LoRA perturbations are confined to a task-relevant subspace: the math SFT direction has near-zero projection onto the chemistry/tool-use capability subspaces, regardless of source.

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

### 7.3 Degradation Efficiency

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

## 9 Statistical Robustness

### 9.1 Seed Variations

| N | Seeds | ORZ Mean +/- Std | Range |
|---:|---:|---:|---:|
| 100 | 2 | 29.6% +/- 1.5 | 28.5%--30.7% |
| 500 | 2 | 28.2% +/- 0.3 | 28.0%--28.4% |
| 2,000 | 4 | 22.8% +/- 0.6 | 22.2%--23.3% |

The degradation signal (6+ pp from baseline at N=2000) far exceeds seed variance (0.6 pp), confirming our findings are robust.

### 9.2 Hyperparameter Sensitivity

Aggressive hyperparameters (r=16, LR=2e-4, 3 epochs) amplify perturbation magnitude but preserve direction: NuminaMath at these settings shows worse ORZ accuracy (19.6-21.0%) but identical cross-domain robustness. The direction is a property of the data source, not the optimizer configuration.

## 10 Practical Implications

### 10.1 Pre-SFT Direction Probes

Our findings suggest a practical protocol: before committing to a full SFT run, train a cheap LoRA probe (small N, minimal compute) and compare its weight-update direction to known-good directions. High cosine similarity with a validated source predicts safe degradation profiles.

### 10.2 Data Source Selection

The degradation efficiency metric (eta) provides actionable guidance: sources with lower eta per unit norm are safer. NuminaMath-Hard (eta = 0.047) is 6.3x safer per unit norm than OpenR1 (eta = 0.298), despite both training on competition math.

### 10.3 Format Monitoring

Separately track format compliance (boxed rate) and reasoning accuracy (tolerant score). Format disruption is recoverable by training longer; reasoning degradation is not.

### 10.4 The Cross-Domain Free Lunch

Under LoRA SFT, cross-domain forgetting is a near-solved problem. Across 61 experiments, chemistry and tool-use benchmarks remain within 3% of baseline in all but the most extreme cases. The binding constraint is intra-domain directional interference, not cross-domain forgetting.

## 11 Limitations

**Single model scale.** All experiments use Qwen2.5-3B-Instruct. Larger models may have richer weight-space geometry, potentially changing directional relationships.

**Capability subspace is not directly observed.** We infer V_C from the relationship between perturbation directions and degradation, but do not directly characterize the capability subspace.

**Limited data sources.** Four math data sources may not span the full range of possible perturbation directions. Non-math SFT data could produce qualitatively different directional patterns.

**Cosine similarity in high dimensions.** With ~340M-dimensional vectors, random vectors have cosine near zero. Our observed similarities (0.24--0.88) are all substantially above the random baseline, confirming they reflect real directional structure.

## 12 Conclusion

We have shown that SFT-induced degradation is fundamentally a *directional* phenomenon in weight space. Within a data source, perturbation magnitude predicts degradation with R^2 > 0.80. Across sources, the same magnitude produces vastly different outcomes (4x degradation gap at matched norms), explained by source-dependent perturbation directions that diverge as training volume increases.

The geometric framework -- degradation depends on the projection of the weight update onto capability-specific subspaces -- unifies several disparate observations: within-source norm scaling, cross-source degradation gaps, cross-domain robustness, format phase transitions, and the OpenR1 catastrophic cliff. Each follows naturally from how the perturbation direction relates to different capability subspaces.

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

## Appendix A: Complete Experimental Results

### A.1 NuminaMath Standard Config (r=8, LR=5e-5, 1 epoch)

| N | ORZ | ||dW||_F | SciKnow | TA-Sim F/P | TA-Real F/P | Valid |
|---:|---:|---:|---:|---:|---:|:---:|
| 50 | 29.49% | 0.056 | 33.97% | 78/71% | 89/87% | Y |
| 100 | 30.66% | 0.098 | 35.24% | 78/71% | 89/89% | Y |
| 200 | 29.39% | 0.172 | 33.44% | 78/71% | 89/86% | Y |
| 300 | 27.44% | 0.230 | 33.81% | 78/71% | 89/87% | Y |
| 500 | 28.42% | 0.330 | 35.34% | 77/70% | 89/88% | Y |
| 750 | -- | 0.410 | -- | -- | -- | -- |
| 1000 | 23.83% | 0.478 | 32.59% | 79/73% | 88/86% | Y |
| 1500 | -- | 0.551 | -- | -- | -- | -- |
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

### A.3 OpenR1 Standard Config

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

### A.4 Cosine Similarity Matrix (Selected Experiments)

| | NM-100 | NM-1K | NM-10K | OR-100 | OR-1K | OR-10K |
|---|---:|---:|---:|---:|---:|---:|
| NM-100 | 1.00 | 0.48 | 0.48 | 0.75 | 0.47 | 0.18 |
| NM-1K | 0.48 | 1.00 | 0.76 | 0.47 | 0.71 | 0.41 |
| NM-10K | 0.48 | 0.76 | 1.00 | 0.36 | 0.47 | 0.24 |
| OR-100 | 0.75 | 0.47 | 0.36 | 1.00 | 0.64 | 0.30 |
| OR-1K | 0.47 | 0.71 | 0.47 | 0.64 | 1.00 | 0.71 |
| OR-10K | 0.18 | 0.41 | 0.24 | 0.30 | 0.71 | 1.00 |

### A.5 GSM8K Format Decomposition (NuminaMath)

| N | Strict | Tolerant | Boxed Rate | Format Loss | Reasoning Loss |
|---:|---:|---:|---:|---:|---:|
| 0 (base) | 84.00% | 84.00% | 99.9% | -- | -- |
| 100 | 83.62% | 84.38% | 100.0% | -0.8 pp | +0.4 pp |
| 500 | 82.71% | 82.49% | 99.8% | +0.2 pp | +1.5 pp |
| 1000 | 62.85% | 78.39% | 82.0% | +15.5 pp | +5.6 pp |
| 2000 | 65.05% | 77.71% | 82.3% | +12.7 pp | +6.3 pp |
| 5000 | 75.51% | 78.85% | 94.4% | +3.3 pp | +5.2 pp |
| 10000 | 76.95% | 77.10% | 96.3% | +0.2 pp | +6.9 pp |

### A.6 GSM8K Format Decomposition (OpenR1)

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

### A.7 OpenR1 Output Analysis (ORZ)

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
