# Research Agent Sandbox

## Goal

**Study how SFT sample count affects math performance vs. catastrophic forgetting in Qwen2.5-3B-Instruct.**

This is a **continual learning research study**, not a pure optimization task. The core question:

> As you increase the number of SFT samples for math reasoning, how does ORZ math accuracy improve, and at what point does the model start forgetting its chemistry (SciKnowEval) and tool-use (ToolAlpaca) capabilities?

### What to produce

A **sample-count curve** mapping N (number of SFT samples) to:
- ORZ math accuracy (primary metric)
- SciKnowEval accuracy (forgetting indicator)
- ToolAlpaca func accuracy / pass rate (forgetting indicator)

Run SFT experiments at these sample counts: **100, 500, 1000, 2000, 5000, 10000**.

### Validity and the forgetting frontier

A run is **"valid"** if SciKnowEval and ToolAlpaca scores stay within 3% of baseline. But **save ALL results** — even invalid runs where forgetting exceeds 3% — because the full curve reveals the forgetting behavior. The agent should focus exploration effort on the **continual learning frontier**: the region where models are near or crossing the 3% forgetting threshold. More data points in the valid region are especially valuable.

### Freedom to explore

The agent has full freedom over:
- **Data sources** — NuminaMath-CoT, OpenR1 (R1 distillation), self-generated (rejection sampling), or other online math CoT datasets
- **Data quality/selection** — random sampling, difficulty-based curation, shortest-solution filtering, etc.
- **Hyperparameters** — LoRA rank, learning rate, epochs, etc.
- **Any SFT variant** — standard SFT, multi-task regularization, replay, etc.

The key constraint is: at each sample count, find the **best achievable math accuracy** that doesn't break the other tasks (and record what happens when it does break).

### Reporting

Maintain clear, structured reporting throughout. After each experiment, append results to a structured log so progress is always visible. Save all checkpoints and eval results — even "failed" ones where forgetting > 3% — because understanding the full behavior is the point.

## Environment

- **Conda env**: `qwen25` (activate with `conda activate qwen25`)
- **GPUs**: H200 nodes available via SLURM. Multiple nodes can be allocated in parallel.
- **Time budget**: ~8 hours per SLURM job, but multiple jobs run in parallel
- **Internet**: Available (see restrictions below)
- **Model**: `Qwen/Qwen2.5-3B-Instruct` (~6GB FP16), loaded via HuggingFace in `utils.py`
- **Working directory**: Use the repository root (where this CLAUDE.md lives). In SLURM scripts, use `$SLURM_SUBMIT_DIR` or set it explicitly based on where you submit from.

## What You Have

### Data (`data/`)
All data is local JSON. This is your **train set only** — there is a held-out test set you cannot see that will be used for final evaluation.

| Dataset | File | Examples | Description |
|---|---|---|---|
| ORZ Math | `data/orz/train.json` | 65,199 | Math problems. Each entry has `"0"` (question) and `"1"` (ground truth answer). Answers are raw (NOT wrapped in `\boxed{}`). Formats: plain numbers, LaTeX fracs, expressions. |
| SciKnowEval | `data/sciknoweval/train.json` | 1,893 | Chemistry L3 MCQ-4-choices only (filling type removed) |
| ToolAlpaca | `data/toolalpaca/simulated_train.json` | 90 | Simulated tool-use queries |
| ToolAlpaca | `data/toolalpaca/real_train.json` | 102 | Real tool-use queries |

### Eval Scripts
- `eval_orz.py` — Evaluates math accuracy (extract `\boxed{}` answer, compare with gold)
- `eval_sciknoweval.py` — Evaluates Chemistry MCQ accuracy
- `eval_toolalpaca.py` — Evaluates tool-use function accuracy + pass rate
- All support: `--test` (small subset), `--batch_size N`, `--no_resume`

### Utilities (`utils.py`)
- `load_model()` → returns `(model, tokenizer)` on GPU
- `generate_responses_batch(model, tokenizer, messages_list)` → returns `(responses, token_counts)`
- `extract_boxed_answer(text)` — extracts `\boxed{...}` from model output
- Checkpoint/resume helpers: `load_checkpoint()`, `save_checkpoint()`
- `math_grader.py` — `math_equal(pred, gold)` for robust math answer comparison

## Baselines (current Qwen2.5-3B-Instruct, no modifications)

| Benchmark | Metric | Baseline |
|---|---|---|
| **ORZ Math** | Accuracy | **28.91%** (on 1024 samples) |
| **SciKnowEval** | MCQ Accuracy | **34.34%** (650/1893) |
| **ToolAlpaca Simulated** | Func Accuracy / Pass Rate | 78.89% / 72.22% |
| **ToolAlpaca Real** | Func Accuracy / Pass Rate | 89.22% / 87.25% |

## Rules

1. **Primary objective**: Map the SFT sample-count vs. math-accuracy vs. forgetting curve
2. **Sample counts to test**: 100, 500, 1000, 2000, 5000, 10000
3. **Validity threshold**: A run is "valid" if SciKnowEval and ToolAlpaca degrade by ≤3% from baseline. But **save all results regardless** — invalid runs are still informative.
4. **Focus on the continual learning frontier**: Spend more effort exploring the valid region and the transition zone where forgetting starts. More data points here = better understanding.
5. **Save everything**: Every experiment's checkpoint under `checkpoints/`, every eval result under `results/`. Use clear naming: e.g., `sft_numinamath_n1000_r16/` so experiments are traceable.
6. **Structured progress log**: After each experiment completes, append to `results/experiment_log.json` (or similar) with: sample count, data source, hyperparams, all three benchmark scores, and whether it's valid.
7. **Only use train data** — do not download or access any other splits of the three benchmark datasets
8. **You can use any SFT approach**: different data sources, LoRA configs, multi-task regularization, replay, etc. Internet is available for downloading training data (see restrictions below).
9. **You have access to H200 GPU nodes via SLURM** — multiple parallel jobs allowed. Maximize parallelism.
10. **Document findings** — save a final summary to `results/approach_summary.md` with the full curve, key insights about forgetting dynamics, and what worked best at each sample count.

## STRICTLY FORBIDDEN — Data Leakage Prevention

**NEVER search for, fetch, download, or access any information about the following three datasets via the internet, web search, or any other means:**

- **`Open-Reasoner-Zero/orz_math_72k_collection_extended`** (HuggingFace)
- **`hicai-zju/SciKnowEval`** (HuggingFace)
- **`tangqiaoyu/ToolAlpaca`** (GitHub)

The private test set is part of these datasets. Searching for or fetching them (even to "look at documentation") constitutes data leakage and invalidates the experiment. You already have all the train data you need locally in `data/`. Use only that.

You MAY download other datasets, papers, code, or resources from the internet — just not these three.

## SLURM — Running Jobs on Compute Nodes

This is an HPC cluster managed by SLURM. The login node has no GPUs. To run GPU work, submit jobs to compute nodes.

### Interactive allocation (for debugging)
```bash
salloc -t8:00:00 --gres=gpu:h200:2 --mem=128G -c 24
# Then run commands interactively on the allocated node
```

### Batch jobs (for parallel experiments)
Write a self-contained script, then submit with `sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=experiment_name
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd $SLURM_SUBMIT_DIR

python your_script.py
```

Submit: `sbatch script.sh`
Check status: `squeue -u $USER`
Cancel: `scancel <job_id>`
View output: `tail -f results/logs/<job_name>_<job_id>.out`

### Parallel experiment pattern
Multiple `sbatch` jobs can run simultaneously on separate nodes. Each agent/experiment should:
1. Write a **self-contained** Python script (all logic in one file, no interactive steps)
2. Save results/checkpoints to `results/` or `checkpoints/` on shared storage
3. Submit via `sbatch` with appropriate resources
4. Monitor via log files

Key constraint: each submitted script must be fully autonomous — it cannot be interactively controlled after submission. Design scripts to save intermediate checkpoints and logs.

## Research Pointers

Relevant work for ideas (not prescriptive — read critically):

### Continual Learning & Catastrophic Forgetting
- **LoRA as a forgetting mitigation** — small rank = less perturbation to base weights, but also less capacity for new knowledge. How does rank interact with sample count?
- **Multi-task regularization / replay** — mixing in SciKnowEval + ToolAlpaca data during math SFT. Does this shift the forgetting frontier?
- **Elastic Weight Consolidation (EWC)**, **experience replay**, and other CL techniques may be worth exploring

### Data Sources & Quality
- **NuminaMath-CoT** — large math CoT dataset, verified solutions
- **OpenR1-Math-220k** — DeepSeek R1 distillation traces (high-quality reasoning)
- **Self-generated** — rejection sampling from the base model itself
- **Data selection matters** — at small N, which 1000 samples you pick may matter more than the training recipe
- External math CoT datasets are allowed to download

### Key Considerations
- `math_equal()` in `math_grader.py` can verify answer correctness (useful for filtering/selection)
- The eval scripts use single-pass inference — everything must be in the weights
- The model is small (3B) — forgetting dynamics may differ from larger models
- LoRA rank, learning rate, and number of epochs all interact with sample count

**This is a research study, not just engineering.** The goal is to understand the relationship between SFT data volume and forgetting. Think critically about why certain sample counts cause forgetting and others don't. Look for phase transitions, understand the mechanisms, and propose explanations.

### Important Technical Notes

- `utils.py` uses `torch.bfloat16` and `attn_implementation="sdpa"` for inference
- `generate_responses_batch()` default `max_new_tokens=2048`
- Flash Attention cannot be built (CUDA 13.0 vs PyTorch 12.8 mismatch), but SDPA works fine
- The eval scripts use naive single-pass inference — so inference-time tricks (majority voting, better prompts) won't help at test time. Everything must be baked into the weights.
- `math_grader.py` (`math_equal`) does string + numeric + symbolic (sympy) comparison — robust enough to use as GRPO reward signal

