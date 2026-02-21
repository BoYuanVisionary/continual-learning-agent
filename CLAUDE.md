# Research Agent Sandbox

## Goal

Improve **Qwen2.5-3B-Instruct**'s math reasoning performance (measured on the ORZ math benchmark) while **maintaining** performance on two other benchmarks (SciKnowEval, ToolAlpaca). You have FULL freedom to use ANY approach you can think of.

## Environment

- **Conda env**: `qwen25` (activate with `conda activate qwen25`)
- **GPUs**: H200 nodes available via SLURM. Multiple nodes can be allocated in parallel.
- **Time budget**: ~8 hours per SLURM job, but multiple jobs run in parallel
- **Internet**: Available (see restrictions below)
- **Model**: `Qwen/Qwen2.5-3B-Instruct` (~6GB FP16), loaded via HuggingFace in `utils.py`
- **Working directory**: `/storage/ice-shared/ae3530b/byuan48/research_agent`

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

1. **Primary objective**: Maximize ORZ math accuracy
2. **Constraint**: SciKnowEval and ToolAlpaca scores must not degrade by more than 3% from baseline
3. **Only use train data** — do not download or access any other splits of these datasets
4. **Final checkpoint**: Save your best model checkpoint under `checkpoints/` — this will be tested manually against a private held-out test set to measure math, chemistry, and tool-calling ability
5. **Document what you did** — save a summary of your approach and results to `results/approach_summary.md`
6. **You can use any approach**: fine-tuning, LoRA, prompt tuning, data augmentation, distillation, downloading additional training data from the internet, etc. Internet is available — use it freely.
7. **You have access to H200 GPU nodes via SLURM** — multiple parallel jobs allowed. Plan your experiments to maximize parallelism.

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
cd /storage/ice-shared/ae3530b/byuan48/research_agent

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

Relevant recent work to draw ideas from (not prescriptive — read critically and decide what fits):

- **Open-Reasoner-Zero** (https://arxiv.org/abs/2503.24290) — scaling RL on base models, open-source
- **GRPO** (https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization, memory-efficient RL
- **TRL GRPOTrainer** (https://huggingface.co/docs/trl/grpo_trainer) — HuggingFace implementation, works with Qwen2.5
- SFT, LoRA, DPO/ORPO, rejection sampling, curriculum learning, multi-task training are all viable
- External math CoT datasets are allowed to download

Key considerations:
- Pure SFT risks catastrophic forgetting — be careful about SciKnowEval/ToolAlpaca degradation
- `math_equal()` in `math_grader.py` can serve as a verifiable reward signal (string + numeric + symbolic comparison)
- The eval scripts use naive single-pass inference — inference-time tricks won't help at test time, everything must be in the weights
- The model is small (3B) — some approaches that work at 32B+ scale may not transfer directly

**This is a research task, not just engineering.** Don't just replicate existing methods. Think critically about what works, what doesn't, and why. Propose novel ideas — new reward designs, creative data strategies, hybrid training pipelines, or approaches that don't exist in the literature yet. Existing papers are starting points for inspiration, not recipes to follow. Run real experiments to validate or invalidate your hypotheses.

### Important Technical Notes

- `utils.py` uses `torch.bfloat16` and `attn_implementation="sdpa"` for inference
- `generate_responses_batch()` default `max_new_tokens=2048`
- Flash Attention cannot be built (CUDA 13.0 vs PyTorch 12.8 mismatch), but SDPA works fine
- The eval scripts use naive single-pass inference — so inference-time tricks (majority voting, better prompts) won't help at test time. Everything must be baked into the weights.
- `math_grader.py` (`math_equal`) does string + numeric + symbolic (sympy) comparison — robust enough to use as GRPO reward signal

## Evaluation

The three eval scripts — `eval_orz.py`, `eval_sciknoweval.py`, `eval_toolalpaca.py` — are the exact same scripts that will be used to evaluate your final checkpoint against the private held-out test set. Use them freely and wisely to measure your progress. All support `--test` (small subset), `--batch_size N`, and `--no_resume`. Results are saved to `results/` as JSON.

Baseline results (unmodified model) are already available:
- `results/orz_results.json` — ORZ math baseline
- `results/sciknoweval_results.json` — SciKnowEval baseline
- `results/toolalpaca_train_results.json` — ToolAlpaca baseline
