#!/bin/bash
#SBATCH --job-name=rejection_sample
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

# Clean up failed checkpoint
rm -f data/orz_self/rejection_checkpoint.json.tmp

# Run rejection sampling: 15000 problems, 8 attempts each, to get enough data
# Expected yield: ~29% solvable * P(at least 1 correct in 8 attempts) ≈ ~3000-5000 examples
python rejection_sample.py \
    --max_problems 15000 \
    --num_attempts 8 \
    --batch_size 32 \
    --temperature 0.7 \
    --top_p 0.8 \
    --checkpoint_every 50 \
    --seed 42

echo "Rejection sampling complete!"
echo "Output: data/orz_self/train_rejection.json"
wc -l data/orz_self/train_rejection.json 2>/dev/null || echo "No output file found"
