#!/bin/bash
#SBATCH --job-name=gsm8k_fill
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

# Evaluate GSM8K on existing checkpoints that are missing GSM8K eval
for N in 50 200 300; do
    CKPT="checkpoints/sft_numinamath_n${N}_r8_lr5e-5_ep1/final_adapter"
    NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1"
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating ${NAME} on GSM8K (strict) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64
        echo ""
        echo "=== Evaluating ${NAME} on GSM8K (tolerant) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64 --tolerant
        echo ""
    else
        echo "Skipping ${NAME}: checkpoint not found"
    fi
done

echo "All fill GSM8K evaluations complete!"
