#!/bin/bash
#SBATCH --job-name=gsm8k_hard_eval
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

# Evaluate GSM8K on NuminaMath-Hard checkpoints (r=8, lr=5e-5, ep=1)
for N in 100 500 1000 2000 5000 10000; do
    CKPT="checkpoints/sft_numinamath_hard_n${N}_r8_lr5e-5_ep1/final_adapter"
    NAME="sft_numinamath_hard_n${N}_r8_lr5e-5_ep1"
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating ${NAME} on GSM8K ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64
        echo ""
        echo "=== Evaluating ${NAME} on GSM8K (tolerant) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64 --tolerant
        echo ""
    else
        echo "Skipping ${NAME}: checkpoint not found"
    fi
done

# Also evaluate OpenR1 checkpoints at key N values
for N in 100 500 1000 2000; do
    CKPT="checkpoints/sft_openr1_n${N}_r8_lr5e-5_ep1/final_adapter"
    NAME="sft_openr1_n${N}_r8_lr5e-5_ep1"
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating ${NAME} on GSM8K ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64
        echo ""
        echo "=== Evaluating ${NAME} on GSM8K (tolerant) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64 --tolerant
        echo ""
    else
        echo "Skipping ${NAME}: checkpoint not found"
    fi
done

echo "All GSM8K evaluations complete!"
