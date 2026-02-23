#!/bin/bash
#SBATCH --job-name=finegrained_eval
#SBATCH --output=results/logs/finegrained_eval_%j.out
#SBATCH --error=results/logs/finegrained_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Fine-grained OpenR1 Evaluation ==="
echo "Start time: $(date)"

for N in 1250 1500; do
    CKPT="checkpoints/sft_openr1_n${N}_r8_lr5e-5_ep1/final_adapter"

    echo ""
    echo "--- Full eval: OpenR1 N=$N ---"
    python eval_finetuned.py \
        --adapter_path "$CKPT" \
        --experiment_name "sft_openr1_n${N}_r8_lr5e-5_ep1" \
        --batch_size 64

    echo ""
    echo "--- GSM8K strict: OpenR1 N=$N ---"
    python eval_gsm8k.py \
        --adapter_path "$CKPT" \
        --experiment_name "sft_openr1_n${N}_r8_lr5e-5_ep1_strict" \
        --batch_size 64

    echo ""
    echo "--- GSM8K tolerant: OpenR1 N=$N ---"
    python eval_gsm8k.py \
        --adapter_path "$CKPT" \
        --experiment_name "sft_openr1_n${N}_r8_lr5e-5_ep1_tolerant" \
        --batch_size 64 \
        --tolerant

    echo ""
    echo "--- Output analysis: OpenR1 N=$N ---"
    python analyze_outputs.py \
        --checkpoints "sft_openr1_n${N}_r8_lr5e-5_ep1" \
        --batch_size 64
done

echo ""
echo "End time: $(date)"
echo "=== Done ==="
