#!/bin/bash
#SBATCH --job-name=openr1_finegrained
#SBATCH --output=results/logs/openr1_finegrained_%j.out
#SBATCH --error=results/logs/openr1_finegrained_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== OpenR1 Fine-Grained Training (N=1250, 1500) ==="
echo "Start time: $(date)"

# Train OpenR1 at N=1250
echo ""
echo "--- Training OpenR1 N=1250 ---"
python train_sft.py \
    --num_samples 1250 \
    --data_source openr1 \
    --lora_rank 8 \
    --lr 5e-5 \
    --epochs 1 \
    --batch_size 4 \
    --grad_accum 4

# Evaluate OpenR1 N=1250
echo ""
echo "--- Evaluating OpenR1 N=1250 ---"
python eval_finetuned.py \
    --adapter_path checkpoints/sft_openr1_n1250_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1250_r8_lr5e-5_ep1 \
    --batch_size 64

# GSM8K strict for N=1250
echo ""
echo "--- GSM8K (strict) OpenR1 N=1250 ---"
python eval_gsm8k.py \
    --adapter_path checkpoints/sft_openr1_n1250_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1250_r8_lr5e-5_ep1_strict \
    --batch_size 64

# GSM8K tolerant for N=1250
echo ""
echo "--- GSM8K (tolerant) OpenR1 N=1250 ---"
python eval_gsm8k.py \
    --adapter_path checkpoints/sft_openr1_n1250_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1250_r8_lr5e-5_ep1_tolerant \
    --batch_size 64 \
    --tolerant

# Train OpenR1 at N=1500
echo ""
echo "--- Training OpenR1 N=1500 ---"
python train_sft.py \
    --num_samples 1500 \
    --data_source openr1 \
    --lora_rank 8 \
    --lr 5e-5 \
    --epochs 1 \
    --batch_size 4 \
    --grad_accum 4

# Evaluate OpenR1 N=1500
echo ""
echo "--- Evaluating OpenR1 N=1500 ---"
python eval_finetuned.py \
    --adapter_path checkpoints/sft_openr1_n1500_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1500_r8_lr5e-5_ep1 \
    --batch_size 64

# GSM8K strict for N=1500
echo ""
echo "--- GSM8K (strict) OpenR1 N=1500 ---"
python eval_gsm8k.py \
    --adapter_path checkpoints/sft_openr1_n1500_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1500_r8_lr5e-5_ep1_strict \
    --batch_size 64

# GSM8K tolerant for N=1500
echo ""
echo "--- GSM8K (tolerant) OpenR1 N=1500 ---"
python eval_gsm8k.py \
    --adapter_path checkpoints/sft_openr1_n1500_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1500_r8_lr5e-5_ep1_tolerant \
    --batch_size 64 \
    --tolerant

# Output analysis for new checkpoints
echo ""
echo "--- Output Analysis for new checkpoints ---"
python analyze_outputs.py \
    --checkpoints \
    sft_openr1_n1250_r8_lr5e-5_ep1 \
    sft_openr1_n1500_r8_lr5e-5_ep1 \
    --batch_size 64

echo ""
echo "End time: $(date)"
echo "=== Done ==="
