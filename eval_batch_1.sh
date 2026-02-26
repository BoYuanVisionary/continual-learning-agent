#!/bin/bash
#SBATCH --job-name=eval_batch_1
#SBATCH --output=/storage/ice-shared/ae3530b/byuan48/research/research_agent/results/logs/eval_batch_1_%j.out
#SBATCH --error=/storage/ice-shared/ae3530b/byuan48/research/research_agent/results/logs/eval_batch_1_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Evaluating sft_numinamath_n100_r8_lr5e-5_ep1_seed2 ==="
python eval_finetuned.py \
    --adapter_path checkpoints/sft_numinamath_n100_r8_lr5e-5_ep1_seed2/final_adapter \
    --experiment_name sft_numinamath_n100_r8_lr5e-5_ep1_seed2 \
    --batch_size 32
echo "=== Done: sft_numinamath_n100_r8_lr5e-5_ep1_seed2 ==="

echo "=== Evaluating sft_numinamath_n100_r8_lr5e-5_ep1_seed3 ==="
python eval_finetuned.py \
    --adapter_path checkpoints/sft_numinamath_n100_r8_lr5e-5_ep1_seed3/final_adapter \
    --experiment_name sft_numinamath_n100_r8_lr5e-5_ep1_seed3 \
    --batch_size 32
echo "=== Done: sft_numinamath_n100_r8_lr5e-5_ep1_seed3 ==="

echo "=== Evaluating sft_numinamath_n1500_r8_lr5e-5_ep1 ==="
python eval_finetuned.py \
    --adapter_path checkpoints/sft_numinamath_n1500_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_numinamath_n1500_r8_lr5e-5_ep1 \
    --batch_size 32
echo "=== Done: sft_numinamath_n1500_r8_lr5e-5_ep1 ==="

echo "=== Evaluating sft_numinamath_n500_r8_lr5e-5_ep1_kl0p5 ==="
python eval_finetuned.py \
    --adapter_path checkpoints/sft_numinamath_n500_r8_lr5e-5_ep1_kl0p5/final_adapter \
    --experiment_name sft_numinamath_n500_r8_lr5e-5_ep1_kl0p5 \
    --batch_size 32
echo "=== Done: sft_numinamath_n500_r8_lr5e-5_ep1_kl0p5 ==="

