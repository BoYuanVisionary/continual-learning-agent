#!/bin/bash
#SBATCH --job-name=lora_analysis
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== LoRA Weight Analysis ==="
python analyze_lora_weights.py

echo "Done!"
