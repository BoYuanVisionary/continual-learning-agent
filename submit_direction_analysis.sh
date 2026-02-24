#!/bin/bash
#SBATCH --job-name=direction_analysis
#SBATCH --output=results/logs/direction_analysis_%j.out
#SBATCH --error=results/logs/direction_analysis_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH -N 1
#SBATCH -n 1

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Running LoRA direction analysis ==="
python analyze_lora_directions.py

echo "=== Direction analysis complete ==="
echo "Results saved to results/lora_direction_analysis.json"
