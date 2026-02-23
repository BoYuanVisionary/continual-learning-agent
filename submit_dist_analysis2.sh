#!/bin/bash
#SBATCH --job-name=dist_analysis2
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Distribution Analysis (fixed OpenR1 loader) ==="
python analyze_distributions.py

echo "Done!"
