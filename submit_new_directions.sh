#!/bin/bash
#SBATCH --job-name=new_directions
#SBATCH --output=results/logs/new_directions_%j.out
#SBATCH --error=results/logs/new_directions_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -n 1

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Computing directions for new checkpoints ==="
python compute_new_directions.py

echo "=== Done ==="
