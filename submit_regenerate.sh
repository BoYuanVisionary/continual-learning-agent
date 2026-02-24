#!/bin/bash
#SBATCH --job-name=regen_figs_pdf
#SBATCH --output=results/logs/regen_figs_pdf_%j.out
#SBATCH --error=results/logs/regen_figs_pdf_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -n 1

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== Regenerating paper figures ==="
python generate_paper_figures.py

echo ""
echo "=== Regenerating v2 PDF ==="
python md_to_pdf.py results/approach_summary_v2.md results/approach_summary_v2.pdf

echo ""
echo "=== Verifying original files unchanged ==="
md5sum results/approach_summary.md results/approach_summary.pdf 2>/dev/null
echo "=== Done ==="
