#!/bin/bash
# Run 7B baseline evaluation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Submitting 7B Baseline Evaluation ==="

cat > /tmp/slurm_7b_baseline.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=7b_baseline
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== 7B Baseline Evaluation ==="
python eval_7b_baselines.py --batch_size 32
echo "=== Done ==="
SLURM_EOF

BASELINE_JOB=$(sbatch /tmp/slurm_7b_baseline.sh 2>&1 | grep -oP '\d+')
echo "  Submitted 7b_baseline -> Job ${BASELINE_JOB}"
echo ""
echo "After completion, check results/baseline_7b_eval.json"
echo "Then update BASELINES_7B in eval_finetuned.py before running 7B experiments"
