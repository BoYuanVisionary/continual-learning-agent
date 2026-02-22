#!/bin/bash
# Round 2: Multiple strategies to improve ORZ accuracy
# Round 1 showed NuminaMath SFT hurts ORZ with LR=2e-4, epoch=3, rank=16

cd /home/hice1/byuan48/research/research_agent
mkdir -p results/logs

echo "=== Round 2: Exploring hyperparameters and data strategies ==="
echo ""

# Strategy A: Very conservative - minimal perturbation
# LR=5e-5, 1 epoch, rank=8
echo "--- Strategy A: Conservative (LR=5e-5, ep=1, r=8) ---"
for N in 100 500 1000 2000 5000 10000; do
    JOB_NAME="r2a_n${N}"
    sbatch --job-name=${JOB_NAME} \
           --output=results/logs/${JOB_NAME}_%j.out \
           --error=results/logs/${JOB_NAME}_%j.err \
           run_experiment.sh ${N} 8 5e-5 1 numinamath
    echo "  Submitted N=${N}"
done

# Strategy B: Competition math only
# LR=1e-4, 2 epochs, rank=16
echo ""
echo "--- Strategy B: Competition math (LR=1e-4, ep=2, r=16) ---"
for N in 100 500 1000 2000 5000 10000; do
    JOB_NAME="r2b_n${N}"
    sbatch --job-name=${JOB_NAME} \
           --output=results/logs/${JOB_NAME}_%j.out \
           --error=results/logs/${JOB_NAME}_%j.err \
           run_experiment.sh ${N} 16 1e-4 2 numinamath_comp
    echo "  Submitted N=${N}"
done

# Strategy C: Ultra conservative
# LR=2e-5, 1 epoch, rank=4
echo ""
echo "--- Strategy C: Ultra conservative (LR=2e-5, ep=1, r=4) ---"
for N in 100 500 1000 2000 5000 10000; do
    JOB_NAME="r2c_n${N}"
    sbatch --job-name=${JOB_NAME} \
           --output=results/logs/${JOB_NAME}_%j.out \
           --error=results/logs/${JOB_NAME}_%j.err \
           run_experiment.sh ${N} 4 2e-5 1 numinamath
    echo "  Submitted N=${N}"
done

echo ""
echo "Total: 18 jobs submitted (3 strategies x 6 sample counts)"
echo "Monitor: squeue -u \$USER | grep r2"
