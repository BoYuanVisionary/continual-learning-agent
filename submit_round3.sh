#!/bin/bash
# Round 3: Optimize the winning strategy
# Best so far: N=100, LR=5e-5, ep=1, r=8 → 30.66% ORZ
# Try: different LR/epoch combos at small N, hard problems, OpenR1

cd /home/hice1/byuan48/research/research_agent
mkdir -p results/logs

echo "=== Round 3: Targeted optimization ==="
echo ""

# Strategy D: Fine-tune around N=100 sweet spot with varied LR
echo "--- Strategy D: N=100 with varied hyperparams ---"
# Higher LR with fewer steps
sbatch --job-name=r3d_n100_lr1e4 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 100 8 1e-4 1 numinamath
# Lower LR with more epochs
sbatch --job-name=r3d_n100_lr5e5_ep5 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 100 8 5e-5 5 numinamath
# Even lower LR
sbatch --job-name=r3d_n100_lr1e5 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 100 8 1e-5 3 numinamath
# Higher rank with low LR
sbatch --job-name=r3d_n100_r32 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 100 32 5e-5 1 numinamath
echo "  4 jobs submitted"

# Strategy D2: Explore N=50 and N=200 around the sweet spot
echo ""
echo "--- Strategy D2: Near-100 sample counts ---"
sbatch --job-name=r3d_n50 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 50 8 5e-5 1 numinamath
sbatch --job-name=r3d_n200 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 200 8 5e-5 1 numinamath
sbatch --job-name=r3d_n300 --output=results/logs/%x_%j.out --error=results/logs/%x_%j.err \
    run_experiment.sh 300 8 5e-5 1 numinamath
echo "  3 jobs submitted"

# Strategy E: Hard NuminaMath (competition, longest solutions)
echo ""
echo "--- Strategy E: Hard NuminaMath problems ---"
for N in 100 500 1000 2000 5000 10000; do
    JOB_NAME="r3e_n${N}"
    sbatch --job-name=${JOB_NAME} --output=results/logs/${JOB_NAME}_%j.out --error=results/logs/${JOB_NAME}_%j.err \
        run_experiment.sh ${N} 8 5e-5 1 numinamath_hard
    echo "  Submitted N=${N}"
done

# Strategy F: OpenR1 data (if available)
echo ""
echo "--- Strategy F: OpenR1 data ---"
if [ -f "data/openr1/openr1_math.json" ]; then
    for N in 100 500 1000 2000 5000 10000; do
        JOB_NAME="r3f_n${N}"
        sbatch --job-name=${JOB_NAME} --output=results/logs/${JOB_NAME}_%j.out --error=results/logs/${JOB_NAME}_%j.err \
            run_experiment.sh ${N} 8 5e-5 1 openr1
        echo "  Submitted N=${N}"
    done
else
    echo "  OpenR1 data not yet available, skipping"
fi

echo ""
echo "Round 3 jobs submitted."
echo "Monitor: squeue -u \$USER | grep r3"
