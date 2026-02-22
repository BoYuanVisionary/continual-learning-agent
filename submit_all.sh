#!/bin/bash
# Submit all SFT experiments in parallel via SLURM
# Each experiment trains on a different number of samples and then evaluates

cd /home/hice1/byuan48/research/research_agent
mkdir -p results/logs

# Configuration
LORA_RANK=16
LR="2e-4"
EPOCHS=3
DATA="numinamath"

echo "Submitting SFT experiments with LoRA rank=${LORA_RANK}, LR=${LR}, Epochs=${EPOCHS}"
echo ""

for N in 100 500 1000 2000 5000 10000; do
    JOB_NAME="sft_n${N}"
    echo "Submitting: N=${N} (job: ${JOB_NAME})"
    sbatch --job-name=${JOB_NAME} \
           --output=results/logs/${JOB_NAME}_%j.out \
           --error=results/logs/${JOB_NAME}_%j.err \
           run_experiment.sh ${N} ${LORA_RANK} ${LR} ${EPOCHS} ${DATA}
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "View logs with: tail -f results/logs/sft_n*"
