#!/bin/bash
#SBATCH --job-name=sft_experiment
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=coe-gpu

# Usage: sbatch run_experiment.sh <num_samples> [lora_rank] [lr] [epochs]
# Example: sbatch run_experiment.sh 1000 16 2e-4 3

source activate qwen25
cd /home/hice1/byuan48/research/research_agent

NUM_SAMPLES=${1:-1000}
LORA_RANK=${2:-16}
LR=${3:-2e-4}
EPOCHS=${4:-3}
DATA_SOURCE=${5:-numinamath}

EXPERIMENT_NAME="sft_${DATA_SOURCE}_n${NUM_SAMPLES}_r${LORA_RANK}_lr${LR}_ep${EPOCHS}"

echo "====================================="
echo "Starting experiment: ${EXPERIMENT_NAME}"
echo "Samples: ${NUM_SAMPLES}, LoRA rank: ${LORA_RANK}, LR: ${LR}, Epochs: ${EPOCHS}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "====================================="

# Step 1: Train
echo ""
echo "=== TRAINING ==="
python train_sft.py \
    --num_samples ${NUM_SAMPLES} \
    --data_source ${DATA_SOURCE} \
    --lora_rank ${LORA_RANK} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --batch_size 4 \
    --grad_accum 4 \
    --max_seq_length 2048 \
    --experiment_name ${EXPERIMENT_NAME}

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

ADAPTER_PATH="checkpoints/${EXPERIMENT_NAME}/final_adapter"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    exit 1
fi

# Step 2: Evaluate
echo ""
echo "=== EVALUATION ==="
python eval_finetuned.py \
    --adapter_path ${ADAPTER_PATH} \
    --experiment_name ${EXPERIMENT_NAME} \
    --batch_size 64

EVAL_EXIT=$?
echo ""
echo "Experiment ${EXPERIMENT_NAME} completed with eval exit code: $EVAL_EXIT"
echo "Results: results/${EXPERIMENT_NAME}_eval.json"
