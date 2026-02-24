#!/bin/bash
#SBATCH --job-name=trunc_openr1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

# Experiment B: Train truncated OpenR1 at N=1000, 2000, 5000
# Addresses reviewer W3 (confound between solution length/style and direction)

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

mkdir -p results/logs

echo "=== Experiment B: Truncated OpenR1 ==="
echo "Start time: $(date)"

for N in 1000 2000 5000; do
    EXP_NAME="sft_openr1trunc_n${N}_r8_lr5e-5_ep1"
    echo ""
    echo "========================================"
    echo "Training: ${EXP_NAME}"
    echo "========================================"

    # Train
    python train_sft.py \
        --data_source openr1_truncated \
        --num_samples $N \
        --lora_rank 8 \
        --lr 5e-5 \
        --epochs 1 \
        --batch_size 4 \
        --grad_accum 4 \
        --max_seq_length 2048 \
        --experiment_name $EXP_NAME

    ADAPTER_PATH="checkpoints/${EXP_NAME}/final_adapter"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: Adapter not found at $ADAPTER_PATH, skipping eval"
        continue
    fi

    # Full eval (ORZ, SciKnowEval, ToolAlpaca)
    echo "Running full evaluation..."
    python eval_finetuned.py \
        --adapter_path $ADAPTER_PATH \
        --experiment_name $EXP_NAME \
        --batch_size 64

    # GSM8K strict
    echo "Running GSM8K strict..."
    python eval_gsm8k.py \
        --adapter_path $ADAPTER_PATH \
        --experiment_name "${EXP_NAME}_strict" \
        --batch_size 64

    # GSM8K tolerant
    echo "Running GSM8K tolerant..."
    python eval_gsm8k.py \
        --adapter_path $ADAPTER_PATH \
        --experiment_name "${EXP_NAME}_tolerant" \
        --batch_size 64 \
        --tolerant

    echo "Done with ${EXP_NAME}"
done

echo ""
echo "=== All truncated OpenR1 experiments complete ==="
echo "End time: $(date)"
