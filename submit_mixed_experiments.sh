#!/bin/bash
#SBATCH --job-name=mixed_sft
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

# Experiment C: Mixed NM+OpenR1 at N=2000 with varying ratios
# Addresses reviewer W4 (intervention experiments)

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

mkdir -p results/logs

echo "=== Experiment C: Mixed NM+OpenR1 Data ==="
echo "Start time: $(date)"

# Mix ratios: 0.75 (75% NM), 0.50 (50/50), 0.25 (25% NM)
for RATIO in 0.75 0.50 0.25; do
    # Convert ratio to naming convention
    NM_PCT=$(python3 -c "print(int(${RATIO}*100))")
    OR_PCT=$(python3 -c "print(int((1-${RATIO})*100))")
    EXP_NAME="sft_mixed${NM_PCT}nm_${OR_PCT}or_n2000_r8_lr5e-5_ep1"

    echo ""
    echo "========================================"
    echo "Training: ${EXP_NAME} (mix_ratio=${RATIO})"
    echo "========================================"

    # Train
    python train_sft.py \
        --data_source mixed \
        --mix_ratio $RATIO \
        --num_samples 2000 \
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
echo "=== All mixed experiments complete ==="
echo "End time: $(date)"
