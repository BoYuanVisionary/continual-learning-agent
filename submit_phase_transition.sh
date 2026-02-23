#!/bin/bash
#SBATCH --job-name=phase_transition
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

# Train at N=750 and N=1500 to characterize phase transition
# Then evaluate on GSM8K (strict + tolerant)

for N in 750 1500; do
    NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1"
    CKPT="checkpoints/${NAME}/final_adapter"

    # Skip if already trained
    if [ -d "$CKPT" ]; then
        echo "=== ${NAME} already trained, skipping ==="
    else
        echo "=== Training ${NAME} ==="
        python train_sft.py \
            --data_source numinamath \
            --num_samples $N \
            --lora_rank 8 \
            --lr 5e-5 \
            --epochs 1 \
            --batch_size 4 \
            --grad_accum 4 \
            --experiment_name "${NAME}"
    fi

    # Evaluate GSM8K strict
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating ${NAME} on GSM8K (strict) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64
        echo ""
        echo "=== Evaluating ${NAME} on GSM8K (tolerant) ==="
        python eval_gsm8k.py --adapter_path "$CKPT" --experiment_name "$NAME" --batch_size 64 --tolerant
        echo ""
    fi
done

echo "Phase transition analysis complete!"
