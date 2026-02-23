#!/bin/bash
#SBATCH --job-name=openr1_gsm8k
#SBATCH --output=results/logs/openr1_gsm8k_%j.out
#SBATCH --error=results/logs/openr1_gsm8k_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

echo "=== OpenR1 GSM8K Evaluation (strict + tolerant) ==="
echo "Start time: $(date)"

# Run strict GSM8K for all 6 OpenR1 checkpoints
for N in 100 500 1000 2000 5000 10000; do
    CKPT="checkpoints/sft_openr1_n${N}_r8_lr5e-5_ep1/final_adapter"
    EXP="sft_openr1_n${N}_r8_lr5e-5_ep1"

    if [ ! -d "$CKPT" ]; then
        echo "Checkpoint not found: $CKPT, skipping"
        continue
    fi

    # Strict mode
    OUT="results/${EXP}_strict_gsm8k.json"
    if [ ! -f "$OUT" ]; then
        echo ""
        echo "--- Strict GSM8K: $EXP ---"
        python eval_gsm8k.py \
            --adapter_path "$CKPT" \
            --experiment_name "${EXP}_strict" \
            --batch_size 64
        # Rename to consistent name
        if [ -f "results/${EXP}_strict_gsm8k.json" ]; then
            echo "Saved: results/${EXP}_strict_gsm8k.json"
        fi
    else
        echo "Already exists: $OUT"
    fi

    # Tolerant mode (only for N=5000, N=10000 which are missing)
    TOUT="results/${EXP}_tolerant_gsm8k.json"
    if [ ! -f "results/${EXP}_gsm8k.json" ] && [ ! -f "$TOUT" ]; then
        echo ""
        echo "--- Tolerant GSM8K: $EXP ---"
        python eval_gsm8k.py \
            --adapter_path "$CKPT" \
            --experiment_name "${EXP}_tolerant" \
            --batch_size 64 \
            --tolerant
        if [ -f "results/${EXP}_tolerant_gsm8k.json" ]; then
            echo "Saved: results/${EXP}_tolerant_gsm8k.json"
        fi
    else
        echo "Tolerant already exists for $EXP"
    fi
done

# Also run OpenR1 output analysis
echo ""
echo "=== OpenR1 Output Analysis ==="
python analyze_outputs.py \
    --checkpoints \
    sft_openr1_n100_r8_lr5e-5_ep1 \
    sft_openr1_n500_r8_lr5e-5_ep1 \
    sft_openr1_n1000_r8_lr5e-5_ep1 \
    sft_openr1_n2000_r8_lr5e-5_ep1 \
    sft_openr1_n5000_r8_lr5e-5_ep1 \
    sft_openr1_n10000_r8_lr5e-5_ep1 \
    --batch_size 64

echo ""
echo "End time: $(date)"
echo "=== Done ==="
