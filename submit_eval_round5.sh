#!/bin/bash
# Round 5: Evaluate all completed but unevaluated checkpoints
# Multi-seed, KL-regularized, and GSM8K evaluations
# Also runs output format analysis

SCRIPT_DIR="/storage/ice-shared/ae3530b/byuan48/research/research_agent"
cd "$SCRIPT_DIR"

# ===== Job 1: Evaluate multi-seed checkpoints (N=100, 500) =====
cat > /tmp/eval_seeds_small.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=eval_seeds_small
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

for SEED in 1 2 3; do
    for N in 100 500; do
        NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1_seed${SEED}"
        ADAPTER="checkpoints/${NAME}/final_adapter"
        if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
            echo "Evaluating: $NAME"
            python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
        else
            echo "Skipping $NAME (no adapter or already evaluated)"
        fi
    done
done
EVALEOF
sbatch /tmp/eval_seeds_small.sh

# ===== Job 2: Evaluate multi-seed N=2000 seed3 =====
cat > /tmp/eval_seeds_2000.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=eval_seeds_2000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

NAME="sft_numinamath_n2000_r8_lr5e-5_ep1_seed3"
ADAPTER="checkpoints/${NAME}/final_adapter"
if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
    echo "Evaluating: $NAME"
    python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
fi
EVALEOF
sbatch /tmp/eval_seeds_2000.sh

# ===== Job 3: Evaluate KL-regularized checkpoints =====
cat > /tmp/eval_kl.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=eval_kl
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

for KL in "kl0p1" "kl0p5"; do
    for N in 500 1000; do
        NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1_${KL}"
        ADAPTER="checkpoints/${NAME}/final_adapter"
        if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
            echo "Evaluating: $NAME"
            python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
        else
            echo "Skipping $NAME"
        fi
    done
done
EVALEOF
sbatch /tmp/eval_kl.sh

# ===== Job 4: GSM8K evaluation on key checkpoints =====
cat > /tmp/eval_gsm8k.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=eval_gsm8k
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

# Key checkpoints to evaluate on GSM8K
for NAME in "sft_numinamath_n100_r8_lr5e-5_ep1" \
            "sft_numinamath_n500_r8_lr5e-5_ep1" \
            "sft_numinamath_n1000_r8_lr5e-5_ep1" \
            "sft_numinamath_n2000_r8_lr5e-5_ep1" \
            "sft_numinamath_n5000_r8_lr5e-5_ep1" \
            "sft_numinamath_n10000_r8_lr5e-5_ep1"; do
    ADAPTER="checkpoints/${NAME}/final_adapter"
    if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_gsm8k.json" ]; then
        echo "GSM8K eval: $NAME"
        python eval_gsm8k.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
        echo "GSM8K eval with tolerant mode: $NAME"
        python eval_gsm8k.py --adapter_path "$ADAPTER" --experiment_name "${NAME}_tolerant" --batch_size 64 --tolerant
    fi
done
EVALEOF
sbatch /tmp/eval_gsm8k.sh

# ===== Job 5: Output format analysis (key insight for paper) =====
cat > /tmp/analyze_outputs.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=analyze_outputs
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

python analyze_outputs.py --checkpoints \
    baseline \
    sft_numinamath_n100_r8_lr5e-5_ep1 \
    sft_numinamath_n500_r8_lr5e-5_ep1 \
    sft_numinamath_n1000_r8_lr5e-5_ep1 \
    sft_numinamath_n2000_r8_lr5e-5_ep1 \
    sft_numinamath_n5000_r8_lr5e-5_ep1 \
    sft_numinamath_n10000_r8_lr5e-5_ep1 \
    sft_openr1_n100_r8_lr5e-5_ep1 \
    sft_openr1_n1000_r8_lr5e-5_ep1 \
    sft_openr1_n5000_r8_lr5e-5_ep1 \
    --batch_size 64 --num_samples 1024
EVALEOF
sbatch /tmp/analyze_outputs.sh

# ===== Job 6: Complete training for N=2000 seed1, seed2 =====
cat > /tmp/train_seeds_2000.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=train_seeds_2000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

for SEED in 1 2; do
    NAME="sft_numinamath_n2000_r8_lr5e-5_ep1_seed${SEED}"
    if [ ! -d "checkpoints/${NAME}/final_adapter" ]; then
        echo "Training: $NAME"
        python train_sft.py --num_samples 2000 --data_source numinamath \
            --lora_rank 8 --lr 5e-5 --epochs 1 --seed $SEED \
            --experiment_name "$NAME"
    fi
done

# Evaluate the trained models
for SEED in 1 2; do
    NAME="sft_numinamath_n2000_r8_lr5e-5_ep1_seed${SEED}"
    ADAPTER="checkpoints/${NAME}/final_adapter"
    if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
        echo "Evaluating: $NAME"
        python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
    fi
done
EVALEOF
sbatch /tmp/train_seeds_2000.sh

# ===== Job 7: Complete training for N=10000 seed1,2,3 =====
cat > /tmp/train_seeds_10000.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=train_seeds_10000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

for SEED in 1 2 3; do
    NAME="sft_numinamath_n10000_r8_lr5e-5_ep1_seed${SEED}"
    if [ ! -d "checkpoints/${NAME}/final_adapter" ]; then
        echo "Training: $NAME"
        python train_sft.py --num_samples 10000 --data_source numinamath \
            --lora_rank 8 --lr 5e-5 --epochs 1 --seed $SEED \
            --experiment_name "$NAME"
    fi
done

# Evaluate
for SEED in 1 2 3; do
    NAME="sft_numinamath_n10000_r8_lr5e-5_ep1_seed${SEED}"
    ADAPTER="checkpoints/${NAME}/final_adapter"
    if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
        echo "Evaluating: $NAME"
        python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
    fi
done
EVALEOF
sbatch /tmp/train_seeds_10000.sh

# ===== Job 8: Complete KL training for N=2000, 5000 =====
cat > /tmp/train_kl_large.sh << 'EVALEOF'
#!/bin/bash
#SBATCH --job-name=train_kl_large
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd /storage/ice-shared/ae3530b/byuan48/research/research_agent

for N in 2000 5000; do
    for KL in 0.1 0.5; do
        KL_STR=$(echo $KL | sed 's/\./_/g')
        NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1_kl${KL_STR}"
        # Fix naming to match expected: kl0p1, kl0p5
        KL_NAME=$(echo $KL | sed 's/0\./0p/')
        NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1_${KL_NAME}"
        if [ ! -d "checkpoints/${NAME}/final_adapter" ]; then
            echo "Training KL: $NAME (kl_weight=$KL)"
            python train_sft_kl.py --num_samples $N --data_source numinamath \
                --lora_rank 8 --lr 5e-5 --epochs 1 --kl_weight $KL \
                --experiment_name "$NAME"
        fi
    done
done

# Evaluate
for N in 2000 5000; do
    for KL_NAME in "kl0p1" "kl0p5"; do
        NAME="sft_numinamath_n${N}_r8_lr5e-5_ep1_${KL_NAME}"
        ADAPTER="checkpoints/${NAME}/final_adapter"
        if [ -d "$ADAPTER" ] && [ ! -f "results/${NAME}_eval.json" ]; then
            echo "Evaluating: $NAME"
            python eval_finetuned.py --adapter_path "$ADAPTER" --experiment_name "$NAME" --batch_size 64
        fi
    done
done
EVALEOF
sbatch /tmp/train_kl_large.sh

echo ""
echo "Submitted all jobs. Check status with: squeue -u \$USER"
