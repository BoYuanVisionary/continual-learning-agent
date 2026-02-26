#!/bin/bash
# Submit evaluation jobs for all unevaluated checkpoints with trained models

WORKDIR=/storage/ice-shared/ae3530b/byuan48/research/research_agent
LOGDIR=$WORKDIR/results/logs
mkdir -p $LOGDIR

# List of checkpoints to evaluate (confirmed to have final_adapter with safetensors)
CHECKPOINTS=(
    "sft_numinamath_n10000_r8_lr5e-5_ep1_seed1"
    "sft_numinamath_n10000_r8_lr5e-5_ep1_seed2"
    "sft_numinamath_n10000_r8_lr5e-5_ep1_seed3"
    "sft_numinamath_n1000_r8_lr5e-5_ep1_kl0p5"
    "sft_numinamath_n100_r8_lr5e-5_ep1_seed2"
    "sft_numinamath_n100_r8_lr5e-5_ep1_seed3"
    "sft_numinamath_n1500_r8_lr5e-5_ep1"
    "sft_numinamath_n500_r8_lr5e-5_ep1_kl0p5"
    "sft_numinamath_n500_r8_lr5e-5_ep1_seed2"
    "sft_numinamath_n500_r8_lr5e-5_ep1_seed3"
    "sft_numinamath_n750_r8_lr5e-5_ep1"
)

# Submit 3 jobs, each handling ~4 checkpoints (to parallelize)
BATCH_SIZE=4

for i in $(seq 0 $((${#CHECKPOINTS[@]} - 1))); do
    CP=${CHECKPOINTS[$i]}
    BATCH_IDX=$((i / BATCH_SIZE))

    # Check if eval already exists
    if [ -f "$WORKDIR/results/${CP}_eval.json" ]; then
        echo "SKIP: $CP (eval already exists)"
        continue
    fi

    echo "Queuing: $CP"
done

# Create batch scripts
for BATCH_IDX in 0 1 2; do
    START=$((BATCH_IDX * BATCH_SIZE))
    END=$((START + BATCH_SIZE - 1))
    if [ $END -ge ${#CHECKPOINTS[@]} ]; then
        END=$((${#CHECKPOINTS[@]} - 1))
    fi

    # Skip if no checkpoints in this batch
    if [ $START -ge ${#CHECKPOINTS[@]} ]; then
        continue
    fi

    BATCH_CPS=()
    for i in $(seq $START $END); do
        CP=${CHECKPOINTS[$i]}
        if [ ! -f "$WORKDIR/results/${CP}_eval.json" ]; then
            BATCH_CPS+=("$CP")
        fi
    done

    if [ ${#BATCH_CPS[@]} -eq 0 ]; then
        echo "SKIP batch $BATCH_IDX (all evals exist)"
        continue
    fi

    SCRIPT="$WORKDIR/eval_batch_${BATCH_IDX}.sh"
    cat > "$SCRIPT" << EVALEOF
#!/bin/bash
#SBATCH --job-name=eval_batch_${BATCH_IDX}
#SBATCH --output=${LOGDIR}/eval_batch_${BATCH_IDX}_%j.out
#SBATCH --error=${LOGDIR}/eval_batch_${BATCH_IDX}_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd $WORKDIR

EVALEOF

    for CP in "${BATCH_CPS[@]}"; do
        cat >> "$SCRIPT" << EVALEOF
echo "=== Evaluating $CP ==="
python eval_finetuned.py \\
    --adapter_path checkpoints/${CP}/final_adapter \\
    --experiment_name ${CP} \\
    --batch_size 32
echo "=== Done: $CP ==="

EVALEOF
    done

    echo "Submitting batch $BATCH_IDX with ${#BATCH_CPS[@]} checkpoints: ${BATCH_CPS[*]}"
    sbatch "$SCRIPT"
done
