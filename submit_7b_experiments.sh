#!/bin/bash
# Round 3: 7B NuminaMath + OpenR1 experiments (8 total)
# Requires: 7B baselines completed, BASELINES_7B updated in eval_finetuned.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Round 3: 7B Training Experiments ==="

for SOURCE in numinamath openr1; do
  for N in 500 1000 2000 5000; do
    EXP="sft_${SOURCE}_n${N}_r8_lr5e-5_ep1_7b"

    # Skip if checkpoint already exists
    if [ -d "checkpoints/${EXP}/final_adapter" ]; then
      echo "  [SKIP] ${EXP} (already exists)"
      continue
    fi

    cat > /tmp/slurm_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== Training ${EXP} ==="
python train_sft.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --num_samples ${N} \\
    --data_source ${SOURCE} \\
    --lora_rank 8 --lora_alpha 16 \\
    --lr 5e-5 --epochs 1 \\
    --batch_size 2 --grad_accum 8 \\
    --seed 42 \\
    --experiment_name ${EXP} \\
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating ${EXP} ==="
python eval_finetuned.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 32

echo "=== GSM8K Strict Eval ${EXP} ==="
python eval_gsm8k.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP}_strict \\
    --batch_size 32

echo "=== GSM8K Tolerant Eval ${EXP} ==="
python eval_gsm8k.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP}_tolerant \\
    --batch_size 32 --tolerant

echo "=== Done ${EXP} ==="
SLURM_EOF

    JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
    echo "  Submitted ${EXP} -> Job ${JOB_ID}"
  done
done

echo ""
echo "=== All 7B experiments submitted (8 jobs) ==="
echo "Monitor with: squeue -u \$USER"
