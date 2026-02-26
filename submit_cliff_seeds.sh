#!/bin/bash
# Round 3: OpenR1 cliff replication with multiple seeds (3 experiments)
# Tests reproducibility of the OR1 N=2000 cliff across seeds 1, 2, 3
# (seed=42 was original)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Round 3: OpenR1 Cliff Seed Replication ==="

for SEED in 1 2 3; do
  EXP="sft_openr1_n2000_r8_lr5e-5_ep1_seed${SEED}"

  if [ -d "checkpoints/${EXP}/final_adapter" ]; then
    echo "  [SKIP] ${EXP} (already exists)"
    continue
  fi

  cat > /tmp/slurm_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== Training ${EXP} ==="
python train_sft.py \\
    --num_samples 2000 \\
    --data_source openr1 \\
    --lora_rank 8 --lora_alpha 16 \\
    --lr 5e-5 --epochs 1 \\
    --batch_size 2 --grad_accum 8 \\
    --seed ${SEED} \\
    --experiment_name ${EXP} \\
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating ${EXP} ==="
python eval_finetuned.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64

echo "=== GSM8K Strict Eval ${EXP} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP}_strict \\
    --batch_size 64

echo "=== GSM8K Tolerant Eval ${EXP} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP}_tolerant \\
    --batch_size 64 --tolerant

echo "=== Done ${EXP} ==="
SLURM_EOF

  JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
  echo "  Submitted ${EXP} -> Job ${JOB_ID}"
done

echo ""
echo "=== All cliff seed experiments submitted (3 jobs) ==="
echo "Monitor with: squeue -u \$USER"
