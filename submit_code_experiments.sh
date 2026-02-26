#!/bin/bash
# Round 3: Code SFT on 3B (4 experiments)
# Requires: data/codealpaca/codealpaca_20k.json downloaded
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Round 3: Code SFT Experiments (3B) ==="

# Check if CodeAlpaca data exists
if [ ! -f "data/codealpaca/codealpaca_20k.json" ]; then
  echo "ERROR: CodeAlpaca data not found. Run 'python download_codealpaca.py' first."
  exit 1
fi

for N in 500 1000 2000 5000; do
  EXP="sft_codealpaca_n${N}_r8_lr5e-5_ep1"

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
    --num_samples ${N} \\
    --data_source codealpaca \\
    --lora_rank 8 --lora_alpha 16 \\
    --lr 5e-5 --epochs 1 \\
    --batch_size 4 --grad_accum 4 \\
    --seed 42 \\
    --experiment_name ${EXP} \\
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating ${EXP} ==="
python eval_finetuned.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64

echo "=== Done ${EXP} ==="
SLURM_EOF

  JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
  echo "  Submitted ${EXP} -> Job ${JOB_ID}"
done

echo ""
echo "=== All code SFT experiments submitted (4 jobs) ==="
echo "Monitor with: squeue -u \$USER"
