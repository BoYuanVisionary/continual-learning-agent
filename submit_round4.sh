#!/bin/bash
# Round 4: Addressing reviewer feedback
# - Multi-seed runs (3 seeds x 4 sample counts)
# - KL-regularized SFT (4 sample counts)
# - Rejection sampling + SFT on in-distribution data
# - GSM8K eval on key existing checkpoints
# - GSM8K baseline eval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results/logs

echo "=== Round 4: Reviewer Response Experiments ==="

# ============================================================
# 1. Multi-seed runs: N=100, 500, 2000, 10000 x seeds 1,2,3
#    (seed=42 was the original, so we add seeds 1, 2, 3)
# ============================================================
echo "--- Submitting multi-seed training + eval jobs ---"

for SEED in 1 2 3; do
  for N in 100 500 2000 10000; do
    EXP="sft_numinamath_n${N}_r8_lr5e-5_ep1_seed${SEED}"

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
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== Training ${EXP} ==="
python train_sft.py \\
    --num_samples ${N} \\
    --data_source numinamath \\
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

echo "=== GSM8K Eval ${EXP} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64 --tolerant

echo "=== Done ${EXP} ==="
SLURM_EOF

    JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
    echo "  Submitted ${EXP} -> Job ${JOB_ID}"
  done
done

# ============================================================
# 2. KL-regularized SFT: N=500, 1000, 2000, 5000
# ============================================================
echo "--- Submitting KL-regularized SFT jobs ---"

for N in 500 1000 2000 5000; do
  for KL_W in 0.1 0.5; do
    KL_TAG=$(echo ${KL_W} | tr '.' 'p')
    EXP="sft_numinamath_n${N}_r8_lr5e-5_ep1_kl${KL_TAG}"

    if [ -d "checkpoints/${EXP}/final_adapter" ]; then
      echo "  [SKIP] ${EXP} (already exists)"
      continue
    fi

    cat > /tmp/slurm_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== KL-Reg Training ${EXP} ==="
python train_sft_kl.py \\
    --num_samples ${N} \\
    --data_source numinamath \\
    --lora_rank 8 --lora_alpha 16 \\
    --lr 5e-5 --epochs 1 \\
    --batch_size 2 --grad_accum 8 \\
    --seed 42 \\
    --kl_weight ${KL_W} \\
    --experiment_name ${EXP} \\
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating ${EXP} ==="
python eval_finetuned.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64

echo "=== GSM8K Eval ${EXP} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64 --tolerant

echo "=== Done ${EXP} ==="
SLURM_EOF

    JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
    echo "  Submitted ${EXP} -> Job ${JOB_ID}"
  done
done

# ============================================================
# 3. Rejection sampling (generates in-distribution data)
# ============================================================
echo "--- Submitting rejection sampling job ---"

EXP="rejection_sample"
cat > /tmp/slurm_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=rejection_sample
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== Rejection Sampling ==="
python rejection_sample.py --max_problems 5000 --num_attempts 4 --batch_size 32

echo "=== Done ==="
SLURM_EOF

RS_JOB=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
echo "  Submitted rejection_sample -> Job ${RS_JOB}"

# ============================================================
# 4. SFT on rejection-sampled data (depends on step 3)
#    Submit with dependency
# ============================================================
echo "--- Submitting rejection-SFT jobs (depend on rejection sampling) ---"

for N in 100 500 1000 2000; do
  EXP="sft_orz_reject_n${N}_r8_lr5e-5_ep1"

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
    --data_source orz_reject \\
    --lora_rank 8 --lora_alpha 16 \\
    --lr 5e-5 --epochs 1 \\
    --batch_size 2 --grad_accum 8 \\
    --seed 42 \\
    --experiment_name ${EXP} \\
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating ${EXP} ==="
python eval_finetuned.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64

echo "=== GSM8K Eval ${EXP} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${EXP}/final_adapter \\
    --experiment_name ${EXP} \\
    --batch_size 64 --tolerant

echo "=== Done ${EXP} ==="
SLURM_EOF

  if [ -n "${RS_JOB}" ]; then
    JOB_ID=$(sbatch --dependency=afterok:${RS_JOB} /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
  else
    JOB_ID=$(sbatch /tmp/slurm_${EXP}.sh 2>&1 | grep -oP '\d+')
  fi
  echo "  Submitted ${EXP} -> Job ${JOB_ID} (depends on ${RS_JOB})"
done

# ============================================================
# 5. GSM8K eval on existing key checkpoints + baseline
# ============================================================
echo "--- Submitting GSM8K eval jobs for existing checkpoints ---"

# Baseline (no adapter)
cat > /tmp/slurm_gsm8k_baseline.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=gsm8k_baseline
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

# Download GSM8K first
python download_gsm8k.py

echo "=== GSM8K Baseline ==="
python eval_gsm8k.py --baseline --experiment_name baseline --batch_size 64 --tolerant
SLURM_EOF

BASELINE_JOB=$(sbatch /tmp/slurm_gsm8k_baseline.sh 2>&1 | grep -oP '\d+')
echo "  Submitted gsm8k_baseline -> Job ${BASELINE_JOB}"

# Key existing checkpoints
for CKPT in \
  sft_numinamath_n100_r8_lr5e-5_ep1 \
  sft_numinamath_n500_r8_lr5e-5_ep1 \
  sft_numinamath_n1000_r8_lr5e-5_ep1 \
  sft_numinamath_n2000_r8_lr5e-5_ep1 \
  sft_numinamath_n5000_r8_lr5e-5_ep1 \
  sft_numinamath_n10000_r8_lr5e-5_ep1 \
  sft_numinamath_hard_n500_r8_lr5e-5_ep1 \
  sft_numinamath_hard_n2000_r8_lr5e-5_ep1 \
  sft_openr1_n1000_r8_lr5e-5_ep1 \
  sft_openr1_n5000_r8_lr5e-5_ep1; do

  if [ ! -d "checkpoints/${CKPT}/final_adapter" ]; then
    echo "  [SKIP] ${CKPT} (no checkpoint)"
    continue
  fi

  cat > /tmp/slurm_gsm8k_${CKPT}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=gsm8k_${CKPT}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== GSM8K Eval ${CKPT} ==="
python eval_gsm8k.py \\
    --adapter_path checkpoints/${CKPT}/final_adapter \\
    --experiment_name ${CKPT} \\
    --batch_size 64 --tolerant
SLURM_EOF

  JOB_ID=$(sbatch --dependency=afterok:${BASELINE_JOB} /tmp/slurm_gsm8k_${CKPT}.sh 2>&1 | grep -oP '\d+')
  echo "  Submitted gsm8k_${CKPT} -> Job ${JOB_ID}"
done

echo ""
echo "=== All Round 4 jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Check logs: tail -f results/logs/<job_name>_<job_id>.out"
