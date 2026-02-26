#!/bin/bash
# Evaluate the effect of max_new_tokens on the OpenR1 cliff
# Tests whether the cliff is a generation truncation artifact
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Token Length Evaluation Experiments ==="

# 1. Baseline 3B (no adapter) — establishes baseline hit-limit rates
cat > /tmp/slurm_toklen_baseline.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=toklen_baseline_3b
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

python eval_token_length.py \
    --experiment_name baseline_3b \
    --max_tokens 1024 2048 4096 \
    --batch_size 32
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_toklen_baseline.sh
JOB_ID=$(sbatch /tmp/slurm_toklen_baseline.sh 2>&1 | grep -oP '\d+')
echo "  Submitted baseline_3b -> Job ${JOB_ID}"

# 2. OpenR1 N=1000 (pre-cliff, should show minimal effect)
cat > /tmp/slurm_toklen_or1_1k.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=toklen_or1_n1000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

python eval_token_length.py \
    --adapter_path checkpoints/sft_openr1_n1000_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n1000 \
    --max_tokens 1024 2048 4096 \
    --batch_size 32
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_toklen_or1_1k.sh
JOB_ID=$(sbatch /tmp/slurm_toklen_or1_1k.sh 2>&1 | grep -oP '\d+')
echo "  Submitted or1_n1000 -> Job ${JOB_ID}"

# 3. OpenR1 N=2000 (the cliff — CRITICAL test)
cat > /tmp/slurm_toklen_or1_2k.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=toklen_or1_n2000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

python eval_token_length.py \
    --adapter_path checkpoints/sft_openr1_n2000_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n2000 \
    --max_tokens 1024 2048 4096 \
    --batch_size 32
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_toklen_or1_2k.sh
JOB_ID=$(sbatch /tmp/slurm_toklen_or1_2k.sh 2>&1 | grep -oP '\d+')
echo "  Submitted or1_n2000 -> Job ${JOB_ID}"

# 4. OpenR1 N=5000 (deep in cliff)
cat > /tmp/slurm_toklen_or1_5k.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=toklen_or1_n5000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

python eval_token_length.py \
    --adapter_path checkpoints/sft_openr1_n5000_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_openr1_n5000 \
    --max_tokens 1024 2048 4096 \
    --batch_size 16
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_toklen_or1_5k.sh
JOB_ID=$(sbatch /tmp/slurm_toklen_or1_5k.sh 2>&1 | grep -oP '\d+')
echo "  Submitted or1_n5000 -> Job ${JOB_ID}"

# 5. NuminaMath N=2000 (control — should show minimal truncation effect)
cat > /tmp/slurm_toklen_nm_2k.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=toklen_nm_n2000
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

python eval_token_length.py \
    --adapter_path checkpoints/sft_numinamath_n2000_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_numinamath_n2000 \
    --max_tokens 1024 2048 4096 \
    --batch_size 32
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_toklen_nm_2k.sh
JOB_ID=$(sbatch /tmp/slurm_toklen_nm_2k.sh 2>&1 | grep -oP '\d+')
echo "  Submitted nm_n2000 -> Job ${JOB_ID}"

# 6. OpenR1 N=5000 7B (the 7B cliff point)
cat > /tmp/slurm_toklen_or1_5k_7b.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=toklen_or1_n5000_7b
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd ${SCRIPT_DIR}

python eval_token_length.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --adapter_path checkpoints/sft_openr1_n5000_r8_lr5e-5_ep1_7b/final_adapter \\
    --experiment_name sft_openr1_n5000_7b \\
    --max_tokens 1024 2048 4096 \\
    --batch_size 16
SLURM_EOF
JOB_ID=$(sbatch /tmp/slurm_toklen_or1_5k_7b.sh 2>&1 | grep -oP '\d+')
echo "  Submitted or1_n5000_7b -> Job ${JOB_ID}"

echo ""
echo "=== All token length eval jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
