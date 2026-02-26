#!/bin/bash
# Retry failed experiments - v2 without --exclusive, shorter time limits
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results/logs

echo "=== Retrying Failed Experiments (v2) ==="

# 1. CodeAlpaca N=2000 (3B, 1 GPU, 2h)
EXP="sft_codealpaca_n2000_r8_lr5e-5_ep1"
cat > /tmp/slurm_retry2_${EXP}.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=sft_codealpaca_n2000_r8_lr5e-5_ep1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source activate qwen25
cd SCRIPT_DIR_PLACEHOLDER

echo "=== Training sft_codealpaca_n2000_r8_lr5e-5_ep1 ==="
python train_sft.py \
    --num_samples 2000 \
    --data_source codealpaca \
    --lora_rank 8 --lora_alpha 16 \
    --lr 5e-5 --epochs 1 \
    --batch_size 4 --grad_accum 4 \
    --seed 42 \
    --experiment_name sft_codealpaca_n2000_r8_lr5e-5_ep1 \
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "=== Evaluating sft_codealpaca_n2000_r8_lr5e-5_ep1 ==="
python eval_finetuned.py \
    --adapter_path checkpoints/sft_codealpaca_n2000_r8_lr5e-5_ep1/final_adapter \
    --experiment_name sft_codealpaca_n2000_r8_lr5e-5_ep1 \
    --batch_size 64

echo "=== Done ==="
SLURM_EOF
sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" /tmp/slurm_retry2_${EXP}.sh
JOB_ID=$(sbatch /tmp/slurm_retry2_${EXP}.sh 2>&1 | grep -oP '\d+')
echo "  Submitted ${EXP} -> Job ${JOB_ID}"

# 2. NuminaMath N=2000 7B (2 GPUs, 4h)
EXP="sft_numinamath_n2000_r8_lr5e-5_ep1_7b"
cat > /tmp/slurm_retry2_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

source activate qwen25
cd ${SCRIPT_DIR}

echo "=== Training ${EXP} ==="
python train_sft.py \\
    --model_name "Qwen/Qwen2.5-7B-Instruct" \\
    --num_samples 2000 \\
    --data_source numinamath \\
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
JOB_ID=$(sbatch /tmp/slurm_retry2_${EXP}.sh 2>&1 | grep -oP '\d+')
echo "  Submitted ${EXP} -> Job ${JOB_ID}"

# 3-5. OpenR1 7B experiments (2 GPUs, 4h each)
for exp_tuple in \
  "openr1:1000" \
  "openr1:2000" \
  "openr1:5000"; do

  SOURCE=$(echo $exp_tuple | cut -d: -f1)
  N=$(echo $exp_tuple | cut -d: -f2)
  EXP="sft_${SOURCE}_n${N}_r8_lr5e-5_ep1_7b"

  cat > /tmp/slurm_retry2_${EXP}.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=4:00:00
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

  JOB_ID=$(sbatch /tmp/slurm_retry2_${EXP}.sh 2>&1 | grep -oP '\d+')
  echo "  Submitted ${EXP} -> Job ${JOB_ID}"
done

echo ""
echo "=== All retry v2 jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
