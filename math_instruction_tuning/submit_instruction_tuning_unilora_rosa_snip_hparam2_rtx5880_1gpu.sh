#!/bin/bash
#SBATCH --job-name=unilora_rosa_snip_hparam2_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-5
#SBATCH --output=logs/unilora_rosa_snip_hparam2_5880_%A_%a.out
#SBATCH --error=logs/unilora_rosa_snip_hparam2_5880_%A_%a.err

# UniLoRA-RoSA-SNIP hyperparameter search, round 2.
# Round 1 (job 1776569, eval 1779306) found on Gemma-7B + MetaMathQA:
#   8:1  GSM8K 74.83 / MATH 29.78  <- best; ties Uni-LoRA on GSM8K
#   4:1  GSM8K 74.22 / MATH 29.74
#   2:1 variants and stronger sparse LR / shifted warmup were all worse.
# This round zooms in around the 8:1 projection:sparse split:
#   - ratio sweep 6:1, 10:1, 12:1, 16:1 with the baseline LR schedule
#   - at the 8:1 split, vary theta_d LR (1e-3) and sparse LR mult (0.1)
# All combinations keep THETA_D_LENGTH + SPARSE_BUDGET = 524288.

mkdir -p logs

# Activate conda env. Override with: CONDA_ENV=your_env sbatch ...
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

# Prefer the local PEFT implementation in this repo.
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"

# Unbuffered stdout so training logs survive the known abort at interpreter
# shutdown (PyArrow core dump after everything is saved).
export PYTHONUNBUFFERED=1

# Limit CPU thread contention and noisy tokenizer warnings.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AWS_EC2_METADATA_DISABLED=true

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
LORA_RANK="${LORA_RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
DATA_PATH="${DATA_PATH:-meta-math/MetaMathQA}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:100000]}"
SEED="${SEED:-42}"
BASE_LR="${BASE_LR:-2e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${HPARAM_INDEX:-0}}"
case "${TASK_ID}" in
    0)  # 6:1 split, baseline LR schedule
        HPARAM_TAG="ratio6to1"
        THETA_D_LENGTH=449390
        SPARSE_BUDGET=74898
        THETA_D_LR=8e-4
        ROSA_SPARSE_LR_MULT=0.2
        ROSA_WARMUP_STEPS=128
        ;;
    1)  # 10:1 split, baseline LR schedule
        HPARAM_TAG="ratio10to1"
        THETA_D_LENGTH=476626
        SPARSE_BUDGET=47662
        THETA_D_LR=8e-4
        ROSA_SPARSE_LR_MULT=0.2
        ROSA_WARMUP_STEPS=128
        ;;
    2)  # 12:1 split, baseline LR schedule
        HPARAM_TAG="ratio12to1"
        THETA_D_LENGTH=483958
        SPARSE_BUDGET=40330
        THETA_D_LR=8e-4
        ROSA_SPARSE_LR_MULT=0.2
        ROSA_WARMUP_STEPS=128
        ;;
    3)  # 16:1 split, baseline LR schedule
        HPARAM_TAG="ratio16to1"
        THETA_D_LENGTH=493448
        SPARSE_BUDGET=30840
        THETA_D_LR=8e-4
        ROSA_SPARSE_LR_MULT=0.2
        ROSA_WARMUP_STEPS=128
        ;;
    4)  # best 8:1 split with a higher theta_d learning rate
        HPARAM_TAG="ratio8to1_lrtd1e-3"
        THETA_D_LENGTH=466034
        SPARSE_BUDGET=58254
        THETA_D_LR=1e-3
        ROSA_SPARSE_LR_MULT=0.2
        ROSA_WARMUP_STEPS=128
        ;;
    5)  # best 8:1 split with a gentler sparse LR multiplier
        HPARAM_TAG="ratio8to1_lrs0.1"
        THETA_D_LENGTH=466034
        SPARSE_BUDGET=58254
        THETA_D_LR=8e-4
        ROSA_SPARSE_LR_MULT=0.1
        ROSA_WARMUP_STEPS=128
        ;;
    *)
        echo "Unsupported SLURM_ARRAY_TASK_ID/HPARAM_INDEX=${TASK_ID}; expected 0-5." >&2
        exit 1
        ;;
esac
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"

if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
    echo "theta_d_length + sparse_budget must equal total_trainable_budget." >&2
    echo "Got td=${THETA_D_LENGTH}, sb=${SPARSE_BUDGET}, total=${TOTAL_TRAINABLE_BUDGET}" >&2
    exit 1
fi

OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/unilora_rosa_snip_1gpu_${HPARAM_TAG}}"
mkdir -p "${OUTPUT}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo ">>> UniLoRA-RoSA-SNIP single-GPU math instruction tuning, hparam round 2"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> hparam_tag=${HPARAM_TAG} (array task ${TASK_ID})"
echo ">>> model=${BASE_MODEL} data=${DATA_PATH} split=${DATASET_SPLIT}"
echo ">>> total_budget=${TOTAL_TRAINABLE_BUDGET} theta_d=${THETA_D_LENGTH} sparse_budget=${SPARSE_BUDGET}"
echo ">>> base_lr=${BASE_LR} theta_d_lr=${THETA_D_LR} sparse_lr_mult=${ROSA_SPARSE_LR_MULT}"
echo ">>> warmup_steps=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS}"
echo ">>> output=${OUTPUT}"
echo ">>> resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-none}"
echo ">>> visible_gpus=${CUDA_VISIBLE_DEVICES:-set by Slurm}"

python intruction_tuning_unilora_multi_gpu.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT}" \
    --unilora_variant unilora_rosa_snip \
    --lora_r "${LORA_RANK}" \
    --theta_d_length "${THETA_D_LENGTH}" \
    --init_theta_d_bound 0.02 \
    --rosa_sparse_budget "${SPARSE_BUDGET}" \
    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}" \
    --rosa_mask_steps "${ROSA_MASK_STEPS}" \
    --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}" \
    --rosa_reset_optimizer_on_mask True \
    --rosa_decay_sparse_lr_after_activation True \
    --data_path "${DATA_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_field query response \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate "${BASE_LR}" \
    --learning_rate_vector_bank "${THETA_D_LR}" \
    --learning_rate_theta_d "${THETA_D_LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --device_map auto \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU:-44GiB}" \
    --max_memory_cpu "${MAX_MEMORY_CPU:-128GiB}" \
    --report_to tensorboard \
    "${RESUME_ARGS[@]}" \
    --seed "${SEED}"

echo "UniLoRA-RoSA-SNIP single-GPU hparam2 run ${HPARAM_TAG} finished."
