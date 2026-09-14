#!/bin/bash
#SBATCH --job-name=p1_vit_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-29
#SBATCH --output=logs/p1_vit_5880_%A_%a.out
#SBATCH --error=logs/p1_vit_5880_%A_%a.err

# P1 视觉实验: ViT-B/16 + LoRA / Uni-LoRA / ProLoSA, 匹配预算
# (LoRA 全空间 D=147456; Uni-LoRA d=24600; ProLoSA theta_d+sparse 恒等于 24600)。
# 注: 曾按原论文对齐 d=72000 + lr 1e-2/head 1e-2~5e-2 (results/p1_vision_d72k),
# 但在少样本+batch64 设定下全面掉点 1~6, 故回退到 d=24600 + 4e-3/3e-3 作为正式配置。
# 用 COMPRESSED_BUDGET=72000 RESULT_ROOT=results/p1_vision_d72k 可复现 d72k 设定。
# 两阶段策略(与 LLM 实验一致):
#   阶段1(默认 SEED=42): 5 个设定 x 6 个配置 = 30 个组合, 全数组 0-29。
#   阶段2: 确定最优 ProLoSA 比例后, SEED=43/44 只补 lora/unilora/prolosa(best)。
# TASK_ID 映射: setting_idx = id / 6, config_idx = id % 6
#   setting: 0=cifar100@1k, 1=food101@1k, 2=dtd@1k, 3=cifar100@5k, 4=cifar100@full
#   config:  0=lora, 1=unilora, 2=prolosa 2:1, 3=prolosa 4:1, 4=prolosa 8:1, 5=prolosa 12:1

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-unilora_modern}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/ViT

export PYTHONPATH="/home/hzhaobi/Uni-LoRA/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
SETTING_IDX=$((TASK_ID / 6))
CONFIG_IDX=$((TASK_ID % 6))

# 阶段1只跑单种子; 阶段2用 SEED=43/44 重复提交最优配置。
SEED="${SEED:-42}"

case "${SETTING_IDX}" in
    0) DATASET="cifar100"; TRAIN_SIZE=1000 ;;
    1) DATASET="food101";  TRAIN_SIZE=1000 ;;
    2) DATASET="dtd";      TRAIN_SIZE=1000 ;;
    3) DATASET="cifar100"; TRAIN_SIZE=5000 ;;
    4) DATASET="cifar100"; TRAIN_SIZE=-1 ;;
    *) echo "Bad setting idx ${SETTING_IDX}" >&2; exit 1 ;;
esac

MODEL="${MODEL:-google/vit-base-patch16-224-in21k}"
LORA_RANK="${LORA_RANK:-4}"
COMPRESSED_BUDGET="${COMPRESSED_BUDGET:-24600}"
LORA_LR="${LORA_LR:-1e-3}"
UNILORA_LR="${UNILORA_LR:-4e-3}"
PROLOSA_BASE_LR="${PROLOSA_BASE_LR:-1e-3}"
PROLOSA_THETA_D_LR="${PROLOSA_THETA_D_LR:-4e-3}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
HEAD_LR="${HEAD_LR:-3e-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SUBSET_SEED="${SUBSET_SEED:-42}"

if [[ "${TRAIN_SIZE}" == "1000" ]]; then
    NUM_EPOCHS="${NUM_EPOCHS:-20}"
    ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-32}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-10}"
    ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
fi

SIZE_TAG="${TRAIN_SIZE}"
[[ "${TRAIN_SIZE}" == "-1" ]] && SIZE_TAG="full"

# ProLoSA 比例扫描表: theta_d + sparse 恒等于预算 (默认 24600; d72k 时用 72000 表)。
if [[ "${COMPRESSED_BUDGET}" == "72000" ]]; then
    declare -A RATIO_THETA=( [2to1]=48000 [4to1]=57600 [8to1]=64000 [12to1]=66462 )
    declare -A RATIO_SPARSE=( [2to1]=24000 [4to1]=14400 [8to1]=8000  [12to1]=5538 )
else
    declare -A RATIO_THETA=( [2to1]=16400 [4to1]=19680 [8to1]=21867 [12to1]=22708 )
    declare -A RATIO_SPARSE=( [2to1]=8200  [4to1]=4920  [8to1]=2733  [12to1]=1892 )
fi

METHOD_ARGS=()
RATIO_TAG=""
case "${CONFIG_IDX}" in
    0)
        METHOD_TAG="lora"
        METHOD_ARGS=(--unilora_variant lora --learning_rate "${LORA_LR}")
        ;;
    1)
        METHOD_TAG="unilora"
        METHOD_ARGS=(
            --unilora_variant unilora
            --vector_length "${COMPRESSED_BUDGET}"
            --learning_rate "${UNILORA_LR}"
        )
        ;;
    2|3|4|5)
        RATIOS=(2to1 4to1 8to1 12to1)
        RATIO_TAG="${RATIOS[$((CONFIG_IDX - 2))]}"
        THETA_D_LENGTH="${RATIO_THETA[${RATIO_TAG}]}"
        SPARSE_BUDGET="${RATIO_SPARSE[${RATIO_TAG}]}"
        if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${COMPRESSED_BUDGET}" ]]; then
            echo "theta_d_length + sparse_budget must equal compressed_budget." >&2
            exit 1
        fi
        METHOD_TAG="prolosa_r${RATIO_TAG}"
        METHOD_ARGS=(
            --unilora_variant unilora_rosa_snip
            --theta_d_length "${THETA_D_LENGTH}"
            --init_theta_d_bound 0.02
            --rosa_sparse_budget "${SPARSE_BUDGET}"
            --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
            --rosa_mask_steps 1
            --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}"
            --rosa_reset_optimizer_on_mask True
            --rosa_decay_sparse_lr_after_activation True
            --learning_rate "${PROLOSA_BASE_LR}"
            --learning_rate_vector_bank "${PROLOSA_THETA_D_LR}"
            --learning_rate_theta_d "${PROLOSA_THETA_D_LR}"
        )
        ;;
    *) echo "Bad config idx ${CONFIG_IDX}" >&2; exit 1 ;;
esac

RESULT_ROOT="${RESULT_ROOT:-results/p1_vision}"
RESULT_FILE="${RESULT_ROOT}/${DATASET}_${SIZE_TAG}/${METHOD_TAG}_s${SEED}.json"
OUTPUT_DIR="output/${RESULT_ROOT#results/}/${DATASET}_${SIZE_TAG}/${METHOD_TAG}_s${SEED}"

if [[ -s "${RESULT_FILE}" ]]; then
    echo "Skip existing result: ${RESULT_FILE}"
    exit 0
fi

echo ">>> P1 vision task_id=${TASK_ID}"
echo ">>> dataset=${DATASET} size=${SIZE_TAG} method=${METHOD_TAG} seed=${SEED}"
echo ">>> epochs=${NUM_EPOCHS} batch=${BATCH_SIZE} head_lr=${HEAD_LR} result=${RESULT_FILE}"

python train_vit_p1.py \
    --model_name_or_path "${MODEL}" \
    --dataset "${DATASET}" \
    --train_size "${TRAIN_SIZE}" \
    --subset_seed "${SUBSET_SEED}" \
    --head_lr "${HEAD_LR}" \
    --result_file "${RESULT_FILE}" \
    --lora_r "${LORA_RANK}" \
    "${METHOD_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --eval_strategy epoch \
    --save_strategy no \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --bf16 True \
    --dataloader_num_workers 4 \
    --report_to none \
    --seed "${SEED}"

echo "P1 vision finished: ${DATASET} ${SIZE_TAG} ${METHOD_TAG} seed=${SEED}."
