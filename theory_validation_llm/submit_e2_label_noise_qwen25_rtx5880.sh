#!/bin/bash
#SBATCH --job-name=e2_csqa_qwen25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0,1,2,3,4,5,6,7,12,13,18,19,20,21,22,23
#SBATCH --output=logs/e2_csqa_qwen25_%A_%a.out
#SBATCH --error=logs/e2_csqa_qwen25_%A_%a.err

# P4/E2 标签噪声敏感性 (真实 LLM 版): Qwen2.5-1.5B + CommonsenseQA 全量训练集。
# 噪声比例 eta ∈ {0, 0.1, 0.2, 0.3}(均匀翻转到错误选项, 噪声标签跨方法共享),
# 验证集保持干净。
# TASK_ID = eta_idx * 6 + config_idx  (完整网格 0-23)
#   eta:    0=0.0, 1=0.1, 2=0.2, 3=0.3
#   config: 0=lora, 1=unilora, 2=prolosa 2:1, 3=4:1, 4=8:1, 5=12:1
# 两阶段策略(默认 SEED=42):
#   阶段1(默认 array): lora/unilora 全部 4 个 eta + ProLoSA 比例扫描只在 eta∈{0.0,0.3};
#   定最优比例后, 用 --array 补 prolosa(best) 其余 eta, 再 SEED=43/44 补多种子。

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-unilora_modern}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/theory_validation_llm

export PYTHONPATH="/home/hzhaobi/Uni-LoRA/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
ETAS=(0.0 0.1 0.2 0.3)

ETA="${ETAS[$((TASK_ID / 6))]}"
CONFIG_IDX=$((TASK_ID % 6))
# 阶段1只跑单种子; 阶段2用 SEED=43/44 重复提交最优配置。
SEED="${SEED:-42}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
LORA_RANK="${LORA_RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
LORA_LR="${LORA_LR:-2e-4}"
UNILORA_LR="${UNILORA_LR:-2e-3}"
PROLOSA_BASE_LR="${PROLOSA_BASE_LR:-2e-4}"
PROLOSA_THETA_D_LR="${PROLOSA_THETA_D_LR:-8e-4}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
MAX_STEPS="${MAX_STEPS:-900}"
NOISE_SEED="${NOISE_SEED:-5678}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"

# ProLoSA 比例扫描表: theta_d + sparse 恒等于 524288。
declare -A RATIO_THETA=( [2to1]=349525 [4to1]=419430 [8to1]=466034 [12to1]=483958 )
declare -A RATIO_SPARSE=( [2to1]=174763 [4to1]=104858 [8to1]=58254  [12to1]=40330 )

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
            --num_vectors 2048
            --vector_length "${TOTAL_TRAINABLE_BUDGET}"
            --learning_rate "${UNILORA_LR}"
        )
        ;;
    2|3|4|5)
        RATIOS=(2to1 4to1 8to1 12to1)
        RATIO_TAG="${RATIOS[$((CONFIG_IDX - 2))]}"
        THETA_D_LENGTH="${RATIO_THETA[${RATIO_TAG}]}"
        SPARSE_BUDGET="${RATIO_SPARSE[${RATIO_TAG}]}"
        METHOD_TAG="prolosa_r${RATIO_TAG}"
        METHOD_ARGS=(
            --unilora_variant unilora_rosa_snip
            --theta_d_length "${THETA_D_LENGTH}"
            --init_theta_d_bound 0.02
            --rosa_sparse_budget "${SPARSE_BUDGET}"
            --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
            --rosa_mask_steps 1
            --rosa_reset_optimizer_on_mask True
            --rosa_decay_sparse_lr_after_activation True
            --learning_rate "${PROLOSA_BASE_LR}"
            --learning_rate_vector_bank "${PROLOSA_THETA_D_LR}"
            --learning_rate_theta_d "${PROLOSA_THETA_D_LR}"
        )
        ;;
    *) echo "Bad config idx ${CONFIG_IDX}" >&2; exit 1 ;;
esac

RESULT_ROOT="${RESULT_ROOT:-results/e2_label_noise}"
RESULT_FILE="${RESULT_ROOT}/eta${ETA}/${METHOD_TAG}_s${SEED}.json"
OUTPUT_DIR="output/e2_label_noise/eta${ETA}_${METHOD_TAG}_s${SEED}"

if [[ -s "${RESULT_FILE}" ]]; then
    echo "Skip existing result: ${RESULT_FILE}"
    exit 0
fi

echo ">>> E2 label noise task_id=${TASK_ID}"
echo ">>> eta=${ETA} method=${METHOD_TAG} seed=${SEED} max_steps=${MAX_STEPS}"
echo ">>> result=${RESULT_FILE}"

python train_eval_csqa_qwen.py \
    --experiment e2 \
    --train_fraction 1.0 \
    --label_noise "${ETA}" \
    --noise_seed "${NOISE_SEED}" \
    --result_file "${RESULT_FILE}" \
    --model_name_or_path "${BASE_MODEL}" \
    --lora_r "${LORA_RANK}" \
    "${METHOD_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --model_max_length 512 \
    --save_strategy no \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --seed "${SEED}"

echo "E2 finished: eta=${ETA} ${METHOD_TAG} seed=${SEED}."
