#!/bin/bash
#SBATCH --job-name=p2_alpaca_train_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-5
#SBATCH --output=logs/p2_alpaca_train_5880_%A_%a.out
#SBATCH --error=logs/p2_alpaca_train_5880_%A_%a.err

# P2 chat 实验的 Alpaca checkpoint 补训: 旧 checkpoint 已不存在, 用与 P0 数学/常识
# 完全一致的现代化管线在 Cleaned Alpaca 上重训 Llama-2-7b。单卡 RTX5880 即可。
# 之后用 instruction_tuning/submit_eval_p2_ifeval_mmlu_1gpu.sh 评 IFEval + MMLU,
# MT-Bench 用 FastChat (fastchat_eval 环境) 另行评测。
# 两阶段策略: 阶段1(默认 SEED=42)全数组 0-5; 定最优比例后 SEED=43/44 补 0,1,<best>。
# TASK_ID = config_idx: 0=lora, 1=unilora, 2=prolosa 2:1, 3=4:1, 4=8:1, 5=12:1
# 数据需先在登录节点生成: python data/prepare_alpaca_cleaned.py

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-unilora_modern}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
CONFIG_IDX="${TASK_ID}"

BASE_MODEL="${LLAMA2_BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
MODEL_TAG="llama2"

# 阶段1只跑单种子; 阶段2用 SEED=43/44 重复提交最优配置。
SEED="${SEED:-42}"

LORA_RANK="${LORA_RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
LORA_LR="${LORA_LR:-2e-4}"
UNILORA_LR="${UNILORA_LR:-2e-3}"
PROLOSA_BASE_LR="${PROLOSA_BASE_LR:-2e-4}"
PROLOSA_THETA_D_LR="${PROLOSA_THETA_D_LR:-8e-4}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"

DATA_PATH="${DATA_PATH:-data/alpaca/alpaca_cleaned.json}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/p2_alpaca}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Missing training data: ${DATA_PATH}. Run data/prepare_alpaca_cleaned.py first." >&2
    exit 1
fi

# ProLoSA 比例扫描表: theta_d + sparse 恒等于 524288。
declare -A RATIO_THETA=( [2to1]=349525 [4to1]=419430 [8to1]=466034 [12to1]=483958 )
declare -A RATIO_SPARSE=( [2to1]=174763 [4to1]=104858 [8to1]=58254  [12to1]=40330 )

METHOD_ARGS=()
RATIO_TAG=""
case "${CONFIG_IDX}" in
    0)
        METHOD="lora"
        METHOD_ARGS=(
            --unilora_variant lora
            --learning_rate "${LORA_LR}"
        )
        ;;
    1)
        METHOD="unilora"
        METHOD_ARGS=(
            --unilora_variant unilora
            --num_vectors 2048
            --vector_length "${TOTAL_TRAINABLE_BUDGET}"
            --save_only_topk_weights True
            --learning_rate "${UNILORA_LR}"
        )
        ;;
    2|3|4|5)
        RATIOS=(2to1 4to1 8to1 12to1)
        RATIO_TAG="${RATIOS[$((CONFIG_IDX - 2))]}"
        THETA_D_LENGTH="${RATIO_THETA[${RATIO_TAG}]}"
        SPARSE_BUDGET="${RATIO_SPARSE[${RATIO_TAG}]}"
        if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
            echo "theta_d_length + sparse_budget must equal total_trainable_budget." >&2
            exit 1
        fi
        METHOD="prolosa_r${RATIO_TAG}"
        METHOD_ARGS=(
            --unilora_variant unilora_rosa_snip
            --theta_d_length "${THETA_D_LENGTH}"
            --init_theta_d_bound 0.02
            --rosa_sparse_budget "${SPARSE_BUDGET}"
            --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
            --rosa_mask_steps "${ROSA_MASK_STEPS}"
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

OUTPUT="${OUTPUT_ROOT}/${MODEL_TAG}_${METHOD}_s${SEED}"
mkdir -p "${OUTPUT}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo ">>> P2 alpaca training task_id=${TASK_ID}"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> backbone=${MODEL_TAG} (${BASE_MODEL}) method=${METHOD} seed=${SEED} ratio=${RATIO_TAG:-n/a}"
echo ">>> data=${DATA_PATH} epochs=${NUM_EPOCHS} eff_batch=$((PER_DEVICE_BATCH * GRAD_ACCUM))"
echo ">>> output=${OUTPUT}"

python intruction_tuning_unilora_multi_gpu.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT}" \
    --lora_r "${LORA_RANK}" \
    "${METHOD_ARGS[@]}" \
    --data_path "${DATA_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_field instruction output \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --device_map auto \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}" \
    --max_memory_cpu "${MAX_MEMORY_CPU}" \
    --report_to tensorboard \
    "${RESUME_ARGS[@]}" \
    --seed "${SEED}"

echo "P2 alpaca training finished: ${MODEL_TAG} ${METHOD} seed=${SEED}."
