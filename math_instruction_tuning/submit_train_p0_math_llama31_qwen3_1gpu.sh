#!/bin/bash
#SBATCH --job-name=p0_math_train_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-11
#SBATCH --output=logs/p0_math_train_5880_%A_%a.out
#SBATCH --error=logs/p0_math_train_5880_%A_%a.err

# P0 主实验(数学): Llama-3.1-8B + Qwen3-8B 在 MetaMathQA-100K 上微调。
# 单卡 RTX5880(48GB): 8B 模型 + 梯度检查点 + bf16 单卡即可(与旧 Gemma-7B 单卡扫描同配置)。
# 两阶段策略:
#   阶段1(超参搜索, 默认 SEED=42): 每 backbone 跑 lora / unilora / prolosa 四种
#     theta_d:sparse 比例(2:1, 4:1, 8:1, 12:1), 共 2x6=12 个组合, 全数组 0-11。
#     (Gemma-7B 旧扫描中 12:1 最优, 8:1 次之, 2:1 最差; 新 backbone 需重扫确认。)
#   阶段2(多种子): 确定最优比例后, 只跑 lora / unilora / prolosa(best) 补种子:
#     SEED=43 sbatch --array=0,1,<best_cfg>,6,7,<6+best_cfg> 本脚本 (44 同理)
# 组合映射: TASK_ID = backbone_idx * 6 + config_idx
#   backbone: 0=llama31, 1=qwen3
#   config: 0=lora, 1=unilora, 2=prolosa 2:1, 3=prolosa 4:1, 4=prolosa 8:1, 5=prolosa 12:1
# 预算口径与主表一致: LoRA(全空间, r=4); Uni-LoRA(d=524288) 与 ProLoSA
#   (theta_d+sparse 恒等于 524288, 仅比例不同) 可训练参数量严格相等。

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-unilora_modern}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

# Prefer the local PEFT implementation in this repo.
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
BACKBONE_IDX=$((TASK_ID / 6))
CONFIG_IDX=$((TASK_ID % 6))

# meta-llama/Llama-3.1-8B 是 gated 仓库; 默认用 NousResearch 的同权重镜像,
# 拿到官方访问权限后可用 LLAMA31_BASE_MODEL 覆盖。
case "${BACKBONE_IDX}" in
    0) MODEL_TAG="llama31"; BASE_MODEL="${LLAMA31_BASE_MODEL:-NousResearch/Meta-Llama-3.1-8B}" ;;
    1) MODEL_TAG="qwen3";   BASE_MODEL="${QWEN3_BASE_MODEL:-Qwen/Qwen3-8B}" ;;
    *) echo "Bad backbone idx ${BACKBONE_IDX}" >&2; exit 1 ;;
esac

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

DATA_PATH="${DATA_PATH:-meta-math/MetaMathQA}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:100000]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/p0_math}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# ProLoSA 比例扫描表: theta_d + sparse 恒等于 TOTAL_TRAINABLE_BUDGET。
if [[ "${TOTAL_TRAINABLE_BUDGET}" == "1048576" ]]; then
    declare -A RATIO_THETA=( [2to1]=699051 [4to1]=838861 [8to1]=932068 [12to1]=967916 )
    declare -A RATIO_SPARSE=( [2to1]=349525 [4to1]=209715 [8to1]=116508 [12to1]=80660 )
else
    declare -A RATIO_THETA=( [2to1]=349525 [4to1]=419430 [8to1]=466034 [12to1]=483958 )
    declare -A RATIO_SPARSE=( [2to1]=174763 [4to1]=104858 [8to1]=58254  [12to1]=40330 )
fi

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

echo ">>> P0 math training task_id=${TASK_ID}"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> backbone=${MODEL_TAG} (${BASE_MODEL}) method=${METHOD} seed=${SEED} ratio=${RATIO_TAG:-n/a}"
echo ">>> data=${DATA_PATH} split=${DATASET_SPLIT} rank=${LORA_RANK} budget=${TOTAL_TRAINABLE_BUDGET}"
echo ">>> output=${OUTPUT}"
echo ">>> visible_gpus=${CUDA_VISIBLE_DEVICES:-set by Slurm}"

python intruction_tuning_unilora_multi_gpu.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT}" \
    --lora_r "${LORA_RANK}" \
    "${METHOD_ARGS[@]}" \
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
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --device_map auto \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}" \
    --max_memory_cpu "${MAX_MEMORY_CPU}" \
    --report_to tensorboard \
    "${RESUME_ARGS[@]}" \
    --seed "${SEED}"

echo "P0 math training finished: ${MODEL_TAG} ${METHOD} seed=${SEED}."
