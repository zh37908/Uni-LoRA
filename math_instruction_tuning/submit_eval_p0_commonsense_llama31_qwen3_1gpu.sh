#!/bin/bash
#SBATCH --job-name=p0_cs_eval_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-11
#SBATCH --output=logs/p0_cs_eval_5880_%A_%a.out
#SBATCH --error=logs/p0_cs_eval_5880_%A_%a.err

# P0 常识推理评测: merge checkpoint 后用 vLLM 依次评测 8 个基准
# (BoolQ/PIQA/SIQA/HellaSwag/WinoGrande/ARC-c/ARC-e/OBQA)。
# TASK_ID 映射与训练脚本一致: TASK_ID = backbone_idx * 6 + config_idx,
#   config: 0=lora, 1=unilora, 2=prolosa 2:1, 3=4:1, 4=8:1, 5=12:1;  SEED 环境变量给定(默认42)。

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
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
BACKBONE_IDX=$((TASK_ID / 6))
CONFIG_IDX=$((TASK_ID % 6))

case "${BACKBONE_IDX}" in
    0) MODEL_TAG="llama31"; BASE_MODEL="${LLAMA31_BASE_MODEL:-NousResearch/Meta-Llama-3.1-8B}" ;;
    1) MODEL_TAG="qwen3";   BASE_MODEL="${QWEN3_BASE_MODEL:-Qwen/Qwen3-8B}" ;;
    *) echo "Bad backbone idx ${BACKBONE_IDX}" >&2; exit 1 ;;
esac

METHODS=(lora unilora prolosa_r2to1 prolosa_r4to1 prolosa_r8to1 prolosa_r12to1)
METHOD="${METHODS[${CONFIG_IDX}]}"
SEED="${SEED:-42}"

DATA_PATH="${DATA_PATH:-data/commonsense/commonsense_170k.json}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/p0_commonsense}"
MERGED_ROOT="${MERGED_ROOT:-output_merged/p0_commonsense}"
RESULT_ROOT="${RESULT_ROOT:-results/p0_commonsense}"
COMMONSENSE_DATA_DIR="${COMMONSENSE_DATA_DIR:-data/commonsense}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
FORCE_MERGE="${FORCE_MERGE:-0}"
KEEP_MERGED="${KEEP_MERGED:-0}"
DATASETS=(${DATASETS:-boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa})

RUN_TAG="${MODEL_TAG}_${METHOD}_s${SEED}"
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_TAG}_${METHOD}_s${SEED}/${BASE_MODEL}/${DATA_PATH}_split_${DATASET_SPLIT}"

if [[ -z "${ADAPTER_PATH:-}" ]]; then
    ADAPTER_PATH="$(ls -dt "${TRAIN_DIR}"/*seed_"${SEED}"/output_*/ft 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${ADAPTER_PATH}" || ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "No trained adapter found for ${RUN_TAG} under ${TRAIN_DIR}" >&2
    exit 1
fi

MERGED_PATH="${MERGED_ROOT}/${RUN_TAG}"
RESULT_DIR="${RESULT_ROOT}/${RUN_TAG}"
mkdir -p "${RESULT_DIR}"

echo ">>> P0 commonsense eval task_id=${TASK_ID} run=${RUN_TAG}"
echo ">>> adapter=${ADAPTER_PATH}"
echo ">>> merged=${MERGED_PATH} results=${RESULT_DIR}"

if [[ "${FORCE_MERGE}" == "1" || ! -f "${MERGED_PATH}/config.json" ]]; then
    rm -rf "${MERGED_PATH}"
    mkdir -p "${MERGED_PATH}"
    echo ">>> Merging adapter into base model"
    python -m utils.merge_adapter_to_base_model \
        --base_model "${BASE_MODEL}" \
        --adapter "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}"
else
    echo ">>> Reusing existing merged model at ${MERGED_PATH}"
fi

for DATASET in "${DATASETS[@]}"; do
    DATA_FILE="${COMMONSENSE_DATA_DIR}/${DATASET}_test.json"
    if [[ ! -f "${DATA_FILE}" ]]; then
        echo "Missing test file ${DATA_FILE}, skip ${DATASET}" >&2
        continue
    fi
    if [[ -s "${RESULT_DIR}/${DATASET}.log" ]] && rg -q "acc====" "${RESULT_DIR}/${DATASET}.log"; then
        echo ">>> Skip ${DATASET}: result already exists"
        continue
    fi
    echo ">>> Evaluating ${DATASET}"
    python instruction_tuning_eval/commonsense_eval.py \
        --model "${MERGED_PATH}" \
        --dataset "${DATASET}" \
        --data_file "${DATA_FILE}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --max_model_len "${MAX_MODEL_LEN:-4096}" \
        --output_file "${RESULT_DIR}/${DATASET}_predictions.json" \
        2>&1 | tee "${RESULT_DIR}/${DATASET}.log"
done

if [[ "${KEEP_MERGED}" != "1" ]]; then
    echo ">>> Removing merged model to save disk: ${MERGED_PATH}"
    rm -rf "${MERGED_PATH}"
fi

echo "P0 commonsense eval finished: ${RUN_TAG}."
