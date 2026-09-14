#!/bin/bash
#SBATCH --job-name=unilora_math_eval_2l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_math_eval_2l20_%j.out
#SBATCH --error=logs/unilora_math_eval_2l20_%j.err

mkdir -p logs

# Activate conda env. Override with: CONDA_ENV=your_env sbatch ...
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

# Prefer the local PEFT implementation in this repo.
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"

# Limit CPU thread contention and noisy tokenizer warnings.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
ADAPTER_PATH="${ADAPTER_PATH:-output/google/gemma-7b/meta-math/MetaMathQA_split_train[:100000]/rank_4_lr_0.002_seed_42/output_2026-05-13T17:22:46-327253/ft}"
MERGED_PATH="${MERGED_PATH:-output_merged/google-gemma-7b_unilora_metamath_rank4_seed42}"
GSM8K_DATA_FILE="${GSM8K_DATA_FILE:-data/math_eval/gsm8k_test.jsonl}"
MATH_DATA_FILE="${MATH_DATA_FILE:-data/math_eval/MATH_test.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
START="${START:-0}"
END="${END:-9223372036854775807}"
FORCE_MERGE="${FORCE_MERGE:-0}"
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_MATH="${RUN_MATH:-1}"

echo ">>> Starting UniLoRA math evaluation on 2 L20 GPUs"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> base_model=${BASE_MODEL}"
echo ">>> adapter_path=${ADAPTER_PATH}"
echo ">>> merged_path=${MERGED_PATH}"
echo ">>> tensor_parallel_size=${TENSOR_PARALLEL_SIZE} batch_size=${BATCH_SIZE}"
echo ">>> eval range start=${START} end=${END}"

if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json under ADAPTER_PATH=${ADAPTER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "Missing adapter_model.safetensors under ADAPTER_PATH=${ADAPTER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${GSM8K_DATA_FILE}" ]]; then
    echo "Missing GSM8K data file: ${GSM8K_DATA_FILE}" >&2
    exit 1
fi

if [[ ! -f "${MATH_DATA_FILE}" ]]; then
    echo "Missing MATH data file: ${MATH_DATA_FILE}" >&2
    exit 1
fi

if [[ "${FORCE_MERGE}" == "1" || ! -f "${MERGED_PATH}/config.json" ]]; then
    echo ">>> Merging adapter into base model"
    mkdir -p "${MERGED_PATH}"
    python -m utils.merge_adapter_to_base_model \
        --base_model "${BASE_MODEL}" \
        --adapter "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}"
else
    echo ">>> Reusing existing merged model at ${MERGED_PATH}"
fi

if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ">>> Running GSM8K evaluation"
    python instruction_tuning_eval/gsm8k_eval.py \
        --model "${MERGED_PATH}" \
        --data_file "${GSM8K_DATA_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --start "${START}" \
        --end "${END}"
fi

if [[ "${RUN_MATH}" == "1" ]]; then
    echo ">>> Running MATH evaluation"
    python instruction_tuning_eval/MATH_eval.py \
        --model "${MERGED_PATH}" \
        --data_file "${MATH_DATA_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --start "${START}" \
        --end "${END}"
fi

echo "UniLoRA math evaluation finished."
