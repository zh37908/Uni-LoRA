#!/bin/bash
#SBATCH --job-name=unilora_rosa_snip_mgpu_2to1_gemma_mathonly_2l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_rosa_snip_mgpu_2to1_gemma_mathonly_2l20_%j.out
#SBATCH --error=logs/unilora_rosa_snip_mgpu_2to1_gemma_mathonly_2l20_%j.err

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

DEFAULT_ADAPTER_PATH="output/unilora_rosa_snip_multi_gpu_2to1_gemma/google/gemma-7b/meta-math/MetaMathQA_split_train[:100000]/unilora_rosa_snip_multi_gpu_td_349525_sb_174763_w_128_m_1_rank_4_lr_0.0002_seed_42/output_2026-05-25T01:26:15-267891/ft"

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
ADAPTER_PATH="${ADAPTER_PATH:-${DEFAULT_ADAPTER_PATH}}"
MERGED_PATH="${MERGED_PATH:-output_merged/google-gemma-7b_unilora_rosa_snip_multi_gpu_2to1_metamath_td349525_sb174763_rank4_seed42}"
MATH_DATA_FILE="${MATH_DATA_FILE:-data/math_eval/MATH_test.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
START="${START:-0}"
END="${END:-9223372036854775807}"
FORCE_MERGE="${FORCE_MERGE:-0}"

merged_model_complete() {
    [[ -f "${MERGED_PATH}/config.json" ]] || return 1
    [[ -f "${MERGED_PATH}/tokenizer_config.json" ]] || return 1
    [[ -f "${MERGED_PATH}/tokenizer.json" || -f "${MERGED_PATH}/tokenizer.model" ]] || return 1

    if [[ -s "${MERGED_PATH}/model.safetensors" || -s "${MERGED_PATH}/pytorch_model.bin" ]]; then
        return 0
    fi

    if [[ -f "${MERGED_PATH}/model.safetensors.index.json" ]]; then
        python - "${MERGED_PATH}" <<'PY'
import json
import os
import sys

merged_path = sys.argv[1]
index_path = os.path.join(merged_path, "model.safetensors.index.json")
with open(index_path, "r", encoding="utf-8") as f:
    index = json.load(f)

shards = set(index.get("weight_map", {}).values())
if not shards:
    raise SystemExit(1)

for shard in shards:
    shard_path = os.path.join(merged_path, shard)
    if not os.path.isfile(shard_path) or not os.path.getsize(shard_path):
        raise SystemExit(1)
PY
        return $?
    fi

    return 1
}

echo ">>> Starting UniLoRA-RoSA-SNIP multi-gpu 2:1 Gemma MATH-only evaluation on 2 L20 GPUs"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> base_model=${BASE_MODEL}"
echo ">>> adapter_path=${ADAPTER_PATH}"
echo ">>> merged_path=${MERGED_PATH}"
echo ">>> math_data_file=${MATH_DATA_FILE}"
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

if [[ ! -f "${MATH_DATA_FILE}" ]]; then
    echo "Missing MATH data file: ${MATH_DATA_FILE}" >&2
    exit 1
fi

if [[ "${FORCE_MERGE}" == "1" ]] || ! merged_model_complete; then
    if [[ -d "${MERGED_PATH}" && "${FORCE_MERGE}" != "1" ]]; then
        echo ">>> Removing incomplete merged model at ${MERGED_PATH}"
        rm -rf "${MERGED_PATH}"
    fi
    echo ">>> Merging adapter into base model"
    mkdir -p "${MERGED_PATH}"
    python -m utils.merge_adapter_to_base_model \
        --base_model "${BASE_MODEL}" \
        --adapter "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}"
else
    echo ">>> Reusing existing complete merged model at ${MERGED_PATH}"
fi

echo ">>> Running MATH evaluation"
python instruction_tuning_eval/MATH_eval.py \
    --model "${MERGED_PATH}" \
    --data_file "${MATH_DATA_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --start "${START}" \
    --end "${END}"

echo "UniLoRA-RoSA-SNIP multi-gpu 2:1 Gemma MATH-only evaluation finished."
