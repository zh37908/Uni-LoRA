#!/bin/bash
#SBATCH --job-name=unilora_rosa_snip_hparam2_eval_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-5%2
#SBATCH --output=logs/unilora_rosa_snip_hparam2_eval_5880_%A_%a.out
#SBATCH --error=logs/unilora_rosa_snip_hparam2_eval_5880_%A_%a.err

# Batch GSM8K + MATH evaluation for hparam round 2 (job 1782183).
# Tags: 6:1, 10:1, 12:1, 16:1, 8:1+theta_d_lr=1e-3, 8:1+sparse_lr_mult=0.1.
#
# The adapter is merged into the base model in float16 so a single 48 GiB
# RTX 5880 can serve it with vLLM (tensor_parallel_size=1). Merged models
# are ~17 GiB each; concurrency is capped at 2 (%2) and each merged model is
# deleted after a successful eval to stay within the disk quota.

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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${HPARAM_INDEX:-0}}"
case "${TASK_ID}" in
    0) HPARAM_TAG="ratio6to1" ;;
    1) HPARAM_TAG="ratio10to1" ;;
    2) HPARAM_TAG="ratio12to1" ;;
    3) HPARAM_TAG="ratio16to1" ;;
    4) HPARAM_TAG="ratio8to1_lrtd1e-3" ;;
    5) HPARAM_TAG="ratio8to1_lrs0.1" ;;
    *)
        echo "Unsupported SLURM_ARRAY_TASK_ID/HPARAM_INDEX=${TASK_ID}; expected 0-5." >&2
        exit 1
        ;;
esac

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
GSM8K_DATA_FILE="${GSM8K_DATA_FILE:-data/math_eval/gsm8k_test.jsonl}"
MATH_DATA_FILE="${MATH_DATA_FILE:-data/math_eval/MATH_test.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
START="${START:-0}"
END="${END:-9223372036854775807}"
FORCE_MERGE="${FORCE_MERGE:-0}"
CLEANUP_MERGED="${CLEANUP_MERGED:-1}"
MERGED_PATH="${MERGED_PATH:-output_merged/gemma7b_unilora_rosa_snip_1gpu_${HPARAM_TAG}}"

# Locate the latest ft adapter for this hyperparameter tag.
if [[ -z "${ADAPTER_PATH:-}" ]]; then
    ADAPTER_PATH=$(ls -d "output/unilora_rosa_snip_1gpu_${HPARAM_TAG}/google/gemma-7b/meta-math/"*/*/output_*/ft 2>/dev/null | sort | tail -1)
fi

echo ">>> UniLoRA-RoSA-SNIP hparam evaluation (GSM8K + MATH) on 1 RTX 5880"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> hparam_tag=${HPARAM_TAG} (array task ${TASK_ID})"
echo ">>> base_model=${BASE_MODEL}"
echo ">>> adapter_path=${ADAPTER_PATH}"
echo ">>> merged_path=${MERGED_PATH}"
echo ">>> tensor_parallel_size=${TENSOR_PARALLEL_SIZE} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN} batch_size=${BATCH_SIZE}"
echo ">>> cleanup_merged=${CLEANUP_MERGED}"

if [[ -z "${ADAPTER_PATH}" || ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "No trained adapter found for tag ${HPARAM_TAG}." >&2
    exit 1
fi
if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "Missing adapter_model.safetensors under ADAPTER_PATH=${ADAPTER_PATH}" >&2
    exit 1
fi
if [[ ! -f "${GSM8K_DATA_FILE}" || ! -f "${MATH_DATA_FILE}" ]]; then
    echo "Missing eval data file(s): ${GSM8K_DATA_FILE} / ${MATH_DATA_FILE}" >&2
    exit 1
fi

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
with open(os.path.join(merged_path, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
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

if [[ "${FORCE_MERGE}" == "1" ]] || ! merged_model_complete; then
    if [[ -d "${MERGED_PATH}" && "${FORCE_MERGE}" != "1" ]]; then
        echo ">>> Removing incomplete merged model at ${MERGED_PATH}"
        rm -rf "${MERGED_PATH}"
    fi
    echo ">>> Merging adapter into base model (float16)"
    mkdir -p "${MERGED_PATH}"
    python -m utils.merge_adapter_to_base_model \
        --base_model "${BASE_MODEL}" \
        --adapter "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}" \
        --dtype float16
else
    echo ">>> Reusing existing complete merged model at ${MERGED_PATH}"
fi

echo ">>> Running GSM8K evaluation"
python instruction_tuning_eval/gsm8k_eval.py \
    --model "${MERGED_PATH}" \
    --data_file "${GSM8K_DATA_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --start "${START}" \
    --end "${END}"

echo ">>> Running MATH evaluation"
python instruction_tuning_eval/MATH_eval.py \
    --model "${MERGED_PATH}" \
    --data_file "${MATH_DATA_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --start "${START}" \
    --end "${END}"

if [[ "${CLEANUP_MERGED}" == "1" ]]; then
    echo ">>> Removing merged model at ${MERGED_PATH} to reclaim disk space"
    rm -rf "${MERGED_PATH}"
fi

echo "UniLoRA-RoSA-SNIP hparam evaluation for ${HPARAM_TAG} finished."
