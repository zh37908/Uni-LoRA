#!/bin/bash
#SBATCH --job-name=p2_chat_eval_5880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --output=logs/p2_chat_eval_5880_%j.out
#SBATCH --error=logs/p2_chat_eval_5880_%j.err

# P2 chat 评测扩展: 对已有 Alpaca 指令微调 checkpoint 补 IFEval(可验证指令遵循)
# 与 MMLU 5-shot(能力保持/遗忘)两个维度, 复用 checkpoint 不需要新训练。
# 用法(对每个 checkpoint 提交一次):
#   ADAPTER_PATH=/path/to/adapter RUN_TAG=llama2_unilora_s42 \
#     sbatch submit_eval_p2_ifeval_mmlu_1gpu.sh
# 也可以直接评测 base model(不给 ADAPTER_PATH, 作为对照行):
#   RUN_TAG=llama2_7b_base sbatch submit_eval_p2_ifeval_mmlu_1gpu.sh
# 依赖: unilora_modern 环境中安装了 lm_eval(pip install "lm_eval[ifeval]")。

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-unilora_modern}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/instruction_tuning

export PYTHONPATH="/home/hzhaobi/Uni-LoRA/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_TAG="${RUN_TAG:-$(basename "${BASE_MODEL}")_$(date +%s)}"
MERGED_ROOT="${MERGED_ROOT:-output_merged/p2_chat_eval}"
RESULT_ROOT="${RESULT_ROOT:-results/p2_chat_eval}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
RUN_MMLU="${RUN_MMLU:-1}"
RUN_IFEVAL="${RUN_IFEVAL:-1}"
KEEP_MERGED="${KEEP_MERGED:-0}"

RESULT_DIR="${RESULT_ROOT}/${RUN_TAG}"
mkdir -p "${RESULT_DIR}"

if [[ -n "${ADAPTER_PATH}" ]]; then
    if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "Missing adapter_config.json under ADAPTER_PATH=${ADAPTER_PATH}" >&2
        exit 1
    fi
    EVAL_MODEL="${MERGED_ROOT}/${RUN_TAG}"
    if [[ ! -f "${EVAL_MODEL}/config.json" ]]; then
        rm -rf "${EVAL_MODEL}"
        mkdir -p "${EVAL_MODEL}"
        echo ">>> Merging adapter into base model"
        python /home/hzhaobi/Uni-LoRA/math_instruction_tuning/utils/merge_adapter_to_base_model.py \
            --base_model "${BASE_MODEL}" \
            --adapter "${ADAPTER_PATH}" \
            --output_path "${EVAL_MODEL}"
    else
        echo ">>> Reusing merged model at ${EVAL_MODEL}"
    fi
else
    EVAL_MODEL="${BASE_MODEL}"
fi

echo ">>> P2 chat eval run=${RUN_TAG}"
echo ">>> eval_model=${EVAL_MODEL}"
echo ">>> results=${RESULT_DIR}"

MODEL_ARGS="pretrained=${EVAL_MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=0.85,max_model_len=${MAX_MODEL_LEN},dtype=bfloat16"

if [[ "${RUN_MMLU}" == "1" ]]; then
    echo ">>> Running MMLU (5-shot)"
    lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "${RESULT_DIR}/mmlu" \
        2>&1 | tee "${RESULT_DIR}/mmlu.log"
fi

if [[ "${RUN_IFEVAL}" == "1" ]]; then
    echo ">>> Running IFEval"
    lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks ifeval \
        --batch_size auto \
        --output_path "${RESULT_DIR}/ifeval" \
        2>&1 | tee "${RESULT_DIR}/ifeval.log"
fi

if [[ -n "${ADAPTER_PATH}" && "${KEEP_MERGED}" != "1" ]]; then
    echo ">>> Removing merged model to save disk: ${EVAL_MODEL}"
    rm -rf "${EVAL_MODEL}"
fi

echo "P2 chat eval finished: ${RUN_TAG}."
