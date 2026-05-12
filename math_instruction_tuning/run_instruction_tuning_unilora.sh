#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
# BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
LORA_RANK="${LORA_RANK:-4}"
NUM_VECTORS="${NUM_VECTORS:-2048}"
VECTOR_LENGTH="${VECTOR_LENGTH:-524288}"
OUTPUT="${OUTPUT:-output}"
DATA_PATH="${DATA_PATH:-meta-math/MetaMathQA}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:100000]}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
SEED="${SEED:-42}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"
RUN_MERGE_EVAL="${RUN_MERGE_EVAL:-0}"

mkdir -p "${OUTPUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python intruction_tuning_unilora.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT}" \
    --lora_r "${LORA_RANK}" \
    --num_vectors "${NUM_VECTORS}" \
    --vector_length "${VECTOR_LENGTH}" \
    --save_only_topk_weights True \
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
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --device_map auto \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}" \
    --max_memory_cpu "${MAX_MEMORY_CPU}" \
    --report_to tensorboard \
    --seed "${SEED}"

if [[ "${RUN_MERGE_EVAL}" == "1" ]]; then
    MERGED_PATH="${OUTPUT}_merged"
    mkdir -p "${MERGED_PATH}"

    FT_PATH=$(find "${OUTPUT}" -type d -path "*/ft" | grep "${BASE_MODEL}" | grep "rank_${LORA_RANK}" | tail -n 1)
    python -m utils.merge_adapter_to_base_model --base_mode "${BASE_MODEL}" --adapter "${FT_PATH}" --output_path "${MERGED_PATH}"

    python instruction_tuning_eval/gsm8k_eval.py --model "${MERGED_PATH}"
    python instruction_tuning_eval/MATH_eval.py --model "${MERGED_PATH}"
fi
