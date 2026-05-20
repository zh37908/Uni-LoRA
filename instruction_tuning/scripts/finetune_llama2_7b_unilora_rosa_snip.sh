#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTRUCTION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INSTRUCTION_ROOT}/.." && pwd)"
cd "${INSTRUCTION_ROOT}"

export PYTHONPATH="${REPO_ROOT}/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
SPARSE_BUDGET="${SPARSE_BUDGET:-16384}"
THETA_D_LENGTH="${THETA_D_LENGTH:-507904}"
THETA_D_LR="${THETA_D_LR:-8e-4}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
SEED="${SEED:-0}"

if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
    echo "theta_d_length + sparse_budget must equal total_trainable_budget."
    echo "Got td=${THETA_D_LENGTH}, sb=${SPARSE_BUDGET}, total=${TOTAL_TRAINABLE_BUDGET}"
    exit 1
fi

METHOD_NAME="unilora_rosa_snip_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/llama2_7b_${METHOD_NAME}_seed${SEED}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python qlora_unilora.py \
    --model_name_or_path "${MODEL}" \
    --use_auth_token True \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 20 \
    --save_strategy no \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy no \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --unilora_variant unilora_rosa_snip \
    --lora_r 4 \
    --theta_d_length "${THETA_D_LENGTH}" \
    --init_theta_d_bound 0.02 \
    --rosa_sparse_budget "${SPARSE_BUDGET}" \
    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}" \
    --rosa_mask_steps "${ROSA_MASK_STEPS}" \
    --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}" \
    --rosa_reset_optimizer_on_mask True \
    --rosa_decay_sparse_lr_after_activation True \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --gradient_checkpointing \
    --dataset alpaca-clean \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 0 \
    --learning_rate_vector_bank "${THETA_D_LR}" \
    --learning_rate_theta_d "${THETA_D_LR}" \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed "${SEED}"
