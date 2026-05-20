#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTRUCTION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INSTRUCTION_ROOT}/.." && pwd)"
cd "${INSTRUCTION_ROOT}"

export PYTHONPATH="${REPO_ROOT}/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-meta-llama/Llama-2-13b-hf}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-1048576}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
SEED="${SEED:-0}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
EVAL_STEPS="${EVAL_STEPS:-187}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
OUT_ROOT="${OUT_ROOT:-./output/llama2_13b_unilora_rosa_snip_sweep5_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-./logs/llama2_13b_unilora_rosa_snip_sweep5_seed${SEED}}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# Config columns:
#   tag sparse_budget warmup_steps mask_steps theta_d_lr sparse_lr_mult base_lr reset_optimizer decay_sparse_lr
#
# The first five configs scale the strongest NLU RoSA-SNIP settings to the 13B
# total budget. NLU used total=23040; the sparse budgets below keep the same
# ratios: 720->32768, 1080->49152, 1260->57344, 1440->65536, 1620->73728.
CONFIGS=(
  # Strong ACF result on CoLA/STSB: sb=720,w=128,slrm=0.2,decay=1.
  "nlu_acf_decay 32768 128 1 8e-4 0.2 2e-4 True True"

  # STSB/RTE sweeps centered around larger sparse budget and earlier activation.
  "stsb_center 65536 64 1 8e-4 0.2 2e-4 True False"
  "stsb_center_decay 65536 64 1 8e-4 0.2 2e-4 True True"

  # Sparse-budget local search from STSB: slightly below/above the center.
  "stsb_sparse_low 57344 64 1 8e-4 0.2 2e-4 True False"
  "stsb_sparse_high 73728 64 1 8e-4 0.2 2e-4 True False"

  # Projection:sparse ~= 2:1 under the default total budget:
  # theta_d_length=699051, sparse_budget=349525.
  "projection_sparse_2to1 349525 64 1 8e-4 0.2 2e-4 True False"
)

tag_value() {
  local value="${1}"
  value="${value//./p}"
  value="${value//-e/-}"
  echo "${value}"
}

TOTAL_RUNS="${#CONFIGS[@]}"
RUN_IDX=0

echo ">>> Running ${TOTAL_RUNS} Llama-2-13B UniLoRA-RoSA-SNIP configs sequentially on GPU ${GPU}"
echo ">>> model=${MODEL} seed=${SEED} total_trainable_budget=${TOTAL_TRAINABLE_BUDGET}"
echo ">>> out_root=${OUT_ROOT}"

for CONFIG in "${CONFIGS[@]}"; do
  read -r TAG SPARSE_BUDGET ROSA_WARMUP_STEPS ROSA_MASK_STEPS THETA_D_LR ROSA_SPARSE_LR_MULT BASE_LR ROSA_RESET_OPTIMIZER_ON_MASK ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION <<< "${CONFIG}"

  THETA_D_LENGTH=$((TOTAL_TRAINABLE_BUDGET - SPARSE_BUDGET))
  if [[ "${THETA_D_LENGTH}" -le 0 ]]; then
    echo "Skip ${TAG}: sparse_budget=${SPARSE_BUDGET} leaves invalid theta_d_length=${THETA_D_LENGTH}."
    continue
  fi

  TDLR_TAG="$(tag_value "${THETA_D_LR}")"
  SLRM_TAG="$(tag_value "${ROSA_SPARSE_LR_MULT}")"
  BLR_TAG="$(tag_value "${BASE_LR}")"
  METHOD_NAME="unilora_rosa_snip_${TAG}_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_tdlr${TDLR_TAG}_slrm${SLRM_TAG}_blr${BLR_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}_sdecay${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"
  OUTPUT_DIR="${OUT_ROOT}/${METHOD_NAME}"
  LOG_FILE="${LOG_ROOT}/${METHOD_NAME}.log"

  if [[ "${FORCE}" != "1" && -f "${OUTPUT_DIR}/completed" ]]; then
    echo "Skip completed: ${OUTPUT_DIR}"
    continue
  fi

  RUN_IDX=$((RUN_IDX + 1))
  echo "=================================================="
  echo "[${RUN_IDX}/${TOTAL_RUNS}] ${TAG}"
  echo "theta_d_length=${THETA_D_LENGTH} sparse_budget=${SPARSE_BUDGET} warmup=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS}"
  echo "theta_d_lr=${THETA_D_LR} sparse_lr_mult=${ROSA_SPARSE_LR_MULT} base_lr=${BASE_LR} reset=${ROSA_RESET_OPTIMIZER_ON_MASK} decay=${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"
  echo "output: ${OUTPUT_DIR}"
  echo "log: ${LOG_FILE}"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES="${GPU}" python qlora_unilora.py \
    --model_name_or_path "${MODEL}" \
    --use_auth_token True \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 10 \
    --save_strategy no \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_steps "${EVAL_STEPS}" \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --unilora_variant unilora_rosa_snip \
    --lora_r 4 \
    --theta_d_length "${THETA_D_LENGTH}" \
    --init_theta_d_bound "${INIT_THETA_D_BOUND}" \
    --rosa_sparse_budget "${SPARSE_BUDGET}" \
    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}" \
    --rosa_mask_steps "${ROSA_MASK_STEPS}" \
    --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}" \
    --rosa_reset_optimizer_on_mask "${ROSA_RESET_OPTIMIZER_ON_MASK}" \
    --rosa_decay_sparse_lr_after_activation "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" \
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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate "${BASE_LR}" \
    --learning_rate_vector_bank "${THETA_D_LR}" \
    --learning_rate_theta_d "${THETA_D_LR}" \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed "${SEED}" \
    > "${LOG_FILE}" 2>&1

  echo "Finished ${TAG} -> ${OUTPUT_DIR}"
done

echo "All Llama-2-13B UniLoRA-RoSA-SNIP sweep configs have been processed."
