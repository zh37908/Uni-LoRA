#!/bin/bash
# Shared LoRA adapter learning-rate protocol for the theory-validation scripts.
#
# Why: Uni-LoRA/ProLoSA train theta_d with their own lr (--theta_d_lr 5e-3) while
# --head_lr only drives the classifier. Plain LoRA, however, defaulted its A/B matrices
# to --head_lr, i.e. the lr tuned for the compressed methods. On RoBERTa-large this
# diverges on CoLA (5e-3 -> Matthews = 0), RTE (5e-3) and SST-2 (1e-3 -> collapsed
# seeds), producing a degenerate baseline. The symmetric protocol is a per-method
# adapter lr with a shared head lr: LoRA A/B use LoRA-paper-scale rates below.
#
# Usage in a submit script (after cd "${WORK_DIR}"):
#   source theory_common_lora_lr.sh
#   ...
#   lora_lr_setup "${TASK}" "${HEAD_LR}"      # sets LORA_LR, LORA_DIR_TAG, LORA_LR_ARGS
#   METHOD_NAME="lora_r${RANK}${LORA_DIR_TAG}"
#   FULL_CMD+="${LORA_LR_ARGS}"
#
# When the LoRA lr equals the head lr (qnli/mrpc/stsb defaults) nothing changes:
# no directory tag, no extra arg, so previously finished runs are reused.

task_lora_lr() {
  local task="${1}"
  local fallback="${2}"
  case "${task}" in
    cola) echo "${LORA_LR_COLA:-4e-4}" ;;
    sst2) echo "${LORA_LR_SST2:-4e-4}" ;;
    mrpc) echo "${LORA_LR_MRPC:-2e-4}" ;;
    qnli) echo "${LORA_LR_QNLI:-5e-4}" ;;
    rte) echo "${LORA_LR_RTE:-4e-4}" ;;
    stsb) echo "${LORA_LR_STSB:-2e-4}" ;;
    *) echo "${fallback}" ;;
  esac
}

lora_lr_setup() {
  local task="${1}"
  local head_lr="${2}"
  LORA_LR="$(task_lora_lr "${task}" "${head_lr}")"
  if [[ "${LORA_LR}" == "${head_lr}" ]]; then
    LORA_DIR_TAG=""
    LORA_LR_ARGS=""
  else
    LORA_DIR_TAG="_lr${LORA_LR}"
    LORA_LR_ARGS=" --lora_lr ${LORA_LR}"
  fi
  export LORA_LR LORA_DIR_TAG LORA_LR_ARGS
}
