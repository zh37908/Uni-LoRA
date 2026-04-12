#!/usr/bin/env bash
set -euo pipefail

# Small ablation grid for UniLoRA soft weight-sharing.
# Usage:
#   bash acf_submit_unilora_variant_unilora_soft_weight_sharing.sh
#
# Each run gets a unique JSON + TensorBoard subfolder (K/tau/grouping in the stem).
# Omit NUM_EPOCHS to use GLUE defaults from run_unilora_variants_glue.py (e.g. MRPC+roberta-base -> 30 epochs).
# For smoke tests: NUM_EPOCHS=5 bash acf_submit_unilora_variant_unilora_soft_weight_sharing.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# peft package: .../NLU/peft/src/peft
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}../../src"

MODEL_NAME="${MODEL_NAME:-roberta-base}"
TASK="${TASK:-mrpc}"
SEED="${SEED:-0}"
HEAD_LR="${HEAD_LR:-1e-3}"
RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
OUT_DIR="${OUT_DIR:-results_unilora_soft_weight_sharing}"
NUM_EPOCHS="${NUM_EPOCHS:-}"

EPOCH_ARG=()
if [[ -n "${NUM_EPOCHS}" ]]; then
  EPOCH_ARG=(--num_epochs "${NUM_EPOCHS}")
fi

K_LIST=("8" "16" "32")
TAU_LIST=("1e-4" "5e-4" "1e-3")
GROUPING_LIST=("global" "ab_split" "per_layer")

for K in "${K_LIST[@]}"; do
  for TAU in "${TAU_LIST[@]}"; do
    for GROUPING in "${GROUPING_LIST[@]}"; do
      python run_unilora_variants_glue.py \
        --model_name "${MODEL_NAME}" \
        --task "${TASK}" \
        --variant unilora_soft_weight_sharing \
        --head_lr "${HEAD_LR}" \
        --seed "${SEED}" \
        --rank "${RANK}" \
        --theta_d_length "${THETA_D_LENGTH}" \
        "${EPOCH_ARG[@]}" \
        --sws_num_components "${K}" \
        --sws_tau "${TAU}" \
        --sws_grouping "${GROUPING}" \
        --out_dir "${OUT_DIR}"
    done
  done
done
