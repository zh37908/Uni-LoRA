#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate nlu

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL="${MODEL:-roberta-large}"
TASK="${TASK:-mrpc}"
SEED="${SEED:-0}"
HEAD_LR="${HEAD_LR:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"

ROSA_DENSITY="${ROSA_DENSITY:-0.01}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-64}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_SPARSE_LR="${ROSA_SPARSE_LR:-${THETA_D_LR}}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-1}"

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
model_name = "${MODEL}"
task = "${TASK}"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
try:
    load_dataset("nyu-mll/glue", task)
except Exception:
    pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa"
METHOD_NAME="${VARIANT}_smoke_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_smoke_acf}"
SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${METHOD_NAME}/seed_${SEED}"
mkdir -p "${SEED_DIR}"

LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"

CMD=(
  python "${SCRIPT}"
  --variant "${VARIANT}"
  --model_name "${MODEL}"
  --task "${TASK}"
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --rank "${RANK}"
  --theta_d_length "${THETA_D_LENGTH}"
  --theta_d_lr "${THETA_D_LR}"
  --init_theta_d_bound "${INIT_THETA_D_BOUND}"
  --rosa_density "${ROSA_DENSITY}"
  --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
  --rosa_mask_steps "${ROSA_MASK_STEPS}"
  --rosa_sparse_lr "${ROSA_SPARSE_LR}"
  --head_lr "${HEAD_LR}"
  --seed "${SEED}"
  --out_dir "${SEED_DIR}"
)

if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
  CMD+=(--rosa_reset_optimizer_on_mask)
fi

echo "=================================================="
echo "UniLoRA-RoSA smoke run"
echo "model=${MODEL} task=${TASK} seed=${SEED} head_lr=${HEAD_LR}"
echo "num_epochs=${NUM_EPOCHS} rank=${RANK} theta_d_length=${THETA_D_LENGTH}"
echo "density=${ROSA_DENSITY} warmup_steps=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS}"
echo "log: ${LOG_FILE}"
echo "=================================================="

CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

echo "Smoke run finished -> ${LOG_FILE}"
