#!/bin/bash
#
# Local / ACF-style sequential runner for an "original RoSA-like"
# UniLoRA-RoSA-Discrete setup on GLUE.
#
# Notes:
# - The RoSA paper does not report GLUE / RoBERTa-large hyperparameters directly.
# - This script therefore matches the original RoSA training recipe as closely as
#   possible for classification:
#     * low-rank rank defaults to 16
#     * sparse density defaults to 0.006 (the most common bf16 paper setting)
#     * wl64-style schedule defaults to warmup=64, mask_steps=1
#     * optimizer reset after mask generation defaults to enabled
#     * low-rank LR defaults to 7e-4, while sparse/head LR default to 2e-4
# - To keep the compression ratio close to the current UniLoRA convention
#   (rank=4 -> theta_d_length=23040), we scale theta_d_length linearly with rank:
#       theta_d_length = 23040 * rank / 4
#   and use the same default for the sparse bank length unless overridden.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

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
BATCH_SIZE="${BATCH_SIZE:-64}"

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola mrpc})
SEEDS=(${SEEDS:-0 1 2})
HEAD_LRS=(${HEAD_LRS:-2e-4})

REFERENCE_RANK="${REFERENCE_RANK:-4}"
REFERENCE_THETA_D_LENGTH="${REFERENCE_THETA_D_LENGTH:-23040}"
RANK="${RANK:-16}"
DEFAULT_THETA_D_LENGTH="$(
  python3 - <<PY
reference_rank = int("${REFERENCE_RANK}")
reference_theta = int("${REFERENCE_THETA_D_LENGTH}")
rank = int("${RANK}")
scaled = round(reference_theta * rank / reference_rank)
print(max(1, int(scaled)))
PY
)"
THETA_D_LENGTH="${THETA_D_LENGTH:-${DEFAULT_THETA_D_LENGTH}}"
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-4 7e-4})
ROSA_SPARSE_THETA_D_LENGTH="${ROSA_SPARSE_THETA_D_LENGTH:-${THETA_D_LENGTH}}"
ROSA_SPARSE_LR_LIST=(${ROSA_SPARSE_LR_LIST:-1e-4 2e-4})
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"

ROSA_DENSITY_LIST=(${ROSA_DENSITY_LIST:-0.006})
ROSA_MASK_STEPS_LIST=(${ROSA_MASK_STEPS_LIST:-1})
ROSA_WARMUP_STEPS_LIST=(${ROSA_WARMUP_STEPS_LIST:-64})
ROSA_RESET_LIST=(${ROSA_RESET_LIST:-1})
NUM_EPOCHS="${NUM_EPOCHS:-}"

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
model_name = "${MODEL}"
tasks = "${TASKS[*]}".split()
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
for task in tasks:
    try:
        load_dataset("nyu-mll/glue", task)
    except Exception:
        pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa_discrete"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_discrete_original_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for ROSA_SPARSE_LR in "${ROSA_SPARSE_LR_LIST[@]}"; do
          for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
            for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
              for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
                for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                  TOTAL_RUNS=$((TOTAL_RUNS + 1))
                done
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} original-like UniLoRA-RoSA-Discrete jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> rank=${RANK} theta_d_length=${THETA_D_LENGTH} sparse_theta_d_length=${ROSA_SPARSE_THETA_D_LENGTH}"
echo ">>> reference compression anchor: rank=${REFERENCE_RANK} -> theta_d_length=${REFERENCE_THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]} theta_d_lr_list=${THETA_D_LR_LIST[*]} sparse_lr_list=${ROSA_SPARSE_LR_LIST[*]}"
echo ">>> original-like RoSA: density_list=${ROSA_DENSITY_LIST[*]} warmup_list=${ROSA_WARMUP_STEPS_LIST[*]} mask_steps_list=${ROSA_MASK_STEPS_LIST[*]}"
echo ">>> original-like RoSA: reset_list=${ROSA_RESET_LIST[*]} num_epochs=${NUM_EPOCHS:-default}"

RUN_IDX=0
CONFIG_TAG="r${RANK}_td${THETA_D_LENGTH}_std${ROSA_SPARSE_THETA_D_LENGTH}"

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for ROSA_SPARSE_LR in "${ROSA_SPARSE_LR_LIST[@]}"; do
          for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
            for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
              for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
                for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                  RUN_IDX=$((RUN_IDX + 1))

                  METHOD_NAME="${VARIANT}_original_${CONFIG_TAG}_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
                  RUN_NAME="headlr_${HEAD_LR}_thetalr_${THETA_D_LR}_sparselr_${ROSA_SPARSE_LR}"
                  SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}/${RUN_NAME}"
                  mkdir -p "${SEED_DIR}"
                  LOG_FILE="${SEED_DIR}/train.log"

                  CMD=(
                    python "${SCRIPT}"
                    --variant "${VARIANT}"
                    --model_name "${MODEL}"
                    --task "${TASK}"
                    --batch_size "${BATCH_SIZE}"
                    --rank "${RANK}"
                    --theta_d_length "${THETA_D_LENGTH}"
                    --theta_d_lr "${THETA_D_LR}"
                    --rosa_sparse_theta_d_length "${ROSA_SPARSE_THETA_D_LENGTH}"
                    --rosa_sparse_lr "${ROSA_SPARSE_LR}"
                    --head_lr "${HEAD_LR}"
                    --seed "${SEED}"
                    --init_theta_d_bound "${INIT_THETA_D_BOUND}"
                    --unilora_dropout "${UNILORA_DROPOUT}"
                    --rosa_density "${ROSA_DENSITY}"
                    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
                    --rosa_mask_steps "${ROSA_MASK_STEPS}"
                    --out_dir "${SEED_DIR}"
                  )

                  if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
                    CMD+=(--rosa_reset_optimizer_on_mask)
                  fi

                  if [[ -n "${NUM_EPOCHS}" ]]; then
                    CMD+=(--num_epochs "${NUM_EPOCHS}")
                  fi

                  echo "=================================================="
                  echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${HEAD_LR} theta_d_lr=${THETA_D_LR} sparse_lr=${ROSA_SPARSE_LR} method=${METHOD_NAME}"
                  echo "out_dir: ${SEED_DIR}"
                  echo "log: ${LOG_FILE}"
                  echo "=================================================="

                  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

                  echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All local original-like UniLoRA-RoSA-Discrete jobs have been processed."
