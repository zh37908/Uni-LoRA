#!/bin/bash

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
BATCH_SIZE="${BATCH_SIZE:-32}"

RANK="${RANK:-4}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
SKETCH_BITS="${SKETCH_BITS:-4}"
SKETCH_GROUPS_PER_ROW="${SKETCH_GROUPS_PER_ROW:-4}"
SKETCH_NUM_BANKS="${SKETCH_NUM_BANKS:-8}"

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
for m in ['roberta-large']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
for t in ['mrpc', 'cola', 'sst2', 'qnli']:
    try:
        load_dataset('nyu-mll/glue', t)
    except Exception:
        pass
"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="roberta-large"
TASKS=(mrpc)
SEEDS=(0 1)
LRS=(1e-4 1e-3 5e-3)

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_shared_sketch_bank"
METHOD_NAME="${VARIANT}_b${SKETCH_BITS}_g${SKETCH_GROUPS_PER_ROW}_banks${SKETCH_NUM_BANKS}"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_shared_sketch_bank_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + ${#LRS[@]}))
  done
done

echo ">>> Running ${TOTAL_RUNS} SharedSketchBank jobs sequentially on local GPU ${GPU}"
echo ">>> Config: model=${MODEL} tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} bits=${SKETCH_BITS} groups=${SKETCH_GROUPS_PER_ROW} banks=${SKETCH_NUM_BANKS} theta_d_lr=${THETA_D_LR}"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"

    for LR in "${LRS[@]}"; do
      RUN_IDX=$((RUN_IDX + 1))
      LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"

      echo "=================================================="
      echo "[${RUN_IDX}/${TOTAL_RUNS}] variant=${VARIANT} model=${MODEL} task=${TASK} seed=${SEED} lr=${LR}"
      echo "log: ${LOG_FILE}"
      echo "=================================================="

      CUDA_VISIBLE_DEVICES="${GPU}" \
      python "${SCRIPT}" \
        --variant "${VARIANT}" \
        --model_name "${MODEL}" \
        --task "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --rank "${RANK}" \
        --theta_d_lr "${THETA_D_LR}" \
        --init_theta_d_bound "${INIT_THETA_D_BOUND}" \
        --sketch_bits "${SKETCH_BITS}" \
        --sketch_groups_per_row "${SKETCH_GROUPS_PER_ROW}" \
        --sketch_num_banks "${SKETCH_NUM_BANKS}" \
        --head_lr "${LR}" \
        --seed "${SEED}" \
        --out_dir "${SEED_DIR}" \
        > "${LOG_FILE}" 2>&1

      echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
    done
  done
done

echo "All local SharedSketchBank jobs have been processed."
