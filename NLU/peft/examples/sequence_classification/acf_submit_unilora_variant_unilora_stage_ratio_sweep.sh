#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

# Activate NLU conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate nlu

# Clear proxy variables to avoid timeout issues.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# Limit CPU thread usage on the local machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# Local single-GPU config
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
for m in ['roberta-large']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
for t in ['cola', 'mrpc']:
    try:
        load_dataset('nyu-mll/glue', t)
    except Exception:
        pass
"

# Force offline mode after cache warmup.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Sweep only this single parameter:
#   --stage_theta_d_ratios FRONT MIDDLE BACK
RATIO_TRIPLES=(
  "0.333 0.333 0.333"
  "0.3 0.35 0.35"
)

# Multi-group sweep for task/seed/lr
MODELS=(roberta-large)
TASKS=( mrpc)
SEEDS=(0 1 2)
LRS=(2e-4 1e-3 5e-3)

SCRIPT=run_unilora_variants_glue.py
OUT_ROOT="${OUT_ROOT:-results_glue_variants_stage_ratio_sweep_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for _ in "${RATIO_TRIPLES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + ${#LRS[@]}))
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} stage-ratio sweep jobs sequentially on local GPU ${GPU}"

RUN_IDX=0

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
    mkdir -p "${TASK_DIR}"

    for RATIO in "${RATIO_TRIPLES[@]}"; do
      read -r R0 R1 R2 <<< "${RATIO}"
      M_NAME="unilora_stage_ratio_ratio_${R0}_${R1}_${R2}"

      for SEED in "${SEEDS[@]}"; do
        SEED_DIR="${TASK_DIR}/${M_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"

        for LR in "${LRS[@]}"; do
          RUN_IDX=$((RUN_IDX + 1))
          LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"

          echo "=================================================="
          echo "[${RUN_IDX}/${TOTAL_RUNS}] variant=unilora_stage_ratio model=${MODEL} task=${TASK} ratio=${R0},${R1},${R2} seed=${SEED} lr=${LR}"
          echo "log: ${LOG_FILE}"
          echo "=================================================="

          CUDA_VISIBLE_DEVICES="${GPU}" \
          python "${SCRIPT}" \
            --variant unilora_stage_ratio \
            --stage_theta_d_ratios "${R0}" "${R1}" "${R2}" \
            --model_name "${MODEL}" \
            --task "${TASK}" \
            --batch_size "${BATCH_SIZE}" \
            --head_lr "${LR}" \
            --seed "${SEED}" \
            --out_dir "${SEED_DIR}" \
            > "${LOG_FILE}" 2>&1

          echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
        done
      done
    done
  done
done

echo "All local stage-ratio sweep jobs have been processed."
