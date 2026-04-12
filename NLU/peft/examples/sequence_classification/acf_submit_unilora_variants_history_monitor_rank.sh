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
BATCH_SIZE="${BATCH_SIZE:-128}"
MONITOR_EVERY="${MONITOR_EVERY:-10}"
MONITOR_TOP_MODULES="${MONITOR_TOP_MODULES:-5}"

# Experiment config
METHODS=(unilora lora)
STAGE_THETA_D_RATIOS=(0.2 0.3 0.5)

MODELS=(roberta-large)
TASKS=(cola mrpc)
SEEDS=(0 1 2)
LRS=(2e-4 5e-3)

SCRIPT=run_unilora_variants_glue_monitor_rank.py
OUT_ROOT="${OUT_ROOT:-results_glue_variants_history_monitor_rank_acf}"
mkdir -p "${OUT_ROOT}"

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

TOTAL_RUNS=0
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      if [[ "${METHOD}" == "unilora_isometric_control" ]]; then
        ALPHAS=(0.25 0.5 0.75)
      else
        ALPHAS=(0.0)
      fi

      for _ in "${ALPHAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          TOTAL_RUNS=$((TOTAL_RUNS + ${#LRS[@]}))
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} monitor-rank jobs sequentially on local GPU ${GPU}"

RUN_IDX=0

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
    mkdir -p "${TASK_DIR}"

    for METHOD in "${METHODS[@]}"; do
      if [[ "${METHOD}" == "unilora_isometric_control" ]]; then
        ALPHAS=(0.25 0.5 0.75)
      else
        ALPHAS=(0.0)
      fi

      for ALPHA in "${ALPHAS[@]}"; do
        if [[ "${METHOD}" == "unilora_isometric_control" ]]; then
          M_NAME="${METHOD}_alpha${ALPHA}"
          EXTRA_ARGS=(--isometry_alpha "${ALPHA}")
        elif [[ "${METHOD}" == "unilora_stage_ratio" ]]; then
          R0="${STAGE_THETA_D_RATIOS[0]}"
          R1="${STAGE_THETA_D_RATIOS[1]}"
          R2="${STAGE_THETA_D_RATIOS[2]}"
          M_NAME="${METHOD}_ratio_${R0}_${R1}_${R2}"
          EXTRA_ARGS=(--stage_theta_d_ratios "${R0}" "${R1}" "${R2}")
        else
          M_NAME="${METHOD}"
          EXTRA_ARGS=()
        fi

        for SEED in "${SEEDS[@]}"; do
          SEED_DIR="${TASK_DIR}/${M_NAME}/seed_${SEED}"
          mkdir -p "${SEED_DIR}"

          for LR in "${LRS[@]}"; do
            RUN_IDX=$((RUN_IDX + 1))
            LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"

            echo "=================================================="
            echo "[${RUN_IDX}/${TOTAL_RUNS}] method=${METHOD} model=${MODEL} task=${TASK} seed=${SEED} lr=${LR}"
            echo "log: ${LOG_FILE}"
            echo "=================================================="

            CUDA_VISIBLE_DEVICES="${GPU}" \
            python "${SCRIPT}" \
              --variant "${METHOD}" \
              "${EXTRA_ARGS[@]}" \
              --model_name "${MODEL}" \
              --task "${TASK}" \
              --batch_size "${BATCH_SIZE}" \
              --monitor_every "${MONITOR_EVERY}" \
              --monitor_top_modules "${MONITOR_TOP_MODULES}" \
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
done

echo "All local monitor-rank jobs have been processed."
