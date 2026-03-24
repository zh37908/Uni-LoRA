#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

# Activate NLU conda env. Some conda activation hooks reference unset vars,
# so temporarily disable nounset around activation.
set +u
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu
set -u

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

# Paper-style local single-GPU config for the trajectory-initial ablation.
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Paper defaults for RoBERTa-large GLUE local-projection ablation.
RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
TRAJECTORY_NUM_BUCKETS="${TRAJECTORY_NUM_BUCKETS:-4}"
TRAJECTORY_BLOCK_ROWS="${TRAJECTORY_BLOCK_ROWS:-4}"
TRAJECTORY_BLOCK_COLS="${TRAJECTORY_BLOCK_COLS:-4}"
TRAJECTORY_KMEANS_ITERS="${TRAJECTORY_KMEANS_ITERS:-15}"

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

# Force offline mode after cache warmup.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Experiment config
MODEL="roberta-large"
TASKS=(mrpc)
SEEDS=(0 1 2)
LRS=(5e-4 1e-3 5e-3 )

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_trajectory_initial"
METHOD_NAME="${VARIANT}_buckets${TRAJECTORY_NUM_BUCKETS}_br${TRAJECTORY_BLOCK_ROWS}_bc${TRAJECTORY_BLOCK_COLS}_k${TRAJECTORY_KMEANS_ITERS}"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_trajectory_initial_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + ${#LRS[@]}))
  done
done

echo ">>> Running ${TOTAL_RUNS} TrajectoryInitial jobs sequentially on local GPU ${GPU}"
echo ">>> Paper-style setup: model=${MODEL} tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH} theta_d_lr=${THETA_D_LR} num_buckets=${TRAJECTORY_NUM_BUCKETS} block_rows=${TRAJECTORY_BLOCK_ROWS} block_cols=${TRAJECTORY_BLOCK_COLS} kmeans_iters=${TRAJECTORY_KMEANS_ITERS}"

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
        --theta_d_length "${THETA_D_LENGTH}" \
        --theta_d_lr "${THETA_D_LR}" \
        --init_theta_d_bound "${INIT_THETA_D_BOUND}" \
        --trajectory_num_buckets "${TRAJECTORY_NUM_BUCKETS}" \
        --trajectory_block_rows "${TRAJECTORY_BLOCK_ROWS}" \
        --trajectory_block_cols "${TRAJECTORY_BLOCK_COLS}" \
        --trajectory_kmeans_iters "${TRAJECTORY_KMEANS_ITERS}" \
        --head_lr "${LR}" \
        --seed "${SEED}" \
        --out_dir "${SEED_DIR}" \
        > "${LOG_FILE}" 2>&1

      echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
    done
  done
done

echo "All local TrajectoryInitial jobs have been processed."
