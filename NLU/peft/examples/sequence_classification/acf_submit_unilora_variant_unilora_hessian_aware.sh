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

# Paper-style local single-GPU config for the Hessian-aware ablation.
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Paper defaults for RoBERTa-large GLUE local-projection ablation.
RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL="${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL:-5}"
HESSIAN_AWARE_WARMUP_EPOCHS="${HESSIAN_AWARE_WARMUP_EPOCHS:-1}"
HESSIAN_AWARE_REASSIGN_RATIO="${HESSIAN_AWARE_REASSIGN_RATIO:-0.01}"
HESSIAN_AWARE_CANDIDATE_POOL_SIZE="${HESSIAN_AWARE_CANDIDATE_POOL_SIZE:-8}"
HESSIAN_AWARE_CAPACITY_PENALTY="${HESSIAN_AWARE_CAPACITY_PENALTY:-0.1}"
HESSIAN_AWARE_CAPACITY_SLACK="${HESSIAN_AWARE_CAPACITY_SLACK:-2.0}"
HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM="${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM:-0.9}"
HESSIAN_AWARE_ACCEPT_TOLERANCE="${HESSIAN_AWARE_ACCEPT_TOLERANCE:-1e-6}"

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
TASKS=(mrpc cola sst2 qnli)
SEEDS=(0 1 2)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_hessian_aware"
METHOD_NAME="${VARIANT}_int${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL}_warm${HESSIAN_AWARE_WARMUP_EPOCHS}_rr${HESSIAN_AWARE_REASSIGN_RATIO}_pool${HESSIAN_AWARE_CANDIDATE_POOL_SIZE}_cap${HESSIAN_AWARE_CAPACITY_PENALTY}_slack${HESSIAN_AWARE_CAPACITY_SLACK}_ema${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM}"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_hessian_aware_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + ${#LRS[@]}))
  done
done

echo ">>> Running ${TOTAL_RUNS} HessianAware jobs sequentially on local GPU ${GPU}"
echo ">>> Paper-style setup: model=${MODEL} tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH} theta_d_lr=${THETA_D_LR} update_interval=${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL} warmup_epochs=${HESSIAN_AWARE_WARMUP_EPOCHS} reassign_ratio=${HESSIAN_AWARE_REASSIGN_RATIO} candidate_pool_size=${HESSIAN_AWARE_CANDIDATE_POOL_SIZE} capacity_penalty=${HESSIAN_AWARE_CAPACITY_PENALTY} capacity_slack=${HESSIAN_AWARE_CAPACITY_SLACK} curvature_ema_momentum=${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM}"

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
        --hessian_aware_structure_update_interval "${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL}" \
        --hessian_aware_warmup_epochs "${HESSIAN_AWARE_WARMUP_EPOCHS}" \
        --hessian_aware_reassign_ratio "${HESSIAN_AWARE_REASSIGN_RATIO}" \
        --hessian_aware_candidate_pool_size "${HESSIAN_AWARE_CANDIDATE_POOL_SIZE}" \
        --hessian_aware_capacity_penalty "${HESSIAN_AWARE_CAPACITY_PENALTY}" \
        --hessian_aware_capacity_slack "${HESSIAN_AWARE_CAPACITY_SLACK}" \
        --hessian_aware_curvature_ema_momentum "${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM}" \
        --hessian_aware_accept_tolerance "${HESSIAN_AWARE_ACCEPT_TOLERANCE}" \
        --head_lr "${LR}" \
        --seed "${SEED}" \
        --out_dir "${SEED_DIR}" \
        > "${LOG_FILE}" 2>&1

      echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
    done
  done
done

echo "All local HessianAware jobs have been processed."
