#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-IGU on GLUE.
# - Keeps the same task sweep style as the UniLoRA-GeLoRA script
# - Uses compressed UniLoRA A/B with explicit IGU-LoRA-style lora_E pruning
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
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola sst2})
SEEDS=(${SEEDS:-0 1 2})

# Reuse the same LR sweep style as the GeLoRA script.
HEAD_LRS=(${HEAD_LRS:-5e-4 1e-3 1e-4})
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-3})

# Keep target rank fixed at 4, while over-parameterizing the initial rank.
RANK="${RANK:-8}"
IGU_TARGET_RANK="${IGU_TARGET_RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-1.88e-2}"
WARMUP_RATIO="${WARMUP_RATIO:-3.04e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5.48e-2}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"

IGU_INIT_WARMUP="${IGU_INIT_WARMUP:-100}"
IGU_FINAL_WARMUP="${IGU_FINAL_WARMUP:-100}"
IGU_MASK_INTERVAL="${IGU_MASK_INTERVAL:-50}"
IGU_BETA1="${IGU_BETA1:-0.85}"
IGU_BETA2="${IGU_BETA2:-0.85}"
IGU_EPS="${IGU_EPS:-1e-6}"
IGU_R_MIN="${IGU_R_MIN:-1}"

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
VARIANT="unilora_igu"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_igu_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-IGU jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} target_rank=${IGU_TARGET_RANK} theta_d_length=${THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> theta_d_lr_list=${THETA_D_LR_LIST[*]}"
echo ">>> igu: init_warmup=${IGU_INIT_WARMUP} final_warmup=${IGU_FINAL_WARMUP} mask_interval=${IGU_MASK_INTERVAL} beta1=${IGU_BETA1} beta2=${IGU_BETA2} r_min=${IGU_R_MIN}"
echo ">>> optimizer: weight_decay=${WEIGHT_DECAY} warmup_ratio=${WARMUP_RATIO} scheduler=${SCHEDULER_TYPE} dropout=${UNILORA_DROPOUT}"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))

        METHOD_NAME="${VARIANT}_r${RANK}_tr${IGU_TARGET_RANK}_iw${IGU_INIT_WARMUP}_fw${IGU_FINAL_WARMUP}_mi${IGU_MASK_INTERVAL}_wu${WARMUP_RATIO}_wd${WEIGHT_DECAY}_drop${UNILORA_DROPOUT}_sched${SCHEDULER_TYPE}"
        SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"
        LOG_FILE="${SEED_DIR}/log_headlr_${HEAD_LR}_thetalr_${THETA_D_LR}.txt"

        CMD=(
          python "${SCRIPT}"
          --variant "${VARIANT}"
          --model_name "${MODEL}"
          --task "${TASK}"
          --batch_size "${BATCH_SIZE}"
          --rank "${RANK}"
          --theta_d_length "${THETA_D_LENGTH}"
          --theta_d_lr "${THETA_D_LR}"
          --head_lr "${HEAD_LR}"
          --seed "${SEED}"
          --init_theta_d_bound "${INIT_THETA_D_BOUND}"
          --unilora_dropout "${UNILORA_DROPOUT}"
          --igu_target_rank "${IGU_TARGET_RANK}"
          --igu_init_warmup "${IGU_INIT_WARMUP}"
          --igu_final_warmup "${IGU_FINAL_WARMUP}"
          --igu_mask_interval "${IGU_MASK_INTERVAL}"
          --igu_beta1 "${IGU_BETA1}"
          --igu_beta2 "${IGU_BETA2}"
          --igu_eps "${IGU_EPS}"
          --igu_r_min "${IGU_R_MIN}"
          --warmup_ratio "${WARMUP_RATIO}"
          --weight_decay "${WEIGHT_DECAY}"
          --scheduler_type "${SCHEDULER_TYPE}"
          --out_dir "${SEED_DIR}"
        )

        echo "=================================================="
        echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${HEAD_LR} theta_d_lr=${THETA_D_LR} method=${METHOD_NAME}"
        echo "log: ${LOG_FILE}"
        echo "=================================================="

        CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

        echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
      done
    done
  done
done

echo "All local UniLoRA-IGU jobs have been processed."
