#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-GeLoRA on GLUE.
# - GeLoRA-style rank allocation from hidden-state intrinsic dimensions
# - Paper-inspired MRPC defaults, adapted to the current UniLoRA training setup
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

# GeLoRA MRPC paper optimums (Table 10), adapted to UniLoRA:
# - paper single LR is mapped to head_lr
# - keep UniLoRA theta_d_lr at the repo's standard default unless overridden
HEAD_LRS=(${HEAD_LRS:-5e-4 1e-3 1e-4})
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-3})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-1.88e-2}"
WARMUP_RATIO="${WARMUP_RATIO:-3.04e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5.48e-2}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"

GELORA_RANK_OFFSET="${GELORA_RANK_OFFSET:-1}"

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
VARIANT="unilora_gelora"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_gelora_acf}"
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

echo ">>> Running ${TOTAL_RUNS} UniLoRA-GeLoRA jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> theta_d_lr_list=${THETA_D_LR_LIST[*]}"
echo ">>> gelora: offset=${GELORA_RANK_OFFSET} (paper-style rank = max(delta_id, 0) + offset)"
echo ">>> optimizer: weight_decay=${WEIGHT_DECAY} warmup_ratio=${WARMUP_RATIO} scheduler=${SCHEDULER_TYPE} dropout=${UNILORA_DROPOUT}"
echo ">>> target modules follow paper-style attention Q/K/V/O only"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))

        METHOD_NAME="${VARIANT}_off${GELORA_RANK_OFFSET}_wu${WARMUP_RATIO}_wd${WEIGHT_DECAY}_drop${UNILORA_DROPOUT}_sched${SCHEDULER_TYPE}"
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
          --gelora_rank_offset "${GELORA_RANK_OFFSET}"
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

echo "All local UniLoRA-GeLoRA jobs have been processed."
