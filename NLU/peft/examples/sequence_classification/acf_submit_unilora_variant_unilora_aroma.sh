#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-AROMA on GLUE.
# This sweep focuses on three targeted follow-up settings:
# 1) Reduce effective final rank by shortening training to ~3 merges
# 2) Lower theta_d learning rate while keeping the original merge schedule
# 3) Keep optimizer state across merge-and-reinit events
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
TASKS=(${TASKS:-cola})
SEEDS=(${SEEDS:-0})
HEAD_LRS=(${HEAD_LRS:-1e-3})

RANK="${RANK:-1}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"
AROMA_T_IN="${AROMA_T_IN:-2680}"

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
VARIANT="unilora_aroma"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_aroma}"
mkdir -p "${OUT_ROOT}"

# Three targeted settings from the follow-up analysis.
CONFIG_NAMES=("rank3_epochs30" "theta2e3" "keepopt")
CONFIG_THETA_D_LRS=("5e-3" "2e-3" "5e-3")
CONFIG_NUM_EPOCHS=("30" "" "")
CONFIG_KEEP_OPTIMIZER=("0" "0" "1")
CONFIG_SUFFIXES=("t2680_r1_ep30_rank3ish" "t2680_r1_td2e3" "t2680_r1_keepopt")

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for CFG_IDX in "${!CONFIG_NAMES[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-AROMA jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> aroma_t_in=${AROMA_T_IN} weight_decay=${WEIGHT_DECAY} warmup_ratio=${WARMUP_RATIO} scheduler=${SCHEDULER_TYPE}"
echo ">>> sweep configs=${CONFIG_NAMES[*]}"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for CFG_IDX in "${!CONFIG_NAMES[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))

        CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
        THETA_D_LR="${CONFIG_THETA_D_LRS[$CFG_IDX]}"
        NUM_EPOCHS="${CONFIG_NUM_EPOCHS[$CFG_IDX]}"
        KEEP_OPT="${CONFIG_KEEP_OPTIMIZER[$CFG_IDX]}"
        RESULT_SUFFIX="${CONFIG_SUFFIXES[$CFG_IDX]}"

        METHOD_NAME="${VARIANT}_${CFG_NAME}_t${AROMA_T_IN}_r${RANK}_wu${WARMUP_RATIO}_wd${WEIGHT_DECAY}_sched${SCHEDULER_TYPE}"
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
          --aroma_t_in "${AROMA_T_IN}"
          --warmup_ratio "${WARMUP_RATIO}"
          --weight_decay "${WEIGHT_DECAY}"
          --scheduler_type "${SCHEDULER_TYPE}"
          --result_suffix "${RESULT_SUFFIX}"
          --out_dir "${SEED_DIR}"
        )

        if [[ -n "${NUM_EPOCHS}" ]]; then
          CMD+=(--num_epochs "${NUM_EPOCHS}")
        fi
        if [[ "${KEEP_OPT}" == "1" ]]; then
          CMD+=(--aroma_keep_optimizer_on_merge)
        fi

        echo "=================================================="
        echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${HEAD_LR} theta_d_lr=${THETA_D_LR} config=${CFG_NAME}"
        echo "log: ${LOG_FILE}"
        echo "=================================================="

        CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

        echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
      done
    done
  done
done

echo "All local UniLoRA-AROMA jobs have been processed."
