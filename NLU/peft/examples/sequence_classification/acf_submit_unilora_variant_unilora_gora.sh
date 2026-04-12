#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-GoRA on GLUE.
# - GoRA-style rank allocation before UniLoRA injection
# - Stable default settings favor less extreme rank concentration
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
TASKS=(${TASKS:-mrpc cola sst2})
SEEDS=(${SEEDS:-0 1 2})

HEAD_LRS=(${HEAD_LRS:-5e-4 1e-3 1e-4})
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-3})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"

GORA_IMPORTANCE_TYPES=(${GORA_IMPORTANCE_TYPES:-union_mean})
GORA_MIN_RANK_LIST=(${GORA_MIN_RANK_LIST:-2})
GORA_MAX_RANK_LIST=(${GORA_MAX_RANK_LIST:-16})
GORA_ALLOCATE_STRATEGIES=(${GORA_ALLOCATE_STRATEGIES:-moderate})
GORA_FEATURES_FUNCS=(${GORA_FEATURES_FUNCS:-sqrt})
GORA_GRADIENT_EST_STEPS_LIST=(${GORA_GRADIENT_EST_STEPS_LIST:-16})
GORA_SOFTMAX_IMPORTANCE="${GORA_SOFTMAX_IMPORTANCE:-0}"
GORA_TEMPERATURE="${GORA_TEMPERATURE:-1.0}"

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
VARIANT="unilora_gora"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_gora_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for IMP in "${GORA_IMPORTANCE_TYPES[@]}"; do
          for MIN_R in "${GORA_MIN_RANK_LIST[@]}"; do
            for MAX_R in "${GORA_MAX_RANK_LIST[@]}"; do
              for STRAT in "${GORA_ALLOCATE_STRATEGIES[@]}"; do
                for FEAT in "${GORA_FEATURES_FUNCS[@]}"; do
                  for GSTEP in "${GORA_GRADIENT_EST_STEPS_LIST[@]}"; do
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
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-GoRA jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> theta_d_lr_list=${THETA_D_LR_LIST[*]}"
echo ">>> gora: imp=${GORA_IMPORTANCE_TYPES[*]} min_rank=${GORA_MIN_RANK_LIST[*]} max_rank=${GORA_MAX_RANK_LIST[*]}"
echo ">>> gora: strategy=${GORA_ALLOCATE_STRATEGIES[*]} features_func=${GORA_FEATURES_FUNCS[*]} grad_steps=${GORA_GRADIENT_EST_STEPS_LIST[*]}"
echo ">>> gora: softmax=${GORA_SOFTMAX_IMPORTANCE} temperature=${GORA_TEMPERATURE} (epochs: run_unilora_variants_glue.py default per task/model)"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for IMP in "${GORA_IMPORTANCE_TYPES[@]}"; do
          for MIN_R in "${GORA_MIN_RANK_LIST[@]}"; do
            for MAX_R in "${GORA_MAX_RANK_LIST[@]}"; do
              for STRAT in "${GORA_ALLOCATE_STRATEGIES[@]}"; do
                for FEAT in "${GORA_FEATURES_FUNCS[@]}"; do
                  for GSTEP in "${GORA_GRADIENT_EST_STEPS_LIST[@]}"; do
                    RUN_IDX=$((RUN_IDX + 1))

                    METHOD_NAME="${VARIANT}_imp${IMP}_min${MIN_R}_max${MAX_R}_strat${STRAT}_feat${FEAT}_gstep${GSTEP}"
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
                      --gora_importance_type "${IMP}"
                      --gora_min_rank "${MIN_R}"
                      --gora_max_rank "${MAX_R}"
                      --gora_allocate_strategy "${STRAT}"
                      --gora_features_func "${FEAT}"
                      --gora_gradient_est_steps "${GSTEP}"
                      --gora_temperature "${GORA_TEMPERATURE}"
                      --out_dir "${SEED_DIR}"
                    )

                    if [[ "${GORA_SOFTMAX_IMPORTANCE}" == "1" ]]; then
                      CMD+=(--gora_softmax_importance)
                    fi

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
          done
        done
      done
    done
  done
done

echo "All local UniLoRA-GoRA jobs have been processed."
