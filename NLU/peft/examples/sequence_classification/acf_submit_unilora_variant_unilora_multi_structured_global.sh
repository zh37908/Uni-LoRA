#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-Multi-Structured-Global on GLUE.
#
# 目标：
# - 使用全局 A/B 大矩阵语义的 multi-structured-global 结构
# - 扫描 head_lr 与 theta_d_lr 的组合，观察稳定性和最优点
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
export PYTHONPATH="${SCRIPT_DIR}/../../src:${PYTHONPATH:-}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-mrpc cola})
SEEDS=(${SEEDS:-0})

# 扫描 head_lr（分类头）
HEAD_LRS=(${HEAD_LRS:-5e-4})
# 扫描 theta_d_lr（multi-structured-global 参数组）
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-1e-3 5e-3 1e-4 1e-5})

RANK="${RANK:-4}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"

# Multi-Structured-Global 结构参数
M_LIST=(${M_LIST:-8 10})
MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS="${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS:-}"
LAYERWISE_SCALE="${LAYERWISE_SCALE:-1}" # 1=开启 --multi_structured_layerwise_learnable_scale
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
VARIANT="unilora_multi_structured_global"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_multi_structured_global_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for M in "${M_LIST[@]}"; do
      for HEAD_LR in "${HEAD_LRS[@]}"; do
        for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
          TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-Multi-Structured-Global jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> theta_d_lr_list=${THETA_D_LR_LIST[*]}"
echo ">>> M_list=${M_LIST[*]} target_trainable_params=${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS:-None}"
echo ">>> layerwise_scale=${LAYERWISE_SCALE} init_theta_d_bound=${INIT_THETA_D_BOUND} dropout=${UNILORA_DROPOUT}"
echo ">>> num_epochs=${NUM_EPOCHS:-default}"
echo ">>> JSON results will be saved by ${SCRIPT} into per-run out_dir."

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for M in "${M_LIST[@]}"; do
      for HEAD_LR in "${HEAD_LRS[@]}"; do
        for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
          RUN_IDX=$((RUN_IDX + 1))

          METHOD_NAME="${VARIANT}_M${M}"
          if [[ -n "${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS}" ]]; then
            METHOD_NAME="${METHOD_NAME}_T${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS}"
          fi
          METHOD_NAME="${METHOD_NAME}_ls${LAYERWISE_SCALE}"

          RUN_NAME="headlr_${HEAD_LR}_thetalr_${THETA_D_LR}"
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
            --theta_d_lr "${THETA_D_LR}"
            --head_lr "${HEAD_LR}"
            --seed "${SEED}"
            --init_theta_d_bound "${INIT_THETA_D_BOUND}"
            --unilora_dropout "${UNILORA_DROPOUT}"
            --multi_structured_num_hash_pairs "${M}"
            --out_dir "${SEED_DIR}"
          )

          if [[ -n "${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS}" ]]; then
            CMD+=(--multi_structured_target_trainable_params "${MULTI_STRUCTURED_TARGET_TRAINABLE_PARAMS}")
          fi

          if [[ "${LAYERWISE_SCALE}" == "1" ]]; then
            CMD+=(--multi_structured_layerwise_learnable_scale)
          fi

          if [[ -n "${NUM_EPOCHS}" ]]; then
            CMD+=(--num_epochs "${NUM_EPOCHS}")
          fi

          echo "=================================================="
          echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${HEAD_LR} theta_d_lr=${THETA_D_LR} method=${METHOD_NAME}"
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

echo "All local UniLoRA-Multi-Structured-Global jobs have been processed."
