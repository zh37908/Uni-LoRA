#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-RoSA-Stage on GLUE.
#
# 与 `acf_submit_unilora_variant_unilora_rosa.sh` 的主要区别：
#   - variant 改为 `unilora_rosa_stage`
#   - 不再扫 `rosa_warmup_steps`
#   - 改为扫 `rosa_stage_ratio`，即训练总 epoch 完成到指定比例后，再开始收集 sparse mask 梯度并激活 sparse adapter
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
TASKS=(${TASKS:-mrpc})
SEEDS=(${SEEDS:-0})
LRS=(${LRS:-2e-4})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"

# 稀疏补偿全局密度（槽位向量上的 top-k 比例）
ROSA_DENSITY_LIST=(${ROSA_DENSITY_LIST:-0.01})
# 训练总 epoch 进行到该比例后，开始收集 sparse mask 梯度
ROSA_STAGE_RATIO_LIST=(${ROSA_STAGE_RATIO_LIST:-0.25 0.5 0.75})
# 收集多少个 optimizer step 的梯度来确定 sparse mask
ROSA_MASK_STEPS_LIST=(${ROSA_MASK_STEPS_LIST:-1 8})
# 稀疏组学习率 = MULT * THETA_D_LR
ROSA_SPARSE_LR_MULT_LIST=(${ROSA_SPARSE_LR_MULT_LIST:-0.2 1.0})
# 1 = mask 生成后 --rosa_reset_optimizer_on_mask；0 = 不重置
ROSA_RESET_LIST=(${ROSA_RESET_LIST:-0 1})

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
VARIANT="unilora_rosa_stage"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_stage_acf}"
mkdir -p "${OUT_ROOT}"

sparse_lr_from_mult() {
  python3 -c "print(float('${THETA_D_LR}') * float('${1}'))"
}

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
        for ROSA_STAGE_RATIO in "${ROSA_STAGE_RATIO_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-RoSA-Stage jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH} theta_d_lr=${THETA_D_LR}"
echo ">>> rosa-stage: density_list=${ROSA_DENSITY_LIST[*]} stage_ratio_list=${ROSA_STAGE_RATIO_LIST[*]} mask_steps_list=${ROSA_MASK_STEPS_LIST[*]}"
echo ">>> rosa-stage: sparse_lr_mult_list=${ROSA_SPARSE_LR_MULT_LIST[*]} (sparse_lr = mult * theta_d_lr), reset_list=${ROSA_RESET_LIST[*]}"
echo ">>> head LRS=${LRS[*]}"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
        for ROSA_STAGE_RATIO in "${ROSA_STAGE_RATIO_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))
                ROSA_SPARSE_LR="$(sparse_lr_from_mult "${ROSA_SPARSE_MULT}")"
                MULT_TAG="${ROSA_SPARSE_MULT//./p}"
                STAGE_TAG="${ROSA_STAGE_RATIO//./p}"
                METHOD_NAME="${VARIANT}_d${ROSA_DENSITY}_sr${STAGE_TAG}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
                SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
                mkdir -p "${SEED_DIR}"

                LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"

                CMD=(
                  python "${SCRIPT}"
                  --variant "${VARIANT}"
                  --model_name "${MODEL}"
                  --task "${TASK}"
                  --batch_size "${BATCH_SIZE}"
                  --rank "${RANK}"
                  --theta_d_length "${THETA_D_LENGTH}"
                  --theta_d_lr "${THETA_D_LR}"
                  --init_theta_d_bound "${INIT_THETA_D_BOUND}"
                  --rosa_density "${ROSA_DENSITY}"
                  --rosa_stage_ratio "${ROSA_STAGE_RATIO}"
                  --rosa_mask_steps "${ROSA_MASK_STEPS}"
                  --rosa_sparse_lr "${ROSA_SPARSE_LR}"
                  --head_lr "${LR}"
                  --seed "${SEED}"
                  --out_dir "${SEED_DIR}"
                )

                if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
                  CMD+=(--rosa_reset_optimizer_on_mask)
                fi

                echo "=================================================="
                echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${LR} ${METHOD_NAME} sparse_lr=${ROSA_SPARSE_LR}"
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

echo "All local UniLoRA-RoSA-Stage jobs have been processed."
