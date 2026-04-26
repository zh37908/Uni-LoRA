#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-RoSA-Compression on GLUE.
#
# 基于 `acf_submit_unilora_variant_unilora_rosa.sh` 的 RoSA 网格，
# 额外扫 `--sparse_theta_d_length`，训练 variant=`unilora_rosa_compression`。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

# Activate NLU conda env
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

# 分类头 LR；MRPC 上纯 UniLoRA 常见最优点在 5e-4 附近，务必包含
LRS=(${LRS:-2e-4 5e-4})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"

# Sparse-banks for RoSA sparse residual（压缩稀疏 bank 的长度）
# 对比/排查建议至少覆盖：23040（相对温和）和 1024（更激进）
SPARSE_THETA_D_LENGTH_LIST=(${SPARSE_THETA_D_LENGTH_LIST:-11520 5760 2880})

# 稀疏补偿全局密度（槽位向量上的 top-k 比例）
ROSA_DENSITY_LIST=(${ROSA_DENSITY_LIST:-0.01})
# 对应 peft-rosa 的 spa_num_grads：用多少个 optimizer step 累计（实现为 max 累积）
ROSA_MASK_STEPS_LIST=(${ROSA_MASK_STEPS_LIST:-1 8})
# 低秩-only warmup 步数
ROSA_WARMUP_STEPS_LIST=(${ROSA_WARMUP_STEPS_LIST:-256})
# 稀疏组学习率 = MULT * THETA_D_LR（避免 S 与 theta_d 同量级过猛）
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
VARIANT="unilora_rosa_compression"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_compression_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
        for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_LR_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                for SPARSE_THETA_D_LENGTH in "${SPARSE_THETA_D_LENGTH_LIST[@]}"; do
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

echo ">>> Running ${TOTAL_RUNS} UniLoRA-RoSA-Compression jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH} theta_d_lr=${THETA_D_LR}"
echo ">>> rosa: density_list=${ROSA_DENSITY_LIST[*]} warmup_list=${ROSA_WARMUP_STEPS_LIST[*]} mask_steps_list=${ROSA_MASK_STEPS_LIST[*]}"
echo ">>> rosa: sparse_lr_mult_list=${ROSA_SPARSE_LR_MULT_LIST[*]} (sparse_lr = mult * theta_d_lr), reset_list=${ROSA_RESET_LIST[*]}"
echo ">>> sparse_theta_d_length_list=${SPARSE_THETA_D_LENGTH_LIST[*]}"
echo ">>> head LRS=${LRS[*]}"

RUN_IDX=0
for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for ROSA_DENSITY in "${ROSA_DENSITY_LIST[@]}"; do
        for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_LR_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                for SPARSE_THETA_D_LENGTH in "${SPARSE_THETA_D_LENGTH_LIST[@]}"; do
                  RUN_IDX=$((RUN_IDX + 1))

                  ROSA_SPARSE_LR="$(python3 - <<PY
theta_d_lr = float('${THETA_D_LR}')
mult = float('${ROSA_SPARSE_LR_MULT}')
print(theta_d_lr * mult)
PY
)"

                  # 目录名避免特殊字符：mult 用小数点替换
                  MULT_TAG="${ROSA_SPARSE_LR_MULT//./p}"

                  METHOD_NAME="${VARIANT}_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}_std${SPARSE_THETA_D_LENGTH}"

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
                    --sparse_theta_d_length "${SPARSE_THETA_D_LENGTH}"
                    --rosa_density "${ROSA_DENSITY}"
                    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
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
                  echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${LR} method=${METHOD_NAME}"
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
      done
    done
  done
done

echo "All UniLoRA-RoSA-Compression jobs have been processed."

