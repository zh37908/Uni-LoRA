#!/bin/bash
#
# 精简版 UniLoRA-RoSA 提交脚本：只保留 MRPC（head_lr=2e-4, seed=0）上验证集表现较好的 RoSA 组合，
# 补上纯 UniLoRA 常用的 head_lr=5e-4，并在 MRPC + CoLA 上复跑。
#
# 组合来源（results_glue_variants_unilora_rosa_acf 中 best_score 排名前若干；已去掉明显差的如 m8+slrm1.0+rst*）：
#   w256_m8_slrm0p2_rst1, w64_m1_slrm0p2_rst1, w64_m1_slrm1p0_rst0, w64_m8_slrm0p2_rst1,
#   w256_m1_slrm0p2_rst1, w64_m1_slrm0p2_rst0, w64_m8_slrm0p2_rst0, w256_m1_slrm1p0_rst1
#
# 覆盖方式：环境变量 ROSA_COMBOS 为空格分隔的 "W:M:SPARSE_MULT:RST"，例如 ROSA_COMBOS="64:1:0.2:1"
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
TASKS=(${TASKS:-mrpc cola})
SEEDS=(${SEEDS:-0 1 2})
# 补上 5e-4；保留 2e-4（较好组合原先在该 head_lr 下跑出）
LRS=(${LRS:-5e-4 2e-4})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"

ROSA_DENSITY="${ROSA_DENSITY:-0.01}"

# 默认：表现较好的 8 组；格式 warmup_steps : mask_steps : sparse_lr_mult : reset(0/1)
if [[ -z "${ROSA_COMBOS:-}" ]]; then
  ROSA_COMBOS=(
    "256:8:0.2:1"
    "64:1:0.2:1"
    "64:1:1.0:0"
    "64:8:0.2:1"
    "256:1:0.2:1"
    "64:1:0.2:0"
    "64:8:0.2:0"
    "256:1:1.0:1"
  )
else
  # shellcheck disable=SC2206
  ROSA_COMBOS=(${ROSA_COMBOS})
fi

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
VARIANT="unilora_rosa"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_refined_acf}"
mkdir -p "${OUT_ROOT}"

sparse_lr_from_mult() {
  python3 -c "print(float('${THETA_D_LR}') * float('${1}'))"
}

TOTAL_RUNS=$((${#TASKS[@]} * ${#SEEDS[@]} * ${#LRS[@]} * ${#ROSA_COMBOS[@]}))

echo ">>> UniLoRA-RoSA refined sweep: ${TOTAL_RUNS} jobs on GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} head_lrs=${LRS[*]} density=${ROSA_DENSITY}"
echo ">>> ${#ROSA_COMBOS[@]} RoSA combos (w:m:sparse_mult:rst)"

RUN_IDX=0
for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for COMBO in "${ROSA_COMBOS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        IFS=: read -r ROSA_WARMUP_STEPS ROSA_MASK_STEPS ROSA_SPARSE_MULT ROSA_RESET_OPTIMIZER_ON_MASK <<< "${COMBO}"

        MULT_TAG="${ROSA_SPARSE_MULT//./p}"
        METHOD_NAME="${VARIANT}_refined_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
        SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"

        ROSA_SPARSE_LR="$(sparse_lr_from_mult "${ROSA_SPARSE_MULT}")"
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
        echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${LR} ${METHOD_NAME} sparse_lr=${ROSA_SPARSE_LR}"
        echo "log: ${LOG_FILE}"
        echo "=================================================="

        CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

        echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
      done
    done
  done
done

echo "All refined UniLoRA-RoSA jobs finished."
