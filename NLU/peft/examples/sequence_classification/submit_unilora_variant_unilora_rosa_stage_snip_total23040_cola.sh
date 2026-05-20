#!/bin/bash
#SBATCH --job-name=unilora_rosa_stage_snip_cola
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_rosa_stage_snip_total23040_cola_%j.out
#SBATCH --error=logs/unilora_rosa_stage_snip_total23040_cola_%j.err

mkdir -p logs

# Activate NLU conda env.
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# Limit CPU thread contention across parallel srun jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASK="${TASK:-cola}"
SEEDS=(${SEEDS:-0 1 2 3 4})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
HEAD_LR_LIST=(${HEAD_LR_LIST:-5e-3})

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa_stage_snip"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_stage_snip_total23040_cola_slurm}"
mkdir -p "${OUT_ROOT}"

# Sparse adapter is always introduced after 50% of the total training epochs.
ROSA_STAGE_RATIO="0.5"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"

# Default fixed-budget setting mirrors the ACF stage-snip script.
#   sparse_budget theta_d_lr sparse_lr_mult reset_optimizer
CONFIGS=(
  # sparse_budget theta_d_lr sparse_lr_mult reset_optimizer # 保留当前最强中心点
  "720 5e-3 0.1 1"    # 降 sparse LR，测试是否更稳
  "720 3e-3 0.2 1"    # 中间 theta_d_lr
  "720 3e-3 0.1 1"
  "1080 3e-3 0.1 1"   # 720 和 1440 中间预算
  "1080 5e-3 0.1 1"
  "1080 5e-3 0.2 1"
  "1440 2e-3 0.1 1"   # 目前 sb1440 里较合理的低 LR 设置
)

echo ">>> Pre-warming cache (downloading model and CoLA dataset if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

model_name = "${MODEL}"
AutoTokenizer.from_pretrained(model_name)
try:
    AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
except Exception:
    pass
try:
    load_dataset("nyu-mll/glue", "${TASK}")
except Exception:
    pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Compute total sparse positions for the current model / rank / target-module set.
TOTAL_SPARSE_POSITIONS="$(python - <<PY
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
target_suffixes = ["query", "key", "value", "output.dense", "intermediate.dense"]
rank = int("${RANK}")
total = 0
for name, module in model.named_modules():
    if not hasattr(module, "weight"):
        continue
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    weight = module.weight
    out_features, in_features = weight.shape[0], weight.shape[1]
    total += rank * in_features + out_features * rank
print(total)
PY
)"

echo ">>> total_sparse_positions=${TOTAL_SPARSE_POSITIONS}"

density_from_sparse_budget() {
  python3 - <<PY
sparse_budget = int("${1}")
total_sparse_positions = int("${TOTAL_SPARSE_POSITIONS}")
density = sparse_budget / total_sparse_positions
text = f"{density:.12f}".rstrip("0").rstrip(".")
print(text if text else "0")
PY
}

sparse_lr_from_mult() {
  python3 -c "print(float('${1}') * float('${2}'))"
}

result_json_path() {
  python3 - <<PY
import os
seed_dir = """${1}"""
variant = """${VARIANT}"""
task = """${2}"""
model = """${MODEL}"""
lr = """${3}"""
seed = """${4}"""
print(os.path.join(seed_dir, f"{variant}_{task}_{model}_lr{lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0
TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
mkdir -p "${TASK_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
  read -r SPARSE_BUDGET THETA_D_LR ROSA_SPARSE_LR_MULT ROSA_RESET_OPTIMIZER_ON_MASK <<< "${CONFIG}"

  THETA_D_LENGTH=$((TOTAL_TRAINABLE_BUDGET - SPARSE_BUDGET))
  if [[ "${THETA_D_LENGTH}" -le 0 ]]; then
    echo "Skip sparse_budget=${SPARSE_BUDGET} because theta_d_length=${THETA_D_LENGTH} is invalid."
    continue
  fi

  ROSA_DENSITY="$(density_from_sparse_budget "${SPARSE_BUDGET}")"
  ROSA_SPARSE_LR="$(sparse_lr_from_mult "${THETA_D_LR}" "${ROSA_SPARSE_LR_MULT}")"
  MULT_TAG="${ROSA_SPARSE_LR_MULT//./p}"
  TDLR_TAG="${THETA_D_LR//./p}"
  TDLR_TAG="${TDLR_TAG//-e/-}"
  STAGE_TAG="${ROSA_STAGE_RATIO//./p}"
  METHOD_NAME="${VARIANT}_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_d${ROSA_DENSITY}_sr${STAGE_TAG}_m${ROSA_MASK_STEPS}_tdlr${TDLR_TAG}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}_sdecay${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"

  for HEAD_LR in "${HEAD_LR_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"
      LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
      RESULT_JSON="$(result_json_path "${SEED_DIR}" "${TASK}" "${HEAD_LR}" "${SEED}")"

      if [[ -s "${RESULT_JSON}" ]]; then
        echo "Skip existing result: ${RESULT_JSON}"
        continue
      fi

      FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant ${VARIANT} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --rosa_density ${ROSA_DENSITY} --rosa_stage_ratio ${ROSA_STAGE_RATIO} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR}"
      if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
        FULL_CMD+=" --rosa_reset_optimizer_on_mask"
      fi
      if [[ "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" == "1" ]]; then
        FULL_CMD+=" --rosa_decay_sparse_lr_after_activation"
      fi
      FULL_CMD+=" > ${LOG_FILE} 2>&1"

      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} CoLA UniLoRA-RoSA-Stage-SNIP jobs."
echo ">>> task=${TASK} seeds=${SEEDS[*]} head_lrs=${HEAD_LR_LIST[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} total_trainable_budget=${TOTAL_TRAINABLE_BUDGET}"
echo ">>> sparse adapter stage_ratio=${ROSA_STAGE_RATIO} mask_steps=${ROSA_MASK_STEPS}"
echo ">>> sparse_mask_score=snip_abs_weight_grad decay_sparse_lr_after_activation=${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"
echo ">>> configs:"
for CONFIG in "${CONFIGS[@]}"; do
  echo ">>>   ${CONFIG}"
done

if [[ "${TOTAL_RUNS}" -eq 0 ]]; then
  echo "No jobs to run."
  rm -f "${CMD_LIST}"
  exit 0
fi

echo ">>> Starting parallel queue with 4 slots..."
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"

rm -f "${CMD_LIST}"
echo "All CoLA UniLoRA-RoSA-Stage-SNIP jobs have been processed."
