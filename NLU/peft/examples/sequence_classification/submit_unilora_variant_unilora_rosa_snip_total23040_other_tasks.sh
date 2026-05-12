#!/bin/bash
#SBATCH --job-name=unilora_rosa_snip_tp23040_other
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_rosa_snip_total23040_other_%j.out
#SBATCH --error=logs/unilora_rosa_snip_total23040_other_%j.err

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
# CoLA has already been tested by the ACF script; this SLURM job targets the other GLUE tasks.
TASKS=(${TASKS:-mrpc qnli rte sst2})
SEEDS=(${SEEDS:-0 1 2 3 4})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-23040}"
SPARSE_BUDGET="${SPARSE_BUDGET:-720}"
THETA_D_LENGTH="${THETA_D_LENGTH:-22320}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-1}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa_snip"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_snip_total23040_other_tasks}"
mkdir -p "${OUT_ROOT}"

if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
  echo "theta_d_length + sparse_budget must equal total_trainable_budget."
  echo "Got td=${THETA_D_LENGTH}, sb=${SPARSE_BUDGET}, total=${TOTAL_TRAINABLE_BUDGET}"
  exit 1
fi

task_head_lr() {
  case "${1}" in
    cola) echo "${LR_COLA:-5e-3}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    qnli) echo "${LR_QNLI:-5e-4}" ;;
    rte) echo "${LR_RTE:-5e-3}" ;;
    sst2) echo "${LR_SST2:-1e-3}" ;;
    *)
      echo "Unsupported task for task-specific LR: ${1}" >&2
      return 1
      ;;
  esac
}

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

model_name = "${MODEL}"
tasks = "${TASKS[*]}".split()
AutoTokenizer.from_pretrained(model_name)
for task in tasks:
    num_labels = 1 if task == "stsb" else 2
    try:
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception:
        pass
    try:
        load_dataset("nyu-mll/glue", task)
    except Exception:
        pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

ROSA_DENSITY="$(python - <<PY
sparse_budget = int("${SPARSE_BUDGET}")
total_sparse_positions = int("${TOTAL_SPARSE_POSITIONS}")
density = sparse_budget / total_sparse_positions
text = f"{density:.12f}".rstrip("0").rstrip(".")
print(text if text else "0")
PY
)"

ROSA_SPARSE_LR="$(python3 -c "print(float('${THETA_D_LR}') * float('${ROSA_SPARSE_LR_MULT}'))")"
MULT_TAG="${ROSA_SPARSE_LR_MULT//./p}"
METHOD_NAME="${VARIANT}_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}_sdecay${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"

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

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
    RESULT_JSON="$(result_json_path "${SEED_DIR}" "${TASK}" "${HEAD_LR}" "${SEED}")"

    if [[ -s "${RESULT_JSON}" ]]; then
      echo "Skip existing result: ${RESULT_JSON}"
      continue
    fi

    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant ${VARIANT} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR}"
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

echo ">>> Generated ${TOTAL_RUNS} UniLoRA-RoSA-SNIP other-task jobs."
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} total_trainable_budget=${TOTAL_TRAINABLE_BUDGET}"
echo ">>> td=${THETA_D_LENGTH} sb=${SPARSE_BUDGET} warmup=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS}"
echo ">>> slrm=${ROSA_SPARSE_LR_MULT} sparse_lr=${ROSA_SPARSE_LR} density=${ROSA_DENSITY} reset=${ROSA_RESET_OPTIMIZER_ON_MASK}"
echo ">>> sparse_mask_score=snip_abs_weight_grad decay_sparse_lr_after_activation=${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"
echo ">>> task LRs: cola=$(task_head_lr cola), mrpc=$(task_head_lr mrpc), qnli=$(task_head_lr qnli), rte=$(task_head_lr rte), sst2=$(task_head_lr sst2)"

if [[ "${TOTAL_RUNS}" -eq 0 ]]; then
  echo "No jobs to run."
  rm -f "${CMD_LIST}"
  exit 0
fi

echo ">>> Starting parallel queue with 4 slots..."
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"

rm -f "${CMD_LIST}"
echo "All total-budget UniLoRA-RoSA-SNIP other-task jobs have been processed."
