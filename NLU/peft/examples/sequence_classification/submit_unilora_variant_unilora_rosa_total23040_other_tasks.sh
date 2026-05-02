#!/bin/bash
#SBATCH --job-name=unilora_rosa_tp23040
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_rosa_total23040_%j.out
#SBATCH --error=logs/unilora_rosa_total23040_%j.err

mkdir -p logs

# Activate NLU conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

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
TASKS=(${TASKS:-sst2 qnli rte stsb})
SEEDS=(${SEEDS:-0})
LRS=(${LRS:-2e-4})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
SPARSE_BUDGET_LIST=(${SPARSE_BUDGET_LIST:-1440})
ROSA_WARMUP_STEPS_LIST=(${ROSA_WARMUP_STEPS_LIST:-48 64 96})
ROSA_MASK_STEPS_LIST=(${ROSA_MASK_STEPS_LIST:-1 2})
ROSA_SPARSE_LR_MULT_LIST=(${ROSA_SPARSE_LR_MULT_LIST:-0.2})
ROSA_RESET_LIST=(${ROSA_RESET_LIST:-1})

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_rosa_total23040_acf}"
mkdir -p "${OUT_ROOT}"

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

sparse_lr_from_mult() {
  python3 -c "print(float('${THETA_D_LR}') * float('${1}'))"
}

theta_d_from_sparse_budget() {
  python3 - <<PY
total_budget = int("${TOTAL_TRAINABLE_BUDGET}")
sparse_budget = int("${1}")
theta_d = total_budget - sparse_budget
print(theta_d)
PY
}

matched_density_from_sparse_budget() {
  python3 - <<PY
sparse_budget = int("${1}")
total_sparse_positions = int("${TOTAL_SPARSE_POSITIONS}")
density = sparse_budget / total_sparse_positions
text = f"{density:.12f}".rstrip("0").rstrip(".")
print(text if text else "0")
PY
}

result_json_path() {
  python3 - <<PY
import os
seed_dir = """${1}"""
task = """${2}"""
model = """${MODEL}"""
lr = """${3}"""
seed = """${4}"""
print(os.path.join(seed_dir, f"unilora_rosa_{task}_{model}_lr{lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for SPARSE_BUDGET in "${SPARSE_BUDGET_LIST[@]}"; do
        THETA_D_LENGTH="$(theta_d_from_sparse_budget "${SPARSE_BUDGET}")"
        if [[ "${THETA_D_LENGTH}" -le 0 ]]; then
          echo "Skip sparse_budget=${SPARSE_BUDGET} because theta_d_length=${THETA_D_LENGTH} is invalid."
          continue
        fi

        ROSA_DENSITY="$(matched_density_from_sparse_budget "${SPARSE_BUDGET}")"

        for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                ROSA_SPARSE_LR="$(sparse_lr_from_mult "${ROSA_SPARSE_MULT}")"
                MULT_TAG="${ROSA_SPARSE_MULT//./p}"
                METHOD_NAME="${VARIANT}_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_d${ROSA_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
                SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
                mkdir -p "${SEED_DIR}"
                LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"
                RESULT_JSON="$(result_json_path "${SEED_DIR}" "${TASK}" "${LR}" "${SEED}")"

                if [[ -s "${RESULT_JSON}" ]]; then
                  echo "Skip existing result: ${RESULT_JSON}"
                  continue
                fi

                FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant ${VARIANT} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR} --head_lr ${LR} --seed ${SEED} --out_dir ${SEED_DIR}"
                if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
                  FULL_CMD+=" --rosa_reset_optimizer_on_mask"
                fi
                FULL_CMD+=" > ${LOG_FILE} 2>&1"

                echo "${FULL_CMD}" >> "${CMD_LIST}"
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} UniLoRA-RoSA total-budget jobs."
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} total_trainable_budget=${TOTAL_TRAINABLE_BUDGET} theta_d_lr=${THETA_D_LR}"
echo ">>> sparse_budget_list=${SPARSE_BUDGET_LIST[*]} warmup_list=${ROSA_WARMUP_STEPS_LIST[*]}"
echo ">>> mask_steps_list=${ROSA_MASK_STEPS_LIST[*]} sparse_lr_mult_list=${ROSA_SPARSE_LR_MULT_LIST[*]} reset_list=${ROSA_RESET_LIST[*]}"
echo ">>> head_lrs=${LRS[*]}"

if [[ "${TOTAL_RUNS}" -eq 0 ]]; then
  echo "No jobs to run."
  rm -f "${CMD_LIST}"
  exit 0
fi

echo ">>> Starting parallel queue with 4 slots..."
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"

rm -f "${CMD_LIST}"
echo "All other-task total-budget UniLoRA-RoSA jobs have been processed."
