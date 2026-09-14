#!/bin/bash
#SBATCH --job-name=theory_e4p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/theory_e4_projection_sweep_%j.out
#SBATCH --error=logs/theory_e4_projection_sweep_%j.err

# E4 (primary, controlled): fix task, dataset size and d; vary ONLY the random projection P
# via --proj_seed. Then correlate B_off(P) = ||(I - Pi_P) Delta_theta_LoRA||^2 with the
# per-P compression excess loss L_Comp(P) - L_LoRA. This isolates projection bias from
# cross-task confounders (n, noise, difficulty); the cross-task version stays as a
# secondary analysis. LoRA baseline runs (P-independent) provide Delta_theta_LoRA and L_LoRA.
# Override examples:
#   TASKS=mrpc PROJ_SEEDS="10 11 12 13" sbatch submit_theory_e4_projection_sweep.sh

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1
source theory_common_lora_lr.sh || exit 1

mkdir -p logs || exit 1

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1

set -euo pipefail

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-mrpc cola})
# Projection seeds for Uni-LoRA; opt seeds are averaged out per P.
PROJ_SEEDS=(${PROJ_SEEDS:-10 11 12 13 14 15 16 17})
OPT_SEEDS=(${OPT_SEEDS:-0 1})
LORA_SEEDS=(${LORA_SEEDS:-0 1 2})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
UNILORA_THETA_D="${UNILORA_THETA_D:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

SCRIPT="${SCRIPT:-run_unilora_variants_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_theory_e4_projection_sweep}"
mkdir -p "${OUT_ROOT}"

task_head_lr() {
  case "${1}" in
    cola) echo "${LR_COLA:-5e-3}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    rte) echo "${LR_RTE:-5e-3}" ;;
    stsb) echo "${LR_STSB:-2e-4}" ;;
    *)
      echo "Unsupported task for E4 projection sweep: ${1}" >&2
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

result_json_path() {
  python3 - <<PY
import os
seed_dir = """${1}"""
variant = """${2}"""
task = """${3}"""
model = """${MODEL}"""
lr = str(float("""${4}"""))
seed = """${5}"""
print(os.path.join(seed_dir, f"{variant}_{task}_{model}_lr{lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"
  lora_lr_setup "${TASK}" "${HEAD_LR}"

  # LoRA baseline: independent of P, gives Delta_theta_LoRA and L_LoRA.
  for SEED in "${LORA_SEEDS[@]}"; do
    SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/lora_r${RANK}${LORA_DIR_TAG}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
    RESULT_JSON="$(result_json_path "${SEED_DIR}" "lora" "${TASK}" "${HEAD_LR}" "${SEED}")"
    if [[ -s "${RESULT_JSON}" ]]; then
      echo "Skip existing result: ${RESULT_JSON}"
      continue
    fi
    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant lora --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --save_delta_theta --save_eval_logits${LORA_LR_ARGS} > ${LOG_FILE} 2>&1"
    echo "${FULL_CMD}" >> "${CMD_LIST}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
  done

  # Uni-LoRA: sweep projection seed at fixed d; each P averaged over OPT_SEEDS.
  for PROJ_SEED in "${PROJ_SEEDS[@]}"; do
    METHOD_NAME="unilora_td${UNILORA_THETA_D}_proj${PROJ_SEED}"
    for SEED in "${OPT_SEEDS[@]}"; do
      SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${METHOD_NAME}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"
      LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
      RESULT_JSON="$(result_json_path "${SEED_DIR}" "unilora" "${TASK}" "${HEAD_LR}" "${SEED}")"
      if [[ -s "${RESULT_JSON}" ]]; then
        echo "Skip existing result: ${RESULT_JSON}"
        continue
      fi
      FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant unilora --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --head_lr ${HEAD_LR} --seed ${SEED} --proj_seed ${PROJ_SEED} --out_dir ${SEED_DIR} --theta_d_length ${UNILORA_THETA_D} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --save_delta_theta --save_eval_logits > ${LOG_FILE} 2>&1"
      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} E4 projection-sweep jobs."
echo ">>> tasks=${TASKS[*]} proj_seeds=${PROJ_SEEDS[*]} opt_seeds=${OPT_SEEDS[*]} lora_seeds=${LORA_SEEDS[*]}"
echo ">>> model=${MODEL} rank=${RANK} theta_d=${UNILORA_THETA_D} out_root=${OUT_ROOT}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new E4 projection-sweep jobs to run."
fi

rm -f "${CMD_LIST}"
echo "All E4 projection-sweep jobs have been processed."
