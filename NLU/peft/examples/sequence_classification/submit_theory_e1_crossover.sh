#!/bin/bash
#SBATCH --job-name=theory_e1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/theory_e1_crossover_%j.out
#SBATCH --error=logs/theory_e1_crossover_%j.err

# E1: data-size crossover. Same subset is shared across methods via --subset_seed.
# Best hyperparams follow tab:glue_hyperparams and results_glue 5-seed LR sweeps:
#   Uni-LoRA d=23040; ProLoSA SST-2/QNLI d=22320 K=720; CoLA d=14400 K=8640.
# Override examples:
#   TASKS=sst2 TRAIN_SUBSET_RATIOS=0.01 sbatch submit_theory_e1_crossover.sh

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
TASKS=(${TASKS:-sst2 qnli})
METHODS=(${METHODS:-lora unilora unilora_rosa_snip})
SEEDS=(${SEEDS:-0 1 2 3 4})
TRAIN_SUBSET_RATIOS=(${TRAIN_SUBSET_RATIOS:-0.01 0.02 0.05 0.1 0.25 0.5 1.0})
SUBSET_SEED="${SUBSET_SEED:-12345}"

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
UNILORA_THETA_D="${UNILORA_THETA_D:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-1}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

SCRIPT="${SCRIPT:-run_unilora_variants_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_theory_e1_crossover}"
mkdir -p "${OUT_ROOT}"

task_head_lr() {
  case "${1}" in
    cola) echo "${LR_COLA:-5e-3}" ;;
    sst2) echo "${LR_SST2:-1e-3}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    qnli) echo "${LR_QNLI:-5e-4}" ;;
    rte) echo "${LR_RTE:-5e-3}" ;;
    stsb) echo "${LR_STSB:-2e-4}" ;;
    *)
      echo "Unsupported task: ${1}" >&2
      return 1
      ;;
  esac
}

task_prolosa_budget() {
  case "${1}" in
    cola) echo "${PROLOSA_TD_COLA:-14400} ${PROLOSA_SB_COLA:-8640}" ;;
    *) echo "${PROLOSA_TD:-22320} ${PROLOSA_SB:-720}" ;;
  esac
}

ratio_tag() {
  python3 -c "print(f\"{float('${1}'):g}\")"
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
variant = """${2}"""
task = """${3}"""
model = """${MODEL}"""
lr = str(float("""${4}"""))
seed = """${5}"""
ratio = float("""${6}""")
stem = f"{variant}_{task}_{model}_lr{lr}_seed{seed}"
if ratio < 1.0:
    stem += f"_p{ratio:g}"
print(os.path.join(seed_dir, f"{stem}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0
ROSA_SPARSE_LR="$(sparse_lr_from_mult "${THETA_D_LR}" "${ROSA_SPARSE_LR_MULT}")"
MULT_TAG="${ROSA_SPARSE_LR_MULT//./p}"

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"
  lora_lr_setup "${TASK}" "${HEAD_LR}"
  read -r PROLOSA_TD PROLOSA_SB <<< "$(task_prolosa_budget "${TASK}")"
  ROSA_DENSITY="$(density_from_sparse_budget "${PROLOSA_SB}")"

  for RATIO in "${TRAIN_SUBSET_RATIOS[@]}"; do
    RATIO_TAG="$(ratio_tag "${RATIO}")"
    for METHOD in "${METHODS[@]}"; do
      case "${METHOD}" in
        lora)
          METHOD_NAME="lora_r${RANK}${LORA_DIR_TAG}"
          ;;
        unilora)
          METHOD_NAME="unilora_td${UNILORA_THETA_D}"
          ;;
        unilora_rosa_snip)
          METHOD_NAME="prolosa_td${PROLOSA_TD}_sb${PROLOSA_SB}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}"
          ;;
        *)
          echo "Unsupported method: ${METHOD}" >&2
          exit 1
          ;;
      esac

      for SEED in "${SEEDS[@]}"; do
        SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/p${RATIO_TAG}/${METHOD_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"
        LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
        RESULT_JSON="$(result_json_path "${SEED_DIR}" "${METHOD}" "${TASK}" "${HEAD_LR}" "${SEED}" "${RATIO}")"

        if [[ -s "${RESULT_JSON}" ]]; then
          echo "Skip existing result: ${RESULT_JSON}"
          continue
        fi

        FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant ${METHOD} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --train_subset_ratio ${RATIO} --subset_seed ${SUBSET_SEED} --fix_full_data_steps --save_delta_theta"
        if [[ "${METHOD}" == "lora" ]]; then
          FULL_CMD+="${LORA_LR_ARGS}"
        elif [[ "${METHOD}" == "unilora" ]]; then
          FULL_CMD+=" --theta_d_length ${UNILORA_THETA_D} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND}"
        elif [[ "${METHOD}" == "unilora_rosa_snip" ]]; then
          FULL_CMD+=" --theta_d_length ${PROLOSA_TD} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR}"
          if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
            FULL_CMD+=" --rosa_reset_optimizer_on_mask"
          fi
          if [[ "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" == "1" ]]; then
            FULL_CMD+=" --rosa_decay_sparse_lr_after_activation"
          fi
        fi
        FULL_CMD+=" > ${LOG_FILE} 2>&1"

        echo "${FULL_CMD}" >> "${CMD_LIST}"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
      done
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} E1 crossover jobs."
echo ">>> tasks=${TASKS[*]} methods=${METHODS[*]} seeds=${SEEDS[*]} ratios=${TRAIN_SUBSET_RATIOS[*]}"
echo ">>> model=${MODEL} rank=${RANK} subset_seed=${SUBSET_SEED} out_root=${OUT_ROOT}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new E1 jobs to run."
fi

rm -f "${CMD_LIST}"
echo "All E1 crossover jobs have been processed."
