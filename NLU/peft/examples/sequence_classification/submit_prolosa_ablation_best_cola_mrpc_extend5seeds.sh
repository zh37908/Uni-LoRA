#!/bin/bash
#SBATCH --job-name=prolosa_best_cola_mrpc_s3s4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/prolosa_best_cola_mrpc_extend5seeds_%j.out
#SBATCH --error=logs/prolosa_best_cola_mrpc_extend5seeds_%j.err

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1

mkdir -p logs || exit 1

# Activate NLU conda env.
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1

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
TASKS=(${TASKS:-cola mrpc})
# Existing best-result sweeps already have seeds 0/1/2; this script fills them to 5 seeds.
SEEDS=(${SEEDS:-3 4})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-1}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_rosa_snip"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_prolosa_ablation_total23040_cola_mrpc}"
SUBEXP_DIR="${SUBEXP_DIR:-sparse_size_fixed_d}"
mkdir -p "${OUT_ROOT}"

task_config() {
  case "${1}" in
    cola)
      # Best CoLA by 3-seed mean: fixed total budget 23040, sb8640 / td14400.
      echo "5e-3 14400 8640 snip_budget23040_sb8640_tp23040_td14400_sb8640_w128_msscore_tdlr5e-3_slrm0p2_hlr5e-3_rst1_sdecay1"
      ;;
    mrpc)
      # Best MRPC by 3-seed mean: fixed theta_d 22320, sb2160 / total trainable 24480.
      echo "2e-4 22320 2160 snip_fixedd22320_sb2160_tp24480_td22320_sb2160_w128_msscore_tdlr5e-3_slrm0p2_hlr2e-4_rst1_sdecay1"
      ;;
    *)
      echo "Unsupported task for best CoLA/MRPC completion: ${1}" >&2
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
    try:
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
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
variant = """${VARIANT}"""
task = """${2}"""
model = """${MODEL}"""
lr = str(float("""${3}"""))
seed = """${4}"""
print(os.path.join(seed_dir, f"{variant}_{task}_{model}_lr{lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  read -r HEAD_LR THETA_D_LENGTH SPARSE_BUDGET METHOD_NAME <<< "$(task_config "${TASK}")"
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${SUBEXP_DIR}"
  mkdir -p "${TASK_DIR}"

  ROSA_DENSITY="$(density_from_sparse_budget "${SPARSE_BUDGET}")"
  ROSA_SPARSE_LR="$(sparse_lr_from_mult "${THETA_D_LR}" "${ROSA_SPARSE_LR_MULT}")"

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

echo ">>> Generated ${TOTAL_RUNS} ProLoSA best CoLA/MRPC completion jobs."
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} out_root=${OUT_ROOT}/${MODEL}"
echo ">>> theta_d_lr=${THETA_D_LR} warmup=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS}"
echo ">>> slrm=${ROSA_SPARSE_LR_MULT} sparse_lr=$(sparse_lr_from_mult "${THETA_D_LR}" "${ROSA_SPARSE_LR_MULT}") reset=${ROSA_RESET_OPTIMIZER_ON_MASK} sdecay=${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}"

if [[ "${TOTAL_RUNS}" -eq 0 ]]; then
  echo "No jobs to run."
  rm -f "${CMD_LIST}"
  exit 0
fi

echo ">>> Starting parallel queue with 4 slots..."
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"

rm -f "${CMD_LIST}"
echo "All ProLoSA best CoLA/MRPC completion jobs have been processed."
