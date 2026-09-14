#!/bin/bash
#SBATCH --job-name=profile_lora_cola_mrpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/profile_lora_cola_mrpc_%j.out
#SBATCH --error=logs/profile_lora_cola_mrpc_%j.err

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1

mkdir -p logs || exit 1

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1

set -euo pipefail

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola mrpc})
SEEDS=(${SEEDS:-0})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-1.0}"
# Keep this space-free because commands are dispatched through xargs.
GPU_SETUP="${GPU_SETUP:-1xNVIDIA_L20}"
GPU_COUNT="${GPU_COUNT:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_unilora_variants_glue.py}"
PROFILER="${PROFILER:-profile_unilora_prolosa_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_efficiency_profile_lora_cola_mrpc}"
mkdir -p "${OUT_ROOT}"

if [[ -n "${NUM_EPOCHS:-}" ]]; then
  NUM_EPOCHS_ARG="--num_epochs ${NUM_EPOCHS}"
else
  NUM_EPOCHS_ARG=""
fi

task_head_lr() {
  case "${1}" in
    cola) echo "${LR_COLA:-5e-3}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    *)
      echo "Unsupported task for profiling: ${1}" >&2
      return 1
      ;;
  esac
}

echo ">>> Pre-warming model and dataset caches..."
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

LORA_PARAM_COUNT="$(python - <<PY
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
target_suffixes = ["query", "key", "value", "output.dense", "intermediate.dense"]
rank = int("${RANK}")
total = 0
for name, module in model.named_modules():
    if not hasattr(module, "weight") or not any(name.endswith(s) for s in target_suffixes):
        continue
    out_features, in_features = module.weight.shape[:2]
    total += rank * in_features + out_features * rank
print(total)
PY
)"

echo ">>> LoRA A/B parameter count=${LORA_PARAM_COUNT}"

profile_json_path() {
  python3 - <<PY
import os
seed_dir = """${1}"""
task = """${2}"""
model = """${MODEL}"""
lr = """${3}"""
seed = """${4}"""
safe_lr = "".join(ch if ch.isalnum() or ch in "_.+-" else "_" for ch in lr)
print(os.path.join(seed_dir, f"profile_lora_{task}_{model}_lr{safe_lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
trap 'rm -f "${CMD_LIST}"' EXIT
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"
  for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/lora_rank${RANK}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    PROFILE_JSON="$(profile_json_path "${SEED_DIR}" "${TASK}" "${HEAD_LR}" "${SEED}")"
    TRAIN_LOG="${SEED_DIR}/profile_train_lora_${TASK}_lr_${HEAD_LR}.log"
    RUNNER_LOG="${SEED_DIR}/profile_runner_lora_${TASK}_lr_${HEAD_LR}.log"

    if [[ -s "${PROFILE_JSON}" ]]; then
      echo "Skip existing profile: ${PROFILE_JSON}"
      continue
    fi

    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${PROFILER} run --train_script ${TRAIN_SCRIPT} --method lora --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length 0 --unilora_dropout ${LORA_DROPOUT} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --profile_json ${PROFILE_JSON} --train_log ${TRAIN_LOG} --gpu_setup ${GPU_SETUP} --gpu_count ${GPU_COUNT} --total_sparse_positions ${LORA_PARAM_COUNT} --memory_sample_interval ${MEMORY_SAMPLE_INTERVAL} ${NUM_EPOCHS_ARG} > ${RUNNER_LOG} 2>&1"
    echo "${FULL_CMD}" >> "${CMD_LIST}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
  done
done

echo ">>> Generated ${TOTAL_RUNS} LoRA profiling jobs."
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK}"
echo ">>> model=${MODEL} gpu_setup=${GPU_SETUP} out_root=${OUT_ROOT}/${MODEL}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new profiling jobs to run; summarizing existing profiles."
fi

SUMMARY_CSV="${OUT_ROOT}/lora_efficiency_summary.csv"
SUMMARY_MD="${OUT_ROOT}/lora_efficiency_summary.md"
python "${PROFILER}" summarize \
  --input_root "${OUT_ROOT}" \
  --output_csv "${SUMMARY_CSV}" \
  --output_md "${SUMMARY_MD}"

echo "All LoRA profiling jobs have been processed."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary Markdown: ${SUMMARY_MD}"
