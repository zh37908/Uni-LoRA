#!/bin/bash
#SBATCH --job-name=profile_unilora_prolosa_rest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/profile_unilora_prolosa_cola_mrpc_%j.out
#SBATCH --error=logs/profile_unilora_prolosa_cola_mrpc_%j.err

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
METHODS=(${METHODS:-unilora prolosa})
SEEDS=(${SEEDS:-0})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
UNILORA_THETA_D_LENGTH="${UNILORA_THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-1}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"
# Keep this space-free: commands are executed via `xargs ... bash -c`, which
# does not preserve nested quotes around values that contain whitespace.
GPU_SETUP="${GPU_SETUP:-1xNVIDIA_L20}"
GPU_COUNT="${GPU_COUNT:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_unilora_variants_glue.py}"
PROFILER="${PROFILER:-profile_unilora_prolosa_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_efficiency_profile_unilora_prolosa_cola_mrpc}"
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

prolosa_task_config() {
  case "${1}" in
    cola)
      # Best CoLA ProLoSA ablation: fixed total budget 23040, sb8640 / td14400.
      echo "14400 8640 snip_budget23040_sb8640_tp23040_td14400_sb8640_w128_msscore_tdlr5e-3_slrm0p2_hlr5e-3_rst1_sdecay1"
      ;;
    mrpc)
      # Best MRPC ProLoSA ablation: fixed theta_d 22320, sb2160 / total trainable 24480.
      echo "22320 2160 snip_fixedd22320_sb2160_tp24480_td22320_sb2160_w128_msscore_tdlr5e-3_slrm0p2_hlr2e-4_rst1_sdecay1"
      ;;
    *)
      echo "Unsupported task for ProLoSA profiling: ${1}" >&2
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

profile_json_path() {
  python3 - <<PY
import os
seed_dir = """${1}"""
method = """${2}"""
task = """${3}"""
model = """${MODEL}"""
lr = """${4}"""
seed = """${5}"""
safe_lr = "".join(ch if ch.isalnum() or ch in "_.+-" else "_" for ch in lr)
print(os.path.join(seed_dir, f"profile_{method}_{task}_{model}_lr{safe_lr}_seed{seed}.json"))
PY
}

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for METHOD in "${METHODS[@]}"; do
    case "${METHOD}" in
      unilora)
        METHOD_NAME="unilora_td${UNILORA_THETA_D_LENGTH}_rank${RANK}"
        THETA_D_LENGTH="${UNILORA_THETA_D_LENGTH}"
        SPARSE_BUDGET="0"
        ROSA_DENSITY="0"
        ROSA_SPARSE_LR=""
        ;;
      prolosa)
        read -r THETA_D_LENGTH SPARSE_BUDGET METHOD_NAME <<< "$(prolosa_task_config "${TASK}")"
        ROSA_DENSITY="$(density_from_sparse_budget "${SPARSE_BUDGET}")"
        ROSA_SPARSE_LR="$(sparse_lr_from_mult "${THETA_D_LR}" "${ROSA_SPARSE_LR_MULT}")"
        ;;
      *)
        echo "Unsupported method: ${METHOD}" >&2
        exit 1
        ;;
    esac

    for SEED in "${SEEDS[@]}"; do
      SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"
      PROFILE_JSON="$(profile_json_path "${SEED_DIR}" "${METHOD}" "${TASK}" "${HEAD_LR}" "${SEED}")"
      TRAIN_LOG="${SEED_DIR}/profile_train_${METHOD}_${TASK}_lr_${HEAD_LR}.log"
      RUNNER_LOG="${SEED_DIR}/profile_runner_${METHOD}_${TASK}_lr_${HEAD_LR}.log"

      if [[ -s "${PROFILE_JSON}" ]]; then
        echo "Skip existing profile: ${PROFILE_JSON}"
        continue
      fi

      FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${PROFILER} run --train_script ${TRAIN_SCRIPT} --method ${METHOD} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --profile_json ${PROFILE_JSON} --train_log ${TRAIN_LOG} --gpu_setup ${GPU_SETUP} --gpu_count ${GPU_COUNT} --sparse_budget ${SPARSE_BUDGET} --total_sparse_positions ${TOTAL_SPARSE_POSITIONS} ${NUM_EPOCHS_ARG}"

      if [[ "${METHOD}" == "prolosa" ]]; then
        FULL_CMD+=" --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR}"
        if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
          FULL_CMD+=" --rosa_reset_optimizer_on_mask"
        fi
        if [[ "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" == "1" ]]; then
          FULL_CMD+=" --rosa_decay_sparse_lr_after_activation"
        fi
      fi

      FULL_CMD+=" > ${RUNNER_LOG} 2>&1"
      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} Uni-LoRA/ProLoSA profiling jobs."
echo ">>> tasks=${TASKS[*]} methods=${METHODS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} gpu_setup=${GPU_SETUP} out_root=${OUT_ROOT}/${MODEL}"
echo ">>> ProLoSA warmup=${ROSA_WARMUP_STEPS} mask_steps=${ROSA_MASK_STEPS} slrm=${ROSA_SPARSE_LR_MULT}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new profiling jobs to run; summarizing existing profiles."
fi

rm -f "${CMD_LIST}"

SUMMARY_CSV="${OUT_ROOT}/unilora_prolosa_efficiency_summary.csv"
SUMMARY_MD="${OUT_ROOT}/unilora_prolosa_efficiency_summary.md"
python "${PROFILER}" summarize \
  --input_root "${OUT_ROOT}" \
  --output_csv "${SUMMARY_CSV}" \
  --output_md "${SUMMARY_MD}"

echo "All Uni-LoRA/ProLoSA profiling jobs have been processed."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary Markdown: ${SUMMARY_MD}"
