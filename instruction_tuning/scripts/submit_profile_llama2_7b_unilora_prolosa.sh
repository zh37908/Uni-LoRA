#!/bin/bash
#SBATCH --job-name=profile_llama2_7b_ul_pl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=/home/hzhaobi/Uni-LoRA/instruction_tuning/logs/profile_llama2_7b_unilora_prolosa_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/instruction_tuning/logs/profile_llama2_7b_unilora_prolosa_%j.err

# Prefer SLURM_SUBMIT_DIR: sbatch copies the script to a spool path, so
# BASH_SOURCE-based dirname can land outside the repo (Permission denied on mkdir).
INSTRUCTION_ROOT="${INSTRUCTION_ROOT:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/instruction_tuning}}"
REPO_ROOT="$(cd "${INSTRUCTION_ROOT}/.." && pwd)"
cd "${INSTRUCTION_ROOT}" || exit 1

mkdir -p logs output || exit 1

# Activate instruction-tuning conda env.
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate instruction_tuning || exit 1

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export PYTHONPATH="${REPO_ROOT}/math_instruction_tuning/peft/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
METHODS=(${METHODS:-unilora prolosa})
SEEDS=(${SEEDS:-0})

TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
UNILORA_NUM_VECTORS="${UNILORA_NUM_VECTORS:-2048}"
THETA_D_LENGTH="${THETA_D_LENGTH:-507904}"
SPARSE_BUDGET="${SPARSE_BUDGET:-16384}"
THETA_D_LR="${THETA_D_LR:-8e-4}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-True}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-True}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SOURCE_MAX_LEN="${SOURCE_MAX_LEN:-16}"
TARGET_MAX_LEN="${TARGET_MAX_LEN:-512}"
DATASET="${DATASET:-alpaca-clean}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-80000}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
# Keep this free of spaces so xargs/bash -c does not split it.
GPU_SETUP="${GPU_SETUP:-1xL20}"
GPU_COUNT="${GPU_COUNT:-1}"

PROFILER="${PROFILER:-profile_llama2_7b_unilora_prolosa.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-qlora_unilora.py}"
OUT_ROOT="${OUT_ROOT:-output/profile_llama2_7b_unilora_prolosa}"
mkdir -p "${OUT_ROOT}"

if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
  echo "theta_d_length + sparse_budget must equal total_trainable_budget."
  echo "Got td=${THETA_D_LENGTH}, sb=${SPARSE_BUDGET}, total=${TOTAL_TRAINABLE_BUDGET}"
  exit 1
fi

if [[ $((UNILORA_NUM_VECTORS * 256)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
  echo "unilora_num_vectors * 256 should match total_trainable_budget for a fair profile."
  echo "Got vectors=${UNILORA_NUM_VECTORS}, total=$((UNILORA_NUM_VECTORS * 256)), budget=${TOTAL_TRAINABLE_BUDGET}"
  exit 1
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  MAX_TRAIN_SAMPLES_ARG="--max_train_samples ${MAX_TRAIN_SAMPLES}"
else
  MAX_TRAIN_SAMPLES_ARG=""
fi

echo ">>> Pre-warming cache (downloading model/tokenizer if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "${MODEL}"
AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=True)
try:
    AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        use_auth_token=True,
    )
except Exception as exc:
    print(f"Model prewarm skipped/failed: {exc}")
PY

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0

for METHOD in "${METHODS[@]}"; do
  case "${METHOD}" in
    unilora)
      METHOD_NAME="unilora_nv${UNILORA_NUM_VECTORS}_tp${TOTAL_TRAINABLE_BUDGET}"
      ;;
    prolosa)
      METHOD_NAME="prolosa_tp${TOTAL_TRAINABLE_BUDGET}_td${THETA_D_LENGTH}_sb${SPARSE_BUDGET}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}"
      ;;
    *)
      echo "Unsupported method: ${METHOD}" >&2
      exit 1
      ;;
  esac

  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="${OUT_ROOT}/${METHOD_NAME}/seed_${SEED}"
    mkdir -p "${RUN_DIR}"
    PROFILE_JSON="${RUN_DIR}/profile_llama2_7b_${METHOD}_seed${SEED}.json"
    TRAIN_LOG="${RUN_DIR}/profile_train_llama2_7b_${METHOD}_seed${SEED}.log"
    RUNNER_LOG="${RUN_DIR}/profile_runner_llama2_7b_${METHOD}_seed${SEED}.log"

    if [[ -s "${PROFILE_JSON}" ]]; then
      echo "Skip existing profile: ${PROFILE_JSON}"
      continue
    fi

    # Build argv with printf %q so spaces/special chars survive xargs -> bash -c.
    FULL_CMD="$(
      printf '%q ' \
        srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 \
        --cpu-bind=none --gpu-bind=single:1 \
        python "${PROFILER}" run \
        --train_script "${TRAIN_SCRIPT}" \
        --method "${METHOD}" \
        --model_name_or_path "${MODEL}" \
        --output_dir "${RUN_DIR}" \
        --profile_json "${PROFILE_JSON}" \
        --train_log "${TRAIN_LOG}" \
        --seed "${SEED}" \
        --lora_r 4 \
        --num_vectors "${UNILORA_NUM_VECTORS}" \
        --theta_d_length "${THETA_D_LENGTH}" \
        --total_trainable_budget "${TOTAL_TRAINABLE_BUDGET}" \
        --sparse_budget "${SPARSE_BUDGET}" \
        --theta_d_lr "${THETA_D_LR}" \
        --init_theta_d_bound "${INIT_THETA_D_BOUND}" \
        --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}" \
        --rosa_warmup_steps "${ROSA_WARMUP_STEPS}" \
        --rosa_mask_steps "${ROSA_MASK_STEPS}" \
        --rosa_reset_optimizer_on_mask "${ROSA_RESET_OPTIMIZER_ON_MASK}" \
        --rosa_decay_sparse_lr_after_activation "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" \
        --dataset "${DATASET}" \
        --source_max_len "${SOURCE_MAX_LEN}" \
        --target_max_len "${TARGET_MAX_LEN}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --logging_steps "${LOGGING_STEPS}" \
        --max_memory_MB "${MAX_MEMORY_MB}" \
        --gpu_setup "${GPU_SETUP}" \
        --gpu_count "${GPU_COUNT}"
    )"
    if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
      FULL_CMD+=" $(printf '%q ' --max_train_samples "${MAX_TRAIN_SAMPLES}")"
    fi
    FULL_CMD+=" > $(printf '%q' "${RUNNER_LOG}") 2>&1"

    echo "${FULL_CMD}" >> "${CMD_LIST}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
  done
done

echo ">>> Generated ${TOTAL_RUNS} LLaMA2-7B instruction-tuning profiling jobs."
echo ">>> methods=${METHODS[*]} seeds=${SEEDS[*]} dataset=${DATASET}"
echo ">>> batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} epochs=${NUM_TRAIN_EPOCHS}"
echo ">>> gpu_setup=${GPU_SETUP} parallel_jobs=${PARALLEL_JOBS}"
echo ">>> out_root=${OUT_ROOT}"
echo ">>> conda_env=instruction_tuning"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new profiling jobs to run; summarizing existing profiles."
fi

rm -f "${CMD_LIST}"

SUMMARY_CSV="${OUT_ROOT}/llama2_7b_unilora_prolosa_efficiency_summary.csv"
SUMMARY_MD="${OUT_ROOT}/llama2_7b_unilora_prolosa_efficiency_summary.md"
python "${PROFILER}" summarize \
  --input_root "${OUT_ROOT}" \
  --output_csv "${SUMMARY_CSV}" \
  --output_md "${SUMMARY_MD}"

echo "All LLaMA2-7B profiling jobs have been processed."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary Markdown: ${SUMMARY_MD}"
