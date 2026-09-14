#!/bin/bash
#SBATCH --job-name=profile_math_lora_ul_pl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/profile_math_lora_unilora_prolosa_l20_2gpu_%j.out
#SBATCH --error=logs/profile_math_lora_unilora_prolosa_l20_2gpu_%j.err

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AWS_EC2_METADATA_DISABLED=true

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
DATA_PATH="${DATA_PATH:-meta-math/MetaMathQA}"
FULL_DATASET_SAMPLES="${FULL_DATASET_SAMPLES:-100000}"
FULL_NUM_TRAIN_EPOCHS="${FULL_NUM_TRAIN_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-1}"
SEED="${SEED:-42}"

LORA_RANK="${LORA_RANK:-4}"
NUM_VECTORS="${NUM_VECTORS:-2048}"
VECTOR_LENGTH="${VECTOR_LENGTH:-524288}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
SPARSE_BUDGET="${SPARSE_BUDGET:-16384}"
THETA_D_LENGTH="${THETA_D_LENGTH:-507904}"
THETA_D_LR="${THETA_D_LR:-8e-4}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"

# LoRA and UniLoRA need only a short steady-state window. ProLoSA must run its
# complete structural warmup and mask-selection window, then activated steps.
LORA_PROFILE_STEPS="${LORA_PROFILE_STEPS:-12}"
UNILORA_PROFILE_STEPS="${UNILORA_PROFILE_STEPS:-12}"
PROLOSA_POST_ACTIVATION_STEPS="${PROLOSA_POST_ACTIVATION_STEPS:-10}"
LORA_LR="${LORA_LR:-2e-4}"
UNILORA_LR="${UNILORA_LR:-2e-3}"
PROLOSA_LR="${PROLOSA_LR:-2e-4}"

MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"
OUT_ROOT="${OUT_ROOT:-output/profile_math_lora_unilora_prolosa_multi_gpu_l20_2gpu}"
PROFILER="${PROFILER:-profile_math_unilora_prolosa.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-intruction_tuning_unilora_multi_gpu.py}"

if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
    echo "theta_d_length + sparse_budget must equal total_trainable_budget." >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}/lora" "${OUT_ROOT}/unilora" "${OUT_ROOT}/prolosa"

COMMON_ARGS=(
    --train_script "${TRAIN_SCRIPT}"
    --model_name_or_path "${BASE_MODEL}"
    --data_path "${DATA_PATH}"
    --seed "${SEED}"
    --full_dataset_samples "${FULL_DATASET_SAMPLES}"
    --full_num_train_epochs "${FULL_NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --lora_r "${LORA_RANK}"
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}"
    --max_memory_cpu "${MAX_MEMORY_CPU}"
)

echo ">>> Profiling standard LoRA with ${LORA_PROFILE_STEPS} optimizer steps"
python "${PROFILER}" run \
    --method lora \
    --output_dir "${OUT_ROOT}/lora/train_output" \
    --profile_json "${OUT_ROOT}/lora/profile.json" \
    --train_log "${OUT_ROOT}/lora/train.log" \
    --profile_max_steps "${LORA_PROFILE_STEPS}" \
    --learning_rate "${LORA_LR}" \
    "${COMMON_ARGS[@]}"

echo ">>> Profiling UniLoRA with ${UNILORA_PROFILE_STEPS} optimizer steps"
python "${PROFILER}" run \
    --method unilora \
    --output_dir "${OUT_ROOT}/unilora/train_output" \
    --profile_json "${OUT_ROOT}/unilora/profile.json" \
    --train_log "${OUT_ROOT}/unilora/train.log" \
    --profile_max_steps "${UNILORA_PROFILE_STEPS}" \
    --learning_rate "${UNILORA_LR}" \
    --num_vectors "${NUM_VECTORS}" \
    --vector_length "${VECTOR_LENGTH}" \
    "${COMMON_ARGS[@]}"

PROLOSA_PROFILE_STEPS=$((ROSA_WARMUP_STEPS + ROSA_MASK_STEPS + PROLOSA_POST_ACTIVATION_STEPS))
echo ">>> Profiling ProLoSA with ${PROLOSA_PROFILE_STEPS} optimizer steps"
echo ">>> Includes warmup=${ROSA_WARMUP_STEPS}, mask=${ROSA_MASK_STEPS}, post-activation=${PROLOSA_POST_ACTIVATION_STEPS}"
python "${PROFILER}" run \
    --method prolosa \
    --output_dir "${OUT_ROOT}/prolosa/train_output" \
    --profile_json "${OUT_ROOT}/prolosa/profile.json" \
    --train_log "${OUT_ROOT}/prolosa/train.log" \
    --profile_max_steps "${PROLOSA_PROFILE_STEPS}" \
    --post_activation_steps "${PROLOSA_POST_ACTIVATION_STEPS}" \
    --learning_rate "${PROLOSA_LR}" \
    --theta_d_length "${THETA_D_LENGTH}" \
    --sparse_budget "${SPARSE_BUDGET}" \
    --theta_d_lr "${THETA_D_LR}" \
    --rosa_sparse_lr_mult "${ROSA_SPARSE_LR_MULT}" \
    --rosa_warmup_steps "${ROSA_WARMUP_STEPS}" \
    --rosa_mask_steps "${ROSA_MASK_STEPS}" \
    "${COMMON_ARGS[@]}"

python "${PROFILER}" summarize \
    --profiles \
        "${OUT_ROOT}/lora/profile.json" \
        "${OUT_ROOT}/unilora/profile.json" \
        "${OUT_ROOT}/prolosa/profile.json" \
    --output_json "${OUT_ROOT}/summary.json" \
    --output_md "${OUT_ROOT}/summary.md"

echo "Profiling finished."
echo "Summary: ${OUT_ROOT}/summary.md"
