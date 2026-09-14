#!/bin/bash
#SBATCH --job-name=prof_math_1gpu_20m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/profile_math_lora_unilora_prolosa_l20_1gpu_20min_%j.out
#SBATCH --error=logs/profile_math_lora_unilora_prolosa_l20_1gpu_20min_%j.err

set -euo pipefail

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
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
PROFILE_SECONDS="${PROFILE_SECONDS:-1200}"
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-0.5}"
SEED="${SEED:-42}"

LORA_RANK="${LORA_RANK:-4}"
NUM_VECTORS="${NUM_VECTORS:-2048}"
VECTOR_LENGTH="${VECTOR_LENGTH:-524288}"
TOTAL_TRAINABLE_BUDGET="${TOTAL_TRAINABLE_BUDGET:-524288}"
SPARSE_BUDGET="${SPARSE_BUDGET:-174763}"
THETA_D_LENGTH="${THETA_D_LENGTH:-349525}"
THETA_D_LR="${THETA_D_LR:-8e-4}"
ROSA_SPARSE_LR_MULT="${ROSA_SPARSE_LR_MULT:-0.2}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"

LORA_LR="${LORA_LR:-2e-4}"
UNILORA_LR="${UNILORA_LR:-2e-3}"
PROLOSA_LR="${PROLOSA_LR:-2e-4}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"

PROFILER="${PROFILER:-profile_math_unilora_prolosa.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-intruction_tuning_unilora_multi_gpu.py}"
OUT_ROOT="${OUT_ROOT:-logs/profile_math_1gpu_20min_${SLURM_JOB_ID}}"

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
    --profile_wall_time_seconds "${PROFILE_SECONDS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --lora_r "${LORA_RANK}"
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}"
    --max_memory_cpu "${MAX_MEMORY_CPU}"
    --memory_sample_interval "${MEMORY_SAMPLE_INTERVAL}"
)

echo ">>> One-GPU fixed-duration profile"
echo ">>> model=${BASE_MODEL}, training window per method=${PROFILE_SECONDS}s"
echo ">>> results=${OUT_ROOT}"

echo ">>> [1/3] LoRA"
python "${PROFILER}" run \
    --method lora \
    --output_dir "${OUT_ROOT}/lora/train_output" \
    --profile_json "${OUT_ROOT}/lora/profile.json" \
    --train_log "${OUT_ROOT}/lora/train.log" \
    --learning_rate "${LORA_LR}" \
    "${COMMON_ARGS[@]}"

echo ">>> [2/3] UniLoRA"
python "${PROFILER}" run \
    --method unilora \
    --output_dir "${OUT_ROOT}/unilora/train_output" \
    --profile_json "${OUT_ROOT}/unilora/profile.json" \
    --train_log "${OUT_ROOT}/unilora/train.log" \
    --learning_rate "${UNILORA_LR}" \
    --num_vectors "${NUM_VECTORS}" \
    --vector_length "${VECTOR_LENGTH}" \
    "${COMMON_ARGS[@]}"

echo ">>> [3/3] ProLoSA"
python "${PROFILER}" run \
    --method prolosa \
    --output_dir "${OUT_ROOT}/prolosa/train_output" \
    --profile_json "${OUT_ROOT}/prolosa/profile.json" \
    --train_log "${OUT_ROOT}/prolosa/train.log" \
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
echo "Detailed report: ${OUT_ROOT}/summary.md"
echo "Machine-readable results: ${OUT_ROOT}/summary.json"
