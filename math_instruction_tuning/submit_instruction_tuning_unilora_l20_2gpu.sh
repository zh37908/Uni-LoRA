#!/bin/bash
#SBATCH --job-name=unilora_math_2l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_math_2l20_%j.out
#SBATCH --error=logs/unilora_math_2l20_%j.err

mkdir -p logs

# Activate conda env. Override with: CONDA_ENV=your_env sbatch ...
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

# Prefer the local PEFT implementation in this repo.
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"

# Limit CPU thread contention and noisy tokenizer warnings.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BASE_MODEL="${BASE_MODEL:-google/gemma-7b}"
LORA_RANK="${LORA_RANK:-4}"
NUM_VECTORS="${NUM_VECTORS:-2048}"
VECTOR_LENGTH="${VECTOR_LENGTH:-524288}"
OUTPUT="${OUTPUT:-output}"
DATA_PATH="${DATA_PATH:-meta-math/MetaMathQA}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:100000]}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
SEED="${SEED:-42}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-44GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-128GiB}"

mkdir -p "${OUTPUT}"

echo ">>> Starting UniLoRA math instruction tuning on 2 L20 GPUs"
echo ">>> conda_env=${CONDA_ENV}"
echo ">>> model=${BASE_MODEL} data=${DATA_PATH} split=${DATASET_SPLIT}"
echo ">>> visible_gpus=${CUDA_VISIBLE_DEVICES:-set by Slurm} max_memory_per_gpu=${MAX_MEMORY_PER_GPU}"

python intruction_tuning_unilora.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT}" \
    --lora_r "${LORA_RANK}" \
    --num_vectors "${NUM_VECTORS}" \
    --vector_length "${VECTOR_LENGTH}" \
    --save_only_topk_weights True \
    --data_path "${DATA_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_field query response \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --device_map auto \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}" \
    --max_memory_cpu "${MAX_MEMORY_CPU}" \
    --report_to tensorboard \
    --seed "${SEED}"

echo "UniLoRA math instruction tuning finished."
