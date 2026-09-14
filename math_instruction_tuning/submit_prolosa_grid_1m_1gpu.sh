#!/bin/bash
#SBATCH --job-name=prolosa_r4_1m_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
# Submit through submit_prolosa_grid_dispatch.sh; dispatcher supplies --array=<task>.
#SBATCH --chdir=/home/hzhaobi/Uni-LoRA/math_instruction_tuning
#SBATCH --output=logs/prolosa_grid_1m_%A_%a.out
#SBATCH --error=logs/prolosa_grid_1m_%A_%a.err
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_modern
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="${PWD}/peft/src:${PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
python prolosa_grid_1m.py run --task-id "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
