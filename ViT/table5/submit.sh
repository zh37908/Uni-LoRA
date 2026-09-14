#!/bin/bash
#SBATCH --job-name=vit_table5_prolosa
#SBATCH --account=shsong
#SBATCH --partition=gpu-rtx5880
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ViT/table5/logs/slurm_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ViT/table5/logs/slurm_%j.err

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_modern
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd /home/hzhaobi/Uni-LoRA/ViT/table5
echo "Table 5 allocation: job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} GPUs=${CUDA_VISIBLE_DEVICES}"
python -u pipeline.py launch
