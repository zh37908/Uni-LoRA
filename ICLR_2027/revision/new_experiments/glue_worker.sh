#!/bin/bash
#SBATCH --job-name=rev5_glue_full
#SBATCH --account=shsong
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/glue_worker_%A_%a.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/glue_worker_%A_%a.err
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/hzhaobi/miniconda3/bin/python ICLR_2027/revision/new_experiments/glue_worker.py "$1" "$SLURM_ARRAY_TASK_ID"
