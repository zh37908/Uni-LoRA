#!/bin/bash
#SBATCH --job-name=rev5_synthetic
#SBATCH --account=shsong
#SBATCH --partition=gpu-rtx4090d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/synthetic_%A_%a.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/synthetic_%A_%a.err
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ "${1:-}" = prepare ]; then
 /home/hzhaobi/miniconda3/envs/unilora_modern/bin/python synthetic/revision_controls/run_controls.py --prepare
else
 /home/hzhaobi/miniconda3/envs/unilora_modern/bin/python synthetic/revision_controls/run_controls.py --index "${SLURM_ARRAY_TASK_ID:-0}" --count "${REV5_SYNTHETIC_BATCH:-1}" --shards "${REV5_SYNTHETIC_SHARDS:-0}" ${1:-}
fi
