#!/bin/bash
#SBATCH --job-name=rev5_glue
#SBATCH --account=shsong
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/glue_%A_%a.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/glue_%A_%a.err
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REV5_MANIFEST=${1:?manifest required}
REV5_SPEC=$(/home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))[int(sys.argv[2])])' "$REV5_MANIFEST" "$SLURM_ARRAY_TASK_ID")
/home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python ICLR_2027/revision/new_experiments/run_glue.py --spec "$REV5_SPEC"
