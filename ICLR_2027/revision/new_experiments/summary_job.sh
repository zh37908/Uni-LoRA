#!/bin/bash
#SBATCH --job-name=rev5_analysis
#SBATCH --account=shsong
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/analysis_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ICLR_2027/revision/new_experiments/logs/analysis_%j.err
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA
if [ "$1" = select ]; then
 /home/hzhaobi/miniconda3/bin/python ICLR_2027/revision/new_experiments/select_glue.py
else
 /home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python ICLR_2027/revision/new_experiments/summarize.py "$1"
fi
