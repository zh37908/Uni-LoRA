#!/bin/bash
#SBATCH --job-name=prolosa_grid_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --partition=amd
#SBATCH --account=shsong
#SBATCH --chdir=/home/hzhaobi/Uni-LoRA/math_instruction_tuning
#SBATCH --output=logs/prolosa_grid_summary_%j.out
#SBATCH --error=logs/prolosa_grid_summary_%j.err
set -euo pipefail
/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python prolosa_grid_1m.py summarize
