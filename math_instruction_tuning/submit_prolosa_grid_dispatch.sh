#!/bin/bash
#SBATCH --job-name=prolosa_grid_dispatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=72:00:00
#SBATCH --partition=amd
#SBATCH --account=shsong
#SBATCH --chdir=/home/hzhaobi/Uni-LoRA/math_instruction_tuning
#SBATCH --output=logs/prolosa_grid_dispatch_%j.out
#SBATCH --error=logs/prolosa_grid_dispatch_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python dispatch_prolosa_grid_1m.py
