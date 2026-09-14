#!/bin/bash
#SBATCH --job-name=vit_table5_summary
#SBATCH --account=shsong
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/hzhaobi/Uni-LoRA/ViT/table5/logs/summary_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/ViT/table5/logs/summary_%j.err
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA/ViT/table5
/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python -u finish.py
