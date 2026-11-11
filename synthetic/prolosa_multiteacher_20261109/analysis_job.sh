#!/usr/bin/env bash
#SBATCH --account=shsong
#SBATCH --partition=amd
#SBATCH --qos=amd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

cd /home/hzhaobi/Uni-LoRA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

python=${MULTITEACHER_PYTHON:-/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python}
exec "$python" -u synthetic/prolosa_multiteacher_20261109/analyze.py
