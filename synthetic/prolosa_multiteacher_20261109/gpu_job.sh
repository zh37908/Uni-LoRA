#!/usr/bin/env bash
#SBATCH --account=shsong
#SBATCH --partition=gpu-a30
#SBATCH --qos=a30_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

cd /home/hzhaobi/Uni-LoRA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

python=${MULTITEACHER_PYTHON:-/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python}
runner=synthetic/prolosa_multiteacher_20261109/run_multiteacher.py

case "${1:-}" in
    smoke)
        [[ "$#" -eq 1 ]] || { echo "Usage: gpu_job.sh smoke" >&2; exit 2; }
        "$python" -u "$runner" prepare
        exec "$python" -u "$runner" smoke
        ;;
    run)
        [[ "$#" -eq 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
            echo "Usage: gpu_job.sh run SHARDS" >&2; exit 2;
        }
        [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
            echo "run requires a Slurm array task ID" >&2; exit 2;
        }
        (( SLURM_ARRAY_TASK_ID < $2 )) || {
            echo "Array task $SLURM_ARRAY_TASK_ID exceeds shard count $2" >&2; exit 2;
        }
        exec "$python" -u "$runner" run --shards "$2"
        ;;
    *)
        echo "Usage: gpu_job.sh {smoke|run SHARDS}" >&2
        exit 2
        ;;
esac
