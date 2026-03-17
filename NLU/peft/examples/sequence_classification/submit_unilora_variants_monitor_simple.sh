#!/bin/bash
#SBATCH --job-name=unilora_variants_monitor_simple
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=results_variants_monitor/unilora_variants_monitor_simple_%j.out
#SBATCH --error=results_variants_monitor/unilora_variants_monitor_simple_%j.err

set -e

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODEL=roberta-large
TASK=mrpc
SEED=42
LR=1e-3
SCRIPT=run_unilora_variants_glue_monitor.py
OUT_ROOT=results_variants_monitor
mkdir -p "${OUT_ROOT}"

METHODS=(unilora_fastfood unilora_nonorm unilora_learnable unilora_isometric_control unilora)

# Create a temporary file to store all commands
CMD_LIST=$(mktemp)

for METHOD in "${METHODS[@]}"; do
  if [[ "$METHOD" == "unilora_isometric_control" ]]; then
    ALPHAS=(0.5)
  else
    ALPHAS=(0.0)
  fi

  for ALPHA in "${ALPHAS[@]}"; do
    if [[ "$METHOD" == "unilora_isometric_control" ]]; then
      M_NAME="${METHOD}_alpha${ALPHA}"
      EXTRA_FLAG="--isometry_alpha ${ALPHA}"
    else
      M_NAME="${METHOD}"
      EXTRA_FLAG=""
    fi

    OUT_DIR="${OUT_ROOT}/${M_NAME}_${TASK}_${MODEL}_lr${LR}_seed${SEED}"
    mkdir -p "${OUT_DIR}"

    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --gpu-bind=single:1 \
python ${SCRIPT} \
  --model_name ${MODEL} \
  --task ${TASK} \
  --variant ${METHOD} \
  ${EXTRA_FLAG} \
  --head_lr ${LR} \
  --seed ${SEED} \
  --out_dir ${OUT_DIR} \
  --lanczos_every 0"
    echo "$FULL_CMD" >> "$CMD_LIST"
  done
done

echo "Total tasks generated: $(wc -l < $CMD_LIST). Starting parallel queue with 4 slots..."
cat "$CMD_LIST" | xargs -I {} -P 4 bash -c "{}"

rm "$CMD_LIST"
echo "All tasks in the queue have been processed."
