#!/bin/bash
#SBATCH --job-name=unilora_stage_ratio_glue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=results_glue_variants/unilora_stage_ratio_%j.out
#SBATCH --error=results_glue_variants/unilora_stage_ratio_%j.err

mkdir -p logs

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

echo ">>> Pre-warming cache (Downloading models and datasets)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
for m in ['roberta-large']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
for t in ['cola', 'mrpc']:
    try:
        load_dataset('nyu-mll/glue', t)
    except Exception:
        pass
"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHODS=(unilora_stage_ratio)
STAGE_THETA_D_RATIOS=(0.2 0.3 0.5)
MODELS=(roberta-large)
TASKS=(cola mrpc sst2 qnli rte stsb)
SEEDS=(0 1 2 3 4)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)

SCRIPT=run_unilora_variants_glue.py
OUT_ROOT=results_glue_variants
mkdir -p "${OUT_ROOT}"

CMD_LIST=$(mktemp)

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR=${OUT_ROOT}/${MODEL}/${TASK}
    mkdir -p "${TASK_DIR}"
    for METHOD in "${METHODS[@]}"; do
      R0="${STAGE_THETA_D_RATIOS[0]}"
      R1="${STAGE_THETA_D_RATIOS[1]}"
      R2="${STAGE_THETA_D_RATIOS[2]}"
      M_NAME="${METHOD}_ratio_${R0}_${R1}_${R2}"
      EXTRA_FLAG="--stage_theta_d_ratios ${R0} ${R1} ${R2}"

      for SEED in "${SEEDS[@]}"; do
        SEED_DIR=${TASK_DIR}/${M_NAME}/seed_${SEED}
        mkdir -p "${SEED_DIR}"
        for LR in "${LRS[@]}"; do
          LOG_FILE=${SEED_DIR}/log_lr_${LR}.txt
          FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --gpu-bind=single:1 python ${SCRIPT} --variant ${METHOD} ${EXTRA_FLAG} --model_name ${MODEL} --task ${TASK} --head_lr ${LR} --seed ${SEED} --out_dir ${SEED_DIR} > ${LOG_FILE} 2>&1"
          echo "$FULL_CMD" >> "$CMD_LIST"
        done
      done
    done
  done
done

echo "Total tasks generated: $(wc -l < $CMD_LIST). Starting parallel queue with 4 slots..."
cat "$CMD_LIST" | xargs -I {} -P 4 bash -c "{}"

rm "$CMD_LIST"
echo "All tasks in the queue have been processed."
