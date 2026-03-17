#!/bin/bash
#SBATCH --job-name=unilora_stage_ratio_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=results_glue_variants/unilora_stage_ratio_sweep_%j.out
#SBATCH --error=results_glue_variants/unilora_stage_ratio_sweep_%j.err

mkdir -p logs

# Activate NLU conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

# Avoid proxy timeout on compute nodes
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# Limit CPU thread contention
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

# Sweep only this single parameter:
#   --stage_theta_d_ratios FRONT MIDDLE BACK
RATIO_TRIPLES=(
  "0.2 0.4 0.4"
  "0.333 0.333 0.333"
  "0.4 0.2 0.4"
  "0.4 0.4 0.2"
  "0.45 0.1 0.45"
  "0.1 0.45 0.45"
)

# Multi-group sweep for task/seed/lr (edit as needed)
MODELS=(roberta-large)
TASKS=(cola mrpc)
SEEDS=(0 1 2)
LRS=(5e-3)

SCRIPT=run_unilora_variants_glue.py
OUT_ROOT=results_glue_variants
mkdir -p "${OUT_ROOT}"

CMD_LIST=$(mktemp)

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
    mkdir -p "${TASK_DIR}"
    for RATIO in "${RATIO_TRIPLES[@]}"; do
      read -r R0 R1 R2 <<< "${RATIO}"
      M_NAME="unilora_stage_ratio_ratio_${R0}_${R1}_${R2}"
      for SEED in "${SEEDS[@]}"; do
        SEED_DIR="${TASK_DIR}/${M_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"
        for LR in "${LRS[@]}"; do
          LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"
          FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --variant unilora_stage_ratio --stage_theta_d_ratios ${R0} ${R1} ${R2} --model_name ${MODEL} --task ${TASK} --head_lr ${LR} --seed ${SEED} --out_dir ${SEED_DIR} > ${LOG_FILE} 2>&1"
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
