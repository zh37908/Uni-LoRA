#!/bin/bash
#SBATCH --job-name=unilora_trajectory_initial_glue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/logs/unilora_trajectory_initial_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/logs/unilora_trajectory_initial_%j.err

# Under sbatch, BASH_SOURCE points at a copy under /var/spool/slurm/... — do NOT use it for SCRIPT_DIR.
SCRIPT_DIR="/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification"
cd "${SCRIPT_DIR}" || { echo "cd SCRIPT_DIR failed: ${SCRIPT_DIR}"; exit 1; }

mkdir -p "${SCRIPT_DIR}/logs"

# Activate NLU conda env
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
export TOKENIZERS_PARALLELISM=false

RANK=4
THETA_D_LENGTH=23040
THETA_D_LR=5e-3
INIT_THETA_D_BOUND=0.02
TRAJECTORY_NUM_BUCKETS=4
TRAJECTORY_BLOCK_ROWS=4
TRAJECTORY_BLOCK_COLS=4
TRAJECTORY_KMEANS_ITERS=15

echo ">>> Pre-warming cache (Downloading models and datasets)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
for m in ['roberta-large']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
for t in ['mrpc', 'cola', 'sst2', 'qnli']:
    try:
        load_dataset('nyu-mll/glue', t)
    except Exception:
        pass
"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHODS=(unilora_trajectory_initial)
MODELS=(roberta-large)
TASKS=(mrpc)
SEEDS=(0 1 2)
LRS=(5e-4 1e-3 5e-3)

SCRIPT="${SCRIPT_DIR}/run_unilora_variants_glue.py"
OUT_ROOT="${SCRIPT_DIR}/results_glue_variants_trajectory_initial"
mkdir -p "${OUT_ROOT}"

CMD_LIST=$(mktemp)

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR=${OUT_ROOT}/${MODEL}/${TASK}
    mkdir -p "${TASK_DIR}"
    for METHOD in "${METHODS[@]}"; do
      M_NAME="${METHOD}_buckets${TRAJECTORY_NUM_BUCKETS}_br${TRAJECTORY_BLOCK_ROWS}_bc${TRAJECTORY_BLOCK_COLS}_k${TRAJECTORY_KMEANS_ITERS}"
      EXTRA_FLAG="--rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --trajectory_num_buckets ${TRAJECTORY_NUM_BUCKETS} --trajectory_block_rows ${TRAJECTORY_BLOCK_ROWS} --trajectory_block_cols ${TRAJECTORY_BLOCK_COLS} --trajectory_kmeans_iters ${TRAJECTORY_KMEANS_ITERS}"

      for SEED in "${SEEDS[@]}"; do
        SEED_DIR=${TASK_DIR}/${M_NAME}/seed_${SEED}
        mkdir -p "${SEED_DIR}"
        for LR in "${LRS[@]}"; do
          LOG_FILE=${SEED_DIR}/log_lr_${LR}.txt
          FULL_CMD="cd \"${SCRIPT_DIR}\" && srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --gpu-bind=single:1 python \"${SCRIPT}\" --variant ${METHOD} ${EXTRA_FLAG} --model_name ${MODEL} --task ${TASK} --head_lr ${LR} --seed ${SEED} --out_dir \"${SEED_DIR}\" > \"${LOG_FILE}\" 2>&1"
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
