#!/bin/bash
#SBATCH --job-name=alpha_lr_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --chdir=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification
#SBATCH --output=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/results_glue_variants/alpha_lr_sweep_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/results_glue_variants/alpha_lr_sweep_%j.err

set -eo pipefail

# Sweep alpha_lr for UniLoRA learnable-layer variant.
# Keeps all other hyperparameters fixed.

mkdir -p logs
mkdir -p results_glue_variants

# Activate conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

# Clear proxy variables to avoid network timeout on compute nodes
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# Limit CPU threads to reduce contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Pre-warm cache to avoid first-run download races
echo ">>> Pre-warming cache..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
AutoTokenizer.from_pretrained('roberta-large')
AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
load_dataset('nyu-mll/glue', 'cola')
"

# Prefer offline after pre-warm for better stability
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fixed settings
VARIANT="unilora_learnable_layer"
MODEL_NAME="roberta-large"
TASK="cola"
HEAD_LR="1e-3"
THETA_D_LR="5e-3"
ALPHA_FREEZE_RATIO="0.1"
ALPHA_INIT="1.0"
ALPHA_MIN="0.1"
ALPHA_MAX="10.0"
SEED="0"

# Only alpha_lr changes
ALPHA_LRS=(
  "1e-5"
  "5e-5"
  "1e-4"
  "2e-4"
  "5e-4"
)

OUT_ROOT="results_glue_variants/alpha_lr_sweep/${MODEL_NAME}/${TASK}/${VARIANT}/seed_${SEED}"
mkdir -p "${OUT_ROOT}"

echo "Start alpha_lr sweep. Output root: ${OUT_ROOT}"
for ALPHA_LR in "${ALPHA_LRS[@]}"; do
  RUN_DIR="${OUT_ROOT}/alpha_lr_${ALPHA_LR}"
  LOG_FILE="${RUN_DIR}/run.log"
  mkdir -p "${RUN_DIR}"

  echo "============================================================"
  echo "Running alpha_lr=${ALPHA_LR}"
  echo "Log: ${LOG_FILE}"

  srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --gpu-bind=single:1 \
  python run_unilora_variants_glue.py \
    --variant "${VARIANT}" \
    --model_name "${MODEL_NAME}" \
    --task "${TASK}" \
    --head_lr "${HEAD_LR}" \
    --theta_d_lr "${THETA_D_LR}" \
    --alpha_lr "${ALPHA_LR}" \
    --alpha_freeze_ratio "${ALPHA_FREEZE_RATIO}" \
    --alpha_init "${ALPHA_INIT}" \
    --alpha_min "${ALPHA_MIN}" \
    --alpha_max "${ALPHA_MAX}" \
    --seed "${SEED}" \
    --out_dir "${RUN_DIR}" \
    > "${LOG_FILE}" 2>&1
done

echo "Done. Sweep finished."
