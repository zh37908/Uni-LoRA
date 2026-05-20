#!/bin/bash
#SBATCH --job-name=unilora_orig_stsb_repro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/unilora_original_stsb_repro_%j.out
#SBATCH --error=logs/unilora_original_stsb_repro_%j.err

mkdir -p logs

# Activate NLU conda env.
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

set -euo pipefail

# Clear proxy settings to avoid network-related hangs on compute nodes.
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# Limit CPU thread contention across parallel srun jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASK="${TASK:-stsb}"
SEEDS=(${SEEDS:-0 1 2 3 4})

# Keep LR set consistent with the original UniLoRA failing run.
HEAD_LRS=(${HEAD_LRS:-1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2})

SCRIPT=run_unilora_glue.py
OUT_ROOT="${OUT_ROOT:-results_glue_original_unilora_stsb_repro}"
mkdir -p "${OUT_ROOT}"

echo ">>> Pre-warming cache (downloading model and STSB dataset if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

model_name = "${MODEL}"
AutoTokenizer.from_pretrained(model_name)
try:
    AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
except Exception:
    pass
try:
    load_dataset("nyu-mll/glue", "${TASK}")
except Exception:
    pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CMD_LIST="$(mktemp)"
TOTAL_RUNS=0
TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
mkdir -p "${TASK_DIR}"

for SEED in "${SEEDS[@]}"; do
  SEED_DIR="${TASK_DIR}/seed_${SEED}"
  mkdir -p "${SEED_DIR}"

  for HEAD_LR in "${HEAD_LRS[@]}"; do
    LOG_FILE="${SEED_DIR}/log_lr_${HEAD_LR}.txt"
    RESULT_JSON="${SEED_DIR}/${TASK}_${MODEL}_lr_${HEAD_LR}_seed_${SEED}.json"

    if [[ -s "${RESULT_JSON}" ]]; then
      echo "Skip existing result: ${RESULT_JSON}"
      continue
    fi

    # run_unilora_glue.py internally keeps:
    # batch_size=32, max_length=128(for roberta-large), num_epochs=40(for stsb), warmup_ratio=0.06, theta_d_lr=5e-3
    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${SCRIPT} --model_name ${MODEL} --task ${TASK} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} > ${LOG_FILE} 2>&1"

    echo "${FULL_CMD}" >> "${CMD_LIST}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
  done
done

echo ">>> Generated ${TOTAL_RUNS} original UniLoRA STSB jobs."
echo ">>> model=${MODEL} task=${TASK} seeds=${SEEDS[*]}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> fixed params in ${SCRIPT}: batch_size=32, max_length=128, num_epochs=40, warmup_ratio=0.06, theta_d_lr=5e-3"

if [[ "${TOTAL_RUNS}" -eq 0 ]]; then
  echo "No jobs to run."
  rm -f "${CMD_LIST}"
  exit 0
fi

echo ">>> Starting parallel queue with 4 slots..."
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"

rm -f "${CMD_LIST}"
echo "All original UniLoRA STSB jobs have been processed."
