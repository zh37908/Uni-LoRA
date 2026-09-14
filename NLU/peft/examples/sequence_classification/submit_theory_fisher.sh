#!/bin/bash
#SBATCH --job-name=theory_fisher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/theory_fisher_%j.out
#SBATCH --error=logs/theory_fisher_%j.err

# Diagonal empirical Fisher in LoRA (A/B) coordinate space, one vector per task, from a
# full-data plain-LoRA artifact (dedicated-lr run preferred). Feeds the curvature-weighted
# E4/E5 statistics: analyze_theory_e4_e5.py --fisher task=fisher_diag/<task>_fisher.pt

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1
mkdir -p logs fisher_diag || exit 1

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1
set -uo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola mrpc rte stsb qnli sst2})
NUM_BATCHES="${NUM_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED_DIR_NAME="${SEED_DIR_NAME:-seed_0}"

# Candidate roots per task; first existing artifact wins. Dedicated-lr dirs (lora_r4_lr*)
# are listed before legacy shared-lr dirs.
pick_artifact() {
  local task="$1"
  local cands=(
    "results_theory_e4_projection_sweep/${MODEL}/${task}/lora_r4_lr*/${SEED_DIR_NAME}/*_theory.pt"
    "results_theory_e4_projection_sweep/${MODEL}/${task}/lora_r4/${SEED_DIR_NAME}/*_theory.pt"
    "results_theory_e4_remaining_glue/${MODEL}/${task}/lora_r4_lr*/${SEED_DIR_NAME}/*_theory.pt"
    "results_theory_e4_remaining_glue/${MODEL}/${task}/lora_r4/${SEED_DIR_NAME}/*_theory.pt"
    "results_theory_e1_crossover/${MODEL}/${task}/p1/lora_r4_lr*/${SEED_DIR_NAME}/*_theory.pt"
    "results_theory_e1_crossover/${MODEL}/${task}/p1/lora_r4/${SEED_DIR_NAME}/*_theory.pt"
  )
  local pat
  for pat in "${cands[@]}"; do
    # shellcheck disable=SC2086
    for f in ${pat}; do
      if [[ -f "${f}" ]]; then
        echo "${f}"
        return 0
      fi
    done
  done
  return 1
}

for TASK in "${TASKS[@]}"; do
  OUT="fisher_diag/${TASK}_fisher.pt"
  if [[ -s "${OUT}" ]]; then
    echo ">>> ${TASK}: ${OUT} exists, skip"
    continue
  fi
  ART="$(pick_artifact "${TASK}")" || { echo ">>> ${TASK}: no LoRA artifact found, skip"; continue; }
  echo ">>> ${TASK}: artifact=${ART}"
  python estimate_fisher_diag_glue.py \
    --artifact "${ART}" --output "${OUT}" \
    --split train --batch_size "${BATCH_SIZE}" --num_batches "${NUM_BATCHES}" \
    || echo ">>> ${TASK}: Fisher estimation FAILED"
done

echo ">>> Fisher estimation done."
ls -la fisher_diag
