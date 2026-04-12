#!/bin/bash
#
# Smoke / sweep runner for Geo-UniLoRA on GLUE (subset: cola, sst2 by default).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

if [[ -f /home/hzhaobi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
  conda activate nlu
fi

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola sst2})
SEEDS=(${SEEDS:-0})
HEAD_LRS=(${HEAD_LRS:-5e-4})
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-3})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"

GEO_CALIBRATION_STEPS="${GEO_CALIBRATION_STEPS:-16}"
GEO_NUM_GROUPS="${GEO_NUM_GROUPS:-8}"
GEO_SHARED_RATIO="${GEO_SHARED_RATIO:-0.5}"
GEO_ID_ESTIMATOR="${GEO_ID_ESTIMATOR:-prank}"
GEO_GROUPING="${GEO_GROUPING:-layer_block}"

SCRIPT=run_unilora_variants_glue.py
VARIANT="geo_unilora"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_geo_unilora_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        OUT_DIR="${OUT_ROOT}/${MODEL}/${TASK}/geo_g${GEO_NUM_GROUPS}_sr${GEO_SHARED_RATIO}_${GEO_ID_ESTIMATOR}/seed_${SEED}"
        mkdir -p "${OUT_DIR}"
        LOG="logs/geo_unilora_${MODEL}_${TASK}_seed${SEED}_lr${HEAD_LR}.log"
        echo ">>> Run TASK=${TASK} SEED=${SEED} HEAD_LR=${HEAD_LR} THETA_D_LR=${THETA_D_LR}"
        CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
          --model_name "${MODEL}" \
          --task "${TASK}" \
          --variant "${VARIANT}" \
          --head_lr "${HEAD_LR}" \
          --theta_d_lr "${THETA_D_LR}" \
          --seed "${SEED}" \
          --batch_size "${BATCH_SIZE}" \
          --rank "${RANK}" \
          --theta_d_length "${THETA_D_LENGTH}" \
          --init_theta_d_bound "${INIT_THETA_D_BOUND}" \
          --unilora_dropout "${UNILORA_DROPOUT}" \
          --geo_calibration_steps "${GEO_CALIBRATION_STEPS}" \
          --geo_num_groups "${GEO_NUM_GROUPS}" \
          --geo_shared_ratio "${GEO_SHARED_RATIO}" \
          --geo_id_estimator "${GEO_ID_ESTIMATOR}" \
          --geo_grouping "${GEO_GROUPING}" \
          --out_dir "${OUT_DIR}" \
          2>&1 | tee "${LOG}"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
      done
    done
  done
done

echo ">>> Done. total_runs=${TOTAL_RUNS} out_root=${OUT_ROOT}"
