#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-Multi-Hashing on GLUE.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

# Activate NLU conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate nlu

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Experiment parameters
METHODS=(${METHODS:-unilora_multi_hashing})
MODELS=(${MODELS:-roberta-large})
TASKS=(${TASKS:-cola mrpc})
SEEDS=(${SEEDS:-0 })
LRS=(${LRS:-2e-4 5e-4})
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU="${GPU:-0}"

BASE_THETA_D_LENGTH_LIST=(${BASE_THETA_D_LENGTH_LIST:-23040 46080 92160})
MULTI_HASHING_NUM_COMPONENTS=(${MULTI_HASHING_NUM_COMPONENTS:-1 2 4})
MULTI_HASHING_INIT_P_BOUNDS=(${MULTI_HASHING_INIT_P_BOUNDS:-default})

SCRIPT="${SCRIPT:-run_unilora_variants_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_multi_hashing_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for BASE_THETA_D_LENGTH in "${BASE_THETA_D_LENGTH_LIST[@]}"; do
        for NUM_COMPONENTS in "${MULTI_HASHING_NUM_COMPONENTS[@]}"; do
          for INIT_P_BOUND in "${MULTI_HASHING_INIT_P_BOUNDS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
              for LR in "${LRS[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-Multi-Hashing jobs sequentially on local GPU ${GPU}"
echo ">>> models=${MODELS[*]} tasks=${TASKS[*]} seeds=${SEEDS[*]} head_lrs=${LRS[*]}"
echo ">>> base_theta_d_length_list=${BASE_THETA_D_LENGTH_LIST[*]} num_components=${MULTI_HASHING_NUM_COMPONENTS[*]} init_p_bounds=${MULTI_HASHING_INIT_P_BOUNDS[*]}"
echo ">>> out_root=${OUT_ROOT}"

RUN_IDX=0

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
    mkdir -p "${TASK_DIR}"
    for METHOD in "${METHODS[@]}"; do
      for BASE_THETA_D_LENGTH in "${BASE_THETA_D_LENGTH_LIST[@]}"; do
        for NUM_COMPONENTS in "${MULTI_HASHING_NUM_COMPONENTS[@]}"; do
          if (( BASE_THETA_D_LENGTH % NUM_COMPONENTS != 0 )); then
            echo "Error: BASE_THETA_D_LENGTH=${BASE_THETA_D_LENGTH} is not divisible by NUM_COMPONENTS=${NUM_COMPONENTS}" >&2
            exit 1
          fi

          COMPONENT_THETA_D_LENGTH=$((BASE_THETA_D_LENGTH / NUM_COMPONENTS))

          for INIT_P_BOUND in "${MULTI_HASHING_INIT_P_BOUNDS[@]}"; do
            if [[ "${INIT_P_BOUND}" == "default" ]]; then
              M_NAME="${METHOD}_b${BASE_THETA_D_LENGTH}_k${NUM_COMPONENTS}_len${COMPONENT_THETA_D_LENGTH}"
              EXTRA_FLAG=(--multi_hashing_num_components "${NUM_COMPONENTS}" --theta_d_length "${COMPONENT_THETA_D_LENGTH}")
            else
              M_NAME="${METHOD}_b${BASE_THETA_D_LENGTH}_k${NUM_COMPONENTS}_len${COMPONENT_THETA_D_LENGTH}_p${INIT_P_BOUND}"
              EXTRA_FLAG=(
                --multi_hashing_num_components "${NUM_COMPONENTS}"
                --theta_d_length "${COMPONENT_THETA_D_LENGTH}"
                --multi_hashing_init_p_bound "${INIT_P_BOUND}"
              )
            fi

            for SEED in "${SEEDS[@]}"; do
              SEED_DIR="${TASK_DIR}/${M_NAME}/seed_${SEED}"
              mkdir -p "${SEED_DIR}"

              for LR in "${LRS[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))
                LOG_FILE="${SEED_DIR}/log_lr_${LR}.txt"

                echo "=================================================="
                echo "[${RUN_IDX}/${TOTAL_RUNS}] model=${MODEL} task=${TASK} seed=${SEED} head_lr=${LR} method=${M_NAME}"
                echo "out_dir: ${SEED_DIR}"
                echo "log: ${LOG_FILE}"
                echo "=================================================="

                CUDA_VISIBLE_DEVICES="${GPU}" \
                python "${SCRIPT}" --variant "${METHOD}" "${EXTRA_FLAG[@]}" --model_name "${MODEL}" \
                  --task "${TASK}" --batch_size "${BATCH_SIZE}" --head_lr "${LR}" --seed "${SEED}" \
                  --out_dir "${SEED_DIR}" > "${LOG_FILE}" 2>&1
              done
            done
          done
        done
      done
    done
  done
done

echo "All UniLoRA-Multi-Hashing tasks finished successfully."
