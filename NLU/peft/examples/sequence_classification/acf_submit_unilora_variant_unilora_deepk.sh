#!/bin/bash
#
# Local / ACF-style sequential runner for UniLoRA-DeepK on GLUE.
# - Layer-wise DeepK regularization during training
# - Optional hard assignment at end (--deepk_assign_stage=end)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate nlu

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-mrpc})
SEEDS=(${SEEDS:-0})

HEAD_LRS=(${HEAD_LRS:-5e-4})
THETA_D_LR_LIST=(${THETA_D_LR_LIST:-5e-3})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"
UNILORA_DROPOUT="${UNILORA_DROPOUT:-0.0}"

DEEPK_NUM_CLUSTERS_A_LIST=(${DEEPK_NUM_CLUSTERS_A_LIST:-16})
DEEPK_NUM_CLUSTERS_B_LIST=(${DEEPK_NUM_CLUSTERS_B_LIST:-16})
DEEPK_TAU_LIST=(${DEEPK_TAU_LIST:-1e-5 5e-5 1e-4})
DEEPK_F_UPDATE_INTERVAL_LIST=(${DEEPK_F_UPDATE_INTERVAL_LIST:-100})
DEEPK_WARMUP_RATIO="${DEEPK_WARMUP_RATIO:-0.1}"
DEEPK_ASSIGN_STAGE="${DEEPK_ASSIGN_STAGE:-none}"   # none | end
DEEPK_SVD_RANK_CAP="${DEEPK_SVD_RANK_CAP:-0}"

echo ">>> Pre-warming cache (downloading models and datasets if needed)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
model_name = "${MODEL}"
tasks = "${TASKS[*]}".split()
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
for task in tasks:
    try:
        load_dataset("nyu-mll/glue", task)
    except Exception:
        pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT=run_unilora_variants_glue.py
VARIANT="unilora_deepk"
OUT_ROOT="${OUT_ROOT:-results_glue_variants_unilora_deepk_acf}"
mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for K_A in "${DEEPK_NUM_CLUSTERS_A_LIST[@]}"; do
          for K_B in "${DEEPK_NUM_CLUSTERS_B_LIST[@]}"; do
            for TAU in "${DEEPK_TAU_LIST[@]}"; do
              for F_INT in "${DEEPK_F_UPDATE_INTERVAL_LIST[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} UniLoRA-DeepK jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE} rank=${RANK} theta_d_length=${THETA_D_LENGTH}"
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> theta_d_lr_list=${THETA_D_LR_LIST[*]}"
echo ">>> deepk: KA=${DEEPK_NUM_CLUSTERS_A_LIST[*]} KB=${DEEPK_NUM_CLUSTERS_B_LIST[*]} tau=${DEEPK_TAU_LIST[*]} f_interval=${DEEPK_F_UPDATE_INTERVAL_LIST[*]}"
echo ">>> deepk: warmup_ratio=${DEEPK_WARMUP_RATIO} assign_stage=${DEEPK_ASSIGN_STAGE} svd_rank_cap=${DEEPK_SVD_RANK_CAP}"

RUN_IDX=0

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for HEAD_LR in "${HEAD_LRS[@]}"; do
      for THETA_D_LR in "${THETA_D_LR_LIST[@]}"; do
        for K_A in "${DEEPK_NUM_CLUSTERS_A_LIST[@]}"; do
          for K_B in "${DEEPK_NUM_CLUSTERS_B_LIST[@]}"; do
            for TAU in "${DEEPK_TAU_LIST[@]}"; do
              for F_INT in "${DEEPK_F_UPDATE_INTERVAL_LIST[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))

                METHOD_NAME="${VARIANT}_ka${K_A}_kb${K_B}_tau${TAU}_fi${F_INT}_as${DEEPK_ASSIGN_STAGE}"
                SEED_DIR="${TASK_DIR}/${METHOD_NAME}/seed_${SEED}"
                mkdir -p "${SEED_DIR}"
                LOG_FILE="${SEED_DIR}/log_headlr_${HEAD_LR}_thetalr_${THETA_D_LR}.txt"

                CMD=(
                  python "${SCRIPT}"
                  --variant "${VARIANT}"
                  --model_name "${MODEL}"
                  --task "${TASK}"
                  --batch_size "${BATCH_SIZE}"
                  --rank "${RANK}"
                  --theta_d_length "${THETA_D_LENGTH}"
                  --theta_d_lr "${THETA_D_LR}"
                  --head_lr "${HEAD_LR}"
                  --seed "${SEED}"
                  --init_theta_d_bound "${INIT_THETA_D_BOUND}"
                  --unilora_dropout "${UNILORA_DROPOUT}"
                  --deepk_num_clusters_a "${K_A}"
                  --deepk_num_clusters_b "${K_B}"
                  --deepk_tau "${TAU}"
                  --deepk_f_update_interval "${F_INT}"
                  --deepk_warmup_ratio "${DEEPK_WARMUP_RATIO}"
                  --deepk_assign_stage "${DEEPK_ASSIGN_STAGE}"
                  --deepk_svd_rank_cap "${DEEPK_SVD_RANK_CAP}"
                  --out_dir "${SEED_DIR}"
                )

                echo "=================================================="
                echo "[${RUN_IDX}/${TOTAL_RUNS}] task=${TASK} seed=${SEED} head_lr=${HEAD_LR} theta_d_lr=${THETA_D_LR} method=${METHOD_NAME}"
                echo "log: ${LOG_FILE}"
                echo "=================================================="

                CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_FILE}" 2>&1

                echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${LOG_FILE}"
              done
            done
          done
        done
      done
    done
  done
done

echo "All local UniLoRA-DeepK jobs have been processed."
