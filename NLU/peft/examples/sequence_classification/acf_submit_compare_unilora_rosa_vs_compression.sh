#!/bin/bash
#
# Local / ACF-style sequential comparison runner:
# compare `unilora_rosa_compression` vs `unilora_rosa` under matched sparse-trainable budgets.
#
# Matching rule:
# - Rosa-Compression: train a compressed sparse bank with `sparse_theta_d_length = std`
# - Original RoSA: choose `rosa_density = std / total_sparse_positions`
#   so the number of active sparse offsets approximately matches the compressed bank size.
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
TASKS=(${TASKS:-cola})
SEEDS=(${SEEDS:-0})

# For CoLA baseline, larger head LR often works better than 2e-4.
LRS=(${LRS:-2e-4 1e-3 5e-3})

RANK="${RANK:-4}"
THETA_D_LENGTH="${THETA_D_LENGTH:-23040}"
THETA_D_LR="${THETA_D_LR:-5e-3}"
INIT_THETA_D_BOUND="${INIT_THETA_D_BOUND:-0.02}"

# Compare these compression budgets.
SPARSE_THETA_D_LENGTH_LIST=(${SPARSE_THETA_D_LENGTH_LIST:-11520 5760 2880})

# Keep the compare focused on the currently promising region; can still override by env.
ROSA_COMPRESSION_DENSITY="${ROSA_COMPRESSION_DENSITY:-0.01}"
ROSA_WARMUP_STEPS_LIST=(${ROSA_WARMUP_STEPS_LIST:-256})
ROSA_MASK_STEPS_LIST=(${ROSA_MASK_STEPS_LIST:-1})
ROSA_SPARSE_LR_MULT_LIST=(${ROSA_SPARSE_LR_MULT_LIST:-0.2 1.0})
ROSA_RESET_LIST=(${ROSA_RESET_LIST:-0 1})

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
OUT_ROOT="${OUT_ROOT:-results_compare_unilora_rosa_vs_compression_acf}"
mkdir -p "${OUT_ROOT}"

# Compute total sparse positions for the current model / rank / target module set.
TOTAL_SPARSE_POSITIONS="$(python - <<PY
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
target_suffixes = ["query", "key", "value", "output.dense", "intermediate.dense"]
rank = int("${RANK}")
total = 0
for name, module in model.named_modules():
    if not hasattr(module, "weight"):
        continue
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    weight = module.weight
    out_features, in_features = weight.shape[0], weight.shape[1]
    total += rank * in_features + out_features * rank
print(total)
PY
)"

echo ">>> total_sparse_positions=${TOTAL_SPARSE_POSITIONS}"

sparse_lr_from_mult() {
  python3 -c "print(float('${THETA_D_LR}') * float('${1}'))"
}

matched_density_from_std() {
  python3 - <<PY
std = int("${1}")
total = int("${TOTAL_SPARSE_POSITIONS}")
density = std / total
text = f"{density:.12f}".rstrip("0").rstrip(".")
print(text if text else "0")
PY
}

TOTAL_RUNS=0
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for STD in "${SPARSE_THETA_D_LENGTH_LIST[@]}"; do
        for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 2))
              done
            done
          done
        done
      done
    done
  done
done

echo ">>> Running ${TOTAL_RUNS} compare jobs sequentially on local GPU ${GPU}"
echo ">>> tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> model=${MODEL} rank=${RANK} theta_d_length=${THETA_D_LENGTH} theta_d_lr=${THETA_D_LR}"
echo ">>> std_list=${SPARSE_THETA_D_LENGTH_LIST[*]} compression_density=${ROSA_COMPRESSION_DENSITY}"
echo ">>> warmup_list=${ROSA_WARMUP_STEPS_LIST[*]} mask_steps_list=${ROSA_MASK_STEPS_LIST[*]}"
echo ">>> sparse_lr_mult_list=${ROSA_SPARSE_LR_MULT_LIST[*]} reset_list=${ROSA_RESET_LIST[*]}"
echo ">>> head_lrs=${LRS[*]}"

RUN_IDX=0
for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${MODEL}/${TASK}"
  mkdir -p "${TASK_DIR}"

  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      for STD in "${SPARSE_THETA_D_LENGTH_LIST[@]}"; do
        MATCHED_DENSITY="$(matched_density_from_std "${STD}")"

        for ROSA_WARMUP_STEPS in "${ROSA_WARMUP_STEPS_LIST[@]}"; do
          for ROSA_MASK_STEPS in "${ROSA_MASK_STEPS_LIST[@]}"; do
            for ROSA_SPARSE_MULT in "${ROSA_SPARSE_LR_MULT_LIST[@]}"; do
              for ROSA_RESET_OPTIMIZER_ON_MASK in "${ROSA_RESET_LIST[@]}"; do
                ROSA_SPARSE_LR="$(sparse_lr_from_mult "${ROSA_SPARSE_MULT}")"
                MULT_TAG="${ROSA_SPARSE_MULT//./p}"

                # 1) RoSA-Compression run
                RUN_IDX=$((RUN_IDX + 1))
                COMP_METHOD_NAME="unilora_rosa_compression_cmpstd${STD}_d${ROSA_COMPRESSION_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
                COMP_SEED_DIR="${TASK_DIR}/${COMP_METHOD_NAME}/seed_${SEED}"
                mkdir -p "${COMP_SEED_DIR}"
                COMP_LOG_FILE="${COMP_SEED_DIR}/log_lr_${LR}.txt"

                COMP_CMD=(
                  python "${SCRIPT}"
                  --variant "unilora_rosa_compression"
                  --model_name "${MODEL}"
                  --task "${TASK}"
                  --batch_size "${BATCH_SIZE}"
                  --rank "${RANK}"
                  --theta_d_length "${THETA_D_LENGTH}"
                  --theta_d_lr "${THETA_D_LR}"
                  --init_theta_d_bound "${INIT_THETA_D_BOUND}"
                  --sparse_theta_d_length "${STD}"
                  --rosa_density "${ROSA_COMPRESSION_DENSITY}"
                  --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
                  --rosa_mask_steps "${ROSA_MASK_STEPS}"
                  --rosa_sparse_lr "${ROSA_SPARSE_LR}"
                  --head_lr "${LR}"
                  --seed "${SEED}"
                  --out_dir "${COMP_SEED_DIR}"
                )
                if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
                  COMP_CMD+=(--rosa_reset_optimizer_on_mask)
                fi

                echo "=================================================="
                echo "[${RUN_IDX}/${TOTAL_RUNS}] COMP task=${TASK} seed=${SEED} head_lr=${LR} std=${STD} matched_density=${MATCHED_DENSITY}"
                echo "out_dir: ${COMP_SEED_DIR}"
                echo "log: ${COMP_LOG_FILE}"
                echo "=================================================="
                CUDA_VISIBLE_DEVICES="${GPU}" "${COMP_CMD[@]}" > "${COMP_LOG_FILE}" 2>&1
                echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${COMP_LOG_FILE}"

                # 2) Original RoSA run with matched density
                RUN_IDX=$((RUN_IDX + 1))
                ROSA_METHOD_NAME="unilora_rosa_matchstd${STD}_d${MATCHED_DENSITY}_w${ROSA_WARMUP_STEPS}_m${ROSA_MASK_STEPS}_slrm${MULT_TAG}_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
                ROSA_SEED_DIR="${TASK_DIR}/${ROSA_METHOD_NAME}/seed_${SEED}"
                mkdir -p "${ROSA_SEED_DIR}"
                ROSA_LOG_FILE="${ROSA_SEED_DIR}/log_lr_${LR}.txt"

                ROSA_CMD=(
                  python "${SCRIPT}"
                  --variant "unilora_rosa"
                  --model_name "${MODEL}"
                  --task "${TASK}"
                  --batch_size "${BATCH_SIZE}"
                  --rank "${RANK}"
                  --theta_d_length "${THETA_D_LENGTH}"
                  --theta_d_lr "${THETA_D_LR}"
                  --init_theta_d_bound "${INIT_THETA_D_BOUND}"
                  --rosa_density "${MATCHED_DENSITY}"
                  --rosa_warmup_steps "${ROSA_WARMUP_STEPS}"
                  --rosa_mask_steps "${ROSA_MASK_STEPS}"
                  --rosa_sparse_lr "${ROSA_SPARSE_LR}"
                  --head_lr "${LR}"
                  --seed "${SEED}"
                  --out_dir "${ROSA_SEED_DIR}"
                )
                if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
                  ROSA_CMD+=(--rosa_reset_optimizer_on_mask)
                fi

                echo "=================================================="
                echo "[${RUN_IDX}/${TOTAL_RUNS}] ROSA task=${TASK} seed=${SEED} head_lr=${LR} matched_std=${STD} density=${MATCHED_DENSITY}"
                echo "out_dir: ${ROSA_SEED_DIR}"
                echo "log: ${ROSA_LOG_FILE}"
                echo "=================================================="
                CUDA_VISIBLE_DEVICES="${GPU}" "${ROSA_CMD[@]}" > "${ROSA_LOG_FILE}" 2>&1
                echo "Finished [${RUN_IDX}/${TOTAL_RUNS}] -> ${ROSA_LOG_FILE}"
              done
            done
          done
        done
      done
    done
  done
done

echo "All compare jobs have been processed."

