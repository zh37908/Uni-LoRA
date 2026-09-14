#!/bin/bash
#SBATCH --job-name=lora_rosa_cola_best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/lora_rosa_cola_favorable_sweep_%j.out
#SBATCH --error=logs/lora_rosa_cola_favorable_sweep_%j.err

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1

mkdir -p logs || exit 1

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1

set -euo pipefail

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# Stage-1 single-seed search for LoRA-RoSA on CoLA.
MODEL="${MODEL:-roberta-large}"
TASK="${TASK:-cola}"
METHOD="${METHOD:-lora_rosa}"
SEEDS=(${SEEDS:-0})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"

# LoRA A/B and the classifier currently share --head_lr.  Sweep from the
# conventional LoRA range up to the previous 5e-3 CoLA setting as a control.
HEAD_LRS=(${HEAD_LRS:-1e-4 2e-4 5e-4 1e-3 2e-3 5e-3})

# The previous sweep peaked at 4320, so search that point and its neighbors.
SPARSE_BUDGETS=(${SPARSE_BUDGETS:-2160 4320 8640})

# Compare a conservative sparse LR with the previous 1e-3 setting.
SPARSE_LRS=(${SPARSE_LRS:-2e-4 1e-3})

LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"

ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-128}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"

# Sparse parameters have no Adam state before activation.  Avoid clearing the
# already learned LoRA A/B optimizer state when the sparse mask is activated.
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-0}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"

GPU_SETUP="${GPU_SETUP:-1xNVIDIA_L20}"
GPU_COUNT="${GPU_COUNT:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_unilora_variants_glue.py}"
PROFILER="${PROFILER:-profile_unilora_prolosa_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_lora_rosa_cola_favorable_sweep}"
mkdir -p "${OUT_ROOT}"

if [[ -n "${NUM_EPOCHS:-}" ]]; then
  NUM_EPOCHS_ARG="--num_epochs ${NUM_EPOCHS}"
else
  NUM_EPOCHS_ARG=""
fi

echo ">>> Pre-warming RoBERTa-large and CoLA cache..."
python - <<PY
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

AutoTokenizer.from_pretrained("${MODEL}")
AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
load_dataset("nyu-mll/glue", "${TASK}")
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

read -r LORA_PARAM_COUNT BASE_SPARSE_POSITIONS <<< "$(python - <<PY
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
target_suffixes = ["query", "key", "value", "output.dense", "intermediate.dense"]
rank = int("${RANK}")
lora_total = 0
base_total = 0
for name, module in model.named_modules():
    if not hasattr(module, "weight") or not any(name.endswith(s) for s in target_suffixes):
        continue
    out_features, in_features = module.weight.shape[:2]
    lora_total += rank * in_features + out_features * rank
    base_total += out_features * in_features
print(lora_total, base_total)
PY
)"

echo ">>> lora_param_count=${LORA_PARAM_COUNT}"
echo ">>> base_sparse_candidate_positions=${BASE_SPARSE_POSITIONS}"

density_from_sparse_budget() {
  python3 - <<PY
budget = int("${1}")
positions = int("${BASE_SPARSE_POSITIONS}")
print(f"{budget / positions:.12f}".rstrip("0").rstrip("."))
PY
}

safe_token() {
  printf '%s' "${1}" | tr '.+-' 'p__'
}

CMD_LIST="$(mktemp)"
trap 'rm -f "${CMD_LIST}"' EXIT
TOTAL_RUNS=0

for HEAD_LR in "${HEAD_LRS[@]}"; do
  for SPARSE_BUDGET in "${SPARSE_BUDGETS[@]}"; do
    ROSA_DENSITY="$(density_from_sparse_budget "${SPARSE_BUDGET}")"

    for SPARSE_LR in "${SPARSE_LRS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        CONFIG_NAME="hlr$(safe_token "${HEAD_LR}")_sb${SPARSE_BUDGET}_slr$(safe_token "${SPARSE_LR}")_r${RANK}_w${ROSA_WARMUP_STEPS}_drop$(safe_token "${LORA_DROPOUT}")_rst${ROSA_RESET_OPTIMIZER_ON_MASK}"
        SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${CONFIG_NAME}/seed_${SEED}"
        mkdir -p "${SEED_DIR}"

        PROFILE_JSON="${SEED_DIR}/profile_${METHOD}_${TASK}_${MODEL}_${CONFIG_NAME}_seed${SEED}.json"
        TRAIN_LOG="${SEED_DIR}/train_${METHOD}_${TASK}_${CONFIG_NAME}_seed${SEED}.log"
        RUNNER_LOG="${SEED_DIR}/runner_${METHOD}_${TASK}_${CONFIG_NAME}_seed${SEED}.log"

        if [[ -s "${PROFILE_JSON}" ]]; then
          echo "Skip existing profile: ${PROFILE_JSON}"
          continue
        fi

        FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python ${PROFILER} run --train_script ${TRAIN_SCRIPT} --method ${METHOD} --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${LORA_PARAM_COUNT} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --profile_json ${PROFILE_JSON} --train_log ${TRAIN_LOG} --gpu_setup ${GPU_SETUP} --gpu_count ${GPU_COUNT} --sparse_budget ${SPARSE_BUDGET} --total_sparse_positions ${BASE_SPARSE_POSITIONS} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${SPARSE_LR} --unilora_dropout ${LORA_DROPOUT} --weight_decay ${WEIGHT_DECAY} --warmup_ratio ${WARMUP_RATIO} --scheduler_type ${SCHEDULER_TYPE} ${NUM_EPOCHS_ARG}"

        if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
          FULL_CMD+=" --rosa_reset_optimizer_on_mask"
        fi
        if [[ "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" == "1" ]]; then
          FULL_CMD+=" --rosa_decay_sparse_lr_after_activation"
        fi

        FULL_CMD+=" > ${RUNNER_LOG} 2>&1"
        echo "${FULL_CMD}" >> "${CMD_LIST}"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
      done
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} LoRA-RoSA CoLA runs."
echo ">>> head_lrs=${HEAD_LRS[*]}"
echo ">>> sparse_budgets=${SPARSE_BUDGETS[*]} sparse_lrs=${SPARSE_LRS[*]}"
echo ">>> dropout=${LORA_DROPOUT} reset_on_mask=${ROSA_RESET_OPTIMIZER_ON_MASK}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
fi

SUMMARY_CSV="${OUT_ROOT}/lora_rosa_cola_favorable_sweep.csv"
SUMMARY_MD="${OUT_ROOT}/lora_rosa_cola_favorable_sweep.md"

python - "${OUT_ROOT}" "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
md_path = pathlib.Path(sys.argv[3])
rows = []

for path in root.glob("**/profile_*.json"):
    profile = json.loads(path.read_text())
    args = profile.get("args", {})
    rows.append(
        {
            "best_score": profile.get("best_score"),
            "head_lr": args.get("head_lr"),
            "sparse_budget": profile.get("sparse_budget", args.get("sparse_budget")),
            "sparse_lr": args.get("rosa_sparse_lr"),
            "dropout": args.get("unilora_dropout"),
            "seed": profile.get("seed"),
            "wall_clock_min": profile.get("wall_clock_total_s", 0.0) / 60.0,
            "peak_gpu_memory_mb": profile.get("peak_gpu_memory_mb"),
            "status": profile.get("status"),
            "profile_json": str(path),
        }
    )

rows.sort(
    key=lambda row: (
        row["best_score"] is not None,
        row["best_score"] if row["best_score"] is not None else float("-inf"),
    ),
    reverse=True,
)

fields = [
    "best_score",
    "head_lr",
    "sparse_budget",
    "sparse_lr",
    "dropout",
    "seed",
    "wall_clock_min",
    "peak_gpu_memory_mb",
    "status",
    "profile_json",
]
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with md_path.open("w") as handle:
    handle.write("# LoRA-RoSA CoLA favorable sweep\n\n")
    handle.write("| Rank | Best MCC | Head LR | Sparse budget | Sparse LR | Dropout | Seed | Time (min) | Peak MB | Status |\n")
    handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for index, row in enumerate(rows, start=1):
        score = "NA" if row["best_score"] is None else f"{row['best_score']:.6f}"
        handle.write(
            f"| {index} | {score} | {row['head_lr']} | {row['sparse_budget']} | "
            f"{row['sparse_lr']} | {row['dropout']} | {row['seed']} | "
            f"{row['wall_clock_min']:.2f} | {row['peak_gpu_memory_mb']} | {row['status']} |\n"
        )

print(f"Wrote {len(rows)} rows to {csv_path}")
print(f"Wrote ranked summary to {md_path}")
PY

echo "All LoRA-RoSA CoLA favorable-sweep jobs have been processed."
echo "Ranked CSV: ${SUMMARY_CSV}"
echo "Ranked Markdown: ${SUMMARY_MD}"
