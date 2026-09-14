#!/bin/bash
#SBATCH --job-name=lora_rosa_fixed_r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/lora_rosa_fixed_params_rank_sweep_%j.out
#SBATCH --error=logs/lora_rosa_fixed_params_rank_sweep_%j.err

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

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-cola mrpc})
# Compare only lower-rank LoRA-RoSA configurations. Their sparse branches
# receive the parameters freed relative to a pure rank-4 LoRA adapter.
RANKS=(${RANKS:-3 2 1})
SEEDS=(${SEEDS:-0 1 2})

BATCH_SIZE="${BATCH_SIZE:-32}"
REFERENCE_RANK="${REFERENCE_RANK:-4}"

# Stable defaults for the official two-process RoSA lifecycle.
ROSA_SPARSE_LR="${ROSA_SPARSE_LR:-2e-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_ALPHA="${LORA_ALPHA:-16}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-20}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"
ROSA_WARMUP_STEPS="${ROSA_WARMUP_STEPS:-64}"
ROSA_MASK_STEPS="${ROSA_MASK_STEPS:-1}"
ROSA_GRAD_ACC_MODE="${ROSA_GRAD_ACC_MODE:-mean_squared}"
OFFICIAL_TWO_STAGE="${OFFICIAL_TWO_STAGE:-1}"
INCLUDE_LORA_BASELINE="${INCLUDE_LORA_BASELINE:-1}"
ROSA_RESET_OPTIMIZER_ON_MASK="${ROSA_RESET_OPTIMIZER_ON_MASK:-0}"
ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION="${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION:-1}"

GPU_SETUP="${GPU_SETUP:-1xNVIDIA_L20}"
GPU_COUNT="${GPU_COUNT:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_unilora_variants_glue.py}"
PROFILER="${PROFILER:-profile_unilora_prolosa_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_lora_rosa_official_two_stage_fixed_params_cola_mrpc}"
mkdir -p "${OUT_ROOT}"

if [[ -n "${NUM_EPOCHS:-}" ]]; then
  NUM_EPOCHS_ARG="--num_epochs ${NUM_EPOCHS}"
else
  NUM_EPOCHS_ARG=""
fi

task_head_lr() {
  case "${1}" in
    cola) echo "${LR_COLA:-5e-4}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    *)
      echo "Unsupported task: ${1}" >&2
      return 1
      ;;
  esac
}

echo ">>> Pre-warming model and dataset caches..."
python - <<PY
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = "${MODEL}"
tasks = "${TASKS[*]}".split()
AutoTokenizer.from_pretrained(model_name)
for task in tasks:
    AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    load_dataset("nyu-mll/glue", task)
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# LoRA A/B parameter count is linear in rank for this fixed target-module set.
read -r LORA_PARAMS_PER_RANK BASE_SPARSE_POSITIONS <<< "$(python - <<PY
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("${MODEL}", num_labels=2)
target_suffixes = ["query", "key", "value", "output.dense", "intermediate.dense"]
per_rank = 0
base_total = 0
for name, module in model.named_modules():
    if not hasattr(module, "weight") or not any(name.endswith(s) for s in target_suffixes):
        continue
    out_features, in_features = module.weight.shape[:2]
    per_rank += in_features + out_features
    base_total += out_features * in_features
print(per_rank, base_total)
PY
)"

# Parameter budget of pure rank-4 LoRA: no sparse parameters in the reference.
TOTAL_ADAPTER_BUDGET=$((REFERENCE_RANK * LORA_PARAMS_PER_RANK))

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

profile_complete() {
  python - "${1}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    profile = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if profile.get("status") == "ok" and profile.get("best_score") is not None else 1)
PY
}

echo ">>> Fixed adapter trainable budget=${TOTAL_ADAPTER_BUDGET}"
echo ">>> LoRA parameters per rank=${LORA_PARAMS_PER_RANK}"
echo ">>> Sparse candidate positions=${BASE_SPARSE_POSITIONS}"
for RANK in "${RANKS[@]}"; do
  LORA_PARAM_COUNT=$((RANK * LORA_PARAMS_PER_RANK))
  SPARSE_BUDGET=$((TOTAL_ADAPTER_BUDGET - LORA_PARAM_COUNT))
  if (( SPARSE_BUDGET <= 0 || SPARSE_BUDGET > BASE_SPARSE_POSITIONS )); then
    echo "Invalid sparse budget for rank ${RANK}: ${SPARSE_BUDGET}" >&2
    exit 1
  fi
  echo ">>> rank=${RANK}: lora=${LORA_PARAM_COUNT}, sparse=${SPARSE_BUDGET}, total=$((LORA_PARAM_COUNT + SPARSE_BUDGET))"
done

CMD_LIST="$(mktemp)"
trap 'rm -f "${CMD_LIST}"' EXIT
TOTAL_RUNS=0

for TASK in "${TASKS[@]}"; do
  HEAD_LR="$(task_head_lr "${TASK}")"

  if [[ "${INCLUDE_LORA_BASELINE}" == "1" ]]; then
    for SEED in "${SEEDS[@]}"; do
      BASELINE_NAME="lora_r${REFERENCE_RANK}_a${LORA_ALPHA}_tp${TOTAL_ADAPTER_BUDGET}_hlr$(safe_token "${HEAD_LR}")_ws${SCHEDULER_WARMUP_STEPS}"
      SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${BASELINE_NAME}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"
      PROFILE_JSON="${SEED_DIR}/profile_lora_${TASK}_${MODEL}_${BASELINE_NAME}_seed${SEED}.json"
      TRAIN_LOG="${SEED_DIR}/train_lora_${TASK}_${BASELINE_NAME}_seed${SEED}.log"
      RUNNER_LOG="${SEED_DIR}/runner_lora_${TASK}_${BASELINE_NAME}_seed${SEED}.log"
      if profile_complete "${PROFILE_JSON}"; then
        echo "Skip existing profile: ${PROFILE_JSON}"
        continue
      fi
      FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 python -u ${PROFILER} run --train_script ${TRAIN_SCRIPT} --method lora --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${REFERENCE_RANK} --theta_d_length ${TOTAL_ADAPTER_BUDGET} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --profile_json ${PROFILE_JSON} --train_log ${TRAIN_LOG} --gpu_setup ${GPU_SETUP} --gpu_count ${GPU_COUNT} --total_sparse_positions ${TOTAL_ADAPTER_BUDGET} --unilora_dropout ${LORA_DROPOUT} --weight_decay ${WEIGHT_DECAY} --warmup_ratio ${WARMUP_RATIO} --scheduler_type ${SCHEDULER_TYPE} ${NUM_EPOCHS_ARG} --extra_train_args --lora_lr ${HEAD_LR} --lora_alpha ${LORA_ALPHA} --scheduler_warmup_steps ${SCHEDULER_WARMUP_STEPS} > ${RUNNER_LOG} 2>&1"
      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  fi

  for RANK in "${RANKS[@]}"; do
    LORA_PARAM_COUNT=$((RANK * LORA_PARAMS_PER_RANK))
    SPARSE_BUDGET=$((TOTAL_ADAPTER_BUDGET - LORA_PARAM_COUNT))
    ROSA_DENSITY="$(density_from_sparse_budget "${SPARSE_BUDGET}")"
    CONFIG_NAME="official${OFFICIAL_TWO_STAGE}_r${RANK}_a${LORA_ALPHA}_lp${LORA_PARAM_COUNT}_sb${SPARSE_BUDGET}_tp${TOTAL_ADAPTER_BUDGET}_hlr$(safe_token "${HEAD_LR}")_slr$(safe_token "${ROSA_SPARSE_LR}")_ga${ROSA_GRAD_ACC_MODE}_ws${SCHEDULER_WARMUP_STEPS}"

    for SEED in "${SEEDS[@]}"; do
      SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${CONFIG_NAME}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"

      PROFILE_JSON="${SEED_DIR}/profile_lora_rosa_${TASK}_${MODEL}_${CONFIG_NAME}_seed${SEED}.json"
      TRAIN_LOG="${SEED_DIR}/train_lora_rosa_${TASK}_${CONFIG_NAME}_seed${SEED}.log"
      RUNNER_LOG="${SEED_DIR}/runner_lora_rosa_${TASK}_${CONFIG_NAME}_seed${SEED}.log"
      MASK_PATH="${SEED_DIR}/rosa_mask.pt"
      MASK_LOG="${SEED_DIR}/mask_generation.log"
      MASK_OUT_DIR="${SEED_DIR}/mask_generation"

      if profile_complete "${PROFILE_JSON}"; then
        echo "Skip existing profile: ${PROFILE_JSON}"
        continue
      fi

      COMMON_TRAIN_ARGS="--model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${LORA_PARAM_COUNT} --head_lr ${HEAD_LR} --lora_lr ${HEAD_LR} --lora_alpha ${LORA_ALPHA} --seed ${SEED} --unilora_dropout ${LORA_DROPOUT} --weight_decay ${WEIGHT_DECAY} --warmup_ratio ${WARMUP_RATIO} --scheduler_warmup_steps ${SCHEDULER_WARMUP_STEPS} --scheduler_type ${SCHEDULER_TYPE} --rosa_sparse_budget ${SPARSE_BUDGET} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_grad_acc_mode ${ROSA_GRAD_ACC_MODE} ${NUM_EPOCHS_ARG}"
      PHASE1_CMD="python -u ${TRAIN_SCRIPT} --variant lora_rosa ${COMMON_TRAIN_ARGS} --out_dir ${MASK_OUT_DIR} --rosa_mask_save_path ${MASK_PATH} --rosa_terminate_after_mask_generation"
      PHASE2_CMD="python -u ${PROFILER} run --train_script ${TRAIN_SCRIPT} --method lora_rosa --model_name ${MODEL} --task ${TASK} --batch_size ${BATCH_SIZE} --rank ${RANK} --theta_d_length ${LORA_PARAM_COUNT} --head_lr ${HEAD_LR} --seed ${SEED} --out_dir ${SEED_DIR} --profile_json ${PROFILE_JSON} --train_log ${TRAIN_LOG} --gpu_setup ${GPU_SETUP} --gpu_count ${GPU_COUNT} --sparse_budget ${SPARSE_BUDGET} --total_sparse_positions ${BASE_SPARSE_POSITIONS} --rosa_density ${ROSA_DENSITY} --rosa_warmup_steps ${ROSA_WARMUP_STEPS} --rosa_mask_steps ${ROSA_MASK_STEPS} --rosa_sparse_lr ${ROSA_SPARSE_LR} --unilora_dropout ${LORA_DROPOUT} --weight_decay ${WEIGHT_DECAY} --warmup_ratio ${WARMUP_RATIO} --scheduler_type ${SCHEDULER_TYPE} ${NUM_EPOCHS_ARG} --extra_train_args --lora_lr ${HEAD_LR} --lora_alpha ${LORA_ALPHA} --scheduler_warmup_steps ${SCHEDULER_WARMUP_STEPS} --rosa_grad_acc_mode ${ROSA_GRAD_ACC_MODE}"
      if [[ "${OFFICIAL_TWO_STAGE}" == "1" ]]; then
        PHASE2_CMD+=" --rosa_mask_load_path ${MASK_PATH}"
      fi

      if [[ "${ROSA_RESET_OPTIMIZER_ON_MASK}" == "1" ]]; then
        PHASE2_CMD+=" --rosa_reset_optimizer_on_mask"
      fi
      if [[ "${ROSA_DECAY_SPARSE_LR_AFTER_ACTIVATION}" == "1" ]]; then
        PHASE2_CMD="${PHASE2_CMD/ --extra_train_args/ --rosa_decay_sparse_lr_after_activation --extra_train_args}"
      fi

      if [[ "${OFFICIAL_TWO_STAGE}" == "1" ]]; then
        FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 bash -c '${PHASE1_CMD} > ${MASK_LOG} 2>&1 && ${PHASE2_CMD} > ${RUNNER_LOG} 2>&1'"
      else
        FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 ${PHASE2_CMD} > ${RUNNER_LOG} 2>&1"
      fi
      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} runs."
echo ">>> tasks=${TASKS[*]} ranks=${RANKS[*]} seeds=${SEEDS[*]}"
echo ">>> task head LRs: cola=$(task_head_lr cola), mrpc=$(task_head_lr mrpc)"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}" || {
    echo "Some runs failed; continuing to summarize completed profiles." >&2
  }
else
  echo "No new runs; summarizing existing profiles."
fi

SUMMARY_CSV="${OUT_ROOT}/lora_rosa_fixed_params_rank_summary.csv"
SUMMARY_MD="${OUT_ROOT}/lora_rosa_fixed_params_rank_summary.md"

python - "${OUT_ROOT}" "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
md_path = pathlib.Path(sys.argv[3])
groups = defaultdict(list)

for path in root.glob("**/profile_*.json"):
    try:
        profile = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"Skipping invalid profile: {path}", file=sys.stderr)
        continue
    if profile.get("status") != "ok" or profile.get("best_score") is None:
        continue
    key = (profile["method"], profile["task"], int(profile["rank"]))
    groups[key].append(profile)

rows = []
for (method, task, rank), profiles in sorted(groups.items()):
    scores = [float(item["best_score"]) for item in profiles]
    first = profiles[0]
    rows.append(
        {
            "method": method,
            "task": task,
            "rank": rank,
            "num_runs": len(scores),
            "seeds": " ".join(str(item["seed"]) for item in sorted(profiles, key=lambda item: item["seed"])),
            "lora_params": (
                int(first["adapter_reported_params"])
                if method == "lora"
                else int(first["theta_d_length"])
            ),
            "sparse_budget": int(first["sparse_budget"]),
            "adapter_trainable_params": int(first["adapter_reported_params"]),
            "head_lr": first["head_lr"],
            "mean_best_score": statistics.mean(scores),
            "std_best_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "max_best_score": max(scores),
            "mean_wall_clock_min": statistics.mean(float(item["wall_clock_total_s"]) for item in profiles) / 60.0,
            "mean_peak_gpu_memory_mb": statistics.mean(float(item["peak_gpu_memory_mb"]) for item in profiles),
        }
    )

fields = [
    "method",
    "task",
    "rank",
    "num_runs",
    "seeds",
    "lora_params",
    "sparse_budget",
    "adapter_trainable_params",
    "head_lr",
    "mean_best_score",
    "std_best_score",
    "max_best_score",
    "mean_wall_clock_min",
    "mean_peak_gpu_memory_mb",
]
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with md_path.open("w") as handle:
    handle.write("# LoRA-RoSA fixed-parameter rank sweep\n\n")
    handle.write("| Method | Task | Rank | LoRA params | Sparse mask | Total adapter params | Head LR | Seeds | Mean score | Std | Max | Time (min) | Peak MB |\n")
    handle.write("|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
    for row in rows:
        handle.write(
            f"| {row['method']} | {row['task']} | {row['rank']} | {row['lora_params']} | "
            f"{row['sparse_budget']} | {row['adapter_trainable_params']} | "
            f"{row['head_lr']} | {row['seeds']} | {row['mean_best_score']:.6f} | "
            f"{row['std_best_score']:.6f} | {row['max_best_score']:.6f} | "
            f"{row['mean_wall_clock_min']:.2f} | {row['mean_peak_gpu_memory_mb']:.0f} |\n"
        )

print(f"Wrote {len(rows)} grouped rows to {csv_path}")
print(f"Wrote summary to {md_path}")
PY

echo "All fixed-parameter LoRA-RoSA rank-sweep runs have been processed."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary Markdown: ${SUMMARY_MD}"
