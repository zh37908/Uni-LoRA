#!/bin/bash
#SBATCH --job-name=est_peak_mem_rest4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/estimate_peak_memory_rest4_%j.out
#SBATCH --error=logs/estimate_peak_memory_rest4_%j.err

# Fast peak-memory estimate for the 4 GLUE tasks not covered by
# submit_profile_lora_cola_mrpc.sh (sst2 / qnli / rte / stsb).
# Each task runs only a few train steps; 4 GPUs in parallel → usually << 20 min.

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1
mkdir -p logs || exit 1

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate unilora_nlu || exit 1

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-roberta-large}"
TASKS=(${TASKS:-sst2 qnli rte stsb})
METHODS=(${METHODS:-lora})
SEEDS=(${SEEDS:-0})

BATCH_SIZE="${BATCH_SIZE:-32}"
RANK="${RANK:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-8}"
EVAL_STEPS="${EVAL_STEPS:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
GPU_SETUP="${GPU_SETUP:-1xNVIDIA_L20}"

ESTIMATOR="${ESTIMATOR:-estimate_peak_memory_glue.py}"
OUT_ROOT="${OUT_ROOT:-results_glue_efficiency_peak_estimate_rest4}"
mkdir -p "${OUT_ROOT}"

task_head_lr() {
  case "${1}" in
    sst2) echo "${LR_SST2:-1e-3}" ;;
    qnli) echo "${LR_QNLI:-5e-4}" ;;
    rte)  echo "${LR_RTE:-5e-3}" ;;
    stsb) echo "${LR_STSB:-2e-4}" ;;
    cola) echo "${LR_COLA:-5e-3}" ;;
    mrpc) echo "${LR_MRPC:-2e-4}" ;;
    *)
      echo "Unsupported task: ${1}" >&2
      return 1
      ;;
  esac
}

echo ">>> Pre-warming tokenizer / model / datasets (best-effort)..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
model_name = "${MODEL}"
tasks = "${TASKS[*]}".split()
AutoTokenizer.from_pretrained(model_name)
for task in tasks:
    num_labels = 1 if task == "stsb" else 2
    try:
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception:
        pass
    try:
        load_dataset("nyu-mll/glue", task, split="train")
    except Exception:
        pass
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CMD_LIST="$(mktemp)"
trap 'rm -f "${CMD_LIST}"' EXIT
TOTAL_RUNS=0

for METHOD in "${METHODS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    HEAD_LR="$(task_head_lr "${TASK}")"
    for SEED in "${SEEDS[@]}"; do
      SEED_DIR="${OUT_ROOT}/${MODEL}/${TASK}/${METHOD}_rank${RANK}/seed_${SEED}"
      mkdir -p "${SEED_DIR}"
      OUT_JSON="${SEED_DIR}/peak_estimate_${METHOD}_${TASK}_${MODEL}_seed${SEED}.json"
      RUN_LOG="${SEED_DIR}/peak_estimate_${METHOD}_${TASK}.log"

      if [[ -s "${OUT_JSON}" ]]; then
        echo "Skip existing: ${OUT_JSON}"
        continue
      fi

      FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --gpu-bind=single:1 python ${ESTIMATOR} --method ${METHOD} --model_name ${MODEL} --task ${TASK} --head_lr ${HEAD_LR} --seed ${SEED} --batch_size ${BATCH_SIZE} --rank ${RANK} --train_steps ${TRAIN_STEPS} --eval_steps ${EVAL_STEPS} --gpu_setup ${GPU_SETUP} --out_json ${OUT_JSON} > ${RUN_LOG} 2>&1"
      echo "${FULL_CMD}" >> "${CMD_LIST}"
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done

echo ">>> Generated ${TOTAL_RUNS} peak-memory estimate jobs."
echo ">>> tasks=${TASKS[*]} methods=${METHODS[*]} seeds=${SEEDS[*]} train_steps=${TRAIN_STEPS}"
echo ">>> model=${MODEL} batch_size=${BATCH_SIZE} rank=${RANK} out_root=${OUT_ROOT}"

if [[ "${TOTAL_RUNS}" -gt 0 ]]; then
  echo ">>> Starting parallel queue with ${PARALLEL_JOBS} slots..."
  xargs -I {} -P "${PARALLEL_JOBS}" bash -c "{}" < "${CMD_LIST}"
else
  echo "No new estimate jobs; summarizing existing files."
fi

SUMMARY_CSV="${OUT_ROOT}/peak_memory_estimate_summary.csv"
SUMMARY_MD="${OUT_ROOT}/peak_memory_estimate_summary.md"
python - <<PY
import csv, json, glob
from pathlib import Path

root = Path("${OUT_ROOT}")
rows = []
for path in sorted(root.glob("**/peak_estimate_*.json")):
    data = json.loads(path.read_text())
    rows.append({
        "method": data.get("method"),
        "model": data.get("model_name"),
        "task": data.get("task"),
        "trainable_params": data.get("trainable_params"),
        "train_steps": data.get("train_steps"),
        "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb"),
        "peak_gpu_memory_gb": data.get("peak_gpu_memory_gb"),
        "torch_max_memory_allocated_mb": data.get("torch_max_memory_allocated_mb"),
        "wall_clock_s": round(float(data.get("wall_clock_total_s", 0.0)), 1),
        "status": data.get("status"),
        "path": str(path),
    })

csv_path = Path("${SUMMARY_CSV}")
md_path = Path("${SUMMARY_MD}")
fields = [
    "method", "model", "task", "trainable_params", "train_steps",
    "peak_gpu_memory_mb", "peak_gpu_memory_gb",
    "torch_max_memory_allocated_mb", "wall_clock_s", "status", "path",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Fast Peak Memory Estimate (rest 4 GLUE tasks)",
    "",
    "| Method | Model | Task | Params | Steps | Peak Memory | Torch Alloc Peak | Wall |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['method']} | {r['model']} | {r['task']} | {r['trainable_params']} | "
        f"{r['train_steps']} | {r['peak_gpu_memory_gb']:.2f} GB | "
        f"{r['torch_max_memory_allocated_mb']} MB | {r['wall_clock_s']} s |"
    )
lines.extend([
    "",
    "Notes:",
    "",
    "- Estimate-only: tiny data slice + a few train/eval steps at full ``max_length`` padding.",
    "- Peak should be comparable to full training for the same batch size / sequence length.",
    "- Existing full profiles: CoLA/MRPC in ``results_glue_efficiency_profile_lora_cola_mrpc``.",
])
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
for r in rows:
    print(f"  {r['method']:8s} {r['task']:5s} peak={r['peak_gpu_memory_gb']:.2f} GB wall={r['wall_clock_s']}s")
PY

echo "All peak-memory estimate jobs finished."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary Markdown: ${SUMMARY_MD}"
