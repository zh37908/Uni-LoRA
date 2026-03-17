#!/bin/bash
#SBATCH --job-name=compare_hash_row
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=results_glue_variants/unilora_variants_%j.out
#SBATCH --error=results_glue_variants/unilora_variants_%j.err





set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results_compare_rows}"

SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-10}"
NH_LAYERS="${NH_LAYERS:-1}"
NHU="${NHU:-1000}"
COMPRESS="${COMPRESS:-0.03125}"
BATCH_SIZE="${BATCH_SIZE:-50}"
LR="${LR:-0.01}"
DROPOUT="${DROPOUT:-0.25}"
MOMENTUM="${MOMENTUM:-0.9}"
L2REG="${L2REG:-0.0}"
DECAY_FACTOR="${DECAY_FACTOR:-0.1}"
PATIENCE="${PATIENCE:-2}"
VALIDATION_PERCENT="${VALIDATION_PERCENT:-0.1}"
HASH_SEED="${HASH_SEED:-2}"
HASH_BIAS="${HASH_BIAS:-0}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  "$SCRIPT_DIR/multi_hash.py"
  --hashed
  --seed "$SEED"
  --epochs "$EPOCHS"
  --nhLayers "$NH_LAYERS"
  --nhu "$NHU"
  --compress "$COMPRESS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --dropout "$DROPOUT"
  --momentum "$MOMENTUM"
  --l2reg "$L2REG"
  --decay-factor "$DECAY_FACTOR"
  --patience "$PATIENCE"
  --validation-percent "$VALIDATION_PERCENT"
  --hash-seed "$HASH_SEED"
)

if [[ "$HASH_BIAS" == "1" ]]; then
  COMMON_ARGS+=(--hash-bias)
fi

ROWS1_JSON="$OUT_DIR/hash_rows1_seed${SEED}.json"
ROWS3_JSON="$OUT_DIR/hash_rows3_seed${SEED}.json"

echo "Running 1-row hash experiment..."
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --num-rows 1 --results-path "$ROWS1_JSON"

echo "Running 3-row hash experiment..."
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --num-rows 3 --results-path "$ROWS3_JSON"

echo
echo "Comparison summary:"
"$PYTHON_BIN" - "$ROWS1_JSON" "$ROWS3_JSON" <<'PY'
import json
import sys

rows1_path, rows3_path = sys.argv[1], sys.argv[2]

with open(rows1_path, "r", encoding="utf-8") as f:
    rows1 = json.load(f)

with open(rows3_path, "r", encoding="utf-8") as f:
    rows3 = json.load(f)

def final_val_acc(payload):
    history = payload.get("history", [])
    return history[-1]["val_accuracy"] if history else None

def format_metric(value):
    return "n/a" if value is None else f"{value:.4f}"

rows1_test = rows1["final_test"]["accuracy"]
rows3_test = rows3["final_test"]["accuracy"]
rows1_val = final_val_acc(rows1)
rows3_val = final_val_acc(rows3)

print(f"rows=1 result: {rows1_path}")
print(f"rows=3 result: {rows3_path}")
print(f"rows=1 final val acc: {format_metric(rows1_val)}")
print(f"rows=3 final val acc: {format_metric(rows3_val)}")
print(f"rows=1 test acc: {rows1_test:.4f}")
print(f"rows=3 test acc: {rows3_test:.4f}")
print(f"test acc delta (rows=3 - rows=1): {rows3_test - rows1_test:.4f}")
PY
