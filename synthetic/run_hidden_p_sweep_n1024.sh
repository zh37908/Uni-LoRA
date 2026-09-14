#!/usr/bin/env bash
# Incremental Hidden-P sweep for n=1024. Results are written to separate
# subdirectories so they can be merged with the existing 64/128/256 sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results_synthetic_theory_hidden_p_sweep}"
TRIALS="${TRIALS:-100}"
SPARSE_BUDGET="${SPARSE_BUDGET:-16}"

COMMON_ARGS=(
  --experiment hidden_p
  --D 512
  --dims 32
  --sample-sizes 1024
  --alphas 0 0.5 1.0 1.5 2.0
  --noise-stds 0.25 0.5 1.0
  --trials "${TRIALS}"
  --ridge 1e-6
  --methods lora unilora prolosa unilora_oracle
  --sparse-budget "${SPARSE_BUDGET}"
  --sparse-mismatch 1.0
  --support snip
  --seed 13
  --no-plot
)

mkdir -p "${OUT_ROOT}"

for angle in 0 5 15 30; do
  out_dir="${OUT_ROOT}/rotated_angle_${angle}_n1024"
  python "${SCRIPT_DIR}/synthetic_validate_theory.py" \
    "${COMMON_ARGS[@]}" \
    --pm-mode rotated \
    --pm-angle-deg "${angle}" \
    --output-dir "${out_dir}"
done

python "${SCRIPT_DIR}/synthetic_validate_theory.py" \
  "${COMMON_ARGS[@]}" \
  --pm-mode independent \
  --pm-angle-deg 0 \
  --output-dir "${OUT_ROOT}/independent_n1024"

python "${SCRIPT_DIR}/plot_hidden_p_sweep.py" \
  --results-root "${OUT_ROOT}" \
  --noise-std 0.5

echo "Done. Incremental n=1024 results in ${OUT_ROOT}"
