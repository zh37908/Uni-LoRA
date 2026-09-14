#!/usr/bin/env bash
# Hidden-P / non-oracle projection experiment.
# Teacher generates theta_star with P_T; Uni-LoRA/ProLoSA only see P_M.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results_synthetic_theory_hidden_p}"
PM_ANGLE_DEG="${PM_ANGLE_DEG:-15}"
PM_MODE="${PM_MODE:-rotated}"
SPARSE_BUDGET="${SPARSE_BUDGET:-16}"

python "${SCRIPT_DIR}/synthetic_validate_theory.py" \
  --experiment hidden_p \
  --D 512 \
  --dims 32 \
  --sample-sizes 64 128 256 \
  --alphas 0 0.5 1.0 1.5 2.0 \
  --noise-stds 0.25 0.5 1.0 \
  --trials 100 \
  --ridge 1e-6 \
  --methods lora unilora prolosa unilora_oracle \
  --sparse-budget "${SPARSE_BUDGET}" \
  --sparse-mismatch 1.0 \
  --pm-mode "${PM_MODE}" \
  --pm-angle-deg "${PM_ANGLE_DEG}" \
  --support snip \
  --seed 13 \
  --output-dir "${OUT_DIR}"

python "${SCRIPT_DIR}/synthetic_plot_fig2_style.py" \
  --results-dir "${OUT_DIR}" \
  --noise-std 0.5 \
  --small-n 64 \
  --large-n 256

echo "Done. Results in ${OUT_DIR}"
