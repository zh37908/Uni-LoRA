#!/bin/bash
# Final analysis pass after all theory SLURM jobs completed (2026-09-06).
#   1. Summarize E2 label-flip results  -> results_theory_e2_label_noise/summary.md
#   2. Report quality of the rank-64 lr1e-4 lora_ref reruns (E3 reference)
#   3. Re-run the E3 bias-variance decomposition with the new reference
#      -> results_theory_e3_bias_variance/e3_decomposition.md
# Run detached:  nohup bash run_final_theory_analyses.sh > logs/final_analyses.log 2>&1 &

cd /home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification || exit 1
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu || exit 1
set -u

log() { echo "[$(date '+%F %T')] $*"; }

log "=== 1/3: E2 label-flip summary ==="
python summarize_theory_results.py \
  --input_root results_theory_e2_label_noise \
  --output_md results_theory_e2_label_noise/summary.md

log "=== 2/3: lora_ref (rank 64, lr 1e-4) per-seed quality ==="
python - <<'PY'
import json
from pathlib import Path

root = Path("results_theory_e3_bias_variance")
for f in sorted(root.rglob("*.json")):
    if "lr1e-4" not in f.as_posix() or f.name.endswith("_theory.json"):
        continue
    d = json.loads(f.read_text())
    parts = f.as_posix().split("/")
    print(parts[2], parts[3], parts[4],
          f"score={d.get('best_score', float('nan')):.4f}",
          f"min_loss={d.get('min_val_loss', float('nan')):.3f}")
PY

log "=== 3/3: E3 decomposition with updated reference ==="
python analyze_theory_e3.py --output_md results_theory_e3_bias_variance/e3_decomposition.md

log "=== all done ==="
