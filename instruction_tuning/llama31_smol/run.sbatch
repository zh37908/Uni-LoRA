#!/usr/bin/env bash
#SBATCH --job-name=instruction_r4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${UNILORA_PACKAGE:?Set UNILORA_PACKAGE to this package directory}"
source "$UNILORA_PACKAGE/env.sh"
config=$1
python_bin=${UNILORA_PYTHON:-python}
method=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["method"])' "$config")
name=$(basename "$config" .json)
out="$UNILORA_STORAGE/runs/llama31_smol/$name"
child_pid=''
trap 'if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi' USR1 TERM
if [[ ! -f "$out/metrics.json" ]]; then
    resume=()
    if [[ -f "$out/checkpoint.pt" ]]; then resume=(--resume); fi
    "$python_bin" -u "$UNILORA_PACKAGE/train.py" --method "$method" --config "$config" --run-dir "$out" "${resume[@]}" &
    child_pid=$!
    status=0
    while true; do
        wait "$child_pid" && status=0 || status=$?
        if ! kill -0 "$child_pid" 2>/dev/null; then break; fi
    done
    [[ $status == 0 ]] || exit "$status"
    [[ -f "$out/metrics.json" ]] || exit 75  # Checkpoint saved; resubmit this configuration.
fi
if [[ ! -f "$out/evaluation/metrics.json" ]]; then
    "$python_bin" -u "$UNILORA_PACKAGE/evaluate.py" --method "$method" --config "$config" --run-dir "$out" &
    child_pid=$!
    wait "$child_pid"
fi
