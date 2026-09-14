#!/bin/bash
# Second auto-submit queue: re-run the E1 SST-2 LoRA baseline with the dedicated
# adapter lr (theory_common_lora_lr.sh: sst2 -> 4e-4). The original SST-2 LoRA runs
# used the shared head lr 1e-3 and collapsed on some seeds (score 0.51, chance-level
# train loss at the end of training). Uni-LoRA/ProLoSA SST-2 results are reused as-is.
# Shares the QOS slot-check logic with auto_submit_theory_waves.sh; sbatch failures
# from a race are retried.
# Run detached:  nohup bash auto_submit_theory_lora_rerun.sh > logs/auto_submit_lora_rerun.log 2>&1 &

set -u

WORK_DIR="${WORK_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}"
cd "${WORK_DIR}" || exit 1

MAX_SUBMIT=4
POLL_SECS=600

log() { echo "[$(date '+%F %T')] $*"; }

theory_jobs_in_queue() {
  squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -c '^theory_' || true
}

wait_for_slot() {
  while true; do
    local count
    count="$(theory_jobs_in_queue)"
    if [[ "${count}" -lt "${MAX_SUBMIT}" ]]; then
      return 0
    fi
    log "queue full (${count}/${MAX_SUBMIT} theory jobs); sleeping ${POLL_SECS}s"
    sleep "${POLL_SECS}"
  done
}

submit_retry() {
  local label="$1"
  shift
  while true; do
    wait_for_slot
    log "submitting: ${label}"
    if out="$("$@" 2>&1)"; then
      log "OK: ${out}"
      return 0
    fi
    log "sbatch failed (${out}); retrying in ${POLL_SECS}s"
    sleep "${POLL_SECS}"
  done
}

log "lora-rerun auto-submit started (pid $$)"

TASKS=sst2 METHODS=lora TRAIN_SUBSET_RATIOS="0.01 0.02 0.05 0.1" \
  submit_retry "E1 sst2 LoRA rerun (lr 4e-4) p=0.01..0.1" \
  sbatch --export=ALL submit_theory_e1_crossover.sh

TASKS=sst2 METHODS=lora TRAIN_SUBSET_RATIOS="0.25 0.5 1.0" \
  submit_retry "E1 sst2 LoRA rerun (lr 4e-4) p=0.25..1.0" \
  sbatch --export=ALL submit_theory_e1_crossover.sh

log "lora-rerun queue done; exiting"
