#!/bin/bash
# Auto-submit remaining theory-validation waves as QOS submit slots free up.
# l20_qos: MaxSubmit=4 counts pending+running jobs, so we submit one job whenever
# fewer than 4 theory_* jobs are in the queue. Order = wave 3 then wave 4.
# Run detached:  nohup bash auto_submit_theory_waves.sh > logs/auto_submit.log 2>&1 &

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

log "auto-submit started (pid $$)"

# ---- WAVE 3 ----
submit_retry "E3 bootstrap bias-variance" \
  sbatch --export=ALL submit_theory_e3_bias_variance.sh

submit_retry "E4 projection sweep (primary)" \
  sbatch --export=ALL submit_theory_e4_projection_sweep.sh

submit_retry "E4 remaining GLUE (rte/stsb artifacts)" \
  sbatch --export=ALL submit_theory_e4_remaining_glue.sh

TASKS="cola mrpc" TRAIN_SUBSET_RATIOS="0.25 0.5 1.0" \
  submit_retry "E1 cola/mrpc supplements" \
  sbatch --export=ALL submit_theory_e1_crossover.sh

# ---- WAVE 4 ----
submit_retry "E2 STS-B additive noise (primary)" \
  sbatch --export=ALL submit_theory_e2_stsb_noise.sh

TASKS=sst2 LABEL_NOISE_RATIOS="0.0 0.1" \
  submit_retry "E2 sst2 flip eta=0,0.1 (robustness)" \
  sbatch --export=ALL submit_theory_e2_label_noise.sh

TASKS=sst2 LABEL_NOISE_RATIOS="0.2 0.3" \
  submit_retry "E2 sst2 flip eta=0.2,0.3 (robustness)" \
  sbatch --export=ALL submit_theory_e2_label_noise.sh

TASKS=mrpc LABEL_NOISE_RATIOS="0.0 0.1 0.2 0.3" \
  submit_retry "E2 mrpc flip (robustness)" \
  sbatch --export=ALL submit_theory_e2_label_noise.sh

log "all theory waves submitted; auto-submit exiting"
