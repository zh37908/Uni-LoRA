#!/bin/bash
# Third auto-submit queue (follow-ups after the first E3/E4 pass):
#   1. diagonal-Fisher estimation job (curvature-weighted E4/E5; 1 GPU, short)
#   2. E3 reference re-run: rank-64 lora_ref at a lower adapter lr (1e-4). At 4e-4 the
#      rank-64 reference collapsed on CoLA (score 0.20-0.59 vs 0.68 for rank 4).
#   3. Fisher retry, to pick up SST-2 once its dedicated-lr LoRA artifact exists.
# Run detached:  nohup bash auto_submit_theory_followups.sh > logs/auto_submit_followups.log 2>&1 &

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

log "follow-up auto-submit started (pid $$)"

submit_retry "Fisher diagonal (cola mrpc rte stsb qnli sst2)" \
  sbatch --export=ALL submit_theory_fisher.sh

METHODS=lora_ref LORA_LR_COLA=1e-4 LORA_LR_MRPC=1e-4 \
  submit_retry "E3 lora_ref rank-64 re-run at lr 1e-4" \
  sbatch --export=ALL submit_theory_e3_bias_variance.sh

# Wait for the SST-2 dedicated-lr full-data LoRA artifact before retrying Fisher.
while ! ls results_theory_e1_crossover/roberta-large/sst2/p1/lora_r4_lr*/seed_0/*_theory.pt >/dev/null 2>&1; do
  log "waiting for SST-2 dedicated-lr LoRA artifact (p=1.0) ..."
  sleep 1800
done
submit_retry "Fisher retry (sst2)" \
  sbatch --export=ALL submit_theory_fisher.sh

log "follow-up queue done; exiting"
