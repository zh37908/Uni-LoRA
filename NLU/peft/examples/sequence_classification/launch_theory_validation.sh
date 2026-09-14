#!/bin/bash
# Launch theory-validation SLURM jobs.
#
# l20_qos limits: MaxSubmit=4, MaxJobs=2, MaxGPU=8. Each script requests 4 GPUs,
# so at most 4 jobs can sit in queue and 2 can run. WAVE=1 is the default first
# batch (E1 SST-2, all 7 ratios packed into 4 jobs). After those finish:
#   WAVE=2 bash launch_theory_validation.sh   # E1 QNLI
#   WAVE=3 bash launch_theory_validation.sh   # E3 bootstrap + E4 projection sweep + E4 remaining + E1 cola/mrpc
#   WAVE=4 bash launch_theory_validation.sh   # E2: STS-B additive noise (primary) + flip robustness

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}"
cd "${WORK_DIR}"
mkdir -p logs

WAVE="${WAVE:-1}"
export PYTHONUNBUFFERED=1

submit() {
  local name="$1"
  shift
  echo "sbatch ${name}: $*"
  sbatch --export=ALL "$@"
}

case "${WAVE}" in
  1)
    echo ">>> WAVE 1: E1 SST-2, packed ratios (4 submit slots)"
    TASKS=sst2 TRAIN_SUBSET_RATIOS="0.01 0.02" submit "E1 sst2 p=0.01,0.02" submit_theory_e1_crossover.sh
    TASKS=sst2 TRAIN_SUBSET_RATIOS="0.05 0.1" submit "E1 sst2 p=0.05,0.1" submit_theory_e1_crossover.sh
    TASKS=sst2 TRAIN_SUBSET_RATIOS="0.25 0.5" submit "E1 sst2 p=0.25,0.5" submit_theory_e1_crossover.sh
    TASKS=sst2 TRAIN_SUBSET_RATIOS="1.0" submit "E1 sst2 p=1.0" submit_theory_e1_crossover.sh
    ;;
  2)
    echo ">>> WAVE 2: E1 QNLI, packed ratios"
    TASKS=qnli TRAIN_SUBSET_RATIOS="0.01 0.02" submit "E1 qnli p=0.01,0.02" submit_theory_e1_crossover.sh
    TASKS=qnli TRAIN_SUBSET_RATIOS="0.05 0.1" submit "E1 qnli p=0.05,0.1" submit_theory_e1_crossover.sh
    TASKS=qnli TRAIN_SUBSET_RATIOS="0.25 0.5" submit "E1 qnli p=0.25,0.5" submit_theory_e1_crossover.sh
    TASKS=qnli TRAIN_SUBSET_RATIOS="1.0" submit "E1 qnli p=1.0" submit_theory_e1_crossover.sh
    ;;
  3)
    echo ">>> WAVE 3: E3 bootstrap bias-variance, E4 projection sweep (primary), E4 remaining GLUE, E1 cola/mrpc"
    submit "E3 bootstrap" submit_theory_e3_bias_variance.sh
    submit "E4 proj sweep" submit_theory_e4_projection_sweep.sh
    submit "E4 remaining" submit_theory_e4_remaining_glue.sh
    TASKS="cola mrpc" TRAIN_SUBSET_RATIOS="0.25 0.5 1.0" submit "E1 cola/mrpc" submit_theory_e1_crossover.sh
    ;;
  4)
    echo ">>> WAVE 4: E2 noise scaling: STS-B additive Gaussian (primary) + label-flip robustness"
    submit "E2 stsb additive noise" submit_theory_e2_stsb_noise.sh
    TASKS=sst2 LABEL_NOISE_RATIOS="0.0 0.1" submit "E2 sst2 eta=0,0.1" submit_theory_e2_label_noise.sh
    TASKS=sst2 LABEL_NOISE_RATIOS="0.2 0.3" submit "E2 sst2 eta=0.2,0.3" submit_theory_e2_label_noise.sh
    TASKS=mrpc LABEL_NOISE_RATIOS="0.0 0.1 0.2 0.3" submit "E2 mrpc" submit_theory_e2_label_noise.sh
    ;;
  *)
    echo "Unknown WAVE=${WAVE}. Use 1, 2, 3, or 4." >&2
    exit 1
    ;;
esac

echo ">>> WAVE ${WAVE} submitted. Track with: squeue -u \$USER"
echo ">>> Next: WAVE=$((WAVE + 1)) bash launch_theory_validation.sh   (after current jobs finish)"
echo ">>> After artifacts exist, run:"
echo "    python summarize_theory_results.py --input_root results_theory_e1_crossover --output_md results_theory_e1_crossover/summary.md"
echo "    python estimate_fisher_diag_glue.py --artifact <lora *_theory.pt> --output fisher_diag/<task>_fisher.pt   # per task, on a GPU node"
echo "    python analyze_theory_e4_e5.py --roots results_theory_e1_crossover results_theory_e3_bias_variance results_theory_e4_remaining_glue results_theory_e4_projection_sweep --output_md results_theory_e4_e5.md --fisher mrpc=fisher_diag/mrpc_fisher.pt cola=fisher_diag/cola_fisher.pt"
