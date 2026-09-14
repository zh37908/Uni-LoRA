#!/bin/bash
#SBATCH --job-name=profile_rosa_mem_20m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/profile_rosa_mem_20m_%j.out
#SBATCH --error=logs/profile_rosa_mem_20m_%j.err

set -euo pipefail

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-math_instruction_tuning}"
conda activate "${CONDA_ENV}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export AWS_EC2_METADATA_DISABLED=true

RUN_SECONDS="${RUN_SECONDS:-1200}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
RESULT_ROOT="${RESULT_ROOT:-logs/profile_actual_memory_20min_${SLURM_JOB_ID}}"
mkdir -p "${RESULT_ROOT}"

MULTI_GPU_SCRIPT="submit_instruction_tuning_unilora_rosa_snip_multi_gpu_2to1_two_models_l20_2gpu.sh"
LEGACY_SCRIPT="submit_instruction_tuning_unilora_mistral_l20_2gpu.sh"

monitor_pid=""

stop_monitor() {
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
        monitor_pid=""
    fi
}
trap stop_monitor EXIT

start_monitor() {
    local output_csv="$1"
    echo "sample_epoch,gpu_index,memory_used_mib,memory_total_mib,utilization_pct" >"${output_csv}"
    (
        while true; do
            sample_epoch="$(date +%s.%N)"
            nvidia-smi \
                -i "${GPU_IDS}" \
                --query-gpu=index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits |
                while IFS= read -r row; do
                    echo "${sample_epoch},${row}" >>"${output_csv}"
                done
            sleep "${SAMPLE_INTERVAL}"
        done
    ) &
    monitor_pid=$!
}

summarize_memory() {
    local input_csv="$1"
    local output_txt="$2"
    local label="$3"
    python - "${input_csv}" "${output_txt}" "${label}" <<'PY'
import csv
import sys
from collections import defaultdict

input_csv, output_txt, label = sys.argv[1:]
per_gpu_peak = defaultdict(float)
per_gpu_total = {}
per_gpu_util_peak = defaultdict(float)
simultaneous = defaultdict(float)
sample_times = []

with open(input_csv, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        epoch = float(row["sample_epoch"])
        gpu = row["gpu_index"].strip()
        used = float(row["memory_used_mib"])
        total = float(row["memory_total_mib"])
        utilization = float(row["utilization_pct"])
        sample_times.append(epoch)
        per_gpu_peak[gpu] = max(per_gpu_peak[gpu], used)
        per_gpu_total[gpu] = total
        per_gpu_util_peak[gpu] = max(per_gpu_util_peak[gpu], utilization)
        simultaneous[epoch] += used

with open(output_txt, "w", encoding="utf-8") as handle:
    handle.write(f"experiment: {label}\n")
    handle.write(f"samples: {len(set(sample_times))}\n")
    if sample_times:
        handle.write(f"sampled_duration_seconds: {max(sample_times) - min(sample_times):.1f}\n")
    handle.write(
        "peak_simultaneous_total_memory_mib: "
        f"{max(simultaneous.values()) if simultaneous else 0.0:.0f}\n"
    )
    for gpu in sorted(per_gpu_peak, key=lambda value: int(value)):
        handle.write(
            f"gpu_{gpu}: peak_memory_mib={per_gpu_peak[gpu]:.0f}, "
            f"capacity_mib={per_gpu_total[gpu]:.0f}, "
            f"peak_utilization_pct={per_gpu_util_peak[gpu]:.0f}\n"
        )
PY
}

wait_for_gpu_release() {
    local attempts=0
    while (( attempts < 60 )); do
        active_pids="$(
            nvidia-smi -i "${GPU_IDS}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
                tr -d '[:space:]'
        )"
        if [[ -z "${active_pids}" ]]; then
            return 0
        fi
        sleep 2
        attempts=$((attempts + 1))
    done
    echo "Warning: GPU processes still exist after 120 seconds." >&2
}

run_experiment() {
    local label="$1"
    local script_path="$2"
    shift 2

    local experiment_dir="${RESULT_ROOT}/${label}"
    local memory_csv="${experiment_dir}/memory_samples.csv"
    local memory_summary="${experiment_dir}/memory_summary.txt"
    local run_log="${experiment_dir}/run.log"
    mkdir -p "${experiment_dir}"

    echo ">>> ${label}: sampling GPUs ${GPU_IDS} for at most ${RUN_SECONDS} seconds"
    start_monitor "${memory_csv}"

    set +e
    timeout --signal=TERM --kill-after=120s "${RUN_SECONDS}s" \
        env "$@" bash "${script_path}" >"${run_log}" 2>&1
    run_code=$?
    set -e

    stop_monitor
    summarize_memory "${memory_csv}" "${memory_summary}" "${label}"

    if [[ ${run_code} -eq 124 || ${run_code} -eq 137 ]]; then
        echo ">>> ${label}: stopped at the configured time limit (exit ${run_code})"
    elif [[ ${run_code} -ne 0 ]]; then
        echo ">>> ${label}: failed before the time limit (exit ${run_code})" >&2
    else
        echo ">>> ${label}: training command completed before the time limit"
    fi
    echo ">>> ${label}: memory summary at ${memory_summary}"

    wait_for_gpu_release
}

# Force the array-style multi-GPU script to its Mistral branch so both
# measurements use the same base model and are directly comparable.
run_experiment \
    "optimized_multi_gpu_mistral" \
    "${MULTI_GPU_SCRIPT}" \
    -u SLURM_ARRAY_TASK_ID \
    MODEL_INDEX=1 \
    OUTPUT="${RESULT_ROOT}/optimized_multi_gpu_mistral/train_output"

run_experiment \
    "legacy_mistral" \
    "${LEGACY_SCRIPT}" \
    OUTPUT="${RESULT_ROOT}/legacy_mistral/train_output"

echo "Sequential 20-minute memory profiling finished."
echo "Results: ${RESULT_ROOT}"
