#!/bin/bash
# 轮询超参扫描的训练输出, 训练完成(存在 ft/adapter_config.json)且尚无评测结果时
# 自动提交对应的评测作业。全部评完后自动退出。
# TARGETS 条目格式: <output_root>:<eval_task_id>:<seed>
# task_id 映射与训练/评测脚本一致: backbone(0=llama31,1=qwen3)*6 + config
#   config: 0=lora, 1=unilora, 2=prolosa2to1, 3=prolosa4to1, 4=prolosa8to1, 5=prolosa12to1
# 用法: nohup ./auto_eval_sweep.sh > auto_eval_sweep.log 2>&1 &

set -uo pipefail
cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning

TARGETS=()
# Llama-3.1 LoRA r=64 baseline (array id 0 = llama31 + lora)
TARGETS+=("output/p0_math_lora_r64:0:42")

MAX_QUEUED=9
POLL_SECS=600

run_tag_for() {
    local id=$1 seed=$2
    local tag method
    if (( id < 6 )); then tag="llama31"; else tag="qwen3"; fi
    case $((id % 6)) in
        0) method="lora" ;;
        1) method="unilora" ;;
        2) method="prolosa_r2to1" ;;
        3) method="prolosa_r4to1" ;;
        4) method="prolosa_r8to1" ;;
        5) method="prolosa_r12to1" ;;
    esac
    echo "${tag}_${method}_s${seed}"
}

while true; do
    remaining=0
    for entry in "${TARGETS[@]}"; do
        IFS=':' read -r root id seed <<< "${entry}"
        run_tag="$(run_tag_for "${id}" "${seed}")"
        result_root="results/${root#output/}"
        result_file="${result_root}/${run_tag}/gsm8k.log"
        marker="${result_root}/${run_tag}/.eval_submitted"

        # 已有结果 → 跳过
        if [[ -s "${result_file}" ]] && rg -q "acc====" "${result_file}"; then
            continue
        fi
        remaining=$((remaining + 1))
        # 已提交过评测 → 等结果
        [[ -f "${marker}" ]] && continue
        # 训练未完成 → 等
        adapter="$(find "${root}/${run_tag}" -name adapter_config.json -path "*/ft/*" 2>/dev/null | head -1)"
        [[ -z "${adapter}" ]] && continue
        # 队列满 → 下轮再试
        queued=$(squeue -u "$USER" -h -p gpu-rtx5880 2>/dev/null | wc -l)
        if (( queued >= MAX_QUEUED )); then
            echo "$(date '+%F %T') queue full (${queued}), wait: ${root} id=${id} seed=${seed}"
            continue
        fi
        echo "$(date '+%F %T') submitting eval: root=${root} id=${id} seed=${seed}"
        out=$(SEED="${seed}" \
            OUTPUT_ROOT="${root}" \
            MERGED_ROOT="output_merged/${root#output/}" \
            RESULT_ROOT="${result_root}" \
            sbatch --array="${id}" submit_eval_p0_math_llama31_qwen3_1gpu.sh 2>&1)
        echo "  ${out}"
        if [[ "${out}" == *"Submitted batch job"* ]]; then
            mkdir -p "${result_root}/${run_tag}"
            touch "${marker}"
        fi
    done
    if (( remaining == 0 )); then
        echo "$(date '+%F %T') all sweep evals finished, exit."
        break
    fi
    sleep "${POLL_SECS}"
done
