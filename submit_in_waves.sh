#!/bin/bash
# 受 QOS 限额约束的分波提交器。
# 集群限制 (sacctmgr show qos): 5880_qos 每用户最多排 10 个作业 / 跑 8 个 / 12 卡;
# l20_qos 4 排 / 2 跑。大 array (18~63 个任务) 无法一次提交, 本脚本轮询队列,
# 有空位就提交下一个 array 元素, 直到全部交完。
#
# 用法 (在登录节点, 用 nohup 挂后台):
#   nohup ./submit_in_waves.sh <submit_script> <array_spec> [sbatch 额外参数...] \
#       > wave_<name>.log 2>&1 &
# 例:
#   cd math_instruction_tuning
#   nohup ../submit_in_waves.sh submit_train_p0_math_llama31_qwen3_2gpu.sh 0-17 > wave_p0_math.log 2>&1 &
#   ROSA_WARMUP_STEPS=128 nohup ../submit_in_waves.sh submit_e1_sample_sweep_qwen25_rtx5880.sh 0-62 &
# array_spec 支持 "0-17" 或 "0,3,6" 或混合 "0-8,12"。
# 环境变量:
#   PARTITION   队列分区 (默认 gpu-rtx5880, 用于统计队列占用)
#   MAX_QUEUED  本用户在该分区的最大排队+运行作业数 (默认 9, 留 1 个余量)
#   POLL_SECS   轮询间隔秒数 (默认 180)
# 其余环境变量原样传给 sbatch 的作业脚本。

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <submit_script> <array_spec> [extra sbatch args...]" >&2
    exit 1
fi

SUBMIT_SCRIPT="$1"
ARRAY_SPEC="$2"
shift 2
EXTRA_ARGS=("$@")

PARTITION="${PARTITION:-gpu-rtx5880}"
MAX_QUEUED="${MAX_QUEUED:-9}"
POLL_SECS="${POLL_SECS:-180}"

if [[ ! -f "${SUBMIT_SCRIPT}" ]]; then
    echo "Submit script not found: ${SUBMIT_SCRIPT}" >&2
    exit 1
fi

# 展开 array spec 成 id 列表
IDS=()
IFS=',' read -ra PARTS <<< "${ARRAY_SPEC}"
for part in "${PARTS[@]}"; do
    if [[ "${part}" == *-* ]]; then
        IDS+=($(seq "${part%%-*}" "${part##*-}"))
    else
        IDS+=("${part}")
    fi
done

echo ">>> wave submitter: script=${SUBMIT_SCRIPT} ids=(${IDS[*]}) partition=${PARTITION} max_queued=${MAX_QUEUED}"

queued_count() {
    squeue -u "${USER}" -p "${PARTITION}" -h 2>/dev/null | wc -l
}

for id in "${IDS[@]}"; do
    while [[ "$(queued_count)" -ge "${MAX_QUEUED}" ]]; do
        sleep "${POLL_SECS}"
    done
    while true; do
        OUT=$(sbatch --array="${id}" "${EXTRA_ARGS[@]}" "${SUBMIT_SCRIPT}" 2>&1)
        if [[ "${OUT}" == *"Submitted batch job"* ]]; then
            echo "$(date '+%F %T') submitted id=${id}: ${OUT}"
            break
        elif [[ "${OUT}" == *"QOSMaxSubmitJobPerUserLimit"* || "${OUT}" == *"job submit limit"* ]]; then
            echo "$(date '+%F %T') queue full at id=${id}, waiting..."
            sleep "${POLL_SECS}"
        else
            echo "$(date '+%F %T') FAILED id=${id}: ${OUT}" >&2
            exit 1
        fi
    done
    sleep 5
done

echo ">>> all ids submitted."
