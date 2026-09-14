#!/bin/bash
# P2 Alpaca 补训编排: 先提交一个 GPU 冒烟 (prolosa, 256 样本), 通过后自动
# 启动全量 wave (array 0-8)。挂后台运行:
#   nohup ./run_p2_alpaca_pipeline.sh > wave_p2_alpaca.log 2>&1 &

set -uo pipefail

cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
WAVES=/home/hzhaobi/Uni-LoRA/submit_in_waves.sh
SMOKE_LOG=$(mktemp /tmp/p2_alpaca_smoke_wave.XXXX.log)

echo ">>> [$(date '+%F %T')] submitting smoke (id=6 prolosa, 256 samples)"
OUTPUT_ROOT=output/p2_alpaca_smoke DATASET_SPLIT='train[:256]' \
    "${WAVES}" submit_train_p2_alpaca_llama2_2gpu.sh 6 --job-name=smoke_alpaca \
    2>&1 | tee "${SMOKE_LOG}"

JOB_ID=$(rg -o "Submitted batch job (\d+)" -r '$1' "${SMOKE_LOG}" | tail -1)
if [[ -z "${JOB_ID}" ]]; then
    echo "!!! smoke submission failed, aborting." >&2
    exit 1
fi
echo ">>> smoke job=${JOB_ID}, waiting for completion..."

while true; do
    STATE=$(sacct -j "${JOB_ID}" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "${STATE}" in
        COMPLETED) echo ">>> [$(date '+%F %T')] smoke COMPLETED"; break ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            echo "!!! smoke ended with state=${STATE}, aborting. Check logs/p2_alpaca_train_5880_${JOB_ID}_6.*" >&2
            exit 1 ;;
        *) sleep 120 ;;
    esac
done

if ! rg -q "P2 alpaca training finished" "logs/p2_alpaca_train_5880_${JOB_ID}_6.out" 2>/dev/null; then
    echo "!!! smoke log missing success marker, aborting." >&2
    exit 1
fi

echo ">>> [$(date '+%F %T')] smoke passed, launching full wave 0-8"
exec "${WAVES}" submit_train_p2_alpaca_llama2_2gpu.sh 0-8
