#!/bin/bash

# =========================
# Config
# =========================

# 变体列表
VARIANTS=(unilora_fastfood unilora_nonorm)

# 学习率列表 (顺序执行)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)

# 指定使用的单个 GPU ID
GPU=0

# 随机种子
SEEDS=(0)

# GLUE 任务
TASKS=(mrpc cola)

# 模型
MODELS=(roberta-large)

# 使用支持变体的脚本
SCRIPT=run_unilora_variants_glue.py

# 根输出目录
OUT_ROOT=results_variants_sweep

mkdir -p ${OUT_ROOT}

# ==================================
# Loop: Variant × Model × Task × Seed
# ==================================
for VARIANT in "${VARIANTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for TASK in "${TASKS[@]}"; do

            echo "=================================================="
            echo "Variant: ${VARIANT} | Model: ${MODEL} | Task: ${TASK}"
            echo "=================================================="

            # 输出路径包含变体名称
            TASK_DIR=${OUT_ROOT}/${VARIANT}/${MODEL}/${TASK}
            mkdir -p ${TASK_DIR}

            for SEED in "${SEEDS[@]}"; do
                echo "----------------------------------------"
                echo "Running SEED=${SEED}"
                echo "----------------------------------------"

                SEED_DIR=${TASK_DIR}/seed_${SEED}
                mkdir -p ${SEED_DIR}

                # --------------------------------------------------
                # 顺序执行每个学习率，防止单显存溢出
                # --------------------------------------------------
                for LR in "${LRS[@]}"; do
                    LOG_FILE=${SEED_DIR}/log_lr_${LR}.txt

                    echo "Running ${VARIANT}: task=${TASK}, lr=${LR} on GPU ${GPU}..."

                    CUDA_VISIBLE_DEVICES=${GPU} \
                    python ${SCRIPT} \
                        --model_name ${MODEL} \
                        --task ${TASK} \
                        --variant ${VARIANT} \
                        --head_lr ${LR} \
                        --seed ${SEED} \
                        --out_dir ${SEED_DIR} \
                        > ${LOG_FILE} 2>&1
                    
                    echo "Finished ${VARIANT} lr=${LR}. Log: ${LOG_FILE}"
                done

                echo "Finished all LRs for ${VARIANT} on ${TASK} (Seed ${SEED})"
                echo
            done
        done
    done
done

echo "🎉 All variants, models, tasks, seeds & LRs finished."
