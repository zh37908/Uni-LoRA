#!/bin/bash

# =========================
# Config
# =========================

# Learning rates (8 GPUs → 8 LRs)
LRS=(5e-3)

# GPUs for parallel runs
GPUS=(0 )

# Random seeds
SEEDS=(0)

# GLUE tasks (6 tasks)
TASKS=(cola)

# Models
MODELS=(roberta-large)

# Script
SCRIPT=run_unilora_glue.py

# Root output directory
OUT_ROOT=results_glue_test

mkdir -p ${OUT_ROOT}

# =========================
# Loop: model × task × seed
# =========================
for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do

        echo "=================================================="
        echo "Model: ${MODEL} | Task: ${TASK}"
        echo "=================================================="

        TASK_DIR=${OUT_ROOT}/${MODEL}/${TASK}
        mkdir -p ${TASK_DIR}

        # -------------------------
        # Loop over seeds
        # -------------------------
        for SEED in "${SEEDS[@]}"; do
            echo "----------------------------------------"
            echo "Running MODEL=${MODEL}, TASK=${TASK}, SEED=${SEED}"
            echo "----------------------------------------"

            SEED_DIR=${TASK_DIR}/seed_${SEED}
            mkdir -p ${SEED_DIR}

            # -------------------------
            # Launch 8 LRs in parallel
            # -------------------------
            for i in "${!LRS[@]}"; do
                LR=${LRS[$i]}
                GPU=${GPUS[$i]}

                LOG_FILE=${SEED_DIR}/log_lr_${LR}.txt

                echo "Launching model=${MODEL}, task=${TASK}, seed=${SEED}, lr=${LR} on GPU ${GPU}"

                CUDA_VISIBLE_DEVICES=${GPU} \
                python ${SCRIPT} \
                    --model_name ${MODEL} \
                    --task ${TASK} \
                    --head_lr ${LR} \
                    --seed ${SEED} \
                    --output_dir ${SEED_DIR} \
                    > ${LOG_FILE} 2>&1 &

            done

            # -------------------------
            # Wait for this seed
            # -------------------------
            wait
            echo "Finished all LRs for MODEL=${MODEL}, TASK=${TASK}, SEED=${SEED}"
            echo
        done
    done
done

echo "🎉 All models, tasks, seeds & LRs finished."
