#!/bin/bash
#SBATCH --job-name=unilora_hessian_aware_glue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# 每 GPU 绑定的 CPU 数（4 卡 × 本值 = 节点 CPU 总量的一部分）；需与下面 srun --cpus-per-task 一致。
# 若仍 OOM 或 DataLoader 吃内存，可改为 16（视分区单卡 CPU 上限而定）。
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/logs/unilora_hessian_aware_%j.out
#SBATCH --error=/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/logs/unilora_hessian_aware_%j.err

# Under sbatch, BASH_SOURCE points at a copy under /var/spool/slurm/... — do NOT use it for SCRIPT_DIR.
SCRIPT_DIR="/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification"
cd "${SCRIPT_DIR}" || { echo "cd SCRIPT_DIR failed: ${SCRIPT_DIR}"; exit 1; }

mkdir -p "${SCRIPT_DIR}/logs"

# Activate NLU conda env
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_nlu

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

RANK=4
THETA_D_LENGTH=23040
THETA_D_LR=5e-3
INIT_THETA_D_BOUND=0.02
HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL=5
HESSIAN_AWARE_WARMUP_EPOCHS=1
HESSIAN_AWARE_REASSIGN_RATIO=0.01
HESSIAN_AWARE_CANDIDATE_POOL_SIZE=8
HESSIAN_AWARE_CAPACITY_PENALTY=0.1
HESSIAN_AWARE_CAPACITY_SLACK=2.0
HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM=0.9
HESSIAN_AWARE_ACCEPT_TOLERANCE=1e-6

echo ">>> Pre-warming cache (Downloading models and datasets)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
for m in ['roberta-large']:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2)
for t in ['mrpc', 'cola', 'sst2', 'qnli']:
    try:
        load_dataset('nyu-mll/glue', t)
    except Exception:
        pass
"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHODS=(unilora_hessian_aware)
MODELS=(roberta-large)
TASKS=(mrpc cola sst2 qnli)
SEEDS=(0 1 2)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)

SCRIPT="${SCRIPT_DIR}/run_unilora_variants_glue.py"
OUT_ROOT="${SCRIPT_DIR}/results_glue_variants_hessian_aware"
mkdir -p "${OUT_ROOT}"

CMD_LIST=$(mktemp)

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    TASK_DIR=${OUT_ROOT}/${MODEL}/${TASK}
    mkdir -p "${TASK_DIR}"
    for METHOD in "${METHODS[@]}"; do
      M_NAME="${METHOD}_int${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL}_warm${HESSIAN_AWARE_WARMUP_EPOCHS}_rr${HESSIAN_AWARE_REASSIGN_RATIO}_pool${HESSIAN_AWARE_CANDIDATE_POOL_SIZE}_cap${HESSIAN_AWARE_CAPACITY_PENALTY}_slack${HESSIAN_AWARE_CAPACITY_SLACK}_ema${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM}"
      EXTRA_FLAG="--rank ${RANK} --theta_d_length ${THETA_D_LENGTH} --theta_d_lr ${THETA_D_LR} --init_theta_d_bound ${INIT_THETA_D_BOUND} --hessian_aware_structure_update_interval ${HESSIAN_AWARE_STRUCTURE_UPDATE_INTERVAL} --hessian_aware_warmup_epochs ${HESSIAN_AWARE_WARMUP_EPOCHS} --hessian_aware_reassign_ratio ${HESSIAN_AWARE_REASSIGN_RATIO} --hessian_aware_candidate_pool_size ${HESSIAN_AWARE_CANDIDATE_POOL_SIZE} --hessian_aware_capacity_penalty ${HESSIAN_AWARE_CAPACITY_PENALTY} --hessian_aware_capacity_slack ${HESSIAN_AWARE_CAPACITY_SLACK} --hessian_aware_curvature_ema_momentum ${HESSIAN_AWARE_CURVATURE_EMA_MOMENTUM} --hessian_aware_accept_tolerance ${HESSIAN_AWARE_ACCEPT_TOLERANCE}"

      for SEED in "${SEEDS[@]}"; do
        SEED_DIR=${TASK_DIR}/${M_NAME}/seed_${SEED}
        mkdir -p "${SEED_DIR}"
        for LR in "${LRS[@]}"; do
          LOG_FILE=${SEED_DIR}/log_lr_${LR}.txt
          FULL_CMD="cd \"${SCRIPT_DIR}\" && srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 --cpus-per-task=12 --cpu-bind=none --gpu-bind=single:1 python \"${SCRIPT}\" --variant ${METHOD} ${EXTRA_FLAG} --model_name ${MODEL} --task ${TASK} --head_lr ${LR} --seed ${SEED} --out_dir \"${SEED_DIR}\" > \"${LOG_FILE}\" 2>&1"
          echo "$FULL_CMD" >> "$CMD_LIST"
        done
      done
    done
  done
done

echo "Total tasks generated: $(wc -l < $CMD_LIST). Starting parallel queue with 4 slots..."
cat "$CMD_LIST" | xargs -I {} -P 4 bash -c "{}"

rm "$CMD_LIST"
echo "All tasks in the queue have been processed."
