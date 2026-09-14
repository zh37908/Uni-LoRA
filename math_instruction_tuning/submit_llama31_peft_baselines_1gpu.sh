#!/bin/bash
#SBATCH --job-name=llama31_peft_r4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=shsong
#SBATCH --array=0-2
#SBATCH --chdir=/home/hzhaobi/Uni-LoRA/math_instruction_tuning
#SBATCH --output=logs/llama31_peft_r4_%A_%a.out
#SBATCH --error=logs/llama31_peft_r4_%A_%a.err

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate unilora_modern
set -euo pipefail
cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# Reuse the repository's existing VB-LoRA/VeRA/FourierFT implementation.
export PYTHONPATH="/home/hzhaobi/Uni-LoRA/NLU/peft/src:${PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

METHODS=(vblora vera fourierft)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
METHOD="${METHODS[$TASK_ID]}"
RUN_TAG="llama31_${METHOD}_r4_b1048576_s42"
FOURIER_TARGET_SCOPE="${FOURIER_TARGET_SCOPE:-qv}"
if [[ "$METHOD" == fourierft && "$FOURIER_TARGET_SCOPE" == qv ]]; then
  RUN_TAG="llama31_fourierft_qv_r4_b1048576_s42"
fi
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUTPUT="${PWD}/output/llama31_peft_r4_b1m/${RUN_TAG}/job_${RUN_ID}"
RESULT="${PWD}/results/llama31_peft_r4_b1m/${RUN_TAG}/job_${RUN_ID}"
MERGED="${PWD}/output_merged/llama31_peft_r4_b1m/${RUN_TAG}/job_${RUN_ID}"
BASE_MODEL="NousResearch/Meta-Llama-3.1-8B"
mkdir -p "$RESULT"

case "$METHOD" in
  vblora) LR=1e-3; EXTRA=(--num_vectors 93 --vector_length 1024 --learning_rate_vector_bank 1e-3 --learning_rate_logits 1e-2) ;;
  vera) LR=1e-2; EXTRA=() ;;
  fourierft) LR=1e-1; EXTRA=(--fourier_scaling 300 --fourier_target_scope "$FOURIER_TARGET_SCOPE") ;;
esac
printf 'Method=%s output=%s results=%s\n' "$METHOD" "$OUTPUT" "$RESULT"
python train_llama31_peft_baselines.py \
  --method "$METHOD" --model_name_or_path "$BASE_MODEL" --output_dir "$OUTPUT" \
  --lora_r 4 --trainable_budget 1048576 --learning_rate "$LR" "${EXTRA[@]}" \
  --data_path meta-math/MetaMathQA --dataset_split 'train[:100000]' \
  --dataset_field query response --model_max_length 512 \
  --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 64 \
  --gradient_checkpointing True --save_strategy steps --save_steps 100 --save_total_limit 2 \
  --weight_decay 0 --warmup_ratio 0.02 --lr_scheduler_type cosine --logging_steps 10 \
  --bf16 True --tf32 True --fp16 False --report_to tensorboard --seed 42

test -f "$OUTPUT/TRAINING_COMPLETE"
python -m utils.merge_adapter_to_base_model --base_model "$BASE_MODEL" \
  --adapter "$OUTPUT/ft" --output_path "$MERGED" --dtype bfloat16
python instruction_tuning_eval/gsm8k_eval.py --model "$MERGED" \
  --data_file data/math_eval/gsm8k_test.jsonl --batch_size 32 \
  --tensor_parallel_size 1 --max_model_len 4096 2>&1 | tee "$RESULT/gsm8k.log"
python instruction_tuning_eval/MATH_eval.py --model "$MERGED" \
  --data_file data/math_eval/MATH500_test.jsonl --batch_size 32 \
  --tensor_parallel_size 1 --max_model_len 4096 2>&1 | tee "$RESULT/math500.log"
cp "$OUTPUT/run_manifest.json" "$RESULT/run_manifest.json"
printf 'Training and both evaluations finished.\n' > "$RESULT/COMPLETE"
# Keep merged models for reproducibility and manual re-evaluation.
