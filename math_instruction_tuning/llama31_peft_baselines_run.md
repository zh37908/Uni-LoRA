# Llama-3.1-8B additional mathematical reasoning baselines

## FourierFT Q/V replacement (2026-09-13)

At the user's request, cancelled the all-projection FourierFT task 1849991_2
and submitted replacement **1857851_2**, with a 72-hour limit and one RTX 5880.
This is a fresh training run, not a resume of the all-projection checkpoint.
Old logs/checkpoints are retained. Only FourierFT changes target modules:
`q_proj`, `v_proj` (64 matrices), with 16,384 coefficients per matrix, exactly
1,048,576 trainable parameters. All other training/evaluation settings below
are retained, including gradient checkpointing, microbatch 1, accumulation 64,
LR 0.1, scaling 300, seed 42 and two epochs.

Submission: `sbatch --parsable --array=2 submit_llama31_peft_baselines_1gpu.sh`.
The script defaults FourierFT to Q/V; `FOURIER_TARGET_SCOPE=all` restores the old
target scope. The Python entry point retains its all-projection default.
The new run tag is `llama31_fourierft_qv_r4_b1048576_s42`, with `job_1857851`
under the output/results/merged directory roots documented below. Logs are
`logs/llama31_peft_r4_1857851_2.{out,err}`.

Q/V smoke validation passed: exact tiny-model parameter budget, finite nonzero
gradients under gradient checkpointing, two optimizer updates, adapter save/reload,
and merged-output equivalence. See `/tmp/check_llama31_fourier_qv.log`.
Full-model runtime must be measured from the new training log; the reduction in
matrix reconstruction volume is not an end-to-end speedup measurement.
The Q/V experiment must be labelled separately in comparisons with all-projection
Uni-LoRA/ProLoSA and the original baseline configuration below.

## Original submission

Submitted 2026-09-11, Slurm array **1849991**, one RTX 5880 GPU per task.
The scheduler allocates 10 CPUs and 75 GB host memory to each task.

| Array task | Method | Rank | Trainable parameters | One chosen configuration |
|---|---|---:|---:|---|
| 1849991_0 | VB-LoRA | 4 | 1,047,552 | 93 bank vectors, vector length 1024, top-k=2; bank LR 0.001, logits LR 0.01 |
| 1849991_1 | VeRA | 4 | 1,377,152 | LR 0.01, d_initial=0.1, frozen projections saved |
| 1849991_2 | FourierFT | N/A | 1,048,576 | 4,681 coefficients/module plus one extra in 32 modules; LR 0.1, scaling 300, zero-init spectrum |

Reference budget: main-table Uni-LoRA d=1,048,576. All methods adapt the same
224 attention/MLP projections (`q/k/v/o/gate/up/down`) of Llama-3.1-8B.
FourierFT does not have a LoRA rank; the r4 run tag denotes the comparison group.
VeRA cannot match 1,048,576 at rank 4 with these targets: its output vectors
already cost 1,376,256 parameters, plus 224*4 rank-vector parameters.
VB-LoRA count includes BOTH bank (95,232) and trainable logits (952,320),
not only compressed checkpoint storage. Its total is 1,024 below the target.
LoRA-XS was not found in the repository and is not implemented or submitted.

Shared protocol: NousResearch/Meta-Llama-3.1-8B base, MetaMathQA train[:100000],
2 epochs, seed 42, BF16, no quantization, one GPU, gradient checkpointing,
sequence length 512, microbatch 1, gradient accumulation 64, AdamW,
cosine schedule, 2% warmup, zero weight decay. Prompt, answer-only labels,
tokenizer padding, data collation and truncation reuse intruction_tuning_unilora.py.
No hyperparameter sweep or additional training seeds are submitted.
The method-specific learning rates are single initial choices, not validated optima.
VB-LoRA's two learning rates follow the documented recipe:
https://huggingface.co/docs/peft/package_reference/vblora
FourierFT's scaling follows the documented Llama instruction-tuning setting:
https://huggingface.co/docs/peft/package_reference/fourierft

Implementation: reuse **NLU/peft/src** (contains all three methods) in the existing
unilora_modern environment. No tuner code or environment packages were changed.
New entry point: train_llama31_peft_baselines.py.
Submission script: submit_llama31_peft_baselines_1gpu.sh.

Before submission, all three methods passed a tiny Llama CPU smoke check with
gradient checkpointing: two optimizer steps, finite nonzero gradients, exact
parameter-count checks, checkpoint save/reload, and output equivalence after merge.
Each production run asserts its actual trainable parameter count and writes
run_manifest.json, training_arguments.json, resource_usage.json, and ft/.

Training automatically proceeds to merge, GSM8K (1,319 examples), then MATH500
(500 examples); greedy decoding, batch 32, one-way tensor parallelism, context
4096, max generated tokens 1024/2048. Full MATH is not evaluated in this batch.

- Logs: logs/llama31_peft_r4_1849991_{0,1,2}.{out,err}
- Adapters: output/llama31_peft_r4_b1m/llama31_<method>_r4_b1048576_s42/job_1849991/ft
- Results: results/llama31_peft_r4_b1m/llama31_<method>_r4_b1048576_s42/job_1849991/{gsm8k,math500}.log
- Merged weights: output_merged/llama31_peft_r4_b1m/llama31_<method>_r4_b1048576_s42/job_1849991/

Status at submission: all three tasks RUNNING. This indicates allocation/startup,
not completed training or evaluation. The result directory COMPLETE marker is
written only after successful training, merging, and both evaluation processes.
