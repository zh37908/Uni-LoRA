# Llama-3.1-8B ProLoSA grid: r=4, d+K=1,048,576

Started 2026-09-11. Controller job: **1850041**.
GPU tasks request **one RTX 5880**, 10 CPUs, gradient checkpointing,
and a **72-hour per-task time limit**. The controller maintains at most six
outstanding ProLoSA tasks and at most nine total outstanding 5880 tasks,
leaving room under the account's MaxSubmitPU=10. MaxJobsPU=8 may leave one
ProLoSA task pending while the three previous baseline jobs run.
Each 5880 node has six GPUs; jobs may be placed on different nodes.

## Timing semantics verified from implementation

Both `rosa_warmup_steps` and `rosa_mask_steps` count optimizer updates,
not individual forward/backward microbatches. The fixed microbatch is 1 and
gradient accumulation is 64. Total training is 2 epochs over 100,000 examples,
approximately 3,126 optimizer updates.

- Learning-rate scheduler warmup: `warmup_ratio=0.02`, approximately 63 updates.
- Current sparse warmup: 128 updates (8,192 processed examples, about 4.1% of updates).
- Current mask window: 1 update (64 examples).
- Collect scores while global_step is in [warmup, warmup+mask_window).
- Generate the fixed mask at the end of update warmup+mask_window.
- The first sparse optimizer update follows mask generation.
- Thus the original 128+1 setting activates after update 129 and first updates
  sparse values during update 130. The new 512+8 setting activates after update
  520 and first updates sparse values during update 521.

| Sparse warmup | Approx. processed examples before collection | Fraction of 3,126 updates |
|---:|---:|---:|
| 128 | 8,192 | 4.1% |
| 512 | 32,768 | 16.4% |
| 1,024 | 65,536 | 32.8% |

A longer warmup may produce a more informative support estimate, but also leaves
less time to fit sparse residuals. An eight-update window aggregates scores over
approximately 512 examples rather than 64. These are hypotheses being tested,
not conclusions from the existing final accuracies.

## Existing evidence and bounded grid

Reuse eight completed 1M-budget results: d:K in {2:1,4:1,8:1,12:1}, projected LR
in {0.0016,0.0032}, sparse LR multiplier 0.2, sparse warmup 128, mask window 1.
The best two-benchmark mean is the 12:1 / 0.0016 run: GSM8K 75.0569%, MATH500
26.60%, mean 50.8284%. The best GSM8K run is 4:1 / 0.0016: 75.6634% / 24.80%.
Increasing projected LR to 0.0032 did not improve the best mean; at 12:1 it reduced
MATH500 to 21.60%. Hence the added LR range emphasizes 0.0008--0.0024.

25 new configurations, without repeating any of the eight existing ones:

1. Timing/sparse-LR grid at ratio 12:1 and projected LR 0.0016:
   warmup {128,512,1024} x mask window {1,8} x sparse multiplier {0.1,0.2}.
   Twelve combinations, one already evaluated: **11 new**.
2. Ratio/projected-LR grid at warmup 512, mask window 8, sparse multiplier 0.2:
   ratio {4:1,8:1,12:1,24:1} x projected LR {0.0008,0.0016,0.0024}.
   Twelve combinations, one already in grid 1: **11 new**.
3. At ratio 12:1, projected LR 0.0016, warmup 512, window 8, sparse multiplier 0.2:
   initialization bounds 0.01 / 0.04 (default 0.02), and optimizer reset=False
   (default True): **3 new**.

This is a union of two focused Cartesian grids plus three controlled comparisons,
not a full Cartesian product of all hyperparameters. Its anchors are chosen
before seeing new scores. Integer K is round(B/(ratio+1)); d=B-K exactly.
All defaults remain as in the existing runs: rank 4, seed 42, AdamW, two epochs,
BF16, all attention/MLP linear projections, zero dropout/weight decay,
sequence length 512, cosine LR decay, sparse LR decay after activation.
The base LR stays 0.0002; projected and sparse group LRs are controlled explicitly.

Other available controls intentionally held fixed: scheduler warmup, sparse LR
decay policy, total epochs, global scheduler, batch size, gradient clipping,
projection/training seed. Dropout exists in the PEFT config but is hard-coded to
zero in the current entry point; this grid does not modify tuner implementations.

## Submission and reporting

- Exact configurations: prolosa_grid_1m.json (task_id -> all hyperparameters).
- Runner: prolosa_grid_1m.py; invokes the existing training and evaluation scripts.
- Dispatcher: dispatch_prolosa_grid_1m.py; submit via submit_prolosa_grid_dispatch.sh.
- First six tasks: grid_00, grid_03, grid_05, grid_08, grid_10, grid_06.
  They cover warmup 128/512/1024 and windows 1/8 early.
- Per-run logs: results/prolosa_grid_1m/grid_XX/{train,merge,gsm8k,math500}.log.
- Each run immediately evaluates GSM8K and MATH500 after its full two-epoch
  training, validates counts of 1,319 and 500, then writes scores.json and COMPLETE.
- Actual selected K is checked from sparse activation logs; d+K must match the budget.
- Checkpoints/merged weights: output[/_merged]/prolosa_grid_1m/grid_XX/.
- Live summary: results/prolosa_grid_1m/summary.md and summary.json.
- Snapshot reports at 24,48,71 hours: summary_24h.*, summary_48h.*, summary_71h.*.
- Queue/job map: results/prolosa_grid_1m/dispatch.json.
- Controller stops submitting at 71.5 hours and reports any unsent or unfinished
  configurations. Already submitted GPU jobs retain their own 72-hour time limits.

Expected first completed runs: roughly 16--24 hours plus queue time, based on
historical 16-hour ProLoSA training. Three-day preliminary results are the goal;
completion of all 25 configurations in three days is not guaranteed. No partial
training scores are ranked against completed two-epoch scores.

Ranking: arithmetic mean of GSM8K and MATH500, with separate best-GSM8K and
best-MATH500 configurations also reported. These are test-set-selected exploratory
results at seed 42; they are not validation-selected, globally optimal, or evidence
of statistical significance. No new paper scores are filled before completion.

Operational note: five initial tasks inherited a 48-hour limit and were canceled
during startup, then resubmitted at 72 hours; startup artifacts are retained under
results/prolosa_grid_1m/cancelled_48h_startup. No completed experiment was discarded.
