# Llama-3.1-8B instruction tuning: rank 4, three seeds

This package reproduces the completed SmolTalk instruction-tuning experiment
without committing models, datasets, checkpoints, generations, caches, external
repositories, machine-specific paths, job queues, or old experimental copies.
It uses the existing `math_instruction_tuning/peft/src` backend in this repository.
The package consolidates the original experiment scripts; the underlying training,
loss weighting, optimizer, SNIP selection, and serialization functions are unchanged.

## Protocol and fixed configurations

- Base model: `NousResearch/Meta-Llama-3.1-8B`, revision
  `1f47e50cdbe801ad8a5174156ec3a0655108fb9f`.
- Dataset: `HuggingFaceTB/smoltalk`, revision
  `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`, `smol-magpie-ultra` only.
- 50,000 training conversations, 1,000 validation, 1,000 SFT test examples;
  split seed 20260916, first-user deduplication, benchmark prompt-overlap exclusion,
  deterministic SHA256 selection. Complete sequences shorter than 2048 tokens;
  no truncation or packing. `expected_data.json` records the original data hashes.
- All 32 layers' q/k/v/o/gate/up/down projections; frozen embedding and output head.
  Native Llama BOS 128000 and EOS 128001, shared plain-role serialization.
- One epoch / 1,563 optimizer steps, global batch 32, microbatch 1;
  loss weighted by supervised tokens. BF16 base, FP32 adapters, SDPA, TF32,
  non-reentrant gradient checkpointing.
- AdamW `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0, gradient clipping 1;
  cosine scheduler with 0.03 warmup ratio. Training seeds **0, 21, 42**.

| Group | Rank | Effective parameters | Main LR | d | K | Sparse warmup |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 4 | 10,485,760 | 0.00005 | — | — | — |
| Uni-LoRA | 4 | 2,621,440 | 0.003 | 2,621,440 | — | — |
| ProLoSA primary | 4 | 2,621,440 | 0.0024 | 2,419,791 | 201,649 | 128 |
| ProLoSA backup | 4 | 2,621,440 | 0.0024 | 2,097,152 | 524,288 | 128 |

LoRA alpha is 16. Uni/Pro use their own shared-parameter normalization rather than
LoRA alpha scaling, and initialize the shared parameters uniformly in ±0.02.
Both ProLoSA configurations use **max SNIP**: take the coordinate-wise maximum
of `abs(A/B) * abs(gradient)` over all backward microbatches in the 8-step scoring
window. Global Top-K then activates after step 136. Sparse LR is main LR × 0.2
(0.00048); optimizer state is reset at activation. Effective parameter count is
d+K, not the size of the dense storage for the sparse residual.

Evaluation uses pinned official IFEval (541 prompts) and IFBench (300 prompts)
scorers (IFBench uses the pinned `ifbench/data/IFBench_test.jsonl`, not
the repository-root `data/` variant), greedy generation, max 1280 new tokens, native EOS, and unmerged FP32
adapters. It also records validation/test NLL and the synthetic Dev96 diagnostic.
Pinned scorer revisions are in `fetch_verifiers.py`; third-party source is fetched
at runtime into group storage and is not vendored here.

## Selection and interpretation

The seed42 grid in `results/selection_grid.csv` was used for hyperparameter
selection: IFEval prompt-strict first, with IFBench prompt-loose breaking ties.
Uni-LoRA uses its best IFEval configuration. ProLoSA primary ranked first; backup
tied for second on IFEval and won the IFBench tie break. Both configurations were
fixed before running seeds 0 and 21 and are reported below. The final manuscript
comparison uses **ProLoSA backup**, chosen after examining the three-seed results;
the primary results are retained for transparency.

**IFEval/IFBench were used in configuration selection; do not describe them as
untouched final-only evaluation sets.** Seed42 participates in selection and the
three-seed average. The summary also includes new-seed-only means for 0 and 21.
LoRA uses a fixed baseline LR; it did not receive the same grid-search budget.
These are descriptive small-seed comparisons, not a claim of statistical significance.
All formal benchmark scores are from final-epoch adapters, not intermediate
benchmark-based checkpoint selection.

## Recorded results

Percentages, mean ± sample standard deviation (`ddof=1`), seeds 0/21/42:

| Method | IFEval prompt-strict | IFBench prompt-loose |
|---|---:|---:|
| LoRA | 46.64 ± 2.17 | 16.89 ± 1.07 |
| Uni-LoRA | 46.64 ± 4.22 | 15.78 ± 0.96 |
| ProLoSA primary | 47.94 ± 3.37 | 16.11 ± 1.64 |
| **ProLoSA backup (comparison configuration)** | **46.89 ± 2.69** | **18.11 ± 1.17** |

`results/three_seeds.csv` contains all 12 seed-level scores;
`results/summary.json` includes all four official metrics per benchmark;
`results/provenance.json` records original source/prediction hashes. The original
run completed nine new experiments and reused three seed42 results. This standalone
package can rerun all 12 from scratch. Original absolute-path checkpoints are not
resume inputs for this reorganized package: their strict source/config provenance
checks intentionally differ. Resume only checkpoints created by this package.

## Setup and preparation

Use Python 3.10 on a CUDA 12.8-compatible GPU host. Set group storage explicitly;
`env.sh` routes HF, pip, CUDA, torch, NLTK, temporary files and Python caches there.
It does not read `.bashrc`; direct networking is the default. Set
`UNILORA_NETWORK_MODE=inherit` only for an appropriate proxy on the execution host.

```bash
# From the repository root; choose a real group-storage path on your cluster.
export UNILORA_STORAGE=/path/to/group/storage/llama31_smol
export UNILORA_PACKAGE="$PWD/instruction_tuning/llama31_smol"
source "$UNILORA_PACKAGE/env.sh"
python3.10 -m venv "$UNILORA_STORAGE/env"
source "$UNILORA_STORAGE/env/bin/activate"
python -m pip install -r "$UNILORA_PACKAGE/requirements.txt"
export UNILORA_PYTHON="$UNILORA_STORAGE/env/bin/python"

python "$UNILORA_PACKAGE/download.py"
python "$UNILORA_PACKAGE/prepare.py"
python "$UNILORA_PACKAGE/test_data.py"
python "$UNILORA_PACKAGE/development.py"
python "$UNILORA_PACKAGE/test_training.py" --cpu
python "$UNILORA_PACKAGE/test_package.py"
```

Model access remains subject to the model's license and access conditions.
Model and dataset links are created **inside group storage**. Preparation refuses
to overwrite an existing frozen split. The output source-file hashes differ from
the original packaging, while the raw split hashes should match `expected_data.json`.

## Run and resume

```bash
config="$UNILORA_PACKAGE/configs/prolosa_backup_r4_seed0.json"
run="$UNILORA_STORAGE/runs/llama31_smol/prolosa_backup_r4_seed0"
python "$UNILORA_PACKAGE/train.py" --method prolosa --config "$config" --run-dir "$run"
python "$UNILORA_PACKAGE/evaluate.py" --method prolosa --config "$config" --run-dir "$run"
# If interrupted: rerun training with --resume, then rerun evaluation.
# Evaluation resumes from the existing prediction prefix with provenance checks.
```

For a GPU smoke test, use a separate fresh output directory and add `--smoke` to
training. Smoke runs change batch size, warmup/window and length of training; never
use them as reported full-run results.

## Slurm parallelism

Site account/partition/QoS are command-line options, not hardcoded. Each job gets
one GPU and four CPUs. The reference cluster permits eight concurrent normal jobs,
ten running+pending jobs, and 72 hours/job; query your own site's rules first.
Submit two waves to stay within those limits:

```bash
python "$UNILORA_PACKAGE/submit.py" --account YOUR_ACCOUNT --seeds 0,21
# After the first wave has ended:
python "$UNILORA_PACKAGE/submit.py" --account YOUR_ACCOUNT --seeds 42
python "$UNILORA_PACKAGE/summarize.py" --runs "$UNILORA_STORAGE/runs/llama31_smol"
# Recompute the published summary without training or downloading:
python "$UNILORA_PACKAGE/summarize.py" --recorded
```

`submit.py` conservatively counts all of your running/pending jobs and rejects a
wave exceeding `--max-active` (default eight). Keep a single submission process;
do not submit the same configuration simultaneously from multiple terminals.
The job script forwards the pre-timeout signal to save a checkpoint. Resubmit an
interrupted job with the same config; completed training/benchmarks are skipped.
No jobs, downloads, or training are launched merely by importing the package.
