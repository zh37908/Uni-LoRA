# Synthetic Theory Validation

Code and results for the paper's synthetic bias–variance experiments (Figure 2 and the Hidden-P follow-up).

## Fixed-target protocol and current manuscript (2026-09-13)

The current ICLR manuscript uses `results_matched_budget_baselines`,
`results_spike_slab_budget_matched_theory`,
`results_gaussian_synthetic_theory_std_0.125`, and
`results_transformer_lab_level2_v2_full`. The older “Paper Fig. 2” label below
refers to an earlier manuscript.

For each experiment, sample the optimum **once**, outside the noise/training-seed
loop. Keep the target and learner projection fixed across the n/noise sweep.
Compute squared bias and variance using the same estimates and the same fixed
target; the empirical variance denominator is M. `fixed_target.py` enforces this
contract and checks supplied per-trial risks. Distinct target profiles/strengths
require separate decompositions. A declared average of conditional risks is
valid; pooling estimator variance across different targets is a different
quantity and must not be added to conditional bias.

Audit and results: [fixed-target audit](../ICLR_2027/revision/existing_evidence/README.md).
No existing experiment CSV was replaced or relabeled as a new experiment.

**Historical Hidden-P caveat:** the saved angle/mode sweeps hold the target fixed
within each cell, but the old problem seed and RNG consumption also change the
teacher when the learner angle/mode changes. They are not a controlled
same-target projection sweep, and their bias/variance must not be pooled across
scenarios. The updated runner separates teacher/projection/data RNG streams;
new rows carry `target_protocol=fixed_hidden_teacher_v2` and a `target_id`.
Old CSVs remain historical results and have not been rerun. Use a new output
directory for future corrected runs. The older qualitative sweep descriptions
below do not establish an effect caused solely by projection mismatch.

## Layout

```text
synthetic/
├── synthetic_validate_theory.py   # main runner
├── synthetic_plot_fig2_style.py   # Fig. 2-style plots from CSV
├── run_hidden_p.sh                # Hidden-P / non-oracle projection
├── run_hidden_p_sweep.sh          # Hidden-P projection mismatch sweep
├── plot_hidden_p_sweep.py         # tables and figures for the sweep
├── README.md
├── results_synthetic_theory_snip_better/   # paper Fig. 2 source
├── results_synthetic_theory_snip_aniso/
├── results_synthetic_theory_better/
├── results_synthetic_theory_sparse_better/
├── results_synthetic_theory_hidden_p/      # new Hidden-P results
├── results_synthetic_theory_hidden_p_sweep/# Hidden-P sweep tables/figures
└── results_synthetic_theory*/              # earlier / smoke runs
```

## Experiments

| Mode | Flag | Meaning |
|------|------|---------|
| Quadratic noise | `--experiment quadratic_noise` | Local quadratic theory validation |
| Linear regression | `--experiment linear_regression` | Teacher–student setup used for paper Fig. 2 |
| Hidden-P | `--experiment hidden_p` | Teacher uses `P_T`; learners only see `P_M` |

## Paper Fig. 2 source

- Script: `synthetic_validate_theory.py --experiment linear_regression --support snip ...`
- Results: `results_synthetic_theory_snip_better/`
- Plot helper: `synthetic_plot_fig2_style.py --results-dir results_synthetic_theory_snip_better`

## Hidden-P / Non-oracle Projection

Responds to the reviewer concern that ProLoSA cannot observe the true generating projection.

Ground truth:

```text
theta_star = P_T z_star + gamma q_star
```

- `P_T` is used only to synthesize `theta_star`
- Learners may only access `P_M`, built either as a controlled rotation of `P_T`
  (`--pm-mode rotated --pm-angle-deg ...`) or as a fresh random subspace
  (`--pm-mode independent`)
- ProLoSA still fits `theta = P_M z + R a` with warmup SNIP residual support

Compared methods:

- `lora` — full-space estimator
- `unilora` — Uni-LoRA with `P_M`
- `prolosa` — ProLoSA with `P_M` + SNIP
- `unilora_oracle` — Uni-LoRA with `P_T` (upper bound only)

Run:

```bash
bash run_hidden_p.sh
# optional overrides:
# PM_ANGLE_DEG=15 SPARSE_BUDGET=16 OUT_DIR=... bash run_hidden_p.sh

# recommended robustness sweep:
bash run_hidden_p_sweep.sh
```

Default setting in `run_hidden_p.sh`:

- `D=512`, `d=32`, `n ∈ {64,128,256,1024}`
- `gamma ∈ {0,0.5,1,1.5,2}`, `noise_std ∈ {0.25,0.5,1}`
- `pm_angle_deg=15` (subspace overlap ≈ 0.93)
- `sparse_budget=16`, `support=snip`

Sweep setting in `run_hidden_p_sweep.sh`:

- `P_M` modes: rotated angles `{0,5,15,30}` degrees plus `independent`
- `D=512`, `d=32`, `n ∈ {64,128,256}`
- `gamma ∈ {0,0.5,1,1.5,2}`, `noise_std ∈ {0.25,0.5,1}`
- outputs: `hidden_p_results_noise_0.5.{md,tsv}`, `hidden_p_win_rates.{md,tsv}`,
  `hidden_p_delta_noise_0.5.png`, `hidden_p_win_rates.png`,
  `hidden_p_representative_risk_noise_0.5.png`

Observed pattern in `results_synthetic_theory_hidden_p_sweep`:

- `unilora_oracle` is best (as expected; it sees `P_T`)
- under controlled non-oracle mismatch, when residual strength `gamma` is large
  enough and `n` is not too small, **ProLoSA with `P_M` beats Uni-LoRA with `P_M`**
- the ProLoSA win rate decreases as `P_M` moves farther from `P_T`, which is the
  expected stress-test behavior
- `independent` random `P_M` is an extreme stress test (overlap ≈ `d/D`) and
  should be reported separately from the main controlled-mismatch result
- at very small `n` / tiny `gamma`, sparse-branch variance can outweigh the bias reduction (same bias–variance trade-off as in the paper)
