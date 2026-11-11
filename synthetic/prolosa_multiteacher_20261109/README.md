# Q4: independent spike/slab Transformer teachers

This directory implements the [locked Q4 protocol](../../ICLR_2027/revision/reviewer_five_questions_20261109/q4_multiteacher_protocol_zh.txt). It evaluates ten new teacher seeds (100–109), five noise levels, three fixed methods, and twenty paired training repeats: 3,000 fits. Seed 13 remains an external reference. No development search runs on the new teachers.

The selected configurations come from `../prolosa_spike_slab_teacher_20260925/selection.json`. The old `run_search.Adapter` and `run_search.train` implement initialization, AdamW updates, gradient clipping, and sum-SNIP support selection. This experiment uses the old prepared bundle only for adapter initialization; each new teacher receives its own independent physical-weight draw and calibration.

The optional optimization control was **included before seeing results**: for noise 1 and 4, repeats 0–4, all methods and teachers, the same trajectory is extended to 3,000 updates while retaining its 1,000-update primary prediction. This adds 300 extended trajectories.

For the tested Python, PyTorch/CUDA, and analysis dependencies, see [ENVIRONMENT.md](ENVIRONMENT.md) and [environment.yml](environment.yml).

## Run

`submit.sh` queues a GPU preparation and smoke job, then a five-task GPU array with an `afterok` dependency, then a CPU analysis job after the full array succeeds. Five array shards each process two teachers. The GPU jobs target the A30 partition by default; the scripts use `/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python` unless `MULTITEACHER_PYTHON` is set.

```bash
bash synthetic/prolosa_multiteacher_20261109/submit.sh
squeue -u hzhaobi
cat synthetic/prolosa_multiteacher_20261109/job_ids.tsv
```

The preparation job writes `spec.json`, fixed `test.pt`, and each teacher's `teacher.pt`, `truth.npy`, `diagnostics.json`, and `manifest.json`. A smoke run outside the statistical grid checks all three methods and ProLoSA support activation. The main array writes per-fit `result.json`, `predictions.npz`, `factors.npz`, and `support.npz` under `fits/t<seed>/<method>/l<noise>/r<repeat>/`. The analysis job writes complete-fit tables, teacher-level intervals, and figures to `summary/`; `analyze.py` refuses an incomplete grid.

## Required local artifacts

The repository includes the locked protocol, selected configurations, source snapshots, code, and the four frozen binary inputs:

* `synthetic/results_revision5_fixed_teacher/teachers.pt`
* `synthetic/prolosa_all_linear_20260924/prepared.pt`
* `synthetic/prolosa_spike_slab_teacher_20260925/prepared.pt`
* `synthetic/prolosa_spike_slab_teacher_20260925/test.pt`

Before running `submit.sh`, verify them from the repository root with `sha256sum -c synthetic/prolosa_multiteacher_20261109/weights.sha256`.

The Slurm scripts use the original cluster account, partitions, repository path, and Conda environment. Adjust those fields for another cluster before submission. Python dependencies are PyTorch, NumPy, SciPy, and Matplotlib.

`run_multiteacher.py status` reports the number of result files after preparation. Failed array shards can be resubmitted with the same code and specification; completed fits are hash checked before being skipped. A source or dependency change after preparation causes a hard failure, so a changed protocol needs a new experiment directory.

All primary confidence intervals treat the teacher as the independent unit. The test inputs are fixed at the original seed-1000050 design, with new truth computed for every teacher. The old seed-13 fits are shown separately and excluded from ten-teacher inference.
