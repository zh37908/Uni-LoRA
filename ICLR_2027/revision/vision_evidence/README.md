# ViT evidence added to the ProLoSA paper

The main-text vision table and Appendix `app:vit` use completed runs from
`ViT/results/p1_vision` (24,600 compressed-adapter parameters) and
`ViT/results/p1_vision_d72k` (72,000). No training or evaluation was rerun.

Regenerate the evidence from the repository root:

```sh
python ICLR_2027/revision/vision_evidence/build_tables.py
```

The script checks each JSON against its completed training log, verifies sparse
activation and active budgets, exports `runs.csv`, and generates the TeX result
rows plus exact allocation fragments. The CSV preserves original parameter
fields alongside corrected active counts and logged hyperparameters. There are
62 completed runs: 30 seed-42 runs per recipe, plus CIFAR-100@1k LoRA seeds 43/44
for the 24,600 recipe. Smoke runs are excluded.

## Reporting choices

- Every score in the paper is `100 * final_accuracy`, rounded to two decimals.
  `best_accuracy` is retained in the audit CSV only: it is a maximum over repeated
  evaluation on the same held-out split, not validation-selected test accuracy.
- All four ProLoSA ratios are shown. No per-dataset winning ratio is substituted
  into a single synthetic ProLoSA row.
- Both recipes are included because budget, latent learning rate, and head
  learning rate differ. Comparing them does not isolate a budget effect. The
  existing launcher designates the 24,600 recipe as its default after inspecting
  the available results; this is not independent validation selection.
- Complete method comparisons have only seed 42. LoRA's two extra runs cannot
  establish uncertainty or significance for ProLoSA.
- ProLoSA JSON fields `adapter_params` and `trainable_params` were captured
  before sparse activation. The paper counts latent dimension `d` plus the `K`
  coordinates confirmed in each activation log. The trainable classifier head
  adds `769 * number_of_classes`. Active degrees of freedom must not be confused
  with allocated dense sparse-buffer storage or optimizer memory.
- The custom balanced 1,000-image subsets are not the standard VTAB-1k protocol.
- Main-table full-data CIFAR-100: Uni-LoRA 92.10%, ProLoSA 4:1 92.33%, LoRA
  92.41%. All four ProLoSA ratios trail Uni-LoRA on CIFAR-100@1k and @5k.

## Original Uni-LoRA vision experiments

Verified against the [NeurIPS 2025 paper, Section 4.4 and Table 5](https://proceedings.neurips.cc/paper_files/paper/2025/file/3e596a70b60dcf7c6e0ac1a2d73e7470-Paper-Conference.pdf).
That paper already covers ViT-Base and ViT-Large, eight image-classification
datasets, and five runs. Its Table 5 labels the ViT-Base Uni-LoRA adapter budget
72K, although the nearby prose says 74,000; the local legacy
`ViT/fine_tuning_ViT_base.py` explicitly uses 72,000. We use the actual logged
budgets for the new experiments and do not import old numerical baselines into
the new comparisons. The legacy script uses a different batch/accumulation and
checkpoint-selection setup. Completed ProLoSA ViT-Large results are absent.

## Paper files

- `sections/main.tex`: full 24,600-budget vision result table and interpretation.
- `sections/appendix_vision.tex`: model, data, active parameter counts, training,
  support-selection timing, reporting limitations, and full 72,000-budget table.
- `sections/appendix_downstream.tex`: includes the new appendix and scopes the
  earlier 128-step scoring-window convention to allow the documented exceptions.

Build from `ICLR_2027` with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```
