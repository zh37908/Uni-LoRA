# 9-page submission restructuring

The main text was rewritten around the chain **conditions → residual correction → mechanism tests → downstream results**. The original TeX and PDF are preserved in `archive/before_9page_20260912_230439/`.

## Sources and organization

- `iclr2027_conference.tex`: original preamble and submission entry point, statements, bibliography, and appendix inputs. The ICLR style file and page geometry were not changed. `hyperref` is loaded after the float/algorithm packages to avoid duplicate figure/table hyperlink destinations.
- `sections/main.tex`: rewritten abstract, introduction and related work, two core propositions, local sample-size prediction, concise ProLoSA method, three levels of mechanism tests, downstream summary, and matched-budget ablations.
- `sections/appendix_theory.tex`: extended related work, detailed derivations, and proofs. Restated propositions do not consume additional proposition numbers.
- `sections/appendix_method.tex`: original parameterization, framework figure, support selection, and algorithm.
- `sections/appendix_mechanism.tex`: full sequence, Transformer, and real-model experiments, original figures/tables, and controlled-experiment configurations.
- `sections/appendix_downstream.tex`: complete benchmark results, baseline provenance, configurations, Llama-3.1 sweep, and completed allocation/support/warmup ablations.
- `sections/appendix_vision.tex`: ProLoSA ViT-B/16 settings, active parameter accounting, seed/selection limitations, and the supplementary 72,000-budget sweep; included from the downstream appendix. The main text includes all four ProLoSA ratios at the 24,600 budget. Audited result sources and regeneration are in `revision/vision_evidence/README.md`.
- `drafts/planned_experiments.tex`: the original unfilled budget-sweep and same-rank comparison tables, retained as author drafts and excluded from the compiled submission. No experimental measurements were invented or filled in.

## Figure provenance

`figures/make_summary_figures.py` regenerates two vector PDF figures:

- `sequence_summary.pdf`: existing matched-budget CSV and the dense Gaussian CSV with source standard deviation 0.125. Copies of these CSVs are in `figures/data/`. Panel (c) deliberately retains the original unequal-budget failure comparison, explicitly distinguished from the equal-budget comparison in panel (b).
- `real_mechanism.pdf`: the reported paired mean loss gaps in Table E1 and predictive decomposition values in Table E3, parsed from `sections/appendix_mechanism.tex`. No uncertainty for paired gaps is inferred from marginal standard deviations.

The existing `transformer_risk_vs_n.pdf` is used unchanged in the main text. No model training or evaluation was rerun. The newly generated figures require Python, NumPy, and Matplotlib; their PDFs are already included, so these Python dependencies are not needed to compile the paper.

## Preserved qualifications

- Fixed-task residual energy and ensemble-average target energy remain distinct.
- The local sample-size law is conditional on its asymptotic and shared-geometry assumptions.
- The ideal matched-budget theorem assumes noise-independent support selection; warmup selection can violate this.
- Real-model experiments do not establish a reliable crossover. QNLI's weak fit and the absence of bias correction on CoLA are retained.
- Predictive-reference bias and diagonal-Fisher residual measurements are identified as proxies.
- Transformer compression acts in weight-update space, and the two teachers vary both update magnitude and concentration.
- Mixed mathematical-reasoning outcomes, one-seed Llama-3.1 results, and post-hoc test-score configuration selection remain explicit.

## Build

From this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

After adding the ViT experiments, the compiled main text ends on page 9, including three figures and four tables. Statements and references start on page 10. The vision appendix occupies pages 46--47. The initial-submission limit of at most 9 main-text pages is preserved. The final build has no undefined references or overfull boxes.
