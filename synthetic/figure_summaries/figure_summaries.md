# Synthetic Figure Summaries

## Figure S1

- Figure: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s1_known_projection.png`
- Data table: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s1_known_projection_data.md`
- Summary: Figure S1: Known-projection synthetic validation of the bias--variance theory. Left and middle: population excess risk as subspace mismatch gamma increases under different sample sizes. Right: empirical bias--variance decomposition at n=64 and gamma=1. ProLoSA uses warmup SNIP scores to select sparse residual coordinates.

## Figure S2

- Figure: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s2_hidden_p_controlled.png`
- Data table: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s2_hidden_p_controlled_data.md`
- Summary: Figure S2: Hidden-P non-oracle projection with controlled mismatch. The teacher uses P_T to generate the update, while Uni-LoRA and ProLoSA only see P_M, here a 15-degree rotation of P_T (mean squared canonical overlap 0.933). ProLoSA remains below Uni-LoRA for larger gamma at n=256 and n=1024; oracle Uni-LoRA with P_T is shown only as an upper bound.

## Figure S3

- Figure: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s3_projection_mismatch_sweep.png`
- Data table: `/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification/synthetic/figure_summaries/figure_s3_projection_mismatch_sweep_data.md`
- Summary: Figure S3: Projection-mismatch robustness in the Hidden-P setting. Left: ProLoSA win rate over Uni-LoRA for gamma >= 1 across all sample sizes and noise levels. Middle and right: Uni-LoRA minus ProLoSA risk at noise std 0.5 for n=256 and n=1024; values above zero indicate that ProLoSA is better. Independent P_M is an extreme random-subspace stress test.
