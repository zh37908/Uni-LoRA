# E4 / E5 theory artifacts (revised analysis)

## E4-A cross-task: off-subspace energy vs compression gap (secondary)

| task | r_off | r_off^H | gap_score (LoRA-Uni) | gap_loss (Uni-LoRA loss excess) |
|---|---:|---:|---:|---:|
| cola | 0.9870 | - | -0.0167 | -0.0410 |
| mrpc | 0.9870 | - | 0.0041 | -0.0064 |
| qnli | 0.9870 | - | 0.0001 | -0.0352 |
| rte | 0.9870 | - | 0.0217 | 0.0073 |
| sst2 | 0.9870 | - | -0.0053 | -0.0853 |
| stsb | 0.9870 | - | 0.0027 | 0.0137 |

- r_off vs gap_loss: Pearson=-0.7541, Spearman=-0.4857 (n=6)
- r_off vs gap_score: Pearson=-0.1012, Spearman=0.0857 (n=6)

## E4-B controlled projection sweep (primary): B_off(P) vs excess dev loss

### cola

| proj_seed | B_off ratio | B_off^H ratio | Uni dev loss (mean) | excess loss |
|---:|---:|---:|---:|---:|
| 10 | 0.986925 | - | 0.3615 | -0.0636 |
| 11 | 0.987070 | - | 0.3629 | -0.0622 |
| 12 | 0.986978 | - | 0.3630 | -0.0621 |
| 13 | 0.986902 | - | 0.3694 | -0.0557 |
| 14 | 0.987091 | - | 0.3683 | -0.0568 |
| 15 | 0.986974 | - | 0.3757 | -0.0494 |
| 16 | 0.986815 | - | 0.3647 | -0.0604 |
| 17 | 0.986907 | - | 0.3626 | -0.0625 |

- B_off vs excess loss: Pearson=0.1237, Spearman=0.0000 (n=8)

### mrpc

| proj_seed | B_off ratio | B_off^H ratio | Uni dev loss (mean) | excess loss |
|---:|---:|---:|---:|---:|
| 10 | 0.986891 | - | 0.2727 | -0.0086 |
| 11 | 0.986919 | - | 0.2693 | -0.0120 |
| 12 | 0.986855 | - | 0.2746 | -0.0067 |
| 13 | 0.986800 | - | 0.2671 | -0.0142 |
| 14 | 0.987152 | - | 0.2776 | -0.0037 |
| 15 | 0.986826 | - | 0.2701 | -0.0112 |
| 16 | 0.986926 | - | 0.2779 | -0.0034 |
| 17 | 0.986948 | - | 0.2699 | -0.0114 |

- B_off vs excess loss: Pearson=0.6159, Spearman=0.4762 (n=8)

## E5 off-subspace residual concentration (sparse recoverability)

| task | weighting | K | C_K oracle | rho_I^2 SNIP | K/D | K/(D-d) | Enrich_K | q top0.1% | q top1% | q top5% | gauss top1% | kurt(q) | gini(q) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cola | bare | 8640 | 0.0809 | 0.0019 | 0.004883 | 0.004923 | 0.39 | 0.0244 | 0.1357 | 0.3872 | 0.0845 | 4.87 | 0.4711 |
| mrpc | bare | 8640 | 0.0890 | 0.0018 | 0.004883 | 0.004923 | 0.36 | 0.0265 | 0.1504 | 0.4427 | 0.0847 | 5.95 | 0.5326 |
| qnli | bare | 721 | 0.0134 | 0.0002 | 0.000407 | 0.000413 | 0.60 | 0.0258 | 0.1270 | 0.3470 | 0.0844 | 4.34 | 0.4399 |
| rte | bare | 721 | 0.0110 | 0.0001 | 0.000407 | 0.000413 | 0.13 | 0.0223 | 0.1240 | 0.3553 | 0.0847 | 4.29 | 0.4522 |
| sst2 | bare | 0 | - | - | - | - | - | 0.0272 | 0.1351 | 0.3662 | 0.0846 | 4.70 | 0.4500 |
| stsb | bare | 721 | 0.0216 | 0.0001 | 0.000407 | 0.000413 | 0.25 | 0.0409 | 0.1839 | 0.4640 | 0.0846 | 7.41 | 0.5108 |

Notes: q = (I - Pi_P) delta_theta_LoRA with ProLoSA's own grouping; K = SNIP budget; rho_I^2 = off-subspace energy captured by the actual SNIP support; Enrich_K = rho_I^2 / (K/(D-d)). rho_I^2 requires offsets_A/B in the artifact (runs after 2026-08-31); older artifacts show '-'.

## E5 (secondary) total-update energy spectrum

| task | method | top0.1% | top1% | top5% | kurtosis | gini |
|---|---|---:|---:|---:|---:|---:|
| cola | lora | 0.0245 | 0.1363 | 0.3886 | 4.91 | 0.4726 |
| cola | unilora | 0.0141 | 0.0889 | 0.2865 | 3.11 | 0.4169 |
| cola | prolosa | 0.0189 | 0.0952 | 0.2928 | 3.78 | 0.4180 |
| mrpc | lora | 0.0266 | 0.1510 | 0.4448 | 6.00 | 0.5373 |
| mrpc | unilora | 0.0151 | 0.0904 | 0.2873 | 3.15 | 0.4168 |
| mrpc | prolosa | 0.0208 | 0.1012 | 0.3016 | 3.97 | 0.4214 |
| qnli | lora | 0.0260 | 0.1278 | 0.3485 | 4.37 | 0.4406 |
| qnli | unilora | 0.0142 | 0.0896 | 0.2869 | 3.12 | 0.4169 |
| qnli | prolosa | 0.0155 | 0.0890 | 0.2848 | 3.39 | 0.4159 |
| rte | lora | 0.0224 | 0.1247 | 0.3570 | 4.32 | 0.4536 |
| rte | unilora | 0.0140 | 0.0895 | 0.2881 | 3.13 | 0.4178 |
| rte | prolosa | 0.0146 | 0.0909 | 0.2901 | 3.17 | 0.4186 |
| sst2 | lora | 0.0275 | 0.1361 | 0.3681 | 4.75 | 0.4512 |
| sst2 | unilora | 0.0171 | 0.0946 | 0.2921 | 3.26 | 0.4180 |
| sst2 | prolosa | 0.0162 | 0.0901 | 0.2861 | 5.00 | 0.4164 |
| stsb | lora | 0.0414 | 0.1855 | 0.4677 | 7.53 | 0.5156 |
| stsb | unilora | 0.0150 | 0.0913 | 0.2896 | 3.20 | 0.4179 |
| stsb | prolosa | 0.0167 | 0.0942 | 0.2932 | 3.31 | 0.4187 |

