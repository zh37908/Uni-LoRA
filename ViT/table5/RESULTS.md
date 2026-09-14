# Completed ProLoSA Table 5 experiments

80 formal runs: 2 backbones × 8 datasets × seeds 42–46; 20 epochs.
Configuration selection used only validation data, with tuning seeds 101/102.
Each formal checkpoint was selected on validation before evaluating test.
Table cells report test accuracy mean ± sample standard deviation.
The eight-task average gives equal weight to each dataset.
Historical baselines must be identified as quoted results; splits were reconstructed and fixed as documented in README.md.

| Backbone | Eight-task average |
|---|---:|
| ViT-base | 86.91 ± 0.17 |
| ViT-large | 88.58 ± 0.13 |
