# Uni-LoRA / ProLoSA research workspace

This branch packages the current ICLR 2027 manuscript, custom PEFT implementations,
training/evaluation code, frozen experiment configurations, and compact result
summaries. Model weights, downloaded datasets, checkpoints, caches, and raw job
outputs remain outside Git.

- [Two-server, two-week fine-tuning plan](ICLR_2027/two_server_finetuning_plan_20260914.md)
- [Second-server checkout and migration guide](docs/second_server_git_setup_20260914.md)
- [Current manuscript](ICLR_2027/iclr2027_conference.pdf)
- [Ongoing controlled experiments](ICLR_2027/revision/new_experiments/README.md)
- [Observed Python environment](docs/environment_unilora_modern_20260914.json)

The two-server plan is a design, not a claim that its new instruction pipeline
has already been implemented or validated. Existing Slurm launchers contain
source-cluster paths and settings; adapt them before submitting on another cluster.
Use the repository-local PEFT backend required by each experiment.
