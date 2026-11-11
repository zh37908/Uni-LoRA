# Multi-teacher runtime environment

The preparation smoke job ran with Python 3.11.16, PyTorch 2.9.0+cu128,
NumPy 2.2.6, SciPy 1.17.1, and Matplotlib 3.11.1. The experiment uses
PyTorch CUDA 12.8 wheels on NVIDIA GPUs. The PEFT source snapshots are read as
Python source; this experiment does not import PEFT or Transformers at runtime.

From the repository root, create the environment:

```bash
conda env create -f synthetic/prolosa_multiteacher_20261109/environment.yml
conda activate unilora_multiteacher
export MULTITEACHER_PYTHON="$(command -v python)"
"$MULTITEACHER_PYTHON" -c 'import torch, numpy, scipy, matplotlib; print(torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, matplotlib.__version__)'
```

Check CUDA on an allocated GPU node, since login nodes may have no visible GPU:

```bash
"$MULTITEACHER_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The Slurm scripts use `MULTITEACHER_PYTHON` when set and otherwise use the
original cluster environment at
`/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python`. Submit from the
same shell after exporting the variable. Slurm's default environment export
passes it to GPU and analysis jobs. Adjust the account, partition, GPU type,
and repository path in the Slurm scripts for another cluster.

The source release omits the earlier binary artifacts needed to initialize
this experiment; see [README.md](README.md#required-local-artifacts).
