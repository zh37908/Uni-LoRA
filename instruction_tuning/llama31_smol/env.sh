#!/usr/bin/env bash
# Source this after selecting a Python environment. Large artifacts require an
# explicit storage directory; no implicit downloads into a home directory.
: "${UNILORA_STORAGE:?Set UNILORA_STORAGE to your group-storage directory first}"
_pkg_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_repo_dir=$(cd -- "$_pkg_dir/../.." && pwd)
mkdir -p "$UNILORA_STORAGE"
export UNILORA_STORAGE=$(cd -- "$UNILORA_STORAGE" && pwd -P)
export PYTHONPATH="$_repo_dir/math_instruction_tuning/peft/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$UNILORA_STORAGE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="$UNILORA_STORAGE/cache/xdg"
export TORCH_HOME="$UNILORA_STORAGE/cache/torch"
export PIP_CACHE_DIR="$UNILORA_STORAGE/cache/pip"
export PYTHONPYCACHEPREFIX="$UNILORA_STORAGE/cache/pycache"
export NLTK_DATA="$UNILORA_STORAGE/cache/nltk_data"
export CUDA_CACHE_PATH="$UNILORA_STORAGE/cache/cuda"
export TRITON_CACHE_DIR="$UNILORA_STORAGE/cache/triton"
export TMPDIR="$UNILORA_STORAGE/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$UNILORA_STORAGE/logs" "$UNILORA_STORAGE/models" "$UNILORA_STORAGE/datasets" "$UNILORA_STORAGE/manifests"
# A login-node loopback proxy is not reachable on another compute node.
if [[ ${UNILORA_NETWORK_MODE:-direct} == direct ]]; then
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi
unset _pkg_dir _repo_dir
