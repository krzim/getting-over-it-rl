#!/usr/bin/env bash
# Run a command on the AMD RX 9070 XT through ROCm, using the project-local .venv-rocm (ROCm 10.1 wheels + jax 0.12.0
# with jax-rocm10-plugin). Nothing global is changed, so the CUDA setup and its GPU are untouched.
# Needs /dev/dxg (the WSL GPU device).
#   scripts/rocm_run.sh python -c "import jax; print(jax.devices())"
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO_ROOT/scripts/gpu_env.local.sh" ]] && source "$REPO_ROOT/scripts/gpu_env.local.sh"
[[ -x "$REPO_ROOT/.venv-rocm/bin/python" ]] || { echo "error: no .venv-rocm; see the README's ROCm notes" >&2; exit 1; }
export PATH="$REPO_ROOT/.venv-rocm/bin:$PATH"
export JAX_PLATFORMS="${JAX_PLATFORMS:-rocm,cpu}"
export HSA_ENABLE_DXG_DETECTION=1
# The card has 16 GB and also drives the display: cap JAX at ~11 GB (it preallocates this fraction up front)
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
# One job at a time on the card, holding this lock for its lifetime (GOI_ROCM_LOCK, scripts/gpu_env.local.sh).
exec flock "${GOI_ROCM_LOCK:-/tmp/goi_rocm.lock}" env -u LD_LIBRARY_PATH -u CUDA_VISIBLE_DEVICES "$@"
