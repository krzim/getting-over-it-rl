#!/usr/bin/env bash
# Run a command on CUDA device 0 through the project-local .venv-cuda (jax[cuda13] 0.12.0). Every job holds a GPU lock
# while it computes, so jobs take turns on the card; a shared machine sets the lock and turn-taking with another
# project in scripts/gpu_env.local.sh (git-ignored; goi_jax/gpu_lock.py lists the settings).
#   scripts/cuda_run.sh python -c "import jax; print(jax.devices())"
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO_ROOT/scripts/gpu_env.local.sh" ]] && source "$REPO_ROOT/scripts/gpu_env.local.sh"
[[ -x "$REPO_ROOT/.venv-cuda/bin/python" ]] || { echo "error: no .venv-cuda" >&2; exit 1; }
export PATH="$REPO_ROOT/.venv-cuda/bin:$PATH"
# Always the GPU: an inherited JAX_PLATFORMS=cpu once made A/B runs train on the CPU for 15 min while holding the lock.
export JAX_PLATFORMS="cuda,cpu"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
# No while-loop unrolling on the GPU: XLA's unroller copies the bodies of the TOI event loop (several Pallas kernels
# each) and the compile of the training program then takes over an hour; with it off the compile is ~3 min, the
# results are bit-identical and the run time is the same or better (runs/toi_kernel/*_nounroll.log). Only this
# project's CUDA jobs see it; set XLA_FLAGS with another --xla_gpu_enable_while_loop_unrolling value to override.
case " ${XLA_FLAGS:-} " in
  *--xla_gpu_enable_while_loop_unrolling=*) ;;
  *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_NO_UNROLL" ;;
esac
# XLA command buffers (CUDA graphs, incl. data-dependent while loops and conditionals) replay recorded kernel sequences on
# the device instead of launching each kernel from the host: +11% end-to-end training throughput at 16,384 envs
# (runs/speed/cmdbuf_ab.log). Set XLA_FLAGS with another --xla_gpu_enable_command_buffer value to override.
case " ${XLA_FLAGS:-} " in
  *"--xla_gpu_enable_command_buffer="*) ;;
  *) export XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1" ;;
esac
# GOI_LATE_LOCK=1: the program compiles first and takes the lock itself only around GPU work (goi_jax/gpu_lock.py), so a
# long compile never holds the card idle while other jobs wait.
if [[ "${GOI_LATE_LOCK:-0}" == 1 ]]; then exec env -u LD_LIBRARY_PATH "$@"; fi
# Wait for our turn when sharing with a peer project (goi_jax/gpu_lock.py), hold the lock for this one job, then hand
# the turn over.
(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m goi_jax.gpu_lock wait)
flock "${GOI_GPU_LOCK:-/tmp/goi_gpu.lock}" env -u LD_LIBRARY_PATH "$@"
rc=$?
(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m goi_jax.gpu_lock done)
exit $rc
