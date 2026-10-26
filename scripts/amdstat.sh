#!/usr/bin/env bash
# gpustat-style one-shot view of the AMD RX 9070 XT (gpustat only speaks NVIDIA's NVML). Uses the amd-smi that ships
# with the ROCm wheels in .venv-rocm (needs /dev/dxg, the WSL GPU device).
#   scripts/amdstat.sh          # usage + VRAM + processes
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMI="$REPO_ROOT/.venv-rocm/lib/python3.13/site-packages/_rocm_sdk_core/bin/amd-smi"
[[ -x "$SMI" ]] || { echo "error: no amd-smi in .venv-rocm" >&2; exit 1; }
export HSA_ENABLE_DXG_DETECTION=1
env -u LD_LIBRARY_PATH "$SMI" metric -u -m 2>/dev/null | grep -E "GFX_ACTIVITY|UMC_ACTIVITY|TOTAL_VRAM|USED_VRAM|FREE_VRAM" | sed 's/^ *//'
env -u LD_LIBRARY_PATH "$SMI" process 2>/dev/null | grep -E "NAME|PID|VRAM_MEM" | sed 's/^ *//' || true
