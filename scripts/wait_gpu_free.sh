#!/usr/bin/env bash
# Block until the AMD GPU's used VRAM is below $1 MB (default 4000), i.e. no other JAX job holds it; give up after $2 s.
set -euo pipefail
limit="${1:-4000}"; deadline=$(( $(date +%s) + ${2:-3600} ))
while :; do
  used="$("$(dirname "$0")/amdstat.sh" | sed -n 's/^USED_VRAM: \([0-9]*\) MB/\1/p')"
  [[ -n "$used" && "$used" -lt "$limit" ]] && { echo "GPU free: ${used} MB used"; exit 0; }
  (( $(date +%s) > deadline )) && { echo "STOPPED: GPU still busy (${used} MB used)"; exit 1; }
  sleep 20
done
