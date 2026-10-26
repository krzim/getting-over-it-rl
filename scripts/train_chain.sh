#!/usr/bin/env bash
# Run a resumable training command in N chunks (each holds the shared GPU lock only for its own --max-minutes, so other
# GPU users get turns in between). Stops early on DONE or on a STOPPED line that isn't the time budget.
# A --resume PATH in the command applies to the first chunk only: later chunks continue from the run's own latest.pkl
# (train_jax.py's default), not from the original checkpoint again.
#   scripts/train_chain.sh 4 runs/coffee-1.log scripts/cuda_run.sh python -u scripts/train_jax.py --name coffee-1 ...
set -uo pipefail
n="$1"; log="$2"; shift 2
first=("$@")
later=()
skip=0
for a in "$@"; do
  if (( skip )); then skip=0; continue; fi
  if [[ "$a" == "--resume" ]]; then skip=1; continue; fi
  if [[ "$a" == "--fresh" ]]; then continue; fi
  later+=("$a")
done
for i in $(seq 1 "$n"); do
  echo "=== chunk $i/$n $(date +%T)" >> "$log"
  before="$(grep -cE '^(DONE|STOPPED)' "$log")"
  if (( i == 1 )); then "${first[@]}" >> "$log" 2>&1; else "${later[@]}" >> "$log" 2>&1; fi
  # a chunk that crashed without printing its own DONE/STOPPED line must stop the chain, not reuse an older line
  (( $(grep -cE '^(DONE|STOPPED)' "$log") > before )) || { echo "STOPPED: chunk $i ended without a DONE/STOPPED line"; exit 1; }
  last="$(grep -E '^(DONE|STOPPED)' "$log" | tail -1)"
  case "$last" in DONE*) echo "DONE: training finished"; exit 0 ;; *"time budget"*) ;; *) echo "STOPPED: $last"; exit 1 ;; esac
done
echo "DONE: $n chunks"
