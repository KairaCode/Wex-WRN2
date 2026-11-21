#!/usr/bin/env bash
# GPU round + CPU round under one hold of the heavy lock. LOCKSH = the lock helper (take_lock / release_lock).
set -u
cd "$(dirname "$0")"
if [ -n "${LOCKSH:-}" ]; then source "$LOCKSH"; else take_lock() { :; }; release_lock() { :; }; fi
take_lock "wnr4 research: GPU bench round + CPU tests (~25 min)" || exit 3
NOLOCK=1 bash run_gpu_round.sh
NOLOCK=1 bash run_cpu_round.sh
release_lock
echo round8 done
