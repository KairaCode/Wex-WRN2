#!/usr/bin/env bash
# Full ratio measurement, 2048 held-out records per container (256 training records), each run under the
# shared heavy lock (CPU-heavy, large reads), released between runs so GPU sessions can interleave.
# LOCKSH = a lock helper defining take_lock; RUNS = which runs (default: all).
set -u
cd "$(dirname "$0")"
run() {   # dataset out.json args...
  ( if [ -n "${LOCKSH:-}" ]; then source "$LOCKSH"; else take_lock() { :; }; release_lock() { :; }; fi; take_lock "wnr4 research: ratio run $2 (CPU, reads)" || exit 3
    python -I measure.py "$@" > "${2%.json}.log" 2>&1 )
}
want() { [ -z "${RUNS:-}" ] || [[ " $RUNS " == *" $1 "* ]]; }
want q38mx-cz && run q38mx results/measure-q38mx-cz.json --cz 1 --pb 5 --threads 12 --G 32,64,128,256
want q38nv-exact && run q38nv results/measure-q38nv-exact.json --cz 0 --pb 5 --threads 12 --G 32,64,128,256
want v41-cz && run v41 results/measure-v41-cz.json --cz 1 --pb 5 --threads 12 --G 64,128,256
want q38mx-exact && run q38mx results/measure-q38mx-exact.json --cz 0 --pb 5 --threads 12 --G 128
want q38nv-cz && run q38nv results/measure-q38nv-cz.json --cz 1 --pb 5 --threads 12 --G 128
want q38mx-cz-pb1 && run q38mx results/measure-q38mx-cz-pb1.json --cz 1 --pb 1 --threads 12 --G 128 --wnr3 0
echo all done
