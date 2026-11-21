#!/usr/bin/env bash
# CPU tests and CPU decode speed under the heavy lock (quiet machine). LOCKSH = lock helper defining take_lock.
set -u
cd "$(dirname "$0")"
if [ -z "${NOLOCK:-}" ]; then   # NOLOCK=1: the caller already holds the lock
  if [ -n "${LOCKSH:-}" ]; then source "$LOCKSH"; else take_lock() { :; }; release_lock() { :; }; fi
  take_lock "wnr4 research: CPU tests + CPU decode bench (~5 min)" || exit 3
fi
./out/test_wnr4.exe 3000 > results/test_wnr4.log 2>&1
./out/bench_cpu.exe out/packs3/q38mx.pack 64 16 > results/cpu-q38mx.log 2>&1
./out/bench_cpu.exe out/packs3/q38nv.pack 64 16 > results/cpu-q38nv.log 2>&1
./out/bench_cpu.exe out/packs3/v41.pack 8 16 > results/cpu-v41.log 2>&1
echo cpu round done
