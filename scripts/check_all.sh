#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The CPU checks that need only a C compiler and Python (no GPU, no model, no engine tree), in one command: the
# repository check, six standard-library tests (three of them the GGUF converter and the WNR4-GG and WNR4-I4 tools, the
# last two build their library with CC), the WEX1-3/WNR3 CPU check, the WNR4 CPU tests, the WEX4 tests with the conformance vectors, and the
# fast WNR3 decoder's own harness; the entry point of README, CONTRIBUTING and CI.
# Not here: the fuzz replay (wex4/fuzz/run.sh --replay, which also needs g++), the smoke run of bench/wex_io.py, the
# Go and the Rust reader, the sanitizer runs and the benchmarks; CI runs them as steps and jobs of their own
# (scripts/README.md).
# Windows: MSYS2 UCRT64 or MINGW64 gcc and python on PATH. Linux: gcc and PYTHON=python3.
# macOS: PYTHON=python3 and a GNU gcc in CC (Homebrew: CC=gcc-15); the scripts it calls read CC, default gcc.
# Apple clang instead: CC=clang and the libomp of Homebrew in WEX_OPENMP (scripts/build_cpu_check.sh shows the line).
# Usage: bash scripts/check_all.sh     (last line on success: all checks passed)
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" -I check_repo.py --self-test
"$PY" -I check_repo.py
PY_TESTS=(
  scripts/test_verify_wex1.py
  c/tests/test_repack_v4_wex.py
  c/tests/test_build_nr3_picks.py
  wex4/test_gguf2wex4.py
  wex4/test_wnr4gg.py
  wex4/test_wnr4i4.py
)
if [ -n "${WEX_NO_PARALLEL-}" ]; then
  for t in "${PY_TESTS[@]}"; do
    echo "== $t"; "$PY" -I "$t"
  done
  bash scripts/build_cpu_check.sh
  bash scripts/build_wnr4_check.sh
  bash wex4/build_and_test.sh
  bash wex4/research/fast-cpu-decode/synth.sh check
else
  LOGDIR=$(mktemp -d 2>/dev/null || mktemp -d -t wex-check-all)
  trap 'rm -rf "$LOGDIR"' EXIT
  (
    py_pids=()
    for i in "${!PY_TESTS[@]}"; do
      t=${PY_TESTS[$i]}
      (echo "== $t"; "$PY" -I "$t") > "$LOGDIR/py_$i.log" 2>&1 &
      py_pids+=($!)
    done
    prc=0
    for i in "${!PY_TESTS[@]}"; do
      s=0; wait "${py_pids[$i]}" || s=$?
      cat "$LOGDIR/py_$i.log"
      if [ "$s" != 0 ]; then prc=$s; fi
    done
    exit "$prc"
  ) > "$LOGDIR/0_py.log" 2>&1 & p0=$!
  bash scripts/build_cpu_check.sh > "$LOGDIR/1_cpu.log" 2>&1 & p1=$!
  bash scripts/build_wnr4_check.sh > "$LOGDIR/2_wnr4.log" 2>&1 & p2=$!
  (
    bash wex4/build_and_test.sh
    bash wex4/research/fast-cpu-decode/synth.sh check
  ) > "$LOGDIR/3_wex4.log" 2>&1 & p3=$!

  rc=0
  for pair in "$p0:0_py.log" "$p1:1_cpu.log" "$p2:2_wnr4.log" "$p3:3_wex4.log"; do
    pid=${pair%%:*}
    lf=${pair#*:}
    s=0; wait "$pid" || s=$?
    cat "$LOGDIR/$lf"
    if [ "$s" != 0 ]; then rc=$s; fi
  done
  [ "$rc" = 0 ] || exit "$rc"
fi
echo "all checks passed"
