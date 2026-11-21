#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The CPU checks that need only a C compiler and Python (no GPU, no model, no engine tree), in one command: the
# repository check, five standard-library tests (two of them the GGUF converter and the WNR4-GG tool, which builds its
# library with CC), the WEX1-3/WNR3 CPU check, the WNR4 CPU tests, the WEX4 tests with the conformance vectors, and the
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
for t in scripts/test_verify_wex1.py c/tests/test_repack_v4_wex.py c/tests/test_build_nr3_picks.py \
         wex4/test_gguf2wex4.py wex4/test_wnr4gg.py; do
  echo "== $t"; "$PY" -I "$t"
done
bash scripts/build_cpu_check.sh
bash scripts/build_wnr4_check.sh
bash wex4/build_and_test.sh
bash wex4/research/fast-cpu-decode/synth.sh check
echo "all checks passed"
