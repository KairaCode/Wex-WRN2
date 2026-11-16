#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The CPU checks that need only a C compiler and Python (no GPU, no model, no engine tree), in one command: the
# repository check, three standard-library tests, the WEX1-3/WNR3 CPU check, the WEX4 tests with the conformance
# vectors, the fast WNR3 decoder's own harness, and the GLM-5.3 test kit on a small synthetic checkpoint
# (testkits/glm53-wex4); the entry point of README, CONTRIBUTING and CI. Not here: the
# fuzz replay (wex4/fuzz/run.sh --replay, which also needs g++), the smoke run of bench/wex_io.py, the Go and the
# Rust reader, the sanitizer runs and the benchmarks; CI runs them as steps and jobs of their own (scripts/README.md).
# Windows: MSYS2 UCRT64 or MINGW64 gcc and python on PATH. Linux: gcc and PYTHON=python3.
# macOS: PYTHON=python3 and a GNU gcc in CC (Homebrew: CC=gcc-15); the scripts it calls read CC, default gcc.
# Apple clang instead: CC=clang and the libomp of Homebrew in WEX_OPENMP (scripts/build_cpu_check.sh shows the line).
# Usage: bash scripts/check_all.sh     (last line on success: all checks passed)
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" -I check_repo.py --self-test
"$PY" -I check_repo.py
for t in scripts/test_verify_wex1.py c/tests/test_repack_v4_wex.py c/tests/test_build_nr3_picks.py; do
  echo "== $t"; "$PY" -I "$t"
done
bash scripts/build_cpu_check.sh
bash wex4/build_and_test.sh
bash wex4/research/fast-cpu-decode/synth.sh check
# the GLM-5.3-Flash test kit, its stages on a small synthetic checkpoint (the GPU parts run only where a GPU is)
rm -rf out/kit
bash testkits/glm53-wex4/run.sh --synthetic-small --work out/kit --results out/kit/results \
    env fetch build verify roundtrip decode engine report > out/kit.log 2>&1 || { tail -40 out/kit.log; exit 1; }
grep -E '^\[(build|verify|roundtrip|decode|engine)\] ok' out/kit.log
echo "all checks passed"
