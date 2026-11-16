#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The runs of this folder, from the repository root: bash research/cuda-review-20261115/run.sh
# It needs what wex4/fuzz/run.sh --replay needs (gcc, g++, python) and runs that first, for the seeds and the host
# copy of the CUDA source. Then fuzz_wnr3_cuda built twice: without mutator rounds (every file of the vectors'
# manifest, then the four files of impl/go-wex4/testdata, each input once; the harness's own 15 records run first
# in both) and with 100 rounds of the custom mutator on every seed; then the negative control. Logs: vectors.log,
# mutations.log and negative.log next to this script; the builds go to its out/ (git-ignored).
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
O=$ROOT/wex4/fuzz/out
cd "$ROOT"
mkdir -p "$HERE/out"
bash wex4/fuzz/run.sh --replay > "$HERE/out/replay.log" 2>&1 || { tail -20 "$HERE/out/replay.log"; exit 1; }
export TMP="$HERE/out" TEMP="$HERE/out"
W="-Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unknown-pragmas"
INC="-I $ROOT/c -I $ROOT/wex4/research/fast-cpu-decode -I $ROOT/wex4/fuzz/engine -I $ROOT/wex4/fuzz -I $O -pthread"
CXX=${CXX:-g++}
case $(uname -m) in x86_64|amd64) MARCH=-march=haswell ;; *) MARCH= ;; esac
for rounds in 0 100; do
  "$CXX" -std=c++17 -O2 $W $MARCH -DWEX_FUZZ_STANDALONE -DFUZZ_MUTATOR_ROUNDS=$rounds $INC wex4/fuzz/fuzz_wnr3_cuda.cpp \
      -o "$HERE/out/rounds$rounds"
done
# the bridge prints its "[wex] WNR3 record failed its integrity check" line once per build: left out of the logs
quiet() { "$@" 2>&1 | { grep -v '^\[wex\]' || true; }; }
{
  echo "compiler: $("$CXX" --version | sed -n 1p)"
  echo "== the files of wex4/vectors/MANIFEST.tsv"
  quiet "$HERE/out/rounds0" "$O"/seeds/file/*
  echo "== the files of impl/go-wex4/testdata"
  quiet "$HERE/out/rounds0" "$O"/seeds/testdata/*
} > "$HERE/vectors.log"
{
  echo "compiler: $("$CXX" --version | sed -n 1p)"
  echo "== every seed (wnr3, file, testdata), 100 inputs of the custom mutator from each"
  start=$(date +%s)
  quiet "$HERE/out/rounds100" "$O"/seeds/wnr3/* "$O"/seeds/file/* "$O"/seeds/testdata/*
  echo "wall time: $(( $(date +%s) - start )) s"
} > "$HERE/mutations.log"
"${PYTHON:-python}" -I wex4/fuzz/cuda_negative.py > "$HERE/negative.log" 2>&1
tail -1 "$HERE/vectors.log"; tail -2 "$HERE/mutations.log"; tail -1 "$HERE/negative.log"
