#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# CPU check of the WNR4 codecs, no GPU and no real model needed: builds and runs wnr4/tests/test_wnr4.c (kinds MX
# and NV, wnr4/wnr4.h) and wnr4/tests/test_wnr4_gg.c (kind GG, wnr4/wnr4_gg.h): round trips over G 1..256,
# refusals and a mutation fuzz each. Everything built goes to out/wnr4-check.
# CC names the compiler (default gcc), WEX_OPENMP replaces the OpenMP flag (default -fopenmp; WEX_OPENMP=-fno-openmp
# builds without OpenMP: the parallel loops of the codecs are guarded by _OPENMP and then run on one thread),
# WEX_CFLAGS adds flags to every call (the sanitizer CI jobs put theirs there). On x86-64 -march=haswell builds the
# AVX2 table search of wnr4.h; WEX4_MARCH replaces that flag (WEX4_MARCH= builds the scalar search).
# Usage: bash scripts/build_wnr4_check.sh     (last line on success: WNR4 CPU check passed)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=$ROOT/out/wnr4-check
rm -rf "$O" && mkdir -p "$O"
# gcc from Git Bash cannot create temporary files under C:\WINDOWS
export TMP="$ROOT/out" TEMP="$ROOT/out"
CC=${CC:-gcc}
OMP=${WEX_OPENMP:--fopenmp}
case $(uname -m) in
  x86_64|amd64) MARCH=${WEX4_MARCH--march=haswell} ;;
  *) MARCH=${WEX4_MARCH-} ;;
esac
echo "compiler: $("$CC" --version | sed -n 1p), -O2 $MARCH $OMP"
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $MARCH $OMP ${WEX_CFLAGS-}"
cd "$O"
for t in test_wnr4 test_wnr4_gg; do
  echo "== wnr4/tests/$t.c"
  "$CC" $W "$ROOT/wnr4/tests/$t.c" -o "$t.exe"
  "./$t.exe"
done
echo "WNR4 CPU check passed"
