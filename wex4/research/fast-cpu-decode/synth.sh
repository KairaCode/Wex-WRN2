#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The harness on synthetic records, so it runs with no real model: the 4 WNR3 records of wex4/out/synth_nr3.wex
# (written by wex4/build_and_test.sh from fixtures/make_synth_mxfp4.py; 208,896 decoded bytes each).
# Two builds into out/synth/: t_simd.exe with the SIMD kernels of this CPU (AVX2 on x86 through -march=haswell,
# NEON on AArch64; WEX4_MARCH overrides the flag as in wex4/build_and_test.sh) and t_scalar.exe with
# -DWEX_FAST_SCALAR (kernel 0 only).
#   bash synth.sh check [CASES]   both builds: `equal` on the 4 records, then `mutate` with CASES mini-records
#                                 (default 600). Last line on success: synth check passed
#   bash synth.sh speed [RUNS]    both builds: the `speed` table (one thread, median of RUNS passes, default 21)
#   bash synth.sh lanes [CASES]   the SIMD build once for each -DWEX_FAST_LANES (1, 2, 3, 4, 6, 8): `equal`, then
#                                 `mutate` with CASES mini-records (default 200); 5 must not compile. Last line on
#                                 success: synth lanes passed
# Usage: after bash wex4/build_and_test.sh, from any directory. gcc and python on PATH (PYTHON=python3 on Linux);
# CC names another compiler and WEX_OPENMP other OpenMP flags, as in wex4/build_and_test.sh (macOS: a GNU gcc of
# Homebrew, or Apple clang with the libomp of Homebrew in WEX_OPENMP).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PY=${PYTHON:-python}
CC=${CC:-gcc}   # the C compiler, as in wex4/build_and_test.sh
OMP=${WEX_OPENMP:--fopenmp}   # the OpenMP flags, as there
MODE=${1:-check}
O=$HERE/out/synth
C=$ROOT/wex4/out/synth_nr3.wex
if [ ! -f "$C" ]; then echo "$C is missing: run bash wex4/build_and_test.sh first" >&2; exit 1; fi
rm -rf "$O" && mkdir -p "$O"
export TMP="$O" TEMP="$O"   # gcc from Git Bash cannot create temporary files under C:\WINDOWS
case $(uname -m) in
  x86_64|amd64|i?86) MARCH=${WEX4_MARCH--march=haswell} ;;
  *) MARCH=${WEX4_MARCH-} ;;
esac
W="-std=c11 -Wall -Wextra -Wno-unused-function ${WEX_CFLAGS-} -I$ROOT/c -I$HERE"
"$CC" $W -O3 $MARCH $OMP "$HERE/test_fast_decode.c" -o "$O/t_simd.exe"
"$CC" $W -O3 $OMP -DWEX_FAST_SCALAR "$HERE/test_fast_decode.c" -o "$O/t_scalar.exe"
"$PY" -I "$HERE/extract.py" "$C" "$O" 0:0 0:1 0:2 0:3 > "$O/extract.log"
cd "$O"
R="L0_E0.rec L0_E1.rec L0_E2.rec L0_E3.rec"
case $MODE in
  check)
    CASES=${2:-600}
    echo "compiler: $("$CC" --version | sed -n 1p)"
    for t in t_simd t_scalar; do
      echo "== $t equal"
      ./$t.exe equal model.wnm2 $R
      echo "== $t mutate $CASES"
      ./$t.exe mutate model.wnm2 "$CASES" 20261114 $R
    done
    echo "synth check passed"
    ;;
  speed)
    RUNS=${2:-21}
    echo "machine: $(uname -s) $(uname -m), $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?') logical CPUs"
    if [ "$(uname -s)" = Darwin ]; then sysctl machdep.cpu.brand_string hw.model 2>/dev/null || true; fi
    if command -v lscpu > /dev/null 2>&1; then lscpu | grep -E '^(Model name|Vendor ID|CPU part|BogoMIPS)' || true; fi
    echo "compiler: $("$CC" --version | head -1), -O3 $MARCH $OMP"
    for t in t_simd t_scalar; do
      echo "== $t speed $RUNS"
      ./$t.exe speed model.wnm2 "$RUNS" $R
    done
    ;;
  lanes)
    CASES=${2:-200}
    echo "compiler: $("$CC" --version | sed -n 1p)"
    # every build has the same variants, which check already compares; TEST_DEFAULT_ONLY leaves k0 and the
    # default entry point, whose number of streams is the one thing WEX_FAST_LANES changes
    for n in 1 2 3 4 6 8; do
      "$CC" $W -O3 $MARCH $OMP -DWEX_FAST_LANES=$n -DTEST_DEFAULT_ONLY "$HERE/test_fast_decode.c" -o "t_lanes$n.exe"
      echo "== WEX_FAST_LANES=$n equal, mutate $CASES"
      ./t_lanes$n.exe equal model.wnm2 $R > "lanes$n.log" || { cat "lanes$n.log"; exit 1; }
      cat "lanes$n.log"
      # the line proves that the build took the value: the default entry point was compared with it
      grep -q "^equal: .* + default (WEX_FAST_LANES $n), 0 mismatches" "lanes$n.log"
      ./t_lanes$n.exe mutate model.wnm2 "$CASES" 20261115 $R
    done
    if "$CC" $W -O3 $MARCH $OMP -DWEX_FAST_LANES=5 -c "$HERE/test_fast_decode.c" -o t_lanes5.o > lanes5.log 2>&1; then
      echo "-DWEX_FAST_LANES=5 compiled; it must not" >&2; exit 1
    fi
    grep -q 'WEX_FAST_LANES must be' lanes5.log
    echo "synth lanes passed"
    ;;
  *) echo "usage: bash synth.sh check [CASES] | speed [RUNS] | lanes [CASES]" >&2; exit 2 ;;
esac
