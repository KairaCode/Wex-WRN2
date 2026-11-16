#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build and run the WNR4 record experiment (CPU only, at most 8 threads).
#   bash run.sh CONTAINER    the WEX3 WNR3-CZ production container of DeepSeek V4.1 (40 layers x 192 experts), or
#                            the published file, which is that container upgraded to WEX4 in place
# Needs a C compiler with OpenMP (wnr4_proto.c includes omp.h; gcc of MSYS2 UCRT64 on Windows) and python 3 with
# numpy on PATH; set PY to choose the interpreter. CC names another compiler (default gcc) and WEX_OPENMP replaces
# -fopenmp, as in scripts/build_cpu_check.sh. No -march flag: nothing here has SIMD kernels.
# The container is only read (about 1.5 GB in total: 76 records, every record header and length table);
# everything written goes under out/ next to this script (out/cache holds ~1.2 GB of record copies).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PY=${PY:-python}
CC=${CC:-gcc}                  # the C compiler of every build below
OMP=${WEX_OPENMP:--fopenmp}    # the OpenMP flags
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent, as in build_cpu_check.sh
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared" ;;
  *) SO="-shared -fPIC" ;;
esac
cd "$HERE"
echo "compiler: $("$CC" --version | sed -n 1p)"
mkdir -p out/stage/tools out/cache
export TMP="$HERE/out" TEMP="$HERE/out"       # gcc cannot create temporary files under C:\WINDOWS
cp "$ROOT"/c/*.h out/stage/ && cp "$ROOT"/c/tools/wex_nr3_tools.c out/stage/tools/   # "../wex_format.h" resolves
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"
"$CC" $W $OMP -Iout/stage/tools wnr4_proto.c -o out/wnr4_proto.exe -lm
"$CC" $W $SO out/stage/tools/wex_nr3_tools.c -o out/wex_nr3_tools.dll
# one expert per layer (expert = (37 * layer + 11) mod 192) plus L39 E120, a known large record
EXPERTS=$("$PY" -I -c "print(','.join(f'{l}:{(l*37+11)%192}' for l in range(40)) + ',39:120')")
if [ $# -ge 1 ]; then
  "$PY" -I fetch.py "$1" out/cache records "$EXPERTS"
  "$PY" -I fetch.py "$1" out/cache scan out/container_scan.json
fi
OMP_NUM_THREADS=8 ./out/wnr4_proto.exe out/cache out/proto.json $(echo "$EXPERTS" | tr ',' ' ')
if [ $# -ge 1 ]; then "$PY" -I layer_model.py "$1" out/wex_nr3_tools.dll out/layer_model.json out/cache; fi
"$PY" -I report.py out/proto.json out/container_scan.json out/layer_model.json results.json out/cache
