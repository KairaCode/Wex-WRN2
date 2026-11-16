#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Rebuild and rerun the whole study (CPU only, 8 threads, about 10 minutes, reads ~0.4 GiB of the container).
#   bash run_all.sh CONTAINER.wex      (the WEX3 production container of DeepSeek V4.1, 40 layers x 192 experts;
#                                       it is only read)
# Needs: a C compiler (gcc of MSYS2 UCRT64 on Windows), python 3 with numpy as $PY (default: python). Writes out/
# and results.json. CC names another compiler (default gcc), as in scripts/build_cpu_check.sh; nothing here uses
# OpenMP. decbench.c uses AVX2 intrinsics, so it is built with -mavx2 on x86-64 only (WEX4_MARCH overrides the flag,
# as in ../fast-cpu-decode/synth.sh); elsewhere it is not built and the last stage, decbench, is skipped.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python}
CC=${CC:-gcc}   # the C compiler of every build below
case $(uname -m) in
  x86_64|amd64) X86=1; MARCH=${WEX4_MARCH--mavx2} ;;
  *) X86=; MARCH= ;;
esac
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent, and ctxlib.c, which marks
# its exports with __declspec(dllexport), gets that word defined away, as in build_cpu_check.sh
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; FAKE_EXPORT="" ;;
  *) SO="-shared -fPIC"; FAKE_EXPORT="-D__declspec(x)=" ;;
esac
echo "compiler: $("$CC" --version | sed -n 1p)"
mkdir -p out/stage/tools
export TMP="$PWD/out" TEMP="$PWD/out"            # gcc cannot create temporary files under C:\WINDOWS
cp ../../../c/*.h out/stage/ && cp ../../../c/tools/wex_nr3_tools.c out/stage/tools/
"$CC" -O2 $SO out/stage/tools/wex_nr3_tools.c -o out/wex_nr3_tools.dll
"$CC" -O2 -Wall -Wextra $SO $FAKE_EXPORT ctxlib.c -o out/ctxlib.dll
if [ -n "$X86" ]; then "$CC" -O2 $MARCH decbench.c -o out/decbench.exe; fi
[ -f out/manifest.json ] || "$PY" -I extract.py "$1" out
rm -f results.json
"$PY" -I ctxeval.py out validate
"$PY" -I ctxeval.py out contexts
"$PY" -I ctxeval.py out layers
"$PY" -I ctxeval.py out precision
"$PY" -I ctxeval.py out combo =SREL =SREL+RMAX =SREL+PM1 =RMAX =PM1+RMAX =SREL+RMAXC =SREL+POSC =SREL5 =SREL+ROWLVL \
    =SREL+RMAX+ROWLVL =SREL+BMAX3 =SREL+BMAX3+RMAXC ROWLVL COLLVL SROW ROWLVL+COLLVL ROWLVL+POSC SROW+POSC \
    ROWLVL+ZC4 ROWLVL+BIG ROWLVL+DNEXT SROW+ZC4 SROW+DNEXT SROW+POSC+ZC4 ROWLVL+POSC+ZC4 SROW+POS \
    SROW+POSC+BIG+DNEXT =SREL5+PM1+RMAX+ROWLVL =SREL+RMAX+ROWLVL+POSC
"$PY" -I ctxeval.py out kmix SREL+PM1+RMAX
"$PY" -I ctxeval.py out kmix SREL
"$PY" -I ctxeval.py out kmix SREL+RMAX+ROWLVL+POSC
if [ -n "$X86" ]; then
  "$PY" -I ctxeval.py out decbench
else
  echo "decbench.c needs AVX2 (x86-64): decbench skipped"
fi
echo "run_all.sh passed"
