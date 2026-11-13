#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Rebuild and rerun the whole study (CPU only, 8 threads, about 10 minutes, reads ~0.4 GiB of the container).
#   sh run_all.sh CONTAINER.wex        (Windows path of the WEX3 production container; it is only read)
# Needs: gcc (MSYS2 UCRT64) on PATH, python 3 with numpy as $PY (default: python). Writes out/ and results.json.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p out/stage/tools
export TMP="$PWD/out" TEMP="$PWD/out"            # gcc cannot create temporary files under C:\WINDOWS
cp ../../../c/*.h out/stage/ && cp ../../../c/tools/wex_nr3_tools.c out/stage/tools/
gcc -O2 -shared out/stage/tools/wex_nr3_tools.c -o out/wex_nr3_tools.dll
gcc -O2 -Wall -Wextra -shared ctxlib.c -o out/ctxlib.dll
gcc -O2 -mavx2 decbench.c -o out/decbench.exe
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
"$PY" -I ctxeval.py out decbench
