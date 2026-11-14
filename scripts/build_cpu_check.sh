#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# CPU check of the WNR3 sync, no GPU and no real model needed:
#   1. c/tests/test_wex_format.c (codec 5: hash vectors, round trips, mutations, corruption audit, container rules)
#      and c/tests/test_wex4_verify.c (c/tools/wex4_verify.c on small WEX4 files; c/tools/wex4_verify.c also alone,
#      compile only)
#   2. c/tools/wex_nr3_tools.c -> wex_nr3_tools.dll and c/tests/fake_wex_cuda.c -> fake_wex_cuda.dll (compile only)
#   3. c/tools/build_nr3.py smoke: a WEX1 of fixtures/v41-tiny (c/tools/repack_v4_wex.py), then WNR3-CZ bins and a
#      WNR3-CZ container from it, every record CPU-verified by the builder
#   4. with an engine c/ tree as $1: c/tests/test_expert_store_wex.c too (it needs expert_store_registry.c
#      and the engine headers, which are not in this repo)
# c/ has the colibri c/ layout, so the sources build where they are; everything built goes to out/cpu-check.
# Windows: MSYS2 MINGW64 or UCRT64 gcc on PATH, python 3 (stdlib only). Linux: gcc and PYTHON=python3.
# CC names another compiler (CC=clang, which needs its OpenMP runtime: CI runs that on Linux).
# Usage: bash scripts/build_cpu_check.sh [ENGINE_C]
set -eu
set -o pipefail   # the builder and the store test are piped into tail: their exit status must count
ROOT=$(cd "$(dirname "$0")/.." && pwd)
C=$ROOT/c
O=$ROOT/out/cpu-check
if [ $# -ge 1 ]; then K=$(cd "$1" && pwd); fi
rm -rf "$O" && mkdir -p "$O"
# gcc from Git Bash cannot create temporary files under C:\WINDOWS
export TMP="$ROOT/out" TEMP="$ROOT/out"
PY=${PYTHON:-python}
CC=${CC:-gcc}
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent, and the CPU stand-in
# helper, which marks its exports with __declspec(dllexport), gets that word defined away
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; FAKE_EXPORT="" ;;
  *) SO="-shared -fPIC"; FAKE_EXPORT="-D__declspec(x)=" ;;
esac
cd "$O"
# WEX_CFLAGS: extra flags for every compile and link, empty by default (the sanitizer CI job sets
# -fsanitize=address,undefined there)
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function ${WEX_CFLAGS-}"
"$CC" $W "$C/tests/test_wex_format.c" -o test_wex_format.exe
./test_wex_format.exe
"$CC" $W -fopenmp "$C/tests/test_wex4_verify.c" -o test_wex4_verify.exe
# it prints the expected failures, so only its summary line is shown; when it fails the whole output is, since
# a sanitizer report would be in it. its temporary .wex file stays here
./test_wex4_verify.exe > test_wex4_verify.log 2>&1 || { cat test_wex4_verify.log; exit 1; }
tail -1 test_wex4_verify.log
"$CC" $W -fopenmp -c "$C/tools/wex4_verify.c" -o wex4_verify.o
"$CC" $W $SO "$C/tools/wex_nr3_tools.c" -o wex_nr3_tools.dll
"$CC" $W $SO -static-libgcc $FAKE_EXPORT "$C/tests/fake_wex_cuda.c" -o fake_wex_cuda.dll
"$PY" -I "$C/tools/repack_v4_wex.py" --model-dir "$ROOT/fixtures/v41-tiny" --output tiny.wex > tiny.wex.log
P=$("$PY" -I -c "print(','.join(f'{l}:{e}' for l in range(6) for e in range(8)))")
"$PY" -I "$C/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex bins tiny_cz.bin --experts "$P" --train "$P" --cz | tail -2
"$PY" -I "$C/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex container tiny_cz.wex --layers 0,1,2,3,4,5 --train "$P" --cz | tail -2
if [ $# -ge 1 ]; then
  # the engine headers come from that engine tree: "x.h" through -I $K, a test's "../x.h" through -I $K/tests
  F="-D_FILE_OFFSET_BITS=64 $W -fopenmp"
  "$CC" $F -I "$K" -c "$K/expert_store_registry.c" -o r.o
  "$CC" $F -I "$K" -c "$C/expert_store_wex.c" -o w.o
  "$CC" $F -c "$C/wex_decode.c" -o d.o
  "$CC" $F -I "$K" -I "$K/tests" "$C/tests/test_expert_store_wex.c" r.o w.o d.o -o test_expert_store_wex.exe -pthread
  ./test_expert_store_wex.exe | tail -1
fi
echo "WNR3 CPU check passed"
