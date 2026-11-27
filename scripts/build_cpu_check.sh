#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# CPU check of the WNR3 sync, no GPU and no real model needed:
#   1. c/tests/test_wex_format.c (codec 5: hash vectors, round trips, mutations, corruption audit, container rules)
#      and c/tests/test_wex4_verify.c (c/tools/wex4_verify.c on small WEX4 files; c/tools/wex4_verify.c also alone,
#      run on 6 vectors), then c/tests/test_wex_fast_decode.c (the fast WNR3 decoder c/wex_fast_decode.h, and
#      coli_wex_decode_record of c/wex_decode.c with COLI_WEX_FAST_DECODE unset, 1 and 0, against the reference
#      decoder on synthetic records). On x86-64 in three builds: -march=haswell (the AVX2 kernel compiled for the
#      target), the x86-64 baseline (the AVX2 kernel behind a target attribute, chosen at run time by CPUID) and
#      -DWEX_FAST_SCALAR (kernel 0 only); elsewhere in two: the default (NEON on ARM64) and the scalar one
#   2. c/tools/wex_nr3_tools.c -> wex_nr3_tools.dll and c/tests/fake_wex_cuda.c -> fake_wex_cuda.dll (compile only)
#   3. c/tools/build_nr3.py smoke: a WEX1 of fixtures/v41-tiny (c/tools/repack_v4_wex.py), then WNR3-CZ bins and a
#      WNR3-CZ container from it, every record CPU-verified by the builder
#   4. with an engine c/ tree as $1: c/tests/test_expert_store_wex.c too (it needs expert_store_registry.c
#      and the engine headers, which are not in this repo)
# c/ has the colibri c/ layout, so the sources build where they are; everything built goes to out/cpu-check.
# Windows: MSYS2 MINGW64 or UCRT64 gcc on PATH, python 3 (stdlib only). Linux: gcc and PYTHON=python3.
# macOS: a GNU gcc in CC (Homebrew: CC=gcc-15) and PYTHON=python3; the gcc of Xcode is clang and has no OpenMP.
# CC names another compiler (CC=clang, which needs its OpenMP runtime: CI runs that on Linux).
# WEX_OPENMP replaces the OpenMP flag (default -fopenmp). Apple clang refuses -fopenmp and has no OpenMP runtime;
# with P=$(brew --prefix libomp), the OpenMP runtime of Homebrew, the macos-clang CI job runs CC=clang and
# WEX_OPENMP="-Xclang -fopenmp -I$P/include -L$P/lib -lomp -Wno-unused-command-line-argument" (the last flag:
# the compile-only and preprocessor calls would warn that -L and -lomp are not used)
# Usage: bash scripts/build_cpu_check.sh [ENGINE_C]
set -eu
set -o pipefail   # the builder and the store test are piped into tail: their exit status must count
ROOT=$(cd "$(dirname "$0")/.." && pwd)
C=$ROOT/c
O=$ROOT/out/cpu-check
if [ $# -ge 1 ]; then K=$(cd "$1" && pwd); fi
rm -rf "$O" && mkdir -p "$O"
# gcc from Git Bash cannot create temporary files under C:\WINDOWS
export TMP="$O" TEMP="$O"
PY=${PYTHON:-python}
CC=${CC:-gcc}   # the C compiler of every build below
OMP=${WEX_OPENMP:--fopenmp}   # the OpenMP flags of the builds that use it
echo "compiler: $("$CC" --version | sed -n 1p)"
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent, and the CPU stand-in
# helper, which marks its exports with __declspec(dllexport), gets that word defined away. The helper is linked
# with -static-libgcc except on macOS, where Apple clang stops at that option ("unsupported option")
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; FAKE_EXPORT=""; STATIC_GCC="-static-libgcc" ;;
  Darwin) SO="-shared -fPIC"; FAKE_EXPORT="-D__declspec(x)="; STATIC_GCC="" ;;
  *) SO="-shared -fPIC"; FAKE_EXPORT="-D__declspec(x)="; STATIC_GCC="-static-libgcc" ;;
esac
cd "$O"
# WEX_CFLAGS: extra flags for every compile and link, empty by default (the sanitizer CI job sets
# -fsanitize=address,undefined there)
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function ${WEX_CFLAGS-}"
"$CC" $W "$C/tests/test_wex_format.c" -o test_wex_format.exe
./test_wex_format.exe
"$CC" $W $OMP "$C/tests/test_wex4_verify.c" -o test_wex4_verify.exe
# it prints the expected failures, so only its summary line is shown; when it fails the whole output is, since
# a sanitizer report would be in it. its temporary .wex file stays here
./test_wex4_verify.exe > test_wex4_verify.log 2>&1 || { cat test_wex4_verify.log; exit 1; }
tail -1 test_wex4_verify.log
"$CC" $W $OMP -c "$C/tools/wex4_verify.c" -o wex4_verify.o
"$CC" $W $OMP wex4_verify.o -o wex4_verify.exe
# a whole-file check checks every blob, used by a record or not, and opens every WNM2 blob as a model (WEX4-SPEC
# section 6, step 3): exit 0 on a valid vector, 1 on the vectors damaged in a blob under a valid index
V=$ROOT/wex4/vectors
./wex4_verify.exe "$V/valid/tiny2-cz-upgraded.wex4" 1 > wex4_verify_vectors.log
./wex4_verify.exe "$V/valid/wnr4-mx-pb1.wex4" 1 >> wex4_verify_vectors.log
for f in unused-blob-of-kind-16-bit-flip-hash-not-fixed non-zero-byte-in-the-padding-of-an-unused-blob-of-kind-17 \
         unused-wnm2-blob-with-a-table-that-is-not-increasing wnm2-blob-with-a-table-that-is-not-increasing \
         wnm2-blob-id-differs-from-its-model wnr4-wnm4-blob-id-mismatch wnr4-unused-corrupt-wnm4-blob; do
  s=0; ./wex4_verify.exe "$V/invalid/$f.wex4" 1 >> wex4_verify_vectors.log || s=$?
  [ "$s" = 1 ] || { cat wex4_verify_vectors.log; echo "wex4_verify: exit $s on invalid/$f.wex4, expected 1" >&2; exit 1; }
done
echo "wex4_verify: 2 valid vectors pass, 7 vectors with a damaged blob (4 of them unused) refused"
# test_wex_fast_decode.c includes the engine's compat.h (for setenv and unsetenv on Windows), which is not in this
# repository: a stand-in, found as "../compat.h" through -I stand-in/tests. On Windows _putenv_s(name, "") removes
# the variable, as unsetenv does
mkdir -p stand-in/tests
cat > stand-in/compat.h <<'EOF'
/* stand-in for the engine's compat.h, written by scripts/build_cpu_check.sh: setenv and unsetenv on Windows */
#ifdef _WIN32
#include <stdlib.h>
static inline int compat_setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value) ? -1 : 0;
}
static inline int compat_unsetenv(const char *name) { return _putenv_s(name, "") ? -1 : 0; }
#define setenv(name, value, overwrite) compat_setenv(name, value, overwrite)
#define unsetenv(name) compat_unsetenv(name)
#endif
EOF
case $(uname -m) in
  x86_64|amd64) FAST_BUILDS="haswell:-march=haswell baseline: scalar:-DWEX_FAST_SCALAR" ;;
  *) FAST_BUILDS="default: scalar:-DWEX_FAST_SCALAR" ;;
esac
if [ -n "${WEX_NO_PARALLEL-}" ]; then
  for b in $FAST_BUILDS; do
    n=test_wex_fast_decode_${b%%:*}
    "$CC" $W $OMP ${b#*:} -I stand-in/tests "$C/tests/test_wex_fast_decode.c" -o "$n.exe" -pthread
    "./$n.exe" > "$n.log" 2>&1 || { cat "$n.log"; exit 1; }
    echo "$n: $(sed -n 's/^ *fast decoder: //p' "$n.log"); $(tail -1 "$n.log")"
  done
else
  fast_pids=()
  fast_names=()
  for b in $FAST_BUILDS; do
    n=test_wex_fast_decode_${b%%:*}
    fast_names+=("$n")
    (
      "$CC" $W $OMP ${b#*:} -I stand-in/tests "$C/tests/test_wex_fast_decode.c" -o "$n.exe" -pthread
      "./$n.exe"
    ) > "$n.log" 2>&1 &
    fast_pids+=($!)
  done
  fast_rc=0
  for i in "${!fast_names[@]}"; do
    n=${fast_names[$i]}
    s=0
    wait "${fast_pids[$i]}" || s=$?
    if [ "$s" != 0 ]; then
      cat "$n.log"
      fast_rc=$s
    else
      echo "$n: $(sed -n 's/^ *fast decoder: //p' "$n.log"); $(tail -1 "$n.log")"
    fi
  done
  [ "$fast_rc" = 0 ] || exit "$fast_rc"
fi
"$CC" $W $SO "$C/tools/wex_nr3_tools.c" -o wex_nr3_tools.dll
"$CC" $W $SO $STATIC_GCC $FAKE_EXPORT "$C/tests/fake_wex_cuda.c" -o fake_wex_cuda.dll
"$PY" -I "$C/tools/repack_v4_wex.py" --model-dir "$ROOT/fixtures/v41-tiny" --output tiny.wex > tiny.wex.log
P=$("$PY" -I -c "print(','.join(f'{l}:{e}' for l in range(6) for e in range(8)))")
"$PY" -I "$C/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex bins tiny_cz.bin --experts "$P" --train "$P" --cz | tail -2
"$PY" -I "$C/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex container tiny_cz.wex --layers 0,1,2,3,4,5 --train "$P" --cz | tail -2
if [ $# -ge 1 ]; then
  # the engine headers come from that engine tree: "x.h" through -I $K, a test's "../x.h" through -I $K/tests
  F="-D_FILE_OFFSET_BITS=64 $W $OMP"
  "$CC" $F -I "$K" -c "$K/expert_store_registry.c" -o r.o
  "$CC" $F -I "$K" -c "$C/expert_store_wex.c" -o w.o
  "$CC" $F -c "$C/wex_decode.c" -o d.o
  "$CC" $F -I "$K" -I "$K/tests" "$C/tests/test_expert_store_wex.c" r.o w.o d.o -o test_expert_store_wex.exe -pthread
  ./test_expert_store_wex.exe | tail -1
fi
echo "WNR3 CPU check passed"
