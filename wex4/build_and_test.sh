#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# WEX4 prototype: build the tools and run every test. CPU only, no real model needed.
#   1. wex4/wex4_tools.c -> wex4_tools.dll (validator, codecs, in-process fuzzer) with the fast WNR3 CPU decoder
#      (AVX2 + OpenMP on x86, NEON + OpenMP on AArch64, scalar + OpenMP elsewhere), and wex4_tools_ref.dll with the reference decoder: the tests require the same results
#   2. c/tools/wex_nr3_tools.c -> wex_nr3_tools.dll; fixtures/v41-tiny -> tiny.wex (WEX1) -> tiny_cz.wex (WNR3-CZ)
#   3. wex4/test_wex4.py
# Windows: MSYS2 UCRT64 or MINGW64 gcc on PATH, python 3 (stdlib only). Linux: gcc and PYTHON=python3.
# macOS: a GNU gcc in CC (Homebrew: CC=gcc-15) and PYTHON=python3; the gcc of Xcode is clang and has no OpenMP.
# CC names another compiler (CC=clang, which needs its OpenMP runtime: CI runs that on Linux).
# WEX_OPENMP replaces the OpenMP flag (default -fopenmp). Apple clang refuses -fopenmp and has no OpenMP runtime;
# with P=$(brew --prefix libomp), the OpenMP runtime of Homebrew, the macos-clang CI job runs CC=clang and
# WEX_OPENMP="-Xclang -fopenmp -I$P/include -L$P/lib -lomp -Wno-unused-command-line-argument" (the last flag:
# the compile-only and preprocessor calls would warn that -L and -lomp are not used)
# Usage: bash wex4/build_and_test.sh [--fuzz N] [--write-vectors DIR]
set -eu
set -o pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/wex4/out
PY=${PYTHON:-python}
CC=${CC:-gcc}   # the C compiler of every build below
OMP=${WEX_OPENMP:--fopenmp}   # the OpenMP flags of the fast decoder's build
echo "compiler: $("$CC" --version | sed -n 1p)"
rm -rf "$S" && mkdir -p "$S"
export TMP="$S" TEMP="$S"   # gcc from Git Bash cannot create temporary files under C:\WINDOWS
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent and not -static
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; STATIC_SO="-static -shared" ;;
  *) SO="-shared -fPIC"; STATIC_SO="-shared -fPIC" ;;
esac
# WEX_CFLAGS: extra flags for every compile and link, empty by default (the sanitizer CI job sets
# -fsanitize=address,undefined there)
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function ${WEX_CFLAGS-}"
"$CC" $W $SO -I "$ROOT/c" "$ROOT/wex4/wex4_tools.c" -o "$S/wex4_tools_ref.dll"
# the fast decoder picks its kernels when it is compiled: AVX2 on x86 (-march=haswell: AVX2, BMI, BMI2, LZCNT),
# NEON on AArch64 (part of the base instruction set, no flag). On any other CPU (RISC-V, ...) the same source
# builds its scalar kernel 0: the reference loop per stream, parallel across streams, so the fast-versus-reference
# tests still compare two decoders. WEX4_MARCH replaces the flag: WEX4_MARCH= builds the scalar kernel on x86,
# WEX4_MARCH=-DWEX_FAST_SCALAR builds it on any CPU.
case $(uname -m) in
  x86_64|amd64|i?86) MARCH=${WEX4_MARCH--march=haswell} ;;
  *) MARCH=${WEX4_MARCH-} ;;
esac
F="-O3 $MARCH $OMP -DWEX4_FAST_NR3"
"$CC" $W $F $STATIC_SO -I "$ROOT/wex4/research/fast-cpu-decode" -I "$ROOT/c" "$ROOT/wex4/wex4_tools.c" -o "$S/wex4_tools.dll"
# which kernels that was (WEXF_KERNELS of wex_fast_decode.h under the same flags): AVX2, NEON, NEON+SVE (with an
# SVE flag such as WEX4_MARCH=-march=armv8-a+sve) or scalar
KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
          "$CC" $F -E -P -I "$ROOT/wex4/research/fast-cpu-decode" -I "$ROOT/c" -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
echo "fast WNR3 decoder of wex4_tools.dll: $KERNELS kernels"
# c/wex4.h is the engine-side file and can change with the engine: prove that the two DLLs really hold two decoders
# (a sync once dropped the hook, and the fast-versus-reference tests compared the reference with itself)
site() { "$CC" $1 -E -I "$ROOT/wex4/research/fast-cpu-decode" -I "$ROOT/c" "$ROOT/wex4/wex4_tools.c" | grep -c "return $2(&view, &layout, model, scales, weights)" || true; }
if [ "$(site "$F" wex_fast_decode)" != 1 ] || [ "$(site "" wex_nr2_decode)" != 1 ]; then
  echo "wex4_decode_record does not call wex_fast_decode in wex4_tools.dll and wex_nr2_decode in wex4_tools_ref.dll" >&2; exit 1
fi
"$CC" $W $SO "$ROOT/c/tools/wex_nr3_tools.c" -o "$S/wex_nr3_tools.dll"
cd "$S"
"$PY" -I "$ROOT/c/tools/repack_v4_wex.py" --model-dir "$ROOT/fixtures/v41-tiny" --output tiny.wex > tiny.wex.log
P=$("$PY" -I -c "print(','.join(f'{l}:{e}' for l in range(6) for e in range(8)))")
"$PY" -I "$ROOT/c/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex container tiny_cz.wex --layers 0,1,2,3,4,5 --train "$P" --cz \
    --save-model tiny.wnm2 | tail -1
"$PY" -I "$ROOT/c/tools/build_nr3.py" ./wex_nr3_tools.dll tiny.wex container tiny2_cz.wex --layers 0 --max-experts 2     --model tiny.wnm2 --cz | tail -1
# compressible synthetic experts: a container whose non-CZ WNR3 records really are smaller than raw
"$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" synth --layers 1 --experts 4 > synth.log
"$PY" -I "$ROOT/c/tools/repack_v4_wex.py" --model-dir synth --output synth.wex >> synth.log
"$PY" -I "$ROOT/c/tools/build_nr3.py" ./wex_nr3_tools.dll synth.wex container synth_nr3.wex --layers 0 --train 0:0,0:1,0:2,0:3 \
    --save-model synth.wnm2 | tail -1
# naming profiles: the same six experts under the tensor names of every profile (prof/), one WEX1 and one WNR3
# container of them, and a mixed checkpoint per profile (mix/: a first layer without experts, an MXFP4 layer,
# a float layer)
for n in $("$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" --list-names | tr -d '\r'); do
  "$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" "prof/$n" --names "$n" --layers 2 --experts 3 --seed 20261114 >> prof.log
  "$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" "mix/$n" --names "$n" --layers 3 --experts 2 --first-layer 1 --float-layer 2 >> prof.log
done
"$PY" -I "$ROOT/c/tools/repack_v4_wex.py" --model-dir prof/deepseek --output prof.wex >> prof.log
"$PY" -I "$ROOT/c/tools/build_nr3.py" ./wex_nr3_tools.dll prof.wex container prof_nr3.wex --layers 0,1 --model synth.wnm2 | tail -1
"$PY" -I -u "$ROOT/wex4/test_wex4.py" "$S" "$@"
