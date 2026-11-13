#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# WEX4 prototype: build the tools and run every test. CPU only, no real model needed.
#   1. wex4/wex4_tools.c -> wex4_tools.dll (validator, codecs, in-process fuzzer) with the fast WNR3 CPU decoder
#      (AVX2 + OpenMP), and wex4_tools_ref.dll with the reference decoder: the tests require the same results
#   2. c/tools/wex_nr3_tools.c -> wex_nr3_tools.dll; fixtures/v41-tiny -> tiny.wex (WEX1) -> tiny_cz.wex (WNR3-CZ)
#   3. wex4/test_wex4.py
# Windows: MSYS2 UCRT64 or MINGW64 gcc on PATH, python 3 (stdlib only). Linux: gcc and PYTHON=python3.
# Usage: bash wex4/build_and_test.sh [--fuzz N] [--write-vectors DIR]
set -eu
set -o pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/wex4/out
PY=${PYTHON:-python}
rm -rf "$S" && mkdir -p "$S"
export TMP="$S" TEMP="$S"   # gcc from Git Bash cannot create temporary files under C:\WINDOWS
# shared libraries: Windows (MSYS2, Git Bash) as before; elsewhere position-independent and not -static
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; STATIC_SO="-static -shared" ;;
  *) SO="-shared -fPIC"; STATIC_SO="-shared -fPIC" ;;
esac
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"
gcc $W $SO -I "$ROOT/c" "$ROOT/wex4/wex4_tools.c" -o "$S/wex4_tools_ref.dll"
F="-O3 -march=haswell -fopenmp -DWEX4_FAST_NR3"
gcc $W $F $STATIC_SO -I "$ROOT/c" -I "$ROOT/wex4/research/fast-cpu-decode" "$ROOT/wex4/wex4_tools.c" -o "$S/wex4_tools.dll"
# c/wex4.h comes from KAIRA and can change in a sync: prove that the two DLLs really hold two decoders
# (a sync once dropped the hook, and the fast-versus-reference tests compared the reference with itself)
site() { gcc $1 -E -I "$ROOT/c" -I "$ROOT/wex4/research/fast-cpu-decode" "$ROOT/wex4/wex4_tools.c" | grep -c "return $2(&view, &layout, model, scales, weights)" || true; }
if [ "$(site "$F" wex_fast_decode)" != 1 ] || [ "$(site "" wex_nr2_decode)" != 1 ]; then
  echo "wex4_decode_record does not call wex_fast_decode in wex4_tools.dll and wex_nr2_decode in wex4_tools_ref.dll" >&2; exit 1
fi
gcc $W $SO "$ROOT/c/tools/wex_nr3_tools.c" -o "$S/wex_nr3_tools.dll"
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
"$PY" -I -u "$ROOT/wex4/test_wex4.py" "$S" "$@"
