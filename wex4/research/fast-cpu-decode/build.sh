#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Builds the harness three ways into out/ (CPU only):
#   test_fast.exe    AVX2 kernels + OpenMP (-march=haswell: AVX2, BMI, BMI2)
#   test_scalar.exe  -DWEX_FAST_SCALAR: the portable fallback (kernel 0 only) + OpenMP
#   test_ubsan.exe   AVX2 kernels with -fsanitize=undefined -fsanitize-trap=all (traps, no runtime library)
# x86 only (-march=haswell). On AArch64 the NEON kernel needs no flag: synth.sh builds the harness for the CPU it
# runs on.
# Usage: sh build.sh   (gcc on PATH; WEX_SRC = directory of wex_format.h, default ../../../c)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${WEX_SRC:-$HERE/../../../c}
mkdir -p "$HERE/out"
export TMP="$HERE/out" TEMP="$HERE/out"   # gcc cannot create temporary files under the Windows directory
cd "$HERE"
W="-std=c11 -Wall -Wextra -Wno-unused-function -I$SRC"
gcc $W -O3 -march=haswell -fopenmp test_fast_decode.c -o out/test_fast.exe
gcc $W -O3 -fopenmp -DWEX_FAST_SCALAR test_fast_decode.c -o out/test_scalar.exe
gcc $W -O1 -march=haswell -fopenmp -fsanitize=undefined -fsanitize-trap=all test_fast_decode.c -o out/test_ubsan.exe
echo built
