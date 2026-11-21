#!/usr/bin/env bash
# Builds the harness (test_fast_decode.c + wex_fast_decode.h of this directory) into out/:
#   test_z       AVX-512 build (-march=x86-64-v4 -mavx512vbmi2): kernels 0..4, default kernel 4 x 3
#   test_h       AVX2 build (-march=haswell, the i7-7700K's level): kernels 0..3, default kernel 3 x 4
#   test_orig    the unchanged wex_fast_decode.h of Wex-WRN2 (wex_fast_decode.orig.h), AVX2 build
#   test_scalar  -DWEX_FAST_SCALAR
#   test_ubsan_z / test_ubsan_h  UBSan trap builds
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${WEX_SRC:-/workspace/Wex-WRN2/c}
CC=${CC:-gcc}
mkdir -p "$HERE/out"; cd "$HERE"
W="-std=c11 -Wall -Wextra -Wno-unused-function -I$SRC"
Z="-march=x86-64-v4 -mavx512vbmi2"
"$CC" $W -O3 $Z -fopenmp test_fast_decode.c -o out/test_z
"$CC" $W -O3 -march=haswell -fopenmp test_fast_decode.c -o out/test_h
sed 's/#include "wex_fast_decode.h"/#include "wex_fast_decode.orig.h"/' test_fast_decode.c > out/test_orig.c
"$CC" $W -O3 -march=haswell -fopenmp -I"$HERE" -Wno-missing-field-initializers out/test_orig.c -o out/test_orig 2>/dev/null || echo "test_orig: see note in README (harness variants for kernels 3/4 need the new header)"
"$CC" $W -O3 -fopenmp -DWEX_FAST_SCALAR test_fast_decode.c -o out/test_scalar
"$CC" $W -O3 -march=haswell -fopenmp -DTEST_DEFAULT_ONLY test_fast_decode.c -o out/test_h_def
"$CC" $W -O1 $Z -fopenmp -fsanitize=undefined -fsanitize-trap=all test_fast_decode.c -o out/test_ubsan_z
"$CC" $W -O1 -march=haswell -fopenmp -fsanitize=undefined -fsanitize-trap=all test_fast_decode.c -o out/test_ubsan_h
echo built
