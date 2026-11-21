#!/bin/sh
# Supplementary flag-sensitivity builds (same source tdrv.c):
#   *_h5:  -O3 -march=haswell -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mavx512vbmi2 (haswell tuning + AVX-512)
#   *_nat: -O3 -march=native (this Xeon: sapphirerapids-class tuning)
set -e
cd '<TMP>'/wnr3-research/timing
C="-I/workspace/Wex-WRN2/c -I<TMP>/wnr3-research/fastfmt"
ORIG="-I/workspace/Wex-WRN2/wex4/research/fast-cpu-decode"
NEW="-I<TMP>/wnr3-research/fastdec"
H5="-O3 -march=haswell -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mavx512vbmi2 -fopenmp"
NAT="-O3 -march=native -fopenmp"
gcc $H5 $NEW $C tdrv.c -o bin/new_h5
gcc $H5 $ORIG $C tdrv.c -o bin/orig_h5
gcc $NAT $NEW $C tdrv.c -o bin/new_nat
gcc $NAT $ORIG $C tdrv.c -o bin/orig_nat
gcc -march=native -Q --help=target 2>/dev/null | grep -E "^ +-march=|^ +-mtune=" > logs/native_target.txt
