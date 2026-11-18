#!/bin/sh
# Same compiler (gcc 13.3), same source (tdrv.c), two flag sets for every variant:
#   AVX2 set:    -O3 -march=haswell -fopenmp                       (i7-7700K level)
#   AVX-512 set: -O3 -march=x86-64-v4 -mavx512vbmi2 -fopenmp       (F/BW/CD/DQ/VL + VBMI2; Zen 4 has all of them)
# plus ffold_a5 = fastfmt's own AVX-512 flags (-march=haswell -mavx512f/bw/vl/dq), as a cross-check of the WNX kernels.
set -e
cd /tmp/wnr3-research/timing
C="-I/workspace/Wex-WRN2/c -I/tmp/wnr3-research/fastfmt"
ORIG="-I/workspace/Wex-WRN2/wex4/research/fast-cpu-decode"
NEW="-I/tmp/wnr3-research/fastdec"
A2="-O3 -march=haswell -fopenmp"
A5="-O3 -march=x86-64-v4 -mavx512vbmi2 -fopenmp"
gcc $A2 $NEW $C tdrv.c -o bin/new_a2
gcc $A5 $NEW $C tdrv.c -o bin/new_a5
gcc $A2 $ORIG $C tdrv.c -o bin/orig_a2
gcc $A5 $ORIG $C tdrv.c -o bin/orig_a5
gcc -O3 -march=haswell -mavx512f -mavx512bw -mavx512vl -mavx512dq -fopenmp $ORIG $C tdrv.c -o bin/ffold_a5
# which header each binary got
for b in new_a2 new_a5 orig_a2 orig_a5 ffold_a5; do echo "$b: $(strings bin/$b | grep -c 'two-level' || true)"; done
