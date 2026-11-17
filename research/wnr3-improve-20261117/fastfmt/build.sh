#!/bin/sh
# AVX2 build (production i7-7700K class) and AVX-512 build (Ryzen 7950X3D / this Xeon class); same flags otherwise.
cd "$(dirname "$0")"
I="-I/workspace/Wex-WRN2/c -I/workspace/Wex-WRN2/wex4/research/fast-cpu-decode"
CC=${CC:-gcc}
$CC -O2 -march=native $I ff.c -o ff -lm
$CC -O3 -march=haswell -fopenmp $I bench.c -o bench_avx2
$CC -O3 -march=haswell -mavx512f -mavx512bw -mavx512vl -mavx512dq -fopenmp $I bench.c -o bench_avx512
# thread-scaling probe: ./bench_avx512 scal out/mag_2048 -- data/Z_05_080.bin
