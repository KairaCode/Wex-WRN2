#!/usr/bin/env bash
# Timing, interleaved A/B (rows alternate, order rotates each round), 3 real records.
# 1 thread: per-thread CPU time (AB_CPU=1) because the shared machine is oversubscribed; 4 threads: wall clock.
cd "$(dirname "$0")"
R=/tmp/wnr3-data/recs
RECS="$R/rec_00_007.bin $R/rec_20_073.bin $R/rec_39_120.bin"
echo "== 1 thread, every kernel 1/2/3/4 configuration in one AVX-512 build (out/test_z), thread CPU time"
AB_CPU=1 ./out/test_z ab $R/model.bin 9 1 10,7,8,9,11,12,18,19,20,21,22,24,25,26,27 $RECS
echo "== 1 thread, unchanged header (out/test_orig, -march=haswell) vs new defaults, separate binaries, thread CPU time"
AB_CPU=1 python3 abx.py 9 1 1 orig_k1x4=out/test_orig:10 new_avx2_k3x4=out/test_h:22 new_avx512_k4x3=out/test_z:26 new_k1x4_in_z=out/test_z:10 -- $RECS
echo "== 4 threads, wall clock, separate binaries"
python3 abx.py 9 4 1 orig_k1x4=out/test_orig:10 new_avx2_k3x4=out/test_h:22 new_avx512_k4x3=out/test_z:26 -- $RECS
echo "== 1 thread, wall clock, separate binaries"
python3 abx.py 9 1 1 orig_k1x4=out/test_orig:10 new_avx2_k3x4=out/test_h:22 new_avx512_k4x3=out/test_z:26 -- $RECS
