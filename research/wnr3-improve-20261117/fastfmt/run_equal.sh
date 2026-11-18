#!/bin/sh
# Byte equality with Z(source) of every decoder (WNR3 fast + reference; WNX reference + every fast kernel, 1 and 4 threads)
cd "$(dirname "$0")"
D="out/mag_2048 out/joint_2048 out/magt_2048 out/magt_4096"
D2="out/mag_1024 out/mag_4096 out/magt_2048_s512"
./bench_avx512 equal $D -- $(cat test.txt) > logs/equal_avx512.txt 2>&1
./bench_avx2   equal $D -- $(cat test.txt) > logs/equal_avx2.txt 2>&1
./bench_avx512 equal $D2 -- $(cat test.txt) > logs/equal_avx512_b.txt 2>&1
./bench_avx2   equal $D2 -- $(cat test.txt) > logs/equal_avx2_b.txt 2>&1
