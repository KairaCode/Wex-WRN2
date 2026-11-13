#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Verifier's reruns (CPU only). In Git Bash with gcc on PATH: sh run.sh
# Needs ../out/model.wnm2 and ../out/L*.rec (the first experiment's copies) and out/H_*.rec (python -I scan.py SRC OUT).
set -eu
cd "$(dirname "$0")"; mkdir -p out; export TMP="$PWD/out" TEMP="$PWD/out" OMP_NUM_THREADS=8
SRC=../../../../c; W="-std=c11 -Wall -Wextra -Wno-unused-function -I$SRC"
H=$(ls out/H_*.rec | tr '\n' ' ')
gcc $W -I.. -O3 -march=haswell -fopenmp ../test_fast_decode.c -o out/test_fast.exe
gcc $W -I.. -I. -O3 -march=haswell -fopenmp vh.c -o out/vh.exe
out/test_fast.exe equal ../out/model.wnm2 $H                       # held-out byte equality
out/test_fast.exe bench ../out/model.wnm2 5 $H                     # the experiment's table on held-out records
out/vh.exe time ../out/model.wnm2 5 $H                             # own timer, thread scaling, drop-in path
out/vh.exe reenc ../out/model.wnm2 out/H_L38_E10.rec               # full-size re-encodes and boundary mutations
out/vh.exe tailcut ../out/model.wnm2 out/H_L8_E100.rec             # deleted stream tails against a guard page
out/test_fast.exe mutate ../out/model.wnm2 20000 987654321 $H      # about 3 minutes
# hash check removed from BOTH decoders, so byte-mutated records can be accepted and compared
rm -rf nohash && mkdir nohash && cp $SRC/wex_format.h $SRC/wex_palette.h ../test_fast_decode.c nohash/
sed 's/|| state != (nr3 ?/|| 0 \&\& state != (nr3 ?/' $SRC/wex_neural_rans.h > nohash/wex_neural_rans.h
sed 's/|| state != wex_nr3_stream_x0/|| 0 \&\& state != wex_nr3_stream_x0/; s/|| x\[k\] != /|| 0 \&\& x[k] != /; s/|| st\[k\] != /|| 0 \&\& st[k] != /' ../wex_fast_decode.h > nohash/wex_fast_decode.h
gcc -std=c11 -Wno-parentheses -Inohash -O3 -march=haswell -fopenmp nohash/test_fast_decode.c -o out/test_nohash.exe
out/test_nohash.exe mutate ../out/model.wnm2 20000 424242 out/H_L8_E100.rec out/H_L2_E75.rec out/H_L37_E150.rec out/H_L14_E128.rec
rm -rf nohash
