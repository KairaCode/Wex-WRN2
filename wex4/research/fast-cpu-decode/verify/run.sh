#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Verifier's reruns (CPU only). With gcc on PATH (MSYS2 UCRT64 on Windows): bash run.sh
# Needs ../out/model.wnm2 and ../out/L*.rec (the first experiment's copies) and out/H_*.rec (python -I scan.py SRC OUT).
# Other records: VERIFY_MODEL names the WNM2 model and VERIFY_RECORDS the records (72-B entry + record, as
# ../extract.py writes them); then reenc and tailcut take the first of them and the hash-free mutate the first four.
# They must be WNR3-CZ records, as the production container's are: vh.c reenc encodes with the canonical-zero flag,
# so a record whose slab keeps -0 nibbles fails there.
# CC names another compiler (default gcc) and WEX_OPENMP replaces -fopenmp, as in scripts/build_cpu_check.sh;
# set and empty, it builds without OpenMP, as ../synth.sh does (the MSYS2 clang has no OpenMP runtime).
# -march=haswell (the AVX2 kernels) only on x86-64; elsewhere no flag (the NEON kernel needs none on AArch64);
# WEX4_MARCH overrides it, as in ../synth.sh. vh.c is Windows only (windows.h, VirtualAlloc): elsewhere its three
# steps are skipped. Last line on success: verify run.sh passed
set -eu
cd "$(dirname "$0")"; mkdir -p out; export TMP="$PWD/out" TEMP="$PWD/out" OMP_NUM_THREADS=8
CC=${CC:-gcc}                  # the C compiler of every build below
OMP=${WEX_OPENMP--fopenmp}     # the OpenMP flags; set and empty = no OpenMP
case $(uname -m) in
  x86_64|amd64) MARCH=${WEX4_MARCH--march=haswell} ;;
  *) MARCH=${WEX4_MARCH-} ;;
esac
case $(uname -s) in MINGW*|MSYS*|CYGWIN*) WIN=1 ;; *) WIN= ;; esac
echo "compiler: $("$CC" --version | sed -n 1p), -O3 $MARCH $OMP"
SRC=../../../../c; W="-std=c11 -Wall -Wextra -Wno-unused-function -I$SRC"
M=${VERIFY_MODEL:-../out/model.wnm2}
if [ -n "${VERIFY_RECORDS-}" ]; then
  H=$VERIFY_RECORDS
  set -- $H; ONE=$1; FOUR=$(echo $H | cut -d' ' -f1-4)
  REENC=$ONE; TAILCUT=$ONE
else
  H=$(ls out/H_*.rec | tr '\n' ' ')
  REENC=out/H_L38_E10.rec; TAILCUT=out/H_L8_E100.rec
  FOUR="out/H_L8_E100.rec out/H_L2_E75.rec out/H_L37_E150.rec out/H_L14_E128.rec"
fi
"$CC" $W -I.. -O3 $MARCH $OMP ../test_fast_decode.c -o out/test_fast.exe
if [ -n "$WIN" ]; then "$CC" -I.. $W -I. -O3 $MARCH $OMP vh.c -o out/vh.exe; fi
out/test_fast.exe equal "$M" $H                       # held-out byte equality
out/test_fast.exe bench "$M" 5 $H                     # the experiment's table on held-out records
if [ -n "$WIN" ]; then
  out/vh.exe time "$M" 5 $H                           # own timer, thread scaling, drop-in path
  out/vh.exe reenc "$M" "$REENC"                      # full-size re-encodes and boundary mutations
  out/vh.exe tailcut "$M" "$TAILCUT"                  # deleted stream tails against a guard page
else
  echo "vh.c is Windows only (windows.h, VirtualAlloc): time, reenc and tailcut skipped"
fi
out/test_fast.exe mutate "$M" 20000 987654321 $H      # about 3 minutes
# hash check removed from BOTH decoders, so byte-mutated records can be accepted and compared
rm -rf nohash && mkdir nohash && cp $SRC/wex_format.h $SRC/wex_palette.h ../test_fast_decode.c nohash/
sed 's/|| state != (nr3 ?/|| 0 \&\& state != (nr3 ?/' $SRC/wex_neural_rans.h > nohash/wex_neural_rans.h
sed 's/|| state != wex_nr3_stream_x0/|| 0 \&\& state != wex_nr3_stream_x0/; s/|| x\[k\] != /|| 0 \&\& x[k] != /; s/|| st\[k\] != /|| 0 \&\& st[k] != /' ../wex_fast_decode.h > nohash/wex_fast_decode.h
"$CC" -std=c11 -Wno-parentheses -Inohash -O3 $MARCH $OMP nohash/test_fast_decode.c -o out/test_nohash.exe
out/test_nohash.exe mutate "$M" 20000 424242 $FOUR
rm -rf nohash
echo "verify run.sh passed"
