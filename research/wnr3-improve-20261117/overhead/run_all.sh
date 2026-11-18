#!/usr/bin/env bash
# Reproduce every number of RESULTS.md from this directory (reads /tmp/wnr3-data and the Wex-WRN2 checkout;
# writes only here). Heavy steps use 2 threads. Needs gcc with OpenMP, python3 with numpy.
#   bash run_all.sh            all steps (about 1 h on a loaded 4-core machine)
set -eu
cd "$(dirname "$0")"
TH=${TH:-2}
gcc -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -fopenmp ovh_proto.c -o ovh_proto -lm
mkdir -p out
# 0. coded scales + facts of all 121 sample records; the 32-B header of all 7,680 records (HTTP, ~250 KB)
python3 extract.py
[ -f headers.bin ] || python3 fetch_headers.py
# 1. scale-model bounds (ideal code lengths, held-out) and per-record table bounds
python3 scale_bounds.py 512 > scale_bounds_cs512.txt
python3 scale_bounds.py 64 > scale_bounds_cs64.txt
python3 scale_bounds.py 2048 > scale_bounds_cs2048.txt
python3 scale_adaptive.py > scale_adaptive.txt
# 2. TRAIN model (80 records) -> model_train.bin, scale_train.bin
TRAIN=$(python3 -c "print(' '.join(f'{l}:{e}' for l in range(40) for e in sorted({(37*l+11)%192,(37*l+101)%192})))")
OMP_NUM_THREADS=$TH ./ovh_proto train /tmp/wnr3-data/recs . $TRAIN
# 3. every profile on the 41 TEST records: real encode + decode + compare with Z(source)
TEST=$(python3 -c "print(' '.join([f'{l}:{(53*l+7)%192}' for l in range(40)]+['39:120']))")
OMP_NUM_THREADS=$TH ./ovh_proto run /tmp/wnr3-data/recs . out/run_test.json $TEST
python3 report.py out/run_test.json > out/report.txt
python3 final_table.py out/run_test.json > out/final_table.txt
python3 explain_outlier.py > out/explain_outlier.txt
# 4. corruption audit: 4 TEST records x 7 profiles x 6 types x 2500 trials (+ 320 header bits)
OMP_NUM_THREADS=$TH ./ovh_proto audit /tmp/wnr3-data/recs . out/audit.json 2500 20261117 10:153 20:107 30:61 39:120
python3 audit_report.py out/audit.json > out/audit_report.txt
# 5. decoder cost of the reference-style decoder per profile (1 thread, interleaved A/B, 9 rounds)
for R in 10:153 30:61; do OMP_NUM_THREADS=1 ./ovh_proto bench /tmp/wnr3-data/recs . 9 $R; done > out/bench.jsonl
