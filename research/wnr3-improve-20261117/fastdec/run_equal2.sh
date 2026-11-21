#!/usr/bin/env bash
# equal, continued: (1) AVX-512 build, every variant, the records of the 121 not yet in logs/equal_z_121.txt;
# (2) AVX2 build with TEST_DEFAULT_ONLY (k0 + the default entry point = kernel 3 x 4) on all 121;
# (3) AVX2 build, every variant, on 12 records (one per ~3 layers + the outlier).
cd "$(dirname "$0")"
R='<TMP>'/wnr3-data/recs
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 5; done
for f in $R/rec_*.bin; do
  grep -q "^$f: reference" logs/equal_z_121.txt && continue
  EQ_THREADS=2 ./out/test_z equal $R/model.bin $f >> logs/equal_z_121.txt 2>&1 || echo "FAIL $f" >> logs/equal_z_121.txt
done
echo "records: $(grep -c '^equal:' logs/equal_z_121.txt), with 0 mismatches: $(grep -c ' 0 mismatches' logs/equal_z_121.txt), FAIL lines: $(grep -c FAIL logs/equal_z_121.txt)" >> logs/equal_z_121.txt
: > logs/equal_h_default_121.txt
for f in $R/rec_*.bin; do
  EQ_THREADS=2 ./out/test_h_def equal $R/model.bin $f >> logs/equal_h_default_121.txt 2>&1 || echo "FAIL $f" >> logs/equal_h_default_121.txt
done
echo "records: $(grep -c '^equal:' logs/equal_h_default_121.txt), with 0 mismatches: $(grep -c ' 0 mismatches' logs/equal_h_default_121.txt), FAIL lines: $(grep -c FAIL logs/equal_h_default_121.txt)" >> logs/equal_h_default_121.txt
: > logs/equal_h_12.txt
for f in 00_007 03_020 06_133 10_087 13_108 17_064 20_073 24_029 27_140 31_096 35_154 39_120; do
  EQ_THREADS=2 ./out/test_h equal $R/model.bin $R/rec_$f.bin >> logs/equal_h_12.txt 2>&1 || echo "FAIL $f" >> logs/equal_h_12.txt
done
echo "records: $(grep -c '^equal:' logs/equal_h_12.txt), with 0 mismatches: $(grep -c ' 0 mismatches' logs/equal_h_12.txt), FAIL lines: $(grep -c FAIL logs/equal_h_12.txt)" >> logs/equal_h_12.txt
