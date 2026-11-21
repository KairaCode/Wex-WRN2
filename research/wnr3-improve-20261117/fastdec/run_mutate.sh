#!/usr/bin/env bash
# mutation tests (harness `mutate`): mini-records cut from 6 real records, every variant against the reference
cd "$(dirname "$0")"
R='<TMP>'/wnr3-data/recs
RECS="$R/rec_00_007.bin $R/rec_07_078.bin $R/rec_13_108.bin $R/rec_20_073.bin $R/rec_33_080.bin $R/rec_39_120.bin"
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
./out/test_z mutate $R/model.bin 4000 20261117 $RECS > logs/mutate_z_4000.txt 2>&1; echo "exit $?" >> logs/mutate_z_4000.txt
./out/test_h mutate $R/model.bin 3000 777001 $RECS > logs/mutate_h_3000.txt 2>&1; echo "exit $?" >> logs/mutate_h_3000.txt
./out/test_ubsan_z mutate $R/model.bin 800 4242 $RECS > logs/mutate_ubsan_z_800.txt 2>&1; echo "exit $?" >> logs/mutate_ubsan_z_800.txt
