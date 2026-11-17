#!/bin/sh
# Paired A/B timing on 8 TEST records (one every 5 layers): each WNX decoder is timed right next to WNR3
# wex_fast_decode (A B / B A alternating), ROUNDS rounds. Both builds; 1 thread, 4 threads per record, and
# 4 records at once with 1 thread each. Wall and process-CPU time are both reported.
cd "$(dirname "$0")"
R=${ROUNDS:-15}
RECS=$(sed -n '1p;6p;11p;16p;21p;26p;31p;36p' test.txt)
for b in avx512 avx2; do
  for th in 1 4 -4; do
    echo "== build $b, threads $th, load before: $(cat /proc/loadavg)"
    ./bench_$b time $R $th out/mag_2048 out/magt_2048 out/joint_2048 -- $RECS
    echo "   load after: $(cat /proc/loadavg)"
  done
done
