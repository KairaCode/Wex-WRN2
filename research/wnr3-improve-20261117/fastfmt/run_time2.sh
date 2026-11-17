#!/bin/sh
# Second paired timing run (same method as run_time.sh), MAG and MAGT at 2048 only, plus 4 threads with
# OMP_WAIT_POLICY=passive (both decoders use OpenMP; passive avoids spinning at barriers on a shared machine).
cd "$(dirname "$0")"
R=${ROUNDS:-11}
RECS=$(sed -n '1p;6p;11p;16p;21p;26p;31p;36p' test.txt)
for b in avx512 avx2; do
  for th in 1 4 -4; do
    echo "== build $b, threads $th, load before: $(cat /proc/loadavg)"
    ./bench_$b time $R $th out/mag_2048 out/magt_2048 -- $RECS
    echo "   load after: $(cat /proc/loadavg)"
  done
  echo "== build $b, threads 4, OMP_WAIT_POLICY=passive, load before: $(cat /proc/loadavg)"
  OMP_WAIT_POLICY=passive ./bench_$b time $R 4 out/mag_2048 out/magt_2048 -- $RECS
  echo "   load after: $(cat /proc/loadavg)"
done
