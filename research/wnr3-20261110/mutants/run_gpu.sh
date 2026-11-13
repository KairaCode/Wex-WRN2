#!/bin/bash
# WNR3 GPU checks (G2b, G4 parts): waits (poll 2 min, max 3 h) until heavy.lock is free and no deepseek_v4.exe
# runs, takes the lock (same protocol as lab/wnr3-5090-20261107/lock.sh), runs, releases.
LOCK=${KAIRA:?set KAIRA to the KAIRA tree}/lab/v41-gpu-20261107/heavy.lock
B=${KAIRA:?set KAIRA to the KAIRA tree}/lab/wnr3-impl-20261110/build
D='D:\kaira_wnr3_lab'
deadline=$((SECONDS + 10800))
while :; do
  if ! tasklist //FI "IMAGENAME eq deepseek_v4.exe" 2>/dev/null | grep -qi deepseek_v4 &&
     ( set -o noclobber; printf '%s %s\n' "wnr3-impl-20261110" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; then break; fi
  [ $SECONDS -gt $deadline ] && { echo "GPU NOT RUN: lock/engine busy for 3 h (lock: $(cat "$LOCK" 2>/dev/null))"; exit 3; }
  sleep 120
done
trap 'rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
echo "lock: taken $(date +%T)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
for round in 1 2; do
  echo "== G2b round $round: deployed helper"; "$B/test_decode_batch.exe" "$D\deployed\kaira_wex_cuda.dll" "$D\records64.bin" | tail -3
  echo "== G2b round $round: WNR3 helper";    "$B/test_decode_batch.exe" "$B/kaira_wex_cuda.dll" "$D\records64.bin" | tail -3
done
echo "== G4 WNR3 non-CZ (lab CFG model), 52 records"; "$B/test_decode_batch.exe" "$B/kaira_wex_cuda.dll" "$D\g3_cfg_noncz.bin" | tail -3
echo "== G4 WNR3 CZ (lab-80 CZ model), 52 records";   "$B/test_decode_batch.exe" "$B/kaira_wex_cuda.dll" "$D\g4_cz.bin" | tail -3
echo "== CPU/GPU differential"; "$B/test_wex_nr3_gpu.exe" "$B/kaira_wex_cuda.dll" 304 "$D\g3_cfg_noncz.bin" "$D\g4_cz.bin" 2>&1 | tail -8
