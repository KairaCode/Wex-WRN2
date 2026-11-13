#!/bin/bash
# GPU mutation check: the real helper must pass test_wex_nr3_gpu, each mutant (gpu_mutants.py) must fail.
# Same lock protocol as run_gpu.sh (poll 2 min, max 3 h; no deepseek_v4.exe running).
LOCK=${KAIRA:?set KAIRA to the KAIRA tree}/lab/v41-gpu-20261107/heavy.lock
B=${KAIRA:?set KAIRA to the KAIRA tree}/lab/wnr3-impl-20261110/build
G="$1"
D='D:\kaira_wnr3_lab'
deadline=$((SECONDS + 10800))
while :; do
  if ! tasklist //FI "IMAGENAME eq deepseek_v4.exe" 2>/dev/null | grep -qi deepseek_v4 &&
     ( set -o noclobber; printf '%s %s\n' "wnr3-impl-20261110" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; then break; fi
  [ $SECONDS -gt $deadline ] && { echo "GPU NOT RUN: lock/engine busy for 3 h"; exit 3; }
  sleep 120
done
trap 'rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
echo "lock: taken $(date +%T)"
echo "== real helper"; "$B/test_wex_nr3_gpu.exe" "$B/kaira_wex_cuda.dll" 304 "$D\g3_cfg_noncz.bin" "$D\g4_cz.bin" 2>&1 | tail -6
for m in m1_weight_hash m2_scale_hash m3_cz_check m4_ceil_zero; do
  "$G/$m/build/test_wex_nr3_gpu.exe" "$G/$m/build/kaira_wex_cuda.dll" 104 "$D\g4_cz.bin" > "$G/$m/run.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "KILLED   $m (rc $rc): $(grep -m1 -E 'FAIL|mutation' "$G/$m/run.log")" || echo "SURVIVED $m"
done
