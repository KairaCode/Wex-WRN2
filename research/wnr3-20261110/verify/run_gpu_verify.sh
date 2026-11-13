#!/bin/bash
# VERIFY GPU part: waits (poll 2 min, max 3 h) until heavy.lock is free and no deepseek_v4.exe runs,
# takes the lock (protocol of lab/wnr3-5090-20261107/lock.sh), runs, releases on exit.
#  1. GPU decode of the 103 real records (40 held-out + top-64 tail), WNR3-CZ and WNR3 non-CZ
#  2. microbench ABAB: deployed WNR2 helper on the WNR2 production sample (records64.bin) vs the WNR3
#     helper on the WNR3-CZ bin (and on records64.bin)
#  3. GPU corruption audit (audit/audit_gpu.exe) on WNR2 production, WNR3-CZ and WNR3 non-CZ records
LOCK=${KAIRA:?set KAIRA to the KAIRA tree}/lab/v41-gpu-20261107/heavy.lock
V=${KAIRA:?set KAIRA to the KAIRA tree}/lab/wnr3-impl-20261110/verify
G=${KAIRA:?set KAIRA to the KAIRA tree}/lab/wnr3-impl-20261110/fresh-fix3/gpu-build
VW=$(cygpath -w "$V")
D='D:\kaira_wnr3_lab'
# a KAIRA engine = deepseek_v4.exe started from KAIRA\app (lab test binaries of other workflows do not count)
kaira_engine() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='deepseek_v4.exe'\" | Where-Object { \$_.ExecutablePath -like '*\KAIRA\app\*' }).Count" | tr -d '\r'; }
deadline=$((SECONDS + 10800))
while :; do
  if [ "$(kaira_engine)" = 0 ] &&
     ( set -o noclobber; printf '%s %s\n' "wnr3-impl-20261110 VERIFY gpu (~40 min)" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; then break; fi
  [ $SECONDS -gt $deadline ] && { echo "GPU NOT RUN: lock/engine busy for 3 h (lock: $(cat "$LOCK" 2>/dev/null); KAIRA engines: $(kaira_engine))"; exit 3; }
  echo "waiting $(date +%T): lock $(cat "$LOCK" 2>/dev/null | cut -c1-60) / KAIRA engines $(kaira_engine) / other deepseek_v4 $(tasklist //FI "IMAGENAME eq deepseek_v4.exe" | grep -ci deepseek)"
  sleep 120
done
trap 'rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
echo "lock: taken $(date +%T); other deepseek_v4.exe now: $(tasklist //FI "IMAGENAME eq deepseek_v4.exe" | grep -ci deepseek)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
sha256sum "$G/kaira_wex_cuda.dll" /d/kaira_wnr3_lab/deployed/kaira_wex_cuda.dll "$G/test_decode_batch.exe" | cut -c1-16,66-
echo "== 1. GPU decode, WNR3-CZ, 103 records";     "$G/test_decode_batch.exe" "$G/kaira_wex_cuda.dll" "$VW\nr3_cz.bin" | tail -3
echo "== 1. GPU decode, WNR3 non-CZ, 103 records"; "$G/test_decode_batch.exe" "$G/kaira_wex_cuda.dll" "$VW\nr3_noncz.bin" | tail -3
for round in 1 2; do
  echo "== 2. round $round: deployed helper, WNR2 production sample";  "$G/test_decode_batch.exe" "$D\deployed\kaira_wex_cuda.dll" "$D\records64.bin" | tail -2
  echo "== 2. round $round: WNR3 helper, WNR3-CZ 103";                 "$G/test_decode_batch.exe" "$G/kaira_wex_cuda.dll" "$VW\nr3_cz.bin" | tail -2
  echo "== 2. round $round: WNR3 helper, WNR2 production sample";     "$G/test_decode_batch.exe" "$G/kaira_wex_cuda.dll" "$D\records64.bin" | tail -2
done
echo "== 3. GPU corruption audit"
"$V/audit/audit_gpu.exe" "$G/kaira_wex_cuda.dll" "$VW\wnr2_audit.bin" 0 0:0:113 1:39:120
"$V/audit/audit_gpu.exe" "$G/kaira_wex_cuda.dll" "$VW\nr3_cz.bin" 97 0:0:113 40:39:120
"$V/audit/audit_gpu.exe" "$G/kaira_wex_cuda.dll" "$VW\nr3_noncz.bin" 33 0:0:113
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
