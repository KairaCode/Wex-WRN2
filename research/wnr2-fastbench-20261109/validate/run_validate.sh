#!/bin/bash
# Validate (resumed 2026-11-09 evening): one heavy.lock session for
#   1. replay ABBA rerun on the 05:20 WEX1 trace (repeatability across sessions) -> replay/runs/ab2
#   2. step-5 replay variants (B_own, B_pad, B_padl + A_l control), interleaved   -> validate/runs-variants
#   3. short ABBA rerun, WEX1 vs WNR2 on ws-ab (repeatability of the short ABBA)   -> short-abba/runs/<stamp>
# Lock protocol = wnr3-5090-20261107/lock.sh (noclobber create, poll 60 s), but waits up to 6 h and removes
# the lock on exit only if it still holds our own label line.
#   bash run_validate.sh > run_validate.log 2>&1
set -u
F='<KAIRA>'/lab/wnr2-fastbench-20261109
LOCK='<KAIRA>'/lab/v41-gpu-20261107/heavy.lock
MINE="wnr2-fastbench validate (replay ABBA + 3 variants + short ABBA, ~60 min) $(date +%FT%T) pid$$"
deadline=$((SECONDS + 21600))
echo "lock: waiting $(date +%T) (held by: $(cat "$LOCK" 2>/dev/null))"
while ! ( set -o noclobber; printf '%s\n' "$MINE" > "$LOCK" ) 2>/dev/null; do
  [ $SECONDS -gt $deadline ] && { echo "lock: gave up after 6 h (held by: $(cat "$LOCK" 2>/dev/null))"; exit 3; }
  sleep 60
done
trap '[ "$(cat "$LOCK" 2>/dev/null)" = "$MINE" ] && rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
echo "lock: taken $(date +%T)"
snap() { echo "-- $1 $(date +%T): $(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB" | tr -d '\r') GB free RAM;" \
  "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; \
  tasklist | grep -iE 'deepseek|replay|nvcc|cl\.exe|robocopy|python|wex' | awk '{print "   ", $1, $2, $5, $6}'; }

snap start
cd "$F/replay" || exit 2
echo "== 1. replay ABBA rerun $(date +%T)"
LOCK=0 sh ab_replay.sh traces/wex1-3x128.trace "$F/replay/runs/ab2" || exit 1

echo "== 2. replay variants $(date +%T)"
V=$F/validate/runs-variants
mkdir -p "$V"
U1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
U2='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
run() {   # name, replay args
  n=$1; shift
  echo "== $n $(date +%T) $*"
  build/replay.exe "$@" --json "$V/$n.json" > "$V/$n.txt" || { cat "$V/$n.txt"; echo "rc: stopping"; exit 1; }
  cat "$V/$n.txt"
}
run A_l1    --wex "$U1" --trace traces/wex1-3x128.trace --layer-ms 4.9
run B_own1  --wex "$U2" --trace traces/wnr2-3x128.trace --batch
run B_pad1  --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch
run B_padl1 --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch --layer-ms 4.9
run B_padl2 --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch --layer-ms 4.9
run B_pad2  --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch
run B_own2  --wex "$U2" --trace traces/wnr2-3x128.trace --batch
run A_l2    --wex "$U1" --trace traces/wex1-3x128.trace --layer-ms 4.9

snap "before short ABBA"
echo "== 3. short ABBA rerun $(date +%T)"
T0=$SECONDS
bash "$F/short-abba/short_abba.sh" --no-lock || exit 1
echo "short ABBA wall $(( (SECONDS - T0) / 60 )) min $(( (SECONDS - T0) % 60 )) s"
snap end
