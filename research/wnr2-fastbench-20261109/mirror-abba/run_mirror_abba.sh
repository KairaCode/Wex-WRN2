#!/usr/bin/env bash
# Mirror ABBA driver (2026-11-09): A = WEX1 on U:, M (short_abba.sh's arm B) = WNR2 on U: + C: mirror, split 38/62.
#  1. quiet machine (short_abba.sh --check: no build/engine/compute app, GPU < 10 %, and the VRAM gate: GPU
#     memory.used < VRAM_MAX_GB, default 4, with no engine running; it is what sees desktop 3D apps, game editors and
#     3D app UIs, C+G apps on WDDM) and >= 13.3 GB available RAM, 5 good checks in a row, then heavy.lock, held
#     once for steps 2 and 3; retried every 60 s, gives up after WAIT_HOURS (default 10 h; exit 4, NOT_RUN);
#     short_abba.sh then gates each run again (quiet + VRAM + RAM_GB + 0.3 GB, 15 min, else INVALID unstarted);
#  2. smoke of M (short1, 16 tok): short_abba.sh's run check needs "Matching mirror container active",
#     "[v4_wex] read split over 2 drives" and >= 10 % of the OS read bytes on each drive, else stop (exit 6);
#  3. short ABBA A1 B1 B2 A2, w0 + 7 prompts x 64 tok, analysed by analyze_short.py.
# NVMe temperatures of C: (disk 3) and U: (disk 5) every 10 s into each step's dir (nvme_temps.log).
# Engine env = short_abba.sh COMMON (= the v41gpu profile env keys) + the profile's ctx 32768; the installed engine
# (--app-engine: sha256-checked copies in mirror-abba/bin).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SA=$HERE/../short-abba/short_abba.sh
LOCK='<KAIRA>'/lab/v41-gpu-20261107/heavy.lock
TEMP_PY='<KAIRA>'/lab/v41-gpu-20261107/ab/nvme_temp.py
PY=C:/Python314/python.exe
STAMP=$(date +%Y%m%d-%H%M%S)
RUNS=$HERE/runs SMOKE=$HERE/runs/smoke-$STAMP ABBA=$HERE/runs/abba-$STAMP
mkdir -p "$RUNS" || exit 2
exec > >(tee -a "$RUNS/driver-$STAMP.log") 2>&1
ARMS=(--app-engine --a 'U:\kaira_wex\v41-reap\v4_flash_experts.wex' --b 'U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
      --b-env 'COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap COLI_WEX_WEIGHTS=38,62')
export CTX=32768   # v41gpu profile "ctx"
export VRAM_MAX_GB=${VRAM_MAX_GB:-4}   # short_abba.sh's VRAM gate, in --check (this wait) and before each run
WAIT_HOURS=${WAIT_HOURS:-10}
case $WAIT_HOURS in ''|*[!0-9]*) echo "WAIT_HOURS is a whole number of hours, got '$WAIT_HOURS'"; exit 2 ;; esac
END=${DEADLINE_EPOCH:-$(( $(date +%s) + WAIT_HOURS * 3600 ))} MINE= TPID=   # DEADLINE_EPOCH: a restart keeps the first deadline
left() { local l=$(( END - $(date +%s) )); echo $(( l > 60 ? l : 60 )); }
temps() { mkdir -p "$1"; ( while :; do "$PY" -I "$TEMP_PY" 3=C: 5=U: | tr -d '\r' | grep -E ' [CU]: ' | sed "s/^/$(date +%T) /"
                                       sleep 10; done > "$1/nvme_temps.log" 2>&1 ) & TPID=$!; }
cleanup() {
  [ -z "$TPID" ] || kill "$TPID" 2>/dev/null
  if [ -n "$MINE" ]; then
    if [ "$(cat "$LOCK" 2>/dev/null)" = "$MINE" ]; then rm -f "$LOCK"; echo "lock: released $(date +%T)"
    else echo "lock: not ours any more, left alone ($(cat "$LOCK" 2>/dev/null))"; fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "driver $STAMP: CTX=$CTX VRAM_MAX_GB=$VRAM_MAX_GB, waits until $(date -d "@$END" +'%F %T') (WAIT_HOURS=$WAIT_HOURS); ${ARMS[*]}"
last= ok=0
while :; do
  c=$(bash "$SA" --check "${ARMS[@]}" 2>&1)
  q=$(sed -n 's/^quiet-machine check [0-9:]*: //p' <<< "$c")
  # the RAM rule's low arm (13) needs 13.3 GB available at each start, else the engine budget drops (min(RAM_GB, OS
  # available)); without this a non-GPU app holding RAM (desktop 3D apps, 14 GB at 19:40) gets a lock held for 25 min
  r=$(sed -n 's/^RAM check [0-9:]*: \([0-9.]*\) GB available.*/\1/p' <<< "$c")
  [ "$q" = quiet ] && ! awk -v a="$r" 'BEGIN { exit !(a + 0 >= 13.3) }' && q="RAM $r GB available < 13.3"
  # 21:18 run: 16 GB at the check, 3-13 GB a minute later (desktop 3D apps + game editors in use): 5 good checks in a row first
  if [ "$q" = quiet ]; then ok=$((ok + 1)); [ $ok -ge 5 ] || q="quiet + RAM $r GB: good check $ok of 5"; else ok=0; fi
  if [ "$q" = quiet ] && ( set -o noclobber; printf '%s %s\n' "mirror ABBA (wnr2-fastbench): smoke M + A M M A 7x64 tok, ~45 min" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; then
    MINE=$(cat "$LOCK"); echo "lock: taken $(date +%T): $MINE"; break
  fi
  if [ "$q" = quiet ]; then why="heavy.lock held by: $(cat "$LOCK" 2>/dev/null)"; else why="busy: ${q:-check failed}"; fi
  [ "$(date +%s)" -ge "$END" ] && { echo "NOT_RUN $(date +%T): gave up at the deadline (WAIT_HOURS=$WAIT_HOURS); last blocker: $why"; exit 4; }
  [ "$why" = "$last" ] || echo "wait $(date +%T): $why"
  last=$why
  sleep 60
done

temps "$SMOKE"
bash "$SA" --no-lock "${ARMS[@]}" --seq B1 --ids short1 --tokens 16 --warmup '' --dump none --out "$SMOKE" --quiet-max "$(left)"
echo "smoke exit $? (analyze_short.py prints INVALID here: no A/B pair; the run check is what counts)"
kill "$TPID" 2>/dev/null; TPID=
grep -a 'Matching mirror\|read split over\|COLI_WEX_WEIGHTS\|Container v[0-9]* active' "$SMOKE/B1.stderr.log" 2>/dev/null | cut -c1-200
if [ -e "$SMOKE/INVALID" ] || ! grep -q '^rc=0 ' "$SMOKE/short_abba.log" 2>/dev/null; then
  echo "SMOKE FAILED $(date +%T): $(cat "$SMOKE/INVALID" 2>/dev/null || echo 'no run (see short_abba.log)')"; exit 6
fi
echo "smoke OK $(date +%T): mirror registered, read split over 2 drives, both drives served >= 10 % of the bytes"

temps "$ABBA"
bash "$SA" --no-lock "${ARMS[@]}" --out "$ABBA" --quiet-max "$(left)"
rc=$?
kill "$TPID" 2>/dev/null; TPID=
echo "ABBA exit $rc $(date +%T)"
exit $rc
