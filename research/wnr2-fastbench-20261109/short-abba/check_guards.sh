#!/usr/bin/env bash
# Function-level checks of short_abba.sh's guards, no engine and no machine probe: busy, ws_gb, sampler, arm
# and COVERED are taken from short_abba.sh; nvidia-smi, tasklist, avail and sleep are stubs reading files in
# runs/check-guards. Checks: VRAM gate in the quiet check, start gate timeout -> INVALID before the engine,
# refusal of an existing dump, sampler RAM and VRAM hits (and no hit before the baseline), COVERED.
#   bash check_guards.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SA=$HERE/short_abba.sh PY=C:/Python314/python.exe
T=$HERE/runs/check-guards
rm -rf "$T" && mkdir -p "$T/out" "$T/dump" || exit 2
eval "$(sed -n '/^BUSY=/p; /^busy() {/,/^}/p; /^vram_mib() /p; /^ws_gb() {/,/^}/p; /^sampler() {/,/^}/p; /^arm() {/,/^}/p' "$SA")"
eval "$(sed -n "/^COVERED='/,/argv\[2:\]))'\$/p" "$SA")"
nvidia-smi() { case "$*" in *PIDS*) ;; *utilization.gpu,memory.used*) echo "$(cat "$T/util"), $(cat "$T/mem")" ;; *memory.used*) cat "$T/mem" ;; esac; }
tasklist() {   # one engine of ours, PID 4242, 14.0 GB working set; other processes from $T/procs
  if [ "$1" = //fi ]; then [ "$2" != "PID eq 4242" ] || echo '"deepseek_v4.exe","4242","Console","1","14.680.064 K"'
  else cat "$T/procs"; fi
}
avail() { cat "$T/avail"; }
diskbytes() { echo "C:=0 U:=0"; }
pass=0 fail=0
check() { if eval "$2"; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1"; fi; }
set_probe() { echo "$1" > "$T/util"; echo "$2" > "$T/mem"; echo "$3" > "$T/avail"; : > "$T/procs"; }

VLIM=4096 CUDA=1 RAM_GB=16 OUT=$T/out DUMP=$T/dump DUMPS=() MODEL=m PY_REAL=$PY
set_probe 2 5000 20.0
check 'quiet check flags memory.used >= the VRAM gate' '[[ "$(busy " " 1)" == *"5000 MiB >= 4096 MiB (VRAM gate)"* ]]'
set_probe 2 1000 20.0
check 'quiet check passes under the VRAM gate' '[ -z "$(busy " " 1)" ]'

sleep() { :; }   # the start gate's 31 polls run at once
PY=false         # an engine start would run this
set_probe 2 1000 10.0
arm A1 x '' > "$T/arm1.txt" 2>&1; rc=$?
check 'start gate: RAM 10 < 16.3 times out with rc 1' '[ $rc = 1 ]'
check 'start gate: INVALID says the engine was not started' 'grep -q "start gate still closed after 15 min.*engine not started.*available RAM 10.0 GB < 16 + 0.3" "$OUT/INVALID"'
check 'start gate: no meta.json, so nothing started' '[ ! -e "$OUT/A1.meta.json" ]'
rm -f "$OUT/INVALID"; set_probe 2 6000 20.0
arm A1 x '' > "$T/arm2.txt" 2>&1
check 'start gate: VRAM 6000 MiB keeps it closed' 'grep -q "GPU memory used 6000 MiB" "$OUT/INVALID"'
rm -f "$OUT/INVALID"; set_probe 2 1000 20.0; echo keep > "$DUMP/A1.logits"
arm A1 x '' > "$T/arm3.txt" 2>&1; rc=$?
check 'existing dump: refused, rc 1, file left alone' '[ $rc = 1 ] && [ "$(cat "$DUMP/A1.logits")" = keep ] && grep -q "already exists, refused" "$OUT/INVALID"'
check 'meta.json records memory.used and the gate' 'grep -q "\"vram_used_mib\": 1000, \"vram_gate_mib\": 4096" "$OUT/A1.meta.json"'
unset -f sleep

rm -f "$OUT/INVALID" "$OUT/quiet.log"; echo 4242 > "$OUT/A1.pid"; set_probe 50 28000 3.0
sleep() { command sleep 1; }
sampler & S=$!
command sleep 2.5
check 'sampler: no hit before the baseline (first prompt not done)' '[ ! -e "$OUT/INVALID" ]'
echo '{"id": "w0"}' > "$OUT/A1.jsonl"; command sleep 2
echo 31000 > "$T/mem"; command sleep 2
check 'sampler: baseline 28000 MiB, +3000 is no hit' '[ ! -e "$OUT/INVALID" ] && grep -q "baseline 28000 MiB" "$OUT/quiet.log"'
echo 31100 > "$T/mem"; command sleep 2
check 'sampler: +3100 MiB over the baseline is a VRAM hit' 'grep -q "GPU memory used 31100 MiB > A1.s baseline 28000 MiB + 3072" "$OUT/INVALID"'
rm -f "$OUT/INVALID"; echo 30000 > "$T/mem"; echo 0.5 > "$T/avail"; command sleep 2
check 'sampler: available 0.5 + engine 14.0 < 15 is a RAM hit' 'grep -q "RAM available 0.5 GB + engine working set 14.0 GB < RAM_GB 16 - 1" "$OUT/INVALID"'
kill $S; wait $S 2>/dev/null; unset -f sleep
check 'sampler: quiet.log has GPU, RAM and working set' 'grep -q "GPU 50%, 28000 MiB (engine A1); RAM available 3.0 GB + engine working set 14.0 GB" "$OUT/quiet.log"'

PY=$PY_REAL
printf 'abc' > "$DUMP/B1.logits"
s=$("$PY" -I -c 'import hashlib; print(hashlib.sha256(b"abc").hexdigest())')
echo "{\"logits_sha256\": {\"B1\": \"$s\"}, \"tokens\": {}}" > "$OUT/tokens.json"
check 'COVERED: dump in tokens.json with the same sha256' '"$PY" -I -c "$COVERED" "$OUT/tokens.json" "$DUMP/B1.logits"'
check 'COVERED: a dump missing from tokens.json' '! "$PY" -I -c "$COVERED" "$OUT/tokens.json" "$DUMP/B1.logits" "$DUMP/A1.logits"'
printf 'abd' > "$DUMP/B1.logits"
check 'COVERED: another sha256' '! "$PY" -I -c "$COVERED" "$OUT/tokens.json" "$DUMP/B1.logits"'
check 'COVERED: no tokens.json' '! "$PY" -I -c "$COVERED" "$OUT/none.json" "$DUMP/B1.logits" 2>/dev/null'
echo "$pass ok, $fail failed"
[ $fail = 0 ]
