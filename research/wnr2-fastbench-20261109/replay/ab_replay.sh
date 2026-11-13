#!/bin/sh
# WEX1 vs WNR2 replay of one trace on the real containers (U:), ABBA, under heavy.lock.
#   [SEQ="A1 B1 B2 A2"] [A_OPTS=] [B_OPTS=--batch] [VERIFY=64] sh ab_replay.sh TRACE OUTDIR [options for both arms]
# Defaults follow W3: arm A (WEX1) per-record lanes, arm B (WNR2) V4_WEX_BATCH's batched decode.
# VERIFY=N first checks N WNR2 records, decoded by the helper (single and batch API), byte for byte
# against the WEX1 (0 skips it). Each run writes OUTDIR/<run>.json and .txt; the summary goes to
# OUTDIR/summary.txt. WEX1=/WNR2= override the containers; LOCK=0 skips heavy.lock (tiny fixture only).
set -u
cd "$(dirname "$0")"
[ $# -ge 2 ] || { sed -n 2,7p "$0"; exit 2; }
TRACE=$1; OUT=$2; shift 2
EXE=build/replay.exe
PY=C:/Python314/python.exe
WEX1=${WEX1:-'U:\kaira_wex\v41-reap\v4_flash_experts.wex'}
WNR2=${WNR2:-'U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'}
SEQ=${SEQ:-A1 B1 B2 A2}
A_OPTS=${A_OPTS:-}
B_OPTS=${B_OPTS---batch}
mkdir -p "$OUT"
if [ "${LOCK:-1}" != 0 ]; then
  . '<KAIRA>'/lab/wnr3-5090-20261107/lock.sh
  take_lock "wnr2-fastbench ab_replay.sh ($TRACE, $SEQ)" || exit 3
fi

run() {   # out-file, replay arguments: stops the sequence on a failure or a mismatch
  f=$1; shift
  "$EXE" "$@" > "$OUT/$f"; rc=$?
  cat "$OUT/$f"
  [ $rc = 0 ] || { echo "== replay rc=$rc: stopping"; exit 1; }
}
if [ "${VERIFY:-64}" != 0 ]; then
  run verify.txt --wex "$WNR2" --ref "$WEX1" --verify "${VERIFY:-64}"
fi
for s in $SEQ; do
  case $s in A*) w=$WEX1 x=$A_OPTS ;; B*) w=$WNR2 x=$B_OPTS ;; *) echo "bad run $s"; exit 2 ;; esac
  echo "== $s $(date +%T) $w $x $*"
  run "$s.txt" --wex "$w" --trace "$TRACE" $x "$@" --json "$OUT/$s.json"
done
"$PY" -I - "$OUT" $SEQ <<'EOF' | tee "$OUT/summary.txt"
import json, sys
out, runs = sys.argv[1], sys.argv[2:]
arms = {}
for r in runs:
    arms.setdefault(r[0], []).append(json.load(open(f'{out}/{r}.json')))
def avg(xs, f): return sum(f(x) for x in xs) / len(xs)
keys = [('ms/token mean', lambda j: j['ms_per_token']['mean']), ('ms/token p50', lambda j: j['ms_per_token']['p50']),
        ('ms/token p95', lambda j: j['ms_per_token']['p95']), ('layer wait ms/token', lambda j: j['wait_ms_per_token']),
        ('MB/token', lambda j: j['mb_per_token']), ('read GB/s wall', lambda j: j['read_gbs_wall']),
        ('demand read p50 ms', lambda j: j['read_ms']['demand_p50']),
        ('finish ms/record', lambda j: (j['decode']['ms'] + j['h2d']['ms']) / max(1, j['records_finished']))]
print(f"{'':22}" + ''.join(f'{a + " (" + arms[a][0]["format"] + ")":>16}' for a in sorted(arms)) + f"{'B vs A':>10}")
for name, f in keys:
    v = {a: avg(arms[a], f) for a in arms}
    d = f"{100 * (v['B'] / v['A'] - 1):+9.1f}%" if 'A' in v and 'B' in v and v['A'] else ''
    print(f'{name:22}' + ''.join(f'{v[a]:16.2f}' for a in sorted(arms)) + d)
for a in sorted(arms):   # run-to-run spread of the headline number within an arm
    ms = [j['ms_per_token']['mean'] for j in arms[a]]
    print(f'arm {a} ms/token per run: ' + ' '.join(f'{m:.2f}' for m in ms) +
          (f'  spread {100 * (max(ms) - min(ms)) / min(ms):.1f}%' if len(ms) > 1 else ''))
EOF
