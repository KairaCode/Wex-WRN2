#!/bin/sh
# Validate step 5: replay variants on U:, interleaved, under heavy.lock (lock.sh: waits, releases on exit).
#   sh variants.sh OUTDIR
# A_l    WEX1 trace on WEX1 (per-record lanes), --layer-ms 4.9 (record run: (decprof total - moeprof wait) / 40)
# B_own  WNR2-recorded trace on WNR2 --batch (WNR2's own read set, as the tap sees it)
# B_pad  same, padded to the engine's read counter (validate/pad_trace.py, +12,887 wasted reads)
# B_padl B_pad with --layer-ms 4.9
set -u
cd "$(dirname "$0")/../replay"
OUT=$1
mkdir -p "$OUT"
U1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
U2='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
. '<KAIRA>'/lab/wnr3-5090-20261107/lock.sh
take_lock "wnr2-fastbench validate variants.sh (8 replays, ~25 min)" || exit 3
run() {   # name, replay args
  n=$1; shift
  echo "== $n $(date +%T) $*"
  build/replay.exe "$@" --skip 15 --json "$OUT/$n.json" > "$OUT/$n.txt" || { cat "$OUT/$n.txt"; echo "rc $?: stopping"; exit 1; }
  cat "$OUT/$n.txt"
}
run A_l1    --wex "$U1" --trace traces/wex1-3x128.trace --layer-ms 4.9
run B_own1  --wex "$U2" --trace traces/wnr2-3x128.trace --batch
run B_pad1  --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch
run B_padl1 --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch --layer-ms 4.9
run B_padl2 --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch --layer-ms 4.9
run B_pad2  --wex "$U2" --trace traces/wnr2-3x128.pad12887.trace --batch
run B_own2  --wex "$U2" --trace traces/wnr2-3x128.trace --batch
run A_l2    --wex "$U1" --trace traces/wex1-3x128.trace --layer-ms 4.9
