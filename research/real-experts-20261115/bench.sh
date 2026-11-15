#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Ratio and decode speed on the records that run.sh extracted (WORK_DIR/<checkpoint>/rec-all: the WNR3 records
# made with the model trained on the 8 experts, and their slabs), per checkpoint:
#   bench_real.c wnr3    the reference WNR3 decoder and the fast CPU decoder, 1 thread and all threads
#   bench_real.c codecs  lz4, lz4hc, brotli, zstd (libzstd)
#   baselines.py         zstd (compression.zstd), lzma, zlib, bz2 of the Python standard library
# On Windows the machine's total CPU load is logged once a second (cpuload.ps1) into OUT/cpuload.csv, and
# before each checkpoint the mean of 10 s of it is printed. WAIT_IDLE=P waits before each checkpoint until
# that mean is below P percent (checking every 30 s, at most WAIT_MAX minutes, default 120; then it runs
# anyway and says so).
# Needs: gcc with OpenMP and AVX2 (x86-64), the lz4, brotli and zstd libraries, Python 3.14.
# Usage: bash research/real-experts-20261115/bench.sh [WORK_DIR] [RUNS]   (default out/ next to this script, 11)
# Output: WORK_DIR/bench-<date>-<time>/ (system.txt, <checkpoint>-wnr3.txt, -codecs.txt, -stdlib.txt, cpuload.csv)
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
W=$(cd "${1:-$HERE/out}" && pwd)
RUNS=${2:-11}
PY=${PYTHON:-python}
CC=${CC:-gcc}
OUT=$W/bench-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
export TMP="$OUT" TEMP="$OUT"
"$CC" -std=c99 -O3 -march=haswell -fopenmp -Wall -Wextra -Wno-unused-function -I "$ROOT/c" \
    -I "$ROOT/wex4/research/fast-cpu-decode" "$HERE/bench_real.c" -o "$OUT/bench_real.exe" -llz4 -lbrotlienc -lbrotlidec -lzstd
{
  echo "compiler: $("$CC" --version | sed -n 1p), flags -std=c99 -O3 -march=haswell -fopenmp"
  echo "python: $("$PY" -I -c 'import sys; print(sys.version.split()[0])')"
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "runs per row: $RUNS"
} | tee "$OUT/system.txt"
SAMPLER=0
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*)
    SAMPLER=1
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/cpuload.ps1")" \
        "$(cygpath -w "$OUT/cpuload.csv")" "$(cygpath -w "$OUT/stop")" &
    trap 'touch "$OUT/stop"' EXIT ;;
esac
load10() {   # mean CPU load of the next 10 s of the log
  local before after
  before=$(cat "$OUT/cpuload.csv" 2>/dev/null | wc -l)
  sleep 10
  after=$(cat "$OUT/cpuload.csv" 2>/dev/null | wc -l)
  tail -n $((after - before)) "$OUT/cpuload.csv" | awk -F, '{ s += $2; n++ } END { if (n) printf "%.0f", s / n; else print "unknown" }'
}
for n in deepseek-v41 qwen3-235b minimax-m21; do
  R=$W/$n/rec-all
  names=$(ls "$R"/*.rec | sed 's/\.rec$//')
  if [ $SAMPLER = 1 ]; then
    load=$(load10)
    if [ -n "${WAIT_IDLE-}" ]; then
      waited=0
      while [ "$load" = unknown ] || [ "$load" -ge "$WAIT_IDLE" ]; do
        if [ $waited -ge $(( ${WAIT_MAX:-120} * 60 )) ]; then echo "$(date +%H:%M:%S) still at $load% after ${WAIT_MAX:-120} min: running anyway" | tee -a "$OUT/system.txt"; break; fi
        echo "$(date +%H:%M:%S) CPU load $load%, waiting" | tee -a "$OUT/system.txt"
        sleep 20; waited=$((waited + 30))
        load=$(load10)
      done
    fi
    echo "$(date +%H:%M:%S) $n: mean CPU load of the 10 s before: $load%" | tee -a "$OUT/system.txt"
  fi
  echo "$(date +%H:%M:%S) $n: wnr3" | tee -a "$OUT/system.txt"
  "$OUT/bench_real.exe" wnr3 "$RUNS" "$R/model.wnm2" $names > "$OUT/$n-wnr3.txt"
  echo "$(date +%H:%M:%S) $n: codecs" | tee -a "$OUT/system.txt"
  "$OUT/bench_real.exe" codecs "$RUNS" $names > "$OUT/$n-codecs.txt"
  echo "$(date +%H:%M:%S) $n: stdlib" | tee -a "$OUT/system.txt"
  "$PY" -I "$HERE/baselines.py" "$RUNS" "$R" > "$OUT/$n-stdlib.txt"
done
echo "$(date +%H:%M:%S) done" | tee -a "$OUT/system.txt"
echo "bench.sh: results in $OUT"
