#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build the fuzzing harnesses of wex4/fuzz/ and run them. CPU only, no real model needed.
#
#   bash wex4/fuzz/run.sh [SECONDS]     libFuzzer: clang with -fsanitize=fuzzer,address,undefined; every harness
#                                       fuzzes for SECONDS (default 60) from the seed corpus. CC names the clang.
#   bash wex4/fuzz/run.sh --replay      no fuzzing engine: gcc, each harness runs every seed once, and a harness
#                                       with a custom mutator also runs that mutator on its seeds (a regression
#                                       test that needs no clang; WEX_CFLAGS adds flags, as in the other scripts)
#
# WEX_FUZZ_MUTATOR=0 turns the structured mutations of the custom mutators off (the control of a measurement).
# Everything goes to wex4/fuzz/out/ (git-ignored): the harnesses, seeds/ (wex4/fuzz/make_seeds.py), corpus/NAME
# (the inputs libFuzzer keeps: it grows from run to run until out/ is deleted) and crashes/ (the reproducers).
# A harness that stops leaves crashes/NAME-crash-<sha1> (or -oom-, -timeout-); replay it with
#   wex4/fuzz/out/NAME wex4/fuzz/out/crashes/NAME-crash-<sha1>
# The exit status is 0 only when every harness ran its time without a crash and without a failed oracle.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
O=$ROOT/wex4/fuzz/out
PY=${PYTHON:-python}
HARNESSES="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_q38_wex"
# no -fopenmp: one thread per harness, so the OpenMP pragmas of wex4_verify.c and of the fast decoder are skipped
W="-Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unknown-pragmas"
# wex4/fuzz/engine: stand-ins for the engine headers that c/expert_store_wex.c includes (fuzz_store_wex);
# -pthread: the store's cache lock and the model registry of c/wex_decode.c
INC="-I $ROOT/c -I $ROOT/wex4/research/fast-cpu-decode -I $ROOT/wex4/fuzz/engine -pthread"
# the AVX2 kernels of the fast WNR3 decoder exist on x86 only; elsewhere fuzz_wnr3_record compares its scalar kernel
case $(uname -m) in
  x86_64|amd64|i?86) MARCH="-march=haswell" ;;
  *) MARCH="" ;;
esac
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) EXE=.exe ;;
  *) EXE= ;;
esac
mkdir -p "$O/crashes"
export TMP="$O" TEMP="$O"   # gcc from Git Bash cannot create temporary files under C:\WINDOWS
rm -rf "$O/seeds"
"$PY" -I "$ROOT/wex4/fuzz/make_seeds.py" "$O/seeds"
cd "$ROOT"   # fuzz_wnr3_record finds its default base file from here
seeds_of() {   # sets SEEDS, the seed folders of one harness
  case $1 in
    fuzz_wnr3_record) SEEDS=("$O/seeds/wnr3") ;;
    fuzz_store_wex) SEEDS=("$O/seeds/engine") ;;
    fuzz_q38_wex) SEEDS=("$O/seeds/engine" "$O/seeds/file") ;;
    *) SEEDS=("$O/seeds/file") ;;
  esac
}

if [ "${1-}" = --replay ]; then
  for h in $HARNESSES; do
    gcc -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE $INC "$ROOT/wex4/fuzz/$h.c" -o "$O/$h-replay$EXE"
    echo "== $h"
    seeds_of "$h"
    files=()
    for d in "${SEEDS[@]}"; do files+=("$d"/*); done
    "$O/$h-replay$EXE" "${files[@]}"
  done
  echo "fuzz replay passed"
  exit 0
fi

SECONDS_EACH=${1:-60}   # a positive number without leading zeros: libFuzzer reads -max_total_time=0 as no limit
case $SECONDS_EACH in
  ''|*[!0-9]*|0*) echo "usage: bash wex4/fuzz/run.sh [SECONDS | --replay]" >&2; exit 2 ;;
esac
CC=${CC:-clang}
SAN="-g -O1 -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined"
for h in $HARNESSES; do
  "$CC" $W $SAN $MARCH $INC "$ROOT/wex4/fuzz/$h.c" -o "$O/$h$EXE"
done
failed=""
for h in $HARNESSES; do
  # the largest seed file is 843,776 B; a WNR3 record of the base file decodes to 13,056 B (16,384 aligned)
  if [ "$h" = fuzz_wnr3_record ]; then max_len=16384; else max_len=1048576; fi
  # the magic numbers of the containers and records, for the harnesses whose input is a whole file
  if [ "$h" = fuzz_wnr3_record ]; then dict=(); else dict=("-dict=$ROOT/wex4/fuzz/wex.dict"); fi
  seeds_of "$h"
  mkdir -p "$O/corpus/$h"
  echo "== $h: $SECONDS_EACH s"
  "$O/$h$EXE" -max_total_time="$SECONDS_EACH" -max_len=$max_len -timeout=60 -rss_limit_mb=4096 ${dict[@]+"${dict[@]}"} \
      -artifact_prefix="$O/crashes/$h-" -print_final_stats=1 "$O/corpus/$h" "${SEEDS[@]}" \
      > "$O/$h.log" 2>&1 || failed="$failed $h"
  # the last progress line (cov: the instrumented edges reached), libFuzzer's counters, the harness's own summary
  grep -E '^#[0-9]+.* cov: ' "$O/$h.log" | tail -1 | sed -E 's/^#[0-9]+[[:space:]]+[A-Za-z]+[[:space:]]+(cov: [0-9]+ ft: [0-9]+ corp: [^ ]+).*/\1/' || true
  grep -E '^(stat::number_of_executed_units|stat::new_units_added|stat::peak_rss_mb|fuzz_[a-z0-9_]+: )' "$O/$h.log" || true
done
if [ -n "$failed" ]; then
  for h in $failed; do echo "== $h stopped; the end of its log:"; tail -40 "$O/$h.log"; done
  echo "FAILED:$failed (reproducers in $O/crashes)" >&2
  exit 1
fi
echo "fuzz run passed: $(echo $HARNESSES | wc -w) harnesses, $SECONDS_EACH s each"
