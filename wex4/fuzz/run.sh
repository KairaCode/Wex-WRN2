#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build the fuzzing harnesses of wex4/fuzz/ and run them. CPU only, no real model needed.
#
#   bash wex4/fuzz/run.sh [SECONDS]     libFuzzer: clang with -fsanitize=fuzzer,address,undefined; six runs of
#                                       SECONDS each (default 60) from the seed corpus: every harness once, and
#                                       fuzz_store_wex a second time, on the containers of the older codecs with
#                                       a corpus of its own (the run fuzz_store_wex-legacy). CC names the clang
#                                       (macOS: the clang of a Homebrew LLVM; Apple clang has no libFuzzer).
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built
#   bash wex4/fuzz/run.sh --replay      no fuzzing engine: gcc, each harness runs every seed once, and a harness
#                                       with a custom mutator also runs that mutator on its seeds (a regression
#                                       test that needs no clang; CC names another compiler and WEX_CFLAGS adds
#                                       flags, as in the other scripts: scripts/sanitize_c.sh runs it that way).
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built
#
# WEX_FUZZ_MUTATOR=0 turns the structured mutations of the custom mutators off (the control of a measurement).
# Everything goes to wex4/fuzz/out/ (git-ignored): the harnesses, seeds/ (wex4/fuzz/make_seeds.py, with the
# program built from legacy_seeds.c for the containers of WNR and WNR2 records), corpus/RUN
# (the inputs libFuzzer keeps: it grows from run to run until out/ is deleted) and crashes/ (the reproducers).
# A run that stops leaves crashes/RUN-crash-<sha1> (or -oom-, -timeout-); replay it with the harness of the run
#   wex4/fuzz/out/NAME wex4/fuzz/out/crashes/RUN-crash-<sha1>
# (RUN is NAME, or fuzz_store_wex-legacy for NAME fuzz_store_wex).
# The exit status is 0 only when every run ran its time without a crash and without a failed oracle.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
O=$ROOT/wex4/fuzz/out
PY=${PYTHON:-python}
HARNESSES="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_q38_wex"
# the runs: a harness, or HARNESS-NAME for a second run of a harness on other seeds, with a corpus of its own.
# The older codecs of fuzz_store_wex get their own run so that they do not take the time of its WNR3 seeds
RUNS="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_store_wex-legacy fuzz_q38_wex"
# no -fopenmp: one thread per harness, so the OpenMP pragmas of wex4_verify.c and of the fast decoder are skipped,
# unless WEX_CFLAGS brings it in the --replay mode (scripts/sanitize_c.sh tsan does)
W="-Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unknown-pragmas"
# wex4/fuzz/engine: stand-ins for the engine headers that c/expert_store_wex.c includes (fuzz_store_wex);
# -pthread: the store's cache lock and the model registry of c/wex_decode.c
INC="-I $ROOT/c -I $ROOT/wex4/research/fast-cpu-decode -I $ROOT/wex4/fuzz/engine -pthread"
# the AVX2 kernels of the fast WNR3 decoder need -march=haswell on x86; on AArch64 the same source builds its NEON
# kernel with no flag, on any other CPU fuzz_wnr3_record compares its scalar kernel
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
# legacy_seeds writes the seed containers with WNR and WNR2 records (make_seeds.py runs it). It is built with the
# compiler of the mode: gcc, or CC, with WEX_CFLAGS for --replay; clang, or CC, for a fuzzing run
if [ "${1-}" = --replay ]; then SEED_CC="${CC:-gcc} ${WEX_CFLAGS-}"; else SEED_CC="${CC:-clang}"; fi
$SEED_CC -O2 $W $INC "$ROOT/wex4/fuzz/legacy_seeds.c" -o "$O/legacy_seeds$EXE"
"$PY" -I "$ROOT/wex4/fuzz/make_seeds.py" "$O/seeds" "$O/legacy_seeds$EXE"
cd "$ROOT"   # fuzz_wnr3_record finds its default base file from here
seeds_of() {   # sets SEEDS, the seed folders of one run
  case $1 in
    fuzz_wnr3_record) SEEDS=("$O/seeds/wnr3") ;;
    fuzz_store_wex) SEEDS=("$O/seeds/engine") ;;
    fuzz_store_wex-legacy) SEEDS=("$O/seeds/legacy") ;;
    fuzz_q38_wex) SEEDS=("$O/seeds/engine" "$O/seeds/file") ;;
    *) SEEDS=("$O/seeds/file") ;;
  esac
}
harness_of() { H=${1%%-*}; }   # sets H, the harness of a run (no harness name has a dash)

if [ "${1-}" = --replay ]; then
  # the compiler, and which kernels of the fast WNR3 decoder fuzz_wnr3_record compares with the reference under
  # these flags (WEXF_KERNELS of wex_fast_decode.h): AVX2, NEON, NEON+SVE (WEX_CFLAGS=-march=armv8-a+sve) or scalar
  echo "compiler: $("${CC:-gcc}" --version | sed -n 1p)"
  KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
            "${CC:-gcc}" ${WEX_CFLAGS-} $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
  echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
  for h in $HARNESSES; do
    "${CC:-gcc}" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE $INC "$ROOT/wex4/fuzz/$h.c" -o "$O/$h-replay$EXE"
  done
  for run in $RUNS; do
    harness_of "$run"
    echo "== $run"
    seeds_of "$run"
    files=()
    for d in "${SEEDS[@]}"; do files+=("$d"/*); done
    "$O/$H-replay$EXE" "${files[@]}"
  done
  echo "fuzz replay passed"
  exit 0
fi

SECONDS_EACH=${1:-60}   # a positive number without leading zeros: libFuzzer reads -max_total_time=0 as no limit
case $SECONDS_EACH in
  ''|*[!0-9]*|0*) echo "usage: bash wex4/fuzz/run.sh [SECONDS | --replay]" >&2; exit 2 ;;
esac
CC=${CC:-clang}
# as in --replay: the compiler, and the kernels fuzz_wnr3_record compares with the reference under these flags
echo "compiler: $("$CC" --version | sed -n 1p)"
KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
          "$CC" $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
SAN="-g -O1 -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined"
for h in $HARNESSES; do
  "$CC" $W $SAN $MARCH $INC "$ROOT/wex4/fuzz/$h.c" -o "$O/$h$EXE"
done
failed=""
for run in $RUNS; do
  harness_of "$run"
  # the largest seed file is 843,776 B; a WNR3 record of the base file decodes to 13,056 B (16,384 aligned)
  if [ "$H" = fuzz_wnr3_record ]; then max_len=16384; else max_len=1048576; fi
  # the magic numbers of the containers and records, for the harnesses whose input is a whole file
  if [ "$H" = fuzz_wnr3_record ]; then dict=(); else dict=("-dict=$ROOT/wex4/fuzz/wex.dict"); fi
  seeds_of "$run"
  mkdir -p "$O/corpus/$run"
  echo "== $run: $SECONDS_EACH s"
  "$O/$H$EXE" -max_total_time="$SECONDS_EACH" -max_len=$max_len -timeout=60 -rss_limit_mb=4096 ${dict[@]+"${dict[@]}"} \
      -artifact_prefix="$O/crashes/$run-" -print_final_stats=1 "$O/corpus/$run" "${SEEDS[@]}" \
      > "$O/$run.log" 2>&1 || failed="$failed $run"
  # the last progress line (cov: the instrumented edges reached), libFuzzer's counters, the harness's own summary
  grep -E '^#[0-9]+.* cov: ' "$O/$run.log" | tail -1 | sed -E 's/^#[0-9]+[[:space:]]+[A-Za-z]+[[:space:]]+(cov: [0-9]+ ft: [0-9]+ corp: [^ ]+).*/\1/' || true
  grep -E '^(stat::number_of_executed_units|stat::new_units_added|stat::peak_rss_mb|fuzz_[a-z0-9_]+: )' "$O/$run.log" || true
done
if [ -n "$failed" ]; then
  for run in $failed; do echo "== $run stopped; the end of its log:"; tail -40 "$O/$run.log"; done
  echo "FAILED:$failed (reproducers in $O/crashes)" >&2
  exit 1
fi
echo "fuzz run passed: $(echo $HARNESSES | wc -w | tr -d ' ') harnesses in $(echo $RUNS | wc -w | tr -d ' ') runs, $SECONDS_EACH s each"   # macOS pads wc
