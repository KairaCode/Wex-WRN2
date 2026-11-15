#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build the fuzzing harnesses of wex4/fuzz/ and run them. CPU only, no real model needed.
#
#   bash wex4/fuzz/run.sh [SECONDS]     libFuzzer: clang with -fsanitize=fuzzer,address,undefined; every harness
#                                       fuzzes for SECONDS (default 60) from the seed corpus. CC names the clang.
#   bash wex4/fuzz/run.sh --replay      no fuzzing engine: gcc, each harness runs every seed once, and a harness
#                                       with a custom mutator also runs that mutator on its seeds (a regression
#                                       test that needs no clang; CC names another compiler and WEX_CFLAGS adds
#                                       flags, as in the other scripts: scripts/sanitize_c.sh runs it that way).
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built.
#                                       On Windows it then replays the two engine harnesses once more with a CUDA
#                                       helper: c/tests/fake_wex_cuda.c, the CPU stand-in, built as a DLL
#
# WEX_FUZZ_COLIBRI=<dir> (a colibri checkout) or WEX_FUZZ_COLIBRI=fetch (colibri at COLIBRI_COMMIT below, only its
# c/*.h and c/expert_store_registry.c, fetched with git into out/colibri): --replay then also builds the two engine
# harnesses against that tree's c/ folder instead of the stand-ins of engine/ and replays their seeds, and builds
# engine_layout.c against both: the two must print the same declarations, or the replay fails. Nothing of colibri
# is copied into the repository.
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

# The colibri commit the stand-ins of engine/ were checked against (dev, 2026-11-15); WEX_FUZZ_COLIBRI=fetch takes it
COLIBRI_URL=https://github.com/JustVugg/colibri
COLIBRI_COMMIT=98327913f980514970173b6df7bf67210ad0dc16
replay_one() {   # harness, executable, then extra environment assignments: the harness on its seeds
  local h=$1 exe=$2
  shift 2
  seeds_of "$h"
  files=()
  for d in "${SEEDS[@]}"; do files+=("$d"/*); done
  env "$@" "$exe" "${files[@]}"
}
engine_extras() {   # --replay, after the five harnesses
  local h tree log
  if [ -n "$EXE" ]; then
    # c/wex_decode.c loads a CUDA helper on Windows only. With the CPU stand-in the store decodes through the
    # helper's entry points, and the Qwen3.8 reader also runs its load waves (q38_paths.h); the check below fails
    # the replay when no record went through the helper (a DLL that did not load)
    "${CC:-gcc}" -O2 -shared -Wall -Wextra -I "$ROOT/c" "$ROOT/c/tests/fake_wex_cuda.c" -o "$O/fake_wex_cuda.dll"
    for h in fuzz_store_wex fuzz_q38_wex; do
      echo "== $h, CUDA helper c/tests/fake_wex_cuda.c"
      log="$O/$h-helper.log"
      replay_one "$h" "$O/$h-replay$EXE" "WEX_FUZZ_CUDA_LIB=$(cygpath -w "$O/fake_wex_cuda.dll")" > "$log" 2>&1 ||
        { cat "$log"; return 1; }
      cat "$log"
      grep -q 'CUDA helper on, [1-9][0-9]* records decoded through it' "$log" ||
        { echo "$h: no record was decoded through the helper" >&2; return 1; }
    done
  fi
  [ -n "${WEX_FUZZ_COLIBRI-}" ] || return 0
  if [ "$WEX_FUZZ_COLIBRI" = fetch ]; then
    tree=$O/colibri
    if [ "$(git -C "$tree" rev-parse HEAD 2>/dev/null)" != "$COLIBRI_COMMIT" ]; then
      rm -rf "$tree"
      git init -q "$tree"
      # the headers and the backend registry, not the engine's sources: scripts/build_cpu_check.sh out/colibri/c
      # builds c/tests/test_expert_store_wex.c with them (it finds the test's ../ headers through c/tests)
      git -C "$tree" config core.sparseCheckout true
      printf '/c/*.h\n/c/expert_store_registry.c\n' > "$tree/.git/info/sparse-checkout"
      git -C "$tree" fetch -q --depth 1 --filter=blob:none "$COLIBRI_URL" "$COLIBRI_COMMIT"
      git -C "$tree" checkout -q FETCH_HEAD
      mkdir -p "$tree/c/tests"
    fi
  else
    tree=$WEX_FUZZ_COLIBRI
  fi
  if [ ! -f "$tree/c/compat.h" ] || [ ! -f "$tree/c/expert_store.h" ]; then
    echo "WEX_FUZZ_COLIBRI: no colibri headers in $tree/c" >&2
    return 2
  fi
  echo "colibri tree: $tree, commit $(git -C "$tree" rev-parse HEAD 2>/dev/null || echo unknown)"
  # the declarations both harnesses take from the engine headers: the stand-ins and the tree must agree
  "${CC:-gcc}" -O1 $W ${WEX_CFLAGS-} -I "$ROOT/c" -I "$ROOT/wex4/fuzz/engine" "$ROOT/wex4/fuzz/engine_layout.c" \
      -o "$O/layout-standin$EXE"
  "${CC:-gcc}" -O1 $W ${WEX_CFLAGS-} -DWEX_FUZZ_COLIBRI -I "$ROOT/c" -I "$tree/c" "$ROOT/wex4/fuzz/engine_layout.c" \
      -o "$O/layout-colibri$EXE"
  "$O/layout-standin$EXE" > "$O/layout-standin.txt"
  "$O/layout-colibri$EXE" > "$O/layout-colibri.txt"
  if ! diff "$O/layout-standin.txt" "$O/layout-colibri.txt"; then
    echo "engine_layout: the stand-ins of engine/ and the headers of $tree/c differ (lines above)" >&2
    return 1
  fi
  echo "engine_layout: $(wc -l < "$O/layout-colibri.txt" | tr -d ' ') declarations, the same in the stand-ins and in the tree"
  for h in fuzz_store_wex fuzz_q38_wex; do
    # no engine/ on the include path: every engine header comes from the tree
    "${CC:-gcc}" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE -DWEX_FUZZ_COLIBRI \
        -I "$ROOT/c" -I "$ROOT/wex4/research/fast-cpu-decode" -I "$tree/c" -pthread "$ROOT/wex4/fuzz/$h.c" \
        -o "$O/$h-colibri$EXE"
    echo "== $h, against the headers of $tree/c"
    replay_one "$h" "$O/$h-colibri$EXE"
  done
}

if [ "${1-}" = --replay ]; then
  # the compiler, and which kernels of the fast WNR3 decoder fuzz_wnr3_record compares with the reference under
  # these flags (WEXF_KERNELS of wex_fast_decode.h): AVX2, NEON or scalar
  echo "compiler: $("${CC:-gcc}" --version | sed -n 1p)"
  KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
            "${CC:-gcc}" ${WEX_CFLAGS-} $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
  echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
  for h in $HARNESSES; do
    "${CC:-gcc}" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE $INC "$ROOT/wex4/fuzz/$h.c" -o "$O/$h-replay$EXE"
    echo "== $h"
    seeds_of "$h"
    files=()
    for d in "${SEEDS[@]}"; do files+=("$d"/*); done
    "$O/$h-replay$EXE" "${files[@]}"
  done
  engine_extras
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
