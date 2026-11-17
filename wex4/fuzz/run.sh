#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build the fuzzing harnesses of wex4/fuzz/ and run them. CPU only, no real model needed.
#
#   bash wex4/fuzz/run.sh [SECONDS]     libFuzzer: clang with -fsanitize=fuzzer,address,undefined; seven runs of
#                                       SECONDS each (default 60) from the seed corpus: every harness once, and
#                                       fuzz_store_wex a second time, on the containers of the older codecs with
#                                       a corpus of its own (the run fuzz_store_wex-legacy). CC names the clang
#                                       (macOS: the clang of a Homebrew LLVM; Apple clang has no libFuzzer).
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built
#   bash wex4/fuzz/run.sh --replay      no fuzzing engine: gcc, each harness runs every seed once, and a harness
#                                       with a custom mutator also runs that mutator on its seeds (a regression
#                                       test that needs no clang; CC names another compiler and WEX_CFLAGS adds
#                                       flags, as in the other scripts: scripts/sanitize_c.sh runs it that way).
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built.
#                                       On Windows it then replays the two engine harnesses once more with a CUDA
#                                       helper: c/tests/fake_wex_cuda.c, the CPU stand-in, built as a DLL
#   bash wex4/fuzz/run.sh --gpu [ROUNDS] fuzz_wnr3_cuda as in --replay with ROUNDS (default 4) mutator inputs per
#                                       seed, and every record also through the CUDA kernels on an NVIDIA GPU:
#                                       needs the driver and clang with its NVPTX target (CUDA_CLANG names it), not
#                                       the CUDA toolkit (wex4/fuzz/cuda_gpu.h). WEX_FUZZ_GPU_BUILD_ONLY=1 stops it
#                                       after the builds (the PTX and the harness), on a machine without a GPU
#
# fuzz_wnr3_cuda is C++ (the CUDA decoder built for the CPU): it is compiled with CXX, by default the C++ compiler
# that goes with CC (gcc -> g++, gcc-15 -> g++-15, clang -> clang++), from the host copy of c/wex_cuda.cuh and
# c/tools/wex_cuda_bridge.cu that cuda_host.py writes to out/wex_cuda_host/.
#
# WEX_FUZZ_COLIBRI=<dir> (a colibri checkout) or WEX_FUZZ_COLIBRI=fetch (colibri at COLIBRI_COMMIT below, only its
# c/*.h and c/expert_store_registry.c, fetched with git into out/colibri): --replay then also builds the two engine
# harnesses against that tree's c/ folder instead of the stand-ins of engine/ and replays their seeds, and builds
# engine_layout.c against both: the two must print the same declarations, or the replay fails. Nothing of colibri
# is copied into the repository.
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
HARNESSES="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_q38_wex fuzz_wnr3_cuda"
# the runs: a harness, or HARNESS-NAME for a second run of a harness on other seeds, with a corpus of its own.
# The older codecs of fuzz_store_wex get their own run so that they do not take the time of its WNR3 seeds
RUNS="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_store_wex-legacy fuzz_q38_wex fuzz_wnr3_cuda"
# no -fopenmp: one thread per harness, so the OpenMP pragmas of wex4_verify.c and of the fast decoder are skipped,
# unless WEX_CFLAGS brings it in the --replay mode (scripts/sanitize_c.sh tsan does)
W="-Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unknown-pragmas"
# wex4/fuzz/engine: stand-ins for the engine headers that c/expert_store_wex.c includes (fuzz_store_wex);
# -pthread: the store's cache lock and the model registry of c/wex_decode.c
INC="-I $ROOT/wex4/research/fast-cpu-decode -I $ROOT/c -I $ROOT/wex4/fuzz/engine -pthread"
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
# compiler of the mode: gcc, or CC, with WEX_CFLAGS for --replay and --gpu; clang, or CC, for a fuzzing run
case ${1-} in
  --replay|--gpu) SEED_CC="${CC:-gcc} ${WEX_CFLAGS-}" ;;
  *) SEED_CC="${CC:-clang}" ;;
esac
$SEED_CC -O2 $W $INC "$ROOT/wex4/fuzz/legacy_seeds.c" -o "$O/legacy_seeds$EXE"
"$PY" -I "$ROOT/wex4/fuzz/make_seeds.py" "$O/seeds" "$O/legacy_seeds$EXE"
# fuzz_wnr3_cuda also gets the WEX4 files of impl/go-wex4/testdata: ceil-form.wex4 holds the only WNR3 records in
# ceil form that are not synthetic seeds of the mutator
mkdir -p "$O/seeds/testdata"
cp "$ROOT"/impl/go-wex4/testdata/*.wex4 "$O/seeds/testdata/"
"$PY" -I "$ROOT/wex4/fuzz/cuda_host.py" "$ROOT/c" "$O/wex_cuda_host"
cd "$ROOT"   # fuzz_wnr3_record and fuzz_wnr3_cuda find their default base file from here
seeds_of() {   # sets SEEDS, the seed folders of one run
  case $1 in
    fuzz_wnr3_record) SEEDS=("$O/seeds/wnr3") ;;
    fuzz_store_wex) SEEDS=("$O/seeds/engine") ;;
    fuzz_store_wex-legacy) SEEDS=("$O/seeds/legacy") ;;
    fuzz_q38_wex) SEEDS=("$O/seeds/engine" "$O/seeds/file") ;;
    fuzz_wnr3_cuda) SEEDS=("$O/seeds/wnr3" "$O/seeds/file" "$O/seeds/testdata") ;;
    *) SEEDS=("$O/seeds/file") ;;
  esac
}
harness_of() { H=${1%%-*}; }   # sets H, the harness of a run (no harness name has a dash)
cxx_of() {   # the C++ compiler of the C compiler $1
  case $1 in
    *clang*) echo "${1%clang*}clang++${1##*clang}" ;;
    *gcc*) echo "${1%gcc*}g++${1##*gcc}" ;;
    *) echo c++ ;;
  esac
}
# compile harness $1 to $2 with the flags that follow: C with CC_USED, C++ (fuzz_wnr3_cuda) with CXX_USED
build() {
  local h=$1 out=$2
  shift 2
  if [ -f "$ROOT/wex4/fuzz/$h.cpp" ]; then
    "$CXX_USED" -std=c++17 "$@" -I "$ROOT/wex4/fuzz" -I "$O" "$ROOT/wex4/fuzz/$h.cpp" -o "$out"
  else
    "$CC_USED" "$@" "$ROOT/wex4/fuzz/$h.c" -o "$out"
  fi
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
engine_extras() {   # --replay, after the runs
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
        -I "$ROOT/wex4/research/fast-cpu-decode" -I "$ROOT/c" -I "$tree/c" -pthread "$ROOT/wex4/fuzz/$h.c" \
        -o "$O/$h-colibri$EXE"
    echo "== $h, against the headers of $tree/c"
    replay_one "$h" "$O/$h-colibri$EXE"
  done
}

if [ "${1-}" = --replay ]; then
  # the compiler, and which kernels of the fast WNR3 decoder fuzz_wnr3_record compares with the reference under
  # these flags (WEXF_KERNELS of wex_fast_decode.h): AVX2, NEON, NEON+SVE (WEX_CFLAGS=-march=armv8-a+sve) or scalar
  CC_USED=${CC:-gcc}
  CXX_USED=${CXX:-$(cxx_of "$CC_USED")}
  echo "compiler: $("$CC_USED" --version | sed -n 1p)"
  echo "C++ compiler of fuzz_wnr3_cuda: $("$CXX_USED" --version | sed -n 1p)"
  KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
            "$CC_USED" ${WEX_CFLAGS-} $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
  echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
  for h in $HARNESSES; do
    build "$h" "$O/$h-replay$EXE" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE $INC
  done
  for run in $RUNS; do
    harness_of "$run"
    echo "== $run"
    seeds_of "$run"
    files=()
    for d in "${SEEDS[@]}"; do files+=("$d"/*); done
    "$O/$H-replay$EXE" "${files[@]}"
  done
  engine_extras
  echo "fuzz replay passed"
  exit 0
fi

if [ "${1-}" = --gpu ]; then
  # fuzz_wnr3_cuda as in --replay, and every record also through the kernels of c/wex_cuda.cuh on an NVIDIA GPU
  # (cuda_gpu.h): clang (CUDA_CLANG, default clang++, with its NVPTX target) writes their PTX without the CUDA
  # toolkit, the driver compiles it at load time. ROUNDS (default 4) mutator inputs from each seed.
  ROUNDS=${2:-4}
  case $ROUNDS in
    ''|*[!0-9]*) echo "usage: bash wex4/fuzz/run.sh --gpu [ROUNDS]" >&2; exit 2 ;;
  esac
  CC_USED=${CC:-gcc}
  CXX_USED=${CXX:-$(cxx_of "$CC_USED")}
  CLANG_USED=${CUDA_CLANG:-clang++}
  P=wex4/fuzz/out/wex_cuda_ptx   # from the repository root (the working folder here): the harness's default
  mkdir -p "$P"
  printf '#include "cuda_device.h"\n' > "$P/cuda_runtime.h"   # what c/wex_cuda.cuh includes, here
  echo "C++ compiler of fuzz_wnr3_cuda: $("$CXX_USED" --version | sed -n 1p)"
  echo "PTX compiler: $("$CLANG_USED" --version | sed -n 1p)"
  for a in 86 75; do
    "$CLANG_USED" -x cuda --cuda-device-only --cuda-gpu-arch=sm_$a -nocudainc -nocudalib -O2 -S $W -I "$P" -I c \
        -I wex4/fuzz wex4/fuzz/cuda_device.cu -o "$P/sm$a.ptx"
  done
  build fuzz_wnr3_cuda "$O/fuzz_wnr3_cuda-gpu$EXE" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE -DWEX_FUZZ_GPU \
      -DFUZZ_MUTATOR_ROUNDS="$ROUNDS" $INC
  grep -c '\.entry ' "$P/sm86.ptx" "$P/sm75.ptx" | sed 's/^/kernels in /'
  if [ -n "${WEX_FUZZ_GPU_BUILD_ONLY-}" ]; then   # CI: no GPU there
    echo "fuzz GPU build passed (not run: WEX_FUZZ_GPU_BUILD_ONLY is set)"
    exit 0
  fi
  seeds_of fuzz_wnr3_cuda
  files=()
  for d in "${SEEDS[@]}"; do files+=("$d"/*); done
  echo "== fuzz_wnr3_cuda on the GPU, $ROUNDS mutator inputs per seed"
  WEX_FUZZ_GPU_PTX="$P" "$O/fuzz_wnr3_cuda-gpu$EXE" "${files[@]}"
  echo "fuzz GPU run passed"
  exit 0
fi

SECONDS_EACH=${1:-60}   # a positive number without leading zeros: libFuzzer reads -max_total_time=0 as no limit
case $SECONDS_EACH in
  ''|*[!0-9]*|0*) echo "usage: bash wex4/fuzz/run.sh [SECONDS | --replay | --gpu [ROUNDS]]" >&2; exit 2 ;;
esac
CC_USED=${CC:-clang}
CXX_USED=${CXX:-$(cxx_of "$CC_USED")}
# as in --replay: the compilers, and the kernels fuzz_wnr3_record compares with the reference under these flags
echo "compiler: $("$CC_USED" --version | sed -n 1p)"
echo "C++ compiler of fuzz_wnr3_cuda: $("$CXX_USED" --version | sed -n 1p)"
KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
          "$CC_USED" $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
SAN="-g -O1 -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined"
for h in $HARNESSES; do
  build "$h" "$O/$h$EXE" $W $SAN $MARCH $INC
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
  grep -E '^(stat::number_of_executed_units|stat::new_units_added|stat::peak_rss_mb|fuzz_[a-z0-9_]+( other paths| failing reads)?: )' "$O/$run.log" || true
done
if [ -n "$failed" ]; then
  for run in $failed; do echo "== $run stopped; the end of its log:"; tail -40 "$O/$run.log"; done
  echo "FAILED:$failed (reproducers in $O/crashes)" >&2
  exit 1
fi
echo "fuzz run passed: $(echo $HARNESSES | wc -w | tr -d ' ') harnesses in $(echo $RUNS | wc -w | tr -d ' ') runs, $SECONDS_EACH s each"   # macOS pads wc
