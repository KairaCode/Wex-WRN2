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
#                                       It prints the compiler and the kernels of the fast WNR3 decoder it built
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
HARNESSES="fuzz_wex4_file fuzz_wnr3_record fuzz_wex4_verify fuzz_store_wex fuzz_q38_wex fuzz_wnr3_cuda"
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
# fuzz_wnr3_cuda also gets the WEX4 files of impl/go-wex4/testdata: ceil-form.wex4 holds the only WNR3 records in
# ceil form that are not synthetic seeds of the mutator
mkdir -p "$O/seeds/testdata"
cp "$ROOT"/impl/go-wex4/testdata/*.wex4 "$O/seeds/testdata/"
"$PY" -I "$ROOT/wex4/fuzz/cuda_host.py" "$ROOT/c" "$O/wex_cuda_host"
cd "$ROOT"   # fuzz_wnr3_record and fuzz_wnr3_cuda find their default base file from here
seeds_of() {   # sets SEEDS, the seed folders of one harness
  case $1 in
    fuzz_wnr3_record) SEEDS=("$O/seeds/wnr3") ;;
    fuzz_store_wex) SEEDS=("$O/seeds/engine") ;;
    fuzz_q38_wex) SEEDS=("$O/seeds/engine" "$O/seeds/file") ;;
    fuzz_wnr3_cuda) SEEDS=("$O/seeds/wnr3" "$O/seeds/file" "$O/seeds/testdata") ;;
    *) SEEDS=("$O/seeds/file") ;;
  esac
}
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

if [ "${1-}" = --replay ]; then
  # the compiler, and which kernels of the fast WNR3 decoder fuzz_wnr3_record compares with the reference under
  # these flags (WEXF_KERNELS of wex_fast_decode.h): AVX2, NEON or scalar
  CC_USED=${CC:-gcc}
  CXX_USED=${CXX:-$(cxx_of "$CC_USED")}
  echo "compiler: $("$CC_USED" --version | sed -n 1p)"
  echo "C++ compiler of fuzz_wnr3_cuda: $("$CXX_USED" --version | sed -n 1p)"
  KERNELS=$(printf '#include "wex_fast_decode.h"\nkernels WEXF_KERNELS\n' |
            "$CC_USED" ${WEX_CFLAGS-} $MARCH -E -P $INC -x c - | sed -n 's/^kernels "\(.*\)"$/\1/p')
  echo "fast WNR3 decoder of fuzz_wnr3_record: $KERNELS kernels"
  for h in $HARNESSES; do
    build "$h" "$O/$h-replay$EXE" -O2 $W ${WEX_CFLAGS-} $MARCH -DWEX_FUZZ_STANDALONE $INC
    echo "== $h"
    seeds_of "$h"
    files=()
    for d in "${SEEDS[@]}"; do files+=("$d"/*); done
    "$O/$h-replay$EXE" "${files[@]}"
  done
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
SAN="-g -O1 -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined"
for h in $HARNESSES; do
  build "$h" "$O/$h$EXE" $W $SAN $MARCH $INC
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
