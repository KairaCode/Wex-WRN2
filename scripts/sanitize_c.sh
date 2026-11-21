#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The C test executables, and the libraries that the Python tests load, under ThreadSanitizer or
# MemorySanitizer. Linux, clang; CPU only, no real model.
#
#   bash scripts/sanitize_c.sh tsan    -fsanitize=thread -fopenmp: data races in the OpenMP loops of
#                                      c/tools/wex4_verify.c and of the fast WNR3 decoder
#                                      (wex4/research/fast-cpu-decode/wex_fast_decode.h), and between the threads
#                                      that call the libraries. Needs the OpenMP runtime of the same clang (Debian
#                                      and Ubuntu: libomp-N-dev)
#   bash scripts/sanitize_c.sh msan    -fsanitize=memory, no OpenMP: a branch, a pointer or a system call that
#                                      depends on a byte nobody wrote
#
# Both sanitizers need the code under test built with them; MemorySanitizer needs every object of the process.
# The libraries that the Python tests load (wex4/wex4_tools.c with the fast and with the reference WNR3
# decoder, c/tools/wex_nr3_tools.c) are therefore run twice: from scripts/sanitize_libs.c, a C program that
# loads them and calls what the Python tests call (step 5, both modes), and, under tsan only, from Python itself
# with the runtime preloaded (step 6). What runs, each step with the sanitizer:
#   1. two controls: a C file with a deliberate defect, which the sanitizer must report, and (tsan) a correct one
#      with the OpenMP constructs of the sources (parallel, for schedule(dynamic), reduction, critical), which
#      must give no report. They show that the run can fail and that the OpenMP runtime adds no false report
#   2. c/tests/test_wex_format.c (msan only: it has no threads) and c/tests/test_wex4_verify.c (1 and 4 threads)
#   3. c/tools/wex4_verify.c over the valid conformance vectors and over a WEX4 file of 4 synthetic WNR3 records
#      (fixtures/make_synth_mxfp4.py, c/tools/build_nr3.py, wex4/wex4_convert.py upgrade): on 8 threads under
#      tsan, on one thread under msan, whose build has no OpenMP
#   4. the six harnesses of wex4/fuzz/ on their seeds (wex4/fuzz/run.sh --replay; fuzz_wnr3_cuda, the CUDA decoder
#      built for the CPU, is C++ and is built with the clang++ of CC), fuzz_wex4_file with the fast
#      WNR3 decoder (-DWEX4_FAST_NR3); then fuzz_wex4_file on the synthetic file and fuzz_wnr3_record on its first
#      record, where the fast decoder has enough streams for several OpenMP tasks (the records of the vectors
#      have one task per section). Before that a small program prints the streams and the tasks of that record,
#      and the run stops if the weight section has fewer than 2 tasks. Under msan the tasks run one after the other
#   5. the three libraries, built as wex4/build_and_test.sh builds them with the sanitizer added, loaded and called
#      by scripts/sanitize_libs.c (its header lists the eight parts): the conformance vectors and the synthetic
#      file through every function that wex4/wex4.py binds, wex4_fuzz_run from 4 threads at once, the WNR3
#      encoder trained and run on the synthetic experts from 4 threads (its model and records must be those of
#      c/tools/build_nr3.py), and its records decoded by the fast decoder. Before that a control: a library with
#      a deliberate data race and a branch on a byte nobody wrote, loaded the same way, must be reported; and the
#      preprocessed source must show that one build calls the fast decoder and the other the reference one
#   6. tsan only: wex4/build_and_test.sh --fuzz 100 (wex4/test_wex4.py, c/tools/build_nr3.py) with its libraries
#      built under ThreadSanitizer and the shared runtime of the clang preloaded into Python, after two controls
#      through Python (the deliberate race reported, on stderr and in a log_path file as the real run writes
#      them; the correct OpenMP loop clean). It writes wex4/out
# The log of a tsan run also names the Archer library and shows what the OpenMP runtime says about loading it.
# CC names the clang (default clang), PYTHON the interpreter, WEX_CFLAGS adds flags. TSAN_OPTIONS and
# MSAN_OPTIONS are passed through; without TSAN_OPTIONS the script sets ignore_noninstrumented_modules=1, which
# drops what ThreadSanitizer intercepts inside the OpenMP runtime and Python (not instrumented) and nothing of
# the code under test. Seen with clang 18.1.3 and libomp 18 on ubuntu-24.04 (x86-64): without Archer the
# correct control gets 4 reports (the reduction and the end of the region), with Archer and without that option
# 1 report, in pthread_mutex_lock called by the runtime; with both, none. No suppression file is used.
# Everything built goes to out/tsan or out/msan, wex4/fuzz/out and (tsan) wex4/out, all git-ignored.
# Last line on success: "tsan run passed" or "msan run passed".
set -eu
set -o pipefail
MODE=${1-}
case $MODE in
  tsan) SAN="-fsanitize=thread -fopenmp"; REPORT="ThreadSanitizer: data race" ;;
  msan) SAN="-fsanitize=memory -fsanitize-memory-track-origins"; REPORT="MemorySanitizer: use-of-uninitialized-value" ;;
  *) echo "usage: bash scripts/sanitize_c.sh tsan|msan" >&2; exit 2 ;;
esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
C=$ROOT/c
O=$ROOT/out/$MODE
PY=${PYTHON:-python}
export CC=${CC:-clang}
export TSAN_OPTIONS=${TSAN_OPTIONS-ignore_noninstrumented_modules=1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS-4}   # the fast decoder takes its thread count from the OpenMP runtime
# Archer, the tool of the LLVM OpenMP runtime that tells ThreadSanitizer what the runtime synchronizes (the end
# of a parallel region, a reduction, a critical section). The runtime loads libarcher.so by name; where the
# library is not on the loader's path (Ubuntu keeps it in the folder of its clang) it is named here
if [ "$MODE" = tsan ] && [ -z "${OMP_TOOL_LIBRARIES-}" ]; then
  for ARCHER in "$("$CC" -print-file-name=libarcher.so)" "$("$CC" -print-resource-dir)/../../libarcher.so"; do
    if [ -f "$ARCHER" ]; then export OMP_TOOL_LIBRARIES=$ARCHER; break; fi
  done
fi
rm -rf "$O" && mkdir -p "$O"
cd "$O"
W="-Wall -Wextra -Wno-unused-parameter -Wno-unused-function"
S="-g -O1 -fno-omit-frame-pointer $SAN ${WEX_CFLAGS-}"

# 1. controls
if [ "$MODE" = tsan ]; then
  cat > defect.c <<'EOF'
#include <stdio.h>
int main(void) {
    volatile long sum = 0;
    int i;
#pragma omp parallel for num_threads(4) schedule(static)
    for (i = 0; i < 100000; i++) sum += i;   /* four threads write sum with no lock */
    printf("%ld\n", (long)sum);
    return 0;
}
EOF
  cat > clean.c <<'EOF'
#include <stdio.h>
int main(void) {
    long sum = 0, seen = 0;
    int bad = 0, i;
#pragma omp parallel num_threads(4) reduction(+ : sum)
    {
#pragma omp for schedule(dynamic, 1)
        for (i = 0; i < 4000; i++) {
            sum += i;
            if (i % 100 == 0) {
#pragma omp critical
                seen++;
            }
        }
    }
#pragma omp parallel for schedule(dynamic, 1) reduction(| : bad) num_threads(4)
    for (i = 0; i < 4000; i++) bad |= i < 0;
    printf("%ld %ld %d\n", sum, seen, bad);
    return !(sum == 7998000 && seen == 40 && !bad);
}
EOF
  "$CC" $W $S clean.c -o clean
  ./clean > clean.log 2>&1 || { cat clean.log; echo "the correct OpenMP control does not pass under ThreadSanitizer (OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES-}: is Archer loaded?)" >&2; exit 1; }
  echo "controls: the correct OpenMP loop gives no report (OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES-not set})"
  # what the OpenMP runtime itself says about the tool; shown only, the control above is the test
  OMP_TOOL_VERBOSE_INIT=stderr ./clean 2>&1 | grep -E '^(Opening|Tool was started)' || echo "(the OpenMP runtime printed nothing about loading a tool)"
else
  cat > defect.c <<'EOF'
#include <stdlib.h>
int main(void) {
    unsigned char *volatile p = (unsigned char *)malloc(4);
    int r = p[1] ? 3 : 4;   /* a branch on a byte nobody wrote */
    free(p);
    return r == 3;
}
EOF
fi
"$CC" $W $S defect.c -o defect
if ./defect > defect.log 2>&1 || ! grep -q "$REPORT" defect.log; then
  cat defect.log; echo "the control with a deliberate defect was not reported: this run would prove nothing" >&2; exit 1
fi
echo "controls: the deliberate defect is reported ($REPORT)"

# inputs, built without a sanitizer: the WNR3 encoder is loaded into Python, which is not instrumented
"$CC" -O2 $W -shared -fPIC "$C/tools/wex_nr3_tools.c" -o wex_nr3_tools.dll
"$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" synth --layers 1 --experts 4 > synth.log
"$PY" -I "$C/tools/repack_v4_wex.py" --model-dir synth --output synth.wex >> synth.log
"$PY" -I "$C/tools/build_nr3.py" ./wex_nr3_tools.dll synth.wex container synth_nr3.wex --layers 0 --train 0:0,0:1,0:2,0:3 | tail -1
cp synth_nr3.wex synth_nr3.wex3   # the WEX3 form, before the upgrade: step 5 compares its model and records
"$PY" -I "$ROOT/wex4/wex4_convert.py" upgrade synth_nr3.wex >> synth.log
# the stored bytes of its first WNR3 record, as wex4/fuzz/make_seeds.py takes them from the vectors
"$PY" -I -c "
import io, sys
sys.path.insert(0, sys.argv[1])
import wex4
data = open(sys.argv[2], 'rb').read()
record = [r for r in wex4.Reader(io.BytesIO(data)).records if r[3] == 5][0]
open(sys.argv[3], 'wb').write(data[record[0]:record[0] + record[1]])
" "$ROOT/wex4" synth_nr3.wex synth_nr3.rec

# 2. the C tests
if [ "$MODE" = msan ]; then
  "$CC" $W $S "$C/tests/test_wex_format.c" -o test_wex_format
  ./test_wex_format
fi
"$CC" $W $S "$C/tests/test_wex4_verify.c" -o test_wex4_verify
# it prints the expected failures, so only its summary line is shown; when it fails the whole output is
./test_wex4_verify > test_wex4_verify.log 2>&1 || { cat test_wex4_verify.log; exit 1; }
tail -1 test_wex4_verify.log

# 3. the checker: 8 threads under tsan; the msan build has no OpenMP, so there it runs on one thread
if [ "$MODE" = tsan ]; then THREADS=8; HOW="8 threads"; else THREADS=1; HOW="1 thread (no OpenMP)"; fi
"$CC" $W $S "$C/tools/wex4_verify.c" -o wex4_verify
n=0
for f in synth_nr3.wex "$ROOT"/wex4/vectors/valid/*.wex4; do
  ./wex4_verify "$f" $THREADS > wex4_verify.log 2>&1 || { cat wex4_verify.log; exit 1; }
  n=$((n + 1))
done
echo "wex4_verify, $HOW: $n files pass"

# how the fast decoder splits the first synthetic record: the arithmetic of wexf_section (wex_fast_decode.h)
cat > tasks.c <<'EOF'
#include "wex_fast_decode.h"
#include "wex4.h"
#include "fuzz_common.h"
static int tasks_of(uint64_t streams) {
    const uint64_t per = ((streams + WEXF_MAX_TASKS - 1u) / WEXF_MAX_TASKS + 47u) / 48u * 48u;
    return per ? (int)((streams + per - 1u) / per) : 0;
}
int main(int argc, char **argv) {
    Wex4Header h;
    Wex4View v;
    Wex4Record record;
    Wex4Blob blob;
    WexNr2Model model;
    WexNr2Layout layout;
    WexNr2Record view;
    WexEntry entry;
    size_t size = 0;
    uint32_t r;
    int threads = 1;
    unsigned char *file = argc > 1 ? fuzz_read_file(argv[1], &size) : NULL;
    if (!file || size < WEX4_PAGE || wex4_parse_header(file, size, &h) ||
        wex4_open(&h, file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v)) return 2;
    for (r = 0; r < h.records; r++) {
        wex4_get_record(&v, r, &record);
        if (record.codec == WEX4_CODEC_WNR3) break;
    }
    if (r == h.records) return 2;
    wex4_get_blob(&v, record.blob - 1u, &blob);
    if (wex_nr2_model_open(file + blob.offset, (size_t)blob.bytes, &model) || wex4_expert_entry(&v, r, &entry)) return 2;
    wex_nr2_layout(&entry, &layout);
    if (wex_nrx_parse(file + record.offset, record.stored, &layout, model.id, WEX_NR3_MAGIC, &view)) return 2;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("fast decoder, first synthetic record: %u scale streams in %d task(s), %u weight streams in %d task(s); "
           "OpenMP threads: %d\n", (unsigned)view.scale_streams, tasks_of(view.scale_streams),
           (unsigned)view.weight_streams, tasks_of(view.weight_streams), threads);
    free(file);
    return tasks_of(view.weight_streams) >= 2 ? 0 : 1;
}
EOF
"$CC" $W $S -I "$ROOT/wex4/research/fast-cpu-decode" -I "$C" -I "$ROOT/wex4/fuzz" tasks.c -o tasks
./tasks synth_nr3.wex || { echo "the synthetic record gives the fast decoder fewer than 2 tasks: step 4 would not run its parallel path" >&2; exit 1; }

# 4. the fuzzing harnesses on their seeds, then on the synthetic file
WEX_CFLAGS="$S -DWEX4_FAST_NR3" bash "$ROOT/wex4/fuzz/run.sh" --replay
F=$ROOT/wex4/fuzz/out
echo "== synthetic WNR3 records"
"$F/fuzz_wex4_file-replay" synth_nr3.wex
WEX_FUZZ_BASE=$O/synth_nr3.wex "$F/fuzz_wnr3_record-replay" synth_nr3.rec

# 5. the three libraries that the Python tests load through ctypes, built as wex4/build_and_test.sh builds them
#    with the sanitizer added (the fast decoder with OpenMP under tsan, without under msan), loaded and called by
#    scripts/sanitize_libs.c instead of Python, with the inputs of those tests: the vectors and the synthetic file
echo "== the libraries of the Python tests, driven from C"
case $(uname -m) in
  x86_64|amd64|i?86) MARCH=-march=haswell ;;
  *) MARCH= ;;
esac
L="-g -fno-omit-frame-pointer $SAN ${WEX_CFLAGS-}"
REF="-O2 $L -I $C"
FAST="-O3 $MARCH -DWEX4_FAST_NR3 $L -I $ROOT/wex4/research/fast-cpu-decode -I $C"
# as in wex4/build_and_test.sh: prove that the two builds of wex4_tools.c hold two decoders, or the fast decoder
# would drop out of this step without a failure
site() { "$CC" $1 -E "$ROOT/wex4/wex4_tools.c" | grep -c "return $2(&view, &layout, model, scales, weights)" || true; }
if [ "$(site "$FAST" wex_fast_decode)" != 1 ] || [ "$(site "$REF" wex_nr2_decode)" != 1 ]; then
  echo "wex4_decode_record does not call wex_fast_decode in libs/wex4_tools.dll and wex_nr2_decode in libs/wex4_tools_ref.dll" >&2; exit 1
fi
L="$L -shared -fPIC"
mkdir -p libs
"$CC" $W $REF -shared -fPIC "$ROOT/wex4/wex4_tools.c" -o libs/wex4_tools_ref.dll
"$CC" $W $FAST -shared -fPIC "$ROOT/wex4/wex4_tools.c" -o libs/wex4_tools.dll
"$CC" $W -O2 $L "$C/tools/wex_nr3_tools.c" -o libs/wex_nr3_tools.dll
"$CC" $W $S -I "$C" "$ROOT/scripts/sanitize_libs.c" -o sanitize_libs -pthread -ldl
# the control: a library built the same way, with a data race and a branch on a byte nobody wrote, loaded and
# called by the same program, must give the report (so the reports of the libraries above are not lost)
cat > libdefect.c <<'EOF'
#include <stdlib.h>
static int arrived;
volatile long counter;
void race(void) {   /* the callers wait for each other (relaxed atomics: no happens-before), then all write */
    long i;
    __atomic_fetch_add(&arrived, 1, __ATOMIC_RELAXED);
    while (__atomic_load_n(&arrived, __ATOMIC_RELAXED) < 2) {}
    for (i = 0; i < 100000; i++) counter++;
}
int uninit(void) {
    unsigned char *volatile p = (unsigned char *)malloc(4);
    int r = p[1] ? 3 : 4;
    free(p);
    return r;
}
EOF
"$CC" $W -O1 $L libdefect.c -o libs/defect.dll
if ./sanitize_libs --control libs/defect.dll > control_libs.log 2>&1 || ! grep -q "$REPORT" control_libs.log; then
  cat control_libs.log; echo "the library with a deliberate defect was not reported through sanitize_libs" >&2; exit 1
fi
echo "controls: the deliberate defect in a loaded library is reported ($REPORT)"
./sanitize_libs libs/wex4_tools.dll libs/wex4_tools_ref.dll libs/wex_nr3_tools.dll \
    synth.wex synth_nr3.wex3 synth_nr3.wex "$ROOT/wex4/vectors" 4 1000

# 6. tsan only: the WEX4 tests themselves (wex4/build_and_test.sh --fuzz 100) with their three libraries under
#    ThreadSanitizer, loaded into a Python that is not instrumented: the shared runtime of the same clang is
#    preloaded into every Python process of the run (libgcc_s too, which the runtime needs to print a crash). Each
#    process writes its reports to a file of py-reports/; one file fails the step. Two controls through the same
#    Python first: race() of libdefect.c on 2 threads must be reported (on stderr, then in a file of py-control/
#    with the same log_path setting as the real run), the correct OpenMP loop of step 1 (a call
#    on the main thread, then 4 threads at once) must not. No MemorySanitizer counterpart: this clang has no shared
#    MSan runtime, and MSan would take every byte that Python writes for one nobody wrote
if [ "$MODE" = tsan ]; then
  echo "== the WEX4 tests in Python, their libraries under ThreadSanitizer"
  RT=$("$CC" -print-file-name=libclang_rt.tsan-x86_64.so)
  [ -f "$RT" ] || RT=$("$CC" -print-file-name="libclang_rt.tsan-$(uname -m).so")
  [ -f "$RT" ] || { echo "no shared ThreadSanitizer runtime next to $CC" >&2; exit 1; }
  GCC_S=$("$CC" -print-file-name=libgcc_s.so.1)
  [ -f "$GCC_S" ] && RT="$RT $GCC_S"
  printf '#!/bin/sh\nLD_PRELOAD="%s" exec "%s" "$@"\n' "$RT" "$(command -v "$PY")" > py-tsan
  chmod +x py-tsan
  echo "preloaded into $PY: $RT"
  "$CC" $W $S -Dmain=clean_main -shared -fPIC clean.c -o libs/clean.dll
  CALL='import ctypes, sys, threading
fn = getattr(ctypes.CDLL(sys.argv[1]), sys.argv[2])
if sys.argv[4] == "first":
    assert fn() == 0
out = []
threads = [threading.Thread(target=lambda: out.append(fn())) for _ in range(int(sys.argv[3]))]
for t in threads: t.start()
for t in threads: t.join()
assert len(out) == int(sys.argv[3]) and (sys.argv[4] != "first" or not any(out))'
  if ./py-tsan -I -c "$CALL" libs/defect.dll race 2 - > control_py.log 2>&1 || ! grep -q "$REPORT" control_py.log; then
    cat control_py.log; echo "the library with a deliberate data race was not reported through Python" >&2; exit 1
  fi
  # the same control with the log_path of the real run below: the report must land in a file of that kind
  rm -rf py-control && mkdir py-control
  if TSAN_OPTIONS="$TSAN_OPTIONS log_path=$O/py-control/tsan" ./py-tsan -I -c "$CALL" libs/defect.dll race 2 - \
      > control_py_file.log 2>&1 || ! grep -qs "$REPORT" py-control/*; then
    cat control_py_file.log; ls py-control
    echo "the deliberate data race through Python left no report file with log_path" >&2; exit 1
  fi
  ./py-tsan -I -c "$CALL" libs/clean.dll clean_main 4 first > clean_py.log 2>&1 || {
    cat clean_py.log; echo "the correct OpenMP control does not pass in Python under ThreadSanitizer" >&2; exit 1; }
  echo "controls: through Python the deliberate data race is reported and the correct OpenMP loop gives no report"
  rm -rf py-reports && mkdir py-reports
  status=0
  TSAN_OPTIONS="$TSAN_OPTIONS log_path=$O/py-reports/tsan" PYTHON=$O/py-tsan \
      WEX_CFLAGS="-g -fno-omit-frame-pointer -fsanitize=thread ${WEX_CFLAGS-}" \
      bash "$ROOT/wex4/build_and_test.sh" --fuzz 100 > wex4_tests.log 2>&1 || status=$?
  if [ "$status" != 0 ] || [ -n "$(ls py-reports)" ]; then
    tail -40 wex4_tests.log; cat py-reports/* 2>/dev/null || true
    echo "wex4/build_and_test.sh under ThreadSanitizer: exit status $status, $(ls py-reports | wc -l) report file(s)" >&2
    exit 1
  fi
  grep -E '^(fast WNR3 decoder of|fuzz total|WEX4 prototype)' wex4_tests.log
fi
echo "$MODE run passed"
