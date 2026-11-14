#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The C test executables under ThreadSanitizer or MemorySanitizer. Linux, clang; CPU only, no real model.
#
#   bash scripts/sanitize_c.sh tsan    -fsanitize=thread -fopenmp: data races in the OpenMP loops of
#                                      c/tools/wex4_verify.c and of the fast WNR3 decoder
#                                      (wex4/research/fast-cpu-decode/wex_fast_decode.h). Needs the OpenMP runtime
#                                      of the same clang (Debian and Ubuntu: libomp-N-dev)
#   bash scripts/sanitize_c.sh msan    -fsanitize=memory, no OpenMP: a branch, a pointer or a system call that
#                                      depends on a byte nobody wrote
#
# Only executables are instrumented: both sanitizers need every object of the process built with them, so the
# libraries that the other checks load into Python (wex4/test_wex4.py, c/tools/build_nr3.py) are outside this
# script. What runs, each step with the sanitizer:
#   1. two controls: a C file with a deliberate defect, which the sanitizer must report, and (tsan) a correct one
#      with the OpenMP constructs of the sources (parallel, for schedule(dynamic), reduction, critical), which
#      must give no report. They show that the run can fail and that the OpenMP runtime adds no false report
#   2. c/tests/test_wex_format.c (msan only: it has no threads) and c/tests/test_wex4_verify.c (1 and 4 threads)
#   3. c/tools/wex4_verify.c over the valid conformance vectors and over a WEX4 file of 4 synthetic WNR3 records
#      (fixtures/make_synth_mxfp4.py, c/tools/build_nr3.py, wex4/wex4_convert.py upgrade): on 8 threads under
#      tsan, on one thread under msan, whose build has no OpenMP
#   4. the three harnesses of wex4/fuzz/ on their seeds (wex4/fuzz/run.sh --replay), fuzz_wex4_file with the fast
#      WNR3 decoder (-DWEX4_FAST_NR3); then fuzz_wex4_file on the synthetic file and fuzz_wnr3_record on its first
#      record, where the fast decoder has enough streams for several OpenMP tasks (the records of the vectors
#      have one task per section). Before that a small program prints the streams and the tasks of that record,
#      and the run stops if the weight section has fewer than 2 tasks. Under msan the tasks run one after the other
# The log of a tsan run also names the Archer library and shows what the OpenMP runtime says about loading it.
# CC names the clang (default clang), PYTHON the interpreter, WEX_CFLAGS adds flags. TSAN_OPTIONS and
# MSAN_OPTIONS are passed through; without TSAN_OPTIONS the script sets ignore_noninstrumented_modules=1, which
# drops what ThreadSanitizer intercepts inside the OpenMP runtime (not instrumented) and nothing of the code
# under test. Seen with clang 18.1.3 and libomp 18 on ubuntu-24.04 (x86-64): without Archer the correct control
# gets 4 reports (the reduction and the end of the region), with Archer and without that option 1 report, in
# pthread_mutex_lock called by the runtime; with both, none. No suppression file is used.
# Everything built goes to out/tsan or out/msan and wex4/fuzz/out (git-ignored).
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
"$CC" $W $S -I "$C" -I "$ROOT/wex4/research/fast-cpu-decode" -I "$ROOT/wex4/fuzz" tasks.c -o tasks
./tasks synth_nr3.wex || { echo "the synthetic record gives the fast decoder fewer than 2 tasks: step 4 would not run its parallel path" >&2; exit 1; }

# 4. the fuzzing harnesses on their seeds, then on the synthetic file
WEX_CFLAGS="$S -DWEX4_FAST_NR3" bash "$ROOT/wex4/fuzz/run.sh" --replay
F=$ROOT/wex4/fuzz/out
echo "== synthetic WNR3 records"
"$F/fuzz_wex4_file-replay" synth_nr3.wex
WEX_FUZZ_BASE=$O/synth_nr3.wex "$F/fuzz_wnr3_record-replay" synth_nr3.rec
echo "$MODE run passed"
