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
#   3. c/tools/wex4_verify.c on 8 threads over the valid conformance vectors and over a WEX4 file of 4 synthetic
#      WNR3 records (fixtures/make_synth_mxfp4.py, c/tools/build_nr3.py, wex4/wex4_convert.py upgrade)
#   4. the three harnesses of wex4/fuzz/ on their seeds (wex4/fuzz/run.sh --replay), fuzz_wex4_file with the fast
#      WNR3 decoder (-DWEX4_FAST_NR3); then fuzz_wex4_file on the synthetic file and fuzz_wnr3_record on its first
#      record, where the fast decoder has enough streams for several OpenMP tasks (the records of the vectors
#      have one task per section)
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

# 3. the checker on 8 threads
"$CC" $W $S "$C/tools/wex4_verify.c" -o wex4_verify
n=0
for f in synth_nr3.wex "$ROOT"/wex4/vectors/valid/*.wex4; do
  ./wex4_verify "$f" 8 > wex4_verify.log 2>&1 || { cat wex4_verify.log; exit 1; }
  n=$((n + 1))
done
echo "wex4_verify, 8 threads: $n files pass"

# 4. the fuzzing harnesses on their seeds, then on the synthetic file
WEX_CFLAGS="$S -DWEX4_FAST_NR3" bash "$ROOT/wex4/fuzz/run.sh" --replay
F=$ROOT/wex4/fuzz/out
echo "== synthetic WNR3 records"
"$F/fuzz_wex4_file-replay" synth_nr3.wex
WEX_FUZZ_BASE=$O/synth_nr3.wex "$F/fuzz_wnr3_record-replay" synth_nr3.rec
echo "$MODE run passed"
