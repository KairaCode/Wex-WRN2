#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# WNR3 decode on Vulkan compute (prototype): the speed of the device against the CPU decoders, on synthetic
# records of the size of a real expert. A measurement, not a check; not part of scripts/check_all.sh or of CI.
#
# It writes out/bench/big-EXPERTS-HIDDEN-INTER.wex4 once (kept between runs): EXPERTS synthetic experts of
# HIDDEN x INTER (default 4 of 5120 x 2304: 18,800,640 decoded bytes each, the size of a DeepSeek V4.1 expert
# record) from fixtures/make_synth_mxfp4.py, packed as WNR3 records with a model trained on them. The
# bytes are random draws from a skewed distribution, not weights of a model. Then it builds the host program
# with the fast CPU decoder of wex4/research/fast-cpu-decode (-O3, AVX2 on x86 through -march=haswell, OpenMP)
# and runs `wnr3_vk bench` once per shader form: `search` (-DSEARCH, the reference binary search) and `table`
# (the default). Each run decodes the batch RUNS times on the device and on the CPU, compares every output
# with the reference decoder's and prints median, minimum and maximum of each timed part (wnr3_vk.c, bench).
# Then one more run times the two forms against each other in RUNS pairs in one process (device times only,
# the CPU decoders not timed).
#
# Needs: what run.sh needs, python 3, and wex4/out/wex_nr3_tools.dll (bash wex4/build_and_test.sh leaves it).
# Disk and memory grow with EXPERTS: the file holds about 14.5 MB per expert of the default size, and the
# bench keeps the batch in memory several times (records, decoded bytes, the reference's bytes).
# Usage: bash wex4/research/vulkan-decode/bench.sh [RUNS]      (default 15)
#   EXPERTS, HIDDEN, INTER as above. RECORDS = records in the batch (default: all that fit). FORMS = the shader
#   forms to run alone (default "search table"; empty: none). PAIRS=0 leaves the run in pairs out.
#   WNR3_VK_DEVICE, WNR3_VK_STAGE, WNR3_VK_BATCH_BYTES, WNR3_VK_VALIDATE, WNR3_VK_BENCH_CPU: see wnr3_vk.c.
#   WEX4_MARCH overrides -march=haswell as in wex4/build_and_test.sh. CC, PYTHON, VULKAN_LIB as in run.sh.
# Last line on success: vulkan-decode bench: done
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT=$HERE/out
B=$OUT/bench
PY=${PYTHON:-python}
CC=${CC:-gcc}
RUNS=${1:-15}
EXPERTS=${EXPERTS:-4}
HIDDEN=${HIDDEN:-5120}
INTER=${INTER:-2304}
FILE=$B/big-$EXPERTS-$HIDDEN-$INTER.wex4
mkdir -p "$B"
export TMP="$OUT" TEMP="$OUT"   # gcc from Git Bash cannot create temporary files under the Windows directory
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) LIB=${VULKAN_LIB:-vulkan-1} ;;
  *) LIB=${VULKAN_LIB:-vulkan} ;;
esac
case $(uname -m) in
  x86_64|amd64|i?86) MARCH=${WEX4_MARCH--march=haswell} ;;
  *) MARCH=${WEX4_MARCH-} ;;
esac
TOOLS=$ROOT/wex4/out/wex_nr3_tools.dll
if [ ! -f "$FILE" ]; then
  if [ ! -f "$TOOLS" ]; then echo "$TOOLS is missing: run bash wex4/build_and_test.sh first" >&2; exit 1; fi
  echo "writing $EXPERTS synthetic experts of $HIDDEN x $INTER (random bytes, not a model)"
  rm -rf "$B/synth"
  "$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" "$B/synth" --layers 1 --experts "$EXPERTS" \
      --hidden "$HIDDEN" --inter "$INTER"
  "$PY" -I "$ROOT/c/tools/repack_v4_wex.py" --model-dir "$B/synth" --output "$B/big.wex" > "$B/repack.log"
  PICKS=$("$PY" -I -c "print(','.join('0:%d' % e for e in range($EXPERTS)))")
  "$PY" -I "$ROOT/c/tools/build_nr3.py" "$TOOLS" "$B/big.wex" container "$B/big_nr3.wex" --layers 0 --train "$PICKS" | tail -1
  "$PY" -I "$ROOT/wex4/wex4_convert.py" upgrade "$B/big_nr3.wex" | tail -1
  mv "$B/big_nr3.wex" "$FILE"
  rm -rf "$B/synth" "$B/big.wex"
fi
echo "compiler: $("$CC" --version | sed -n 1p), -O3 $MARCH -fopenmp"
"$CC" -std=c11 -O3 $MARCH -fopenmp -Wall -Wextra -Wno-unused-function ${WEX_CFLAGS-} -DWNR3_VK_FAST -I "$ROOT/c" \
    -I "$ROOT/wex4/research/fast-cpu-decode" "$HERE/wnr3_vk.c" -o "$B/wnr3_vk" -l"$LIB"
: > "$B/glslang.log"
glsl() { glslangValidator "$@" >> "$B/glslang.log" 2>&1 || { cat "$B/glslang.log"; exit 1; }; }
for form in search table; do
  D=$([ $form = search ] && echo -DSEARCH || true)
  mkdir -p "$B/$form"
  glsl -V $D "$HERE/wnr3.comp" -o "$B/$form/scales.spv"
  glsl -V -DWEIGHTS $D "$HERE/wnr3.comp" -o "$B/$form/weights.spv"
done
for form in ${FORMS-search table}; do
  case $form in
    search|table) ;;
    *) echo "FORMS: search or table, not $form" >&2; exit 2 ;;
  esac
  echo "== shader form: $form"
  WNR3_VK_SHADERS="$B/$form" "$B/wnr3_vk" bench "$RUNS" "$FILE" ${RECORDS-}
done
if [ "${PAIRS:-1}" != 0 ]; then
  # both forms in one process on the same batch, in RUNS pairs: form A search, form B table
  echo "== pairs: form A search, form B table"
  WNR3_VK_BENCH_CPU=0 WNR3_VK_SHADERS="$B/search" WNR3_VK_SHADERS_B="$B/table" "$B/wnr3_vk" bench "$RUNS" "$FILE" ${RECORDS-}
fi
echo "vulkan-decode bench: done"
