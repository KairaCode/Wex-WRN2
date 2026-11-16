#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The runs of this folder: the WNR3 Vulkan prototype of wex4/research/vulkan-decode/ on the published DeepSeek
# V4.1 Flash container, on an NVIDIA GPU. From the repository root:
#   bash research/vulkan-real-20261116/run.sh STEP CONTAINER
# STEP:
#   every64  run.sh of the prototype with MUTATIONS=0 WNR3_VK_EVERY=64 and the validation layer on (as the
#            prototype's README says for a first run on a real container)
#   bench    the build of bench.sh (fast CPU decoder, -O3 -march=haswell -fopenmp), then `wnr3_vk bench 21` on the
#            first 16 WNR3 records of CONTAINER in one batch of at most 1 GiB: the table form and the search form
#            alone (each with the CPU decoders timed), then both forms in pairs
#   all      `wnr3_vk decode` of every WNR3 record of CONTAINER, the default (table) form only, validation layer off
#   mutate   `wnr3_vk mutate COPIES 20261114` on one WNR3 record in EVERY of CONTAINER, table form (defaults 20 and
#            96: 80 records, 1,600 mutated copies)
# Needs what run.sh of the prototype needs, and an NVIDIA device (WNR3_VK_DEVICE=NVIDIA; another part of a device
# name picks another one). On MSYS2 UCRT64 VK_LAYER_PATH=/ucrt64/bin finds the validation layer. Writes the log of
# the step next to this script, with the paths shortened to <Wex-WRN2>/ and <MODELS>/.
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
STEP=$1
C=$2
V=$ROOT/wex4/research/vulkan-decode
OUT=$V/out
export WNR3_VK_DEVICE=${WNR3_VK_DEVICE:-NVIDIA} VK_LAYER_PATH=${VK_LAYER_PATH:-/ucrt64/bin} TMP="$OUT" TEMP="$OUT"
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) LIB=${VULKAN_LIB:-vulkan-1} ;;
  *) LIB=${VULKAN_LIB:-vulkan} ;;
esac
CDIR=$(dirname "$C")
mixed() { cygpath -m "$1" 2>/dev/null || echo "$1"; }   # the same folder as a native Windows program writes it
short() { sed -e "s#$ROOT/##g" -e "s#$(mixed "$ROOT")/##g" -e "s#$CDIR/#<MODELS>/#g" -e "s#$(mixed "$CDIR")/#<MODELS>/#g"; }
load() { echo "[cpu load $(wmic cpu get loadpercentage 2>/dev/null | tr -d '\r' | sed -n 2p | tr -d ' ')% at $(date -u +%H:%M:%S) UTC]"; }
build() {   # the host program and the two shader forms, as run.sh and bench.sh build them
  mkdir -p "$OUT/real/table" "$OUT/real/search"
  "${CC:-gcc}" -std=c11 -O3 ${WEX4_MARCH--march=haswell} -fopenmp -Wall -Wextra -Wno-unused-function -DWNR3_VK_FAST \
      -I "$ROOT/c" -I "$ROOT/wex4/research/fast-cpu-decode" "$V/wnr3_vk.c" -o "$OUT/real/wnr3_vk" -l"$LIB"
  for form in search table; do
    D=$([ $form = search ] && echo -DSEARCH || true)
    glslangValidator -V $D "$V/wnr3.comp" -o "$OUT/real/$form/scales.spv" > /dev/null
    glslangValidator -V -DWEIGHTS $D "$V/wnr3.comp" -o "$OUT/real/$form/weights.spv" > /dev/null
  done
}
{
  echo "commit $(git rev-parse --short HEAD); $("${CC:-gcc}" --version | sed -n 1p); $(glslangValidator --version | sed -n 1p)"
  load
  start=$(date +%s)
  case $STEP in
    every64) WNR3_VK_VALIDATE=1 MUTATIONS=0 WNR3_VK_EVERY=64 bash "$V/run.sh" "$C" ;;
    bench)
      build
      export WNR3_VK_BATCH_BYTES=1073741824
      for form in table search; do
        echo "== shader form: $form, the first 16 WNR3 records"; load
        WNR3_VK_SHADERS="$OUT/real/$form" "$OUT/real/wnr3_vk" bench 21 "$C" 16
      done
      echo "== pairs: form A search, form B table"; load
      WNR3_VK_BENCH_CPU=0 WNR3_VK_SHADERS="$OUT/real/search" WNR3_VK_SHADERS_B="$OUT/real/table" \
          "$OUT/real/wnr3_vk" bench 21 "$C" 16 ;;
    all) build; WNR3_VK_SHADERS="$OUT/real/table" "$OUT/real/wnr3_vk" decode "$C" ;;
    mutate) build; WNR3_VK_EVERY=${EVERY:-96} WNR3_VK_SHADERS="$OUT/real/table" "$OUT/real/wnr3_vk" mutate "${COPIES:-20}" 20261114 "$C" ;;
    *) echo "STEP: every64, bench, all or mutate" >&2; exit 2 ;;
  esac
  load
  echo "wall time: $(( $(date +%s) - start )) s"
} 2>&1 | short | tee "$HERE/$STEP.log"
