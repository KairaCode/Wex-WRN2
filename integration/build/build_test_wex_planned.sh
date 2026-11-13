#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Build test_wex_planned.exe: the engine's HOT expert store (COLI_V4_TEST_HOOKS) reading a WEX container.
# For a KAIRA-style colibri c/ tree (88ddb07) with the repo's c/ copied over it (guide section 2).
# Run it in an MSYS2 UCRT64 login shell; from Git Bash gcc fails with "Cannot create temporary file in C:\WINDOWS\".
#   MSYSTEM=UCRT64 C:/msys64/usr/bin/bash.exe -lc "sh <repo>/integration/build/build_test_wex_planned.sh <colibri c> [OUT_DIR]"
# Then: OUT_DIR/test_wex_planned.exe MODEL_DIR sample-vN.wex [--expect-fallback|--expect-cuda]
set -eu
C=$(cd "$1" && pwd)
O=${2:-$C/tests}
mkdir -p "$O"
O=$(cd "$O" && pwd)
cd "$C"
F="-D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -DCOLI_V4_TEST_HOOKS -DCOLI_V4_GPU_TIER -DCOLI_V4_MAX_PIN_SLOTS_PER_LAYER=16
   -DCOLI_V4_PIN_RAMP_REQUESTS=24 -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER -O2 -march=x86-64-v3 -fopenmp -ffunction-sections -fdata-sections -include pthread.h"
# 1. the engine objects (COLI_V4_UNIT_ATTENTION.o holds coli_v41_engram_open/close since the V4.1 port; the
#    engine's -flto build of it links, a plain -O2 one pulls in the compressor/indexer units) and the V4 test
#    objects (COLI_V4_UNIT_*_TEST.o, the other units, backend_loader_dsv4.o, wex_decode.o)
make -f Makefile.deepseek-v4 deepseek-v4 -j"${JOBS:-12}"
make tests/test_deepseek_v4.exe -j"${JOBS:-12}"
# 2. units and modules the test needs beyond those
extra=""
for u in LAYER_RESIDENT RUNTIME EXPERT_STORE_AUTO HEAD_CACHE; do
  gcc $F -DCOLI_V4_UNIT_$u -c deepseek_v4.c -o "$O/wp_$u.o"; extra="$extra $O/wp_$u.o"
done
for m in expert_store_registry expert_store_wex v4_route_tap; do
  gcc $F -c $m.c -o "$O/wp_$m.o"; extra="$extra $O/wp_$m.o"
done
# 3. link
units=""
for u in ST CONFIG MATH ROUTE_BF16 SPARSE_ATTENTION RESOURCE_PLAN PROMPT KV_CACHE ATTENTION_CACHE LAYER \
         EXPERT_STORE_HOT_ROWS16_TEST NATIVE_QUANT_BATCH_TEST EXPERT_ROWS16 NATIVE_QUANT NATIVE_QUANT_DUAL \
         NATIVE_QUANT_ROWS16 GPU_TEST ATTENTION; do
  units="$units COLI_V4_UNIT_$u.o"
done
gcc $F tests/test_wex_planned.c $units backend_loader_dsv4.o wex_decode.o $extra -o "$O/test_wex_planned.exe" \
    -Wl,--gc-sections -lm -fopenmp -static -pthread -lws2_32 -lpsapi
echo "built $O/test_wex_planned.exe"
