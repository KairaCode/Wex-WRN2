#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Verification rerun of ../ (CPU only, 8 threads, reads ~0.9 GB read-only). Assembled from the commands used; not run as one script.
#   sh run_verify.sh 'U:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex' 'U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
set -eu
cd "$(dirname "$0")"; ROOT=$(cd ../../../.. && pwd); PY=${PY:-/c/Python314/python.exe}
mkdir -p out/stage/tools; export TMP="$PWD/out" TEMP="$PWD/out"
cp "$ROOT"/c/*.h out/stage/ && cp "$ROOT"/c/tools/wex_nr3_tools.c out/stage/tools/
W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"
gcc $W -fopenmp -Iout/stage/tools ../wnr4_proto.c -o out/wnr4_proto.exe -lm
gcc $W -fopenmp -Iout/stage/tools extra.c -o out/extra.exe -lm
gcc $W -shared out/stage/tools/wex_nr3_tools.c -o out/wex_nr3_tools.dll
"$PY" -I scan.py "$1" ../out/container_scan.json out/scan_verify.json          # every header + length table, own code
"$PY" -I fetch_v.py "$1" out/cache ../out/container_scan.json                  # 16 fresh + 4 overlapping + 3 extreme records
OMP_NUM_THREADS=8 ./out/wnr4_proto.exe out/cache out/proto_verify.json $(cat out/cache/picks.txt)
"$PY" -I compare.py                                                           # vs ../out/proto.json
"$PY" -I st_check.py "$2" out/cache                                           # slab == Z(safetensors); chunk variants via the WNR3 DLL
OMP_NUM_THREADS=8 ./out/extra.exe out/cache 16:177 300                        # timings + bit flips
"$PY" -I -B ../layer_model.py "$1" out/wex_nr3_tools.dll out/layer_model_verify.json out/cache 38,39
