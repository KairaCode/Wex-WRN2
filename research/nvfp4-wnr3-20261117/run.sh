#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The whole run, in the order it was made. Needs the DLLs of wex4/build_and_test.sh in wex4/out
# (bash scripts/check_all.sh builds them), gcc, Python 3.14 with numpy and lz4, and for bench_real.c the lz4,
# brotli and zstd libraries. About 1.85 GB is fetched (range requests only) into DATA.
# Usage: bash research/nvfp4-wnr3-20261117/run.sh DATA WORK [GLM_CONTAINER.wex4]
#   GLM_CONTAINER: the published GLM-5.3-Flash WNR3 container; leave it out to skip that step.
# Results: WORK/results (the files of results/ in this folder) and WORK/results/tables.md.
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$1" "$2"
D=$(cd "$1" && pwd)
W=$(cd "$2" && pwd)
PY=${PYTHON:-python}
X=$ROOT/wex4/out
O=$W/results
mkdir -p "$O" "$W/bin" "$W/tmp"
export TMP="$W/tmp" TEMP="$W/tmp"
$PY -I "$HERE/fetch.py" "$D"
$PY -I "$HERE/fetch.py" "$D" --only glm-nz5
cp "$D/fetch-log.json" "$D/samples.tsv" "$O/"
$PY -I "$HERE/analyze.py" "$D" "$W/out" | tee "$O/analyze.log"
cp "$W/out/estimates.json" "$O/"
$PY -I "$HERE/extra.py" "$D" "$O/extra.json" | tee "$O/extra.log"
$PY -I "$HERE/extra.py" nz5 "$D" "$O/nz5.json"
for spec in qwen-mxfp4:qwen38 glm-mxfp4:qwen38 ds-mxfp4:deepseek ds-quark:deepseek-quark; do
  c=${spec%%:*}
  $PY -I "$HERE/wnr3_real.py" pack "$D" "$W/wnr3" "$X/wex_nr3_tools.dll" "$X/wex4_tools_ref.dll" "$c" "${spec#*:}" | tee -a "$O/wnr3.log"
  cp "$W/wnr3/$c/wnr3.json" "$O/wnr3-$c.json"
done
if [ $# -ge 3 ]; then
  $PY -I "$HERE/wnr3_real.py" container "$3" "$X/wex4_tools_ref.dll" "$D" "$O" | tee -a "$O/wnr3.log"
fi
gcc -std=c99 -O2 -Wall -Wextra "$HERE/nvwnr.c" -o "$W/bin/nvwnr.exe"
$PY -I "$HERE/run_proto.py" "$W/bin/nvwnr.exe" "$D" "$W/out/slabs" "$O/proto.json" 1 | tee "$O/proto.log"
$PY -I "$HERE/run_proto.py" "$W/bin/nvwnr.exe" "$D" "$W/out/slabs" "$O/proto.json" 1 --only qwen-nvfp4,glm-nvfp4,ds-nvfp4 \
    --configs nv-exp7-o0,nv-exp7-o0-cz,nv-exp7-o1p-cz,nv-code6-o1-cz,nv-code12-o1-cz --in-sample | tee "$O/proto3.log"
$PY -I "$HERE/baselines.py" 3 "$X/wex4_tools.dll" "$W"/out/slabs/{ds-nvfp4,ds-mxfp4,ds-quark,qwen-nvfp4,qwen-mxfp4,glm-nvfp4,glm-mxfp4} \
    | tee "$O/baselines.log"
gcc -std=c99 -O3 -march=haswell -fopenmp -Wall -Wextra -Wno-unused-function -I "$ROOT/c" \
    -I "$ROOT/wex4/research/fast-cpu-decode" "$ROOT/research/real-experts-20261115/bench_real.c" -o "$W/bin/bench_real.exe" \
    -llz4 -lbrotlienc -lbrotlidec -lzstd
for c in ds-quark qwen-mxfp4 glm-mxfp4; do
  R=$W/wnr3/$c/rec-all
  "$W/bin/bench_real.exe" wnr3 5 "$R/model.wnm2" $(ls "$R"/*.rec | sed 's/\.rec$//') > "$O/speed-$c.txt"
done
$PY -I "$HERE/summarize.py" "$O" > "$O/tables.md"
echo "run.sh: done, tables in $O/tables.md"
