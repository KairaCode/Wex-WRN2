#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The GPU runs of this folder, from the repository root: bash research/cuda-review-20261115/run_gpu.sh
# It needs an NVIDIA GPU with its driver, clang with its NVPTX target, gcc, g++ and python; not the CUDA toolkit.
# wex4/fuzz/run.sh --gpu with 100 rounds of the custom mutator on every seed (the kernels of c/wex_cuda.cuh as
# PTX that clang writes, on the GPU, next to the emulation and against the CPU reference), then the negative
# control on the GPU (wex4/fuzz/cuda_negative.py --gpu). Logs: gpu.log and gpu-negative.log next to this script.
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
WROOT=$(cygpath -m "$ROOT" 2>/dev/null || echo "$ROOT")   # the same folder as a native Windows program writes it
# the bridge prints its "[wex] WNR3 record failed its integrity check" line once per build: left out of the log,
# with the line that names the seed folders
quiet() { "$@" 2>&1 | { grep -v -e '^\[wex\]' -e '^seeds: ' || true; } | sed -e "s#$ROOT/##g" -e "s#$WROOT/##g"; }
{
  start=$(date +%s)
  quiet bash wex4/fuzz/run.sh --gpu 100
  echo "wall time: $(( $(date +%s) - start )) s"
} > "$HERE/gpu.log"
"${PYTHON:-python}" -I wex4/fuzz/cuda_negative.py --gpu > "$HERE/gpu-negative.log" 2>&1
tail -3 "$HERE/gpu.log"; tail -1 "$HERE/gpu-negative.log"
