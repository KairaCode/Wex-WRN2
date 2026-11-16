#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The GLM-5.3-Flash WEX4 test kit: amd/GLM-5.3-Flash-Quark-MXFP4 as a WEX4 container with WNR3 expert records,
# checked stage by stage (README.md in this folder). Each stage can run alone; each writes RESULTS/<stage>.log and
# RESULTS/<stage>.status (ok, failed or skipped, with the reason) and stops the run when it fails.
#
# Usage: bash testkits/glm53-wex4/run.sh [OPTIONS] STAGE [STAGE ...]
#   stages: env fetch build verify roundtrip decode speed engine report, or all (every stage in that order);
#           clean (not in all) deletes the checkpoint's .safetensors files after a passed round trip, with
#           --delete-checkpoint only. A stage that stopped can be run again: fetch resumes the download, build keeps
#           a finished container (and a trained model) and restarts an interrupted pack; the others start over.
#   --work DIR          tools, the WNM2 model and the container (default ./glm53-work)
#   --results DIR       logs and the report (default ./glm53-results)
#   --checkpoint DIR    an existing download of the checkpoint (default WORK/checkpoint, filled by fetch)
#   --container FILE    a container built before or published (build is then skipped)
#   --threads N         threads of train, pack and wex4_verify (default: every CPU thread)
#   --sample N          decode: check a seeded sample of N expert records instead of all of them
#   --seed S            the seed of every sample (default 20261116)
#   --train-sample N    experts the WNM2 model is trained on (default 80)
#   --colibri DIR       colibri's c/ folder, with glm53 built from the GLM-5.3 WEX4 branch (engine stage; also
#                       the source of the CUDA test on Linux)
#   --tokens N          engine: greedy tokens per run (default 8)
#   --ids LIST          engine: the prompt as token ids, comma-separated (default: the kit's prompt, see README)
#   --cuda-helper FILE  a built kaira_wex_cuda helper (else built with nvcc when an NVIDIA GPU and nvcc are there)
#   --cuda-test FILE    a built test_wex_nr3_gpu (else built with nvcc, as the helper)
#   --unpack-files      roundtrip: write the rebuilt files to WORK/unpacked (needs the checkpoint's size free),
#                       then hash them; without it they are rebuilt into their SHA-256 only and nothing is written
#   --synthetic         a dry run on a synthetic checkpoint with GLM-5.3-Flash's expert shapes (no download)
#   --synthetic-small   the same with small experts (hidden 512, moe_intermediate 256): the check of scripts/check_all.sh
set -eu
set -o pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$KIT/../.." && pwd)
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
CC=${CC:-gcc}
REPO=amd/GLM-5.3-Flash-Quark-MXFP4
REVISION=b5688f25491202978c19c4d036eef579f61bbe07
WORK=$PWD/glm53-work RESULTS=$PWD/glm53-results CHECKPOINT= CONTAINER= SAMPLE=0 SEED=20261116 TRAIN=80
COLIBRI= TOKENS=8 IDS= CUDA_HELPER= CUDA_TEST= UNPACK_FILES=0 SYNTHETIC=0 DELETE_CHECKPOINT=0 SYNTH_H=4096 SYNTH_I=2048
THREADS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# the kit's prompt: "The capital of France is" in GLM-5.3-Flash's tokenizer (README, "The engine stage")
DEFAULT_IDS=785,6722,315,9621,374
STAGES=()
while [ $# -gt 0 ]; do
  case $1 in
    --work) WORK=$2; shift 2 ;;
    --results) RESULTS=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --container) CONTAINER=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --sample) SAMPLE=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --train-sample) TRAIN=$2; shift 2 ;;
    --colibri) COLIBRI=$2; shift 2 ;;
    --tokens) TOKENS=$2; shift 2 ;;
    --ids) IDS=$2; shift 2 ;;
    --cuda-helper) CUDA_HELPER=$2; shift 2 ;;
    --cuda-test) CUDA_TEST=$2; shift 2 ;;
    --unpack-files) UNPACK_FILES=1; shift ;;
    --synthetic) SYNTHETIC=1; shift ;;
    --synthetic-small) SYNTHETIC=1; SYNTH_H=512; SYNTH_I=256; shift ;;
    all) STAGES+=(env fetch build verify roundtrip decode speed engine report); shift ;;
    --delete-checkpoint) DELETE_CHECKPOINT=1; shift ;;
    env|fetch|build|verify|roundtrip|decode|speed|engine|report|clean) STAGES+=("$1"); shift ;;
    -h|--help) sed -n '4,36p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (bash $0 --help)" >&2; exit 2 ;;
  esac
done
[ ${#STAGES[@]} -gt 0 ] || { echo "no stage given (bash $0 --help)" >&2; exit 2; }
mkdir -p "$WORK" "$RESULTS"
WORK=$(cd "$WORK" && pwd) RESULTS=$(cd "$RESULTS" && pwd)
[ -n "$CHECKPOINT" ] || CHECKPOINT=$WORK/checkpoint
[ "$SYNTHETIC" = 0 ] || CHECKPOINT=$WORK/synthetic
TOOLS=$WORK/tools
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) SO="-shared"; STATIC_SO="-static -shared"; EXT=dll; EXE=.exe; WIN=1 ;;
  Darwin) SO="-shared -fPIC"; STATIC_SO="-shared -fPIC"; EXT=dylib; EXE=; WIN=0 ;;
  *) SO="-shared -fPIC"; STATIC_SO="-shared -fPIC"; EXT=so; EXE=; WIN=0 ;;
esac
OMP=${WEX_OPENMP:--fopenmp}
case $(uname -m) in x86_64|amd64) MARCH=${WEX4_MARCH--march=haswell} ;; *) MARCH=${WEX4_MARCH-} ;; esac
FAST=$TOOLS/wex4_tools.$EXT REF=$TOOLS/wex4_tools_ref.$EXT NR3=$TOOLS/wex_nr3_tools.$EXT VERIFY=$TOOLS/wex4_verify$EXE
CONV=$ROOT/wex4/wex4_convert.py
# the container: --container, else the one build writes
[ -n "$CONTAINER" ] || CONTAINER=$WORK/glm53-wnr3.wex4
STAGE=

# ---- stage plumbing -----------------------------------------------------------------------------------------
log() { echo "[$STAGE] $*"; }
finish() {   # trap: a stage that did not reach `ok` failed
  local code=$?
  if [ -n "$STAGE" ] && [ ! -s "$RESULTS/$STAGE.status" ]; then
    echo "failed (exit $code; see $STAGE.log)" > "$RESULTS/$STAGE.status"
    echo "FAILED: stage $STAGE (exit $code), log: $RESULTS/$STAGE.log" >&2
  fi
}
trap finish EXIT
ok() { echo "ok${1:+ ($1)}" > "$RESULTS/$STAGE.status"; log "ok${1:+: $1}"; }
skip() { echo "skipped ($1)" > "$RESULTS/$STAGE.status"; log "skipped: $1"; }
die() { log "FAIL: $*" >&2; exit 1; }
timed() {   # timed NAME CMD...: runs CMD, appends "NAME seconds" to RESULTS/times.tsv
  local name=$1 t0 t1; shift
  t0=$(date +%s)
  "$@"
  t1=$(date +%s)
  printf '%s\t%s\n' "$name" "$((t1 - t0))" >> "$RESULTS/times.tsv"
  log "$name: $((t1 - t0)) s"
}
shard_list() { ls "$CHECKPOINT"/*.safetensors 2>/dev/null | sort; }
free_gb() { "$PY" -I -c "import shutil,sys; print(int(shutil.disk_usage(sys.argv[1]).free/1e9))" "$1"; }
need_container() { [ -f "$CONTAINER" ] || die "no container at $CONTAINER (run the build stage, or give --container)"; }
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }   # a path for a native program
root_of() { sed -n 's/^root \([0-9a-f]\{64\}\)$/\1/p' "$RESULTS/build.log" 2>/dev/null | tail -1; }

tools() {   # the libraries and wex4_verify, built once into WORK/tools (as wex4/build_and_test.sh builds them)
  if [ -f "$FAST" ] && [ -f "$REF" ] && [ -f "$NR3" ] && [ -f "$VERIFY" ]; then return 0; fi
  mkdir -p "$TOOLS"
  local W="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function ${WEX_CFLAGS-}"
  log "building the tools into $TOOLS with $("$CC" --version | sed -n 1p)"
  ( export TMP="$TOOLS" TEMP="$TOOLS"
    "$CC" $W $SO -I "$ROOT/c" "$ROOT/wex4/wex4_tools.c" -o "$REF"
    "$CC" $W -O3 $MARCH $OMP -DWEX4_FAST_NR3 $STATIC_SO -I "$ROOT/c" -I "$ROOT/wex4/research/fast-cpu-decode" \
          "$ROOT/wex4/wex4_tools.c" -o "$FAST"
    "$CC" $W $SO "$ROOT/c/tools/wex_nr3_tools.c" -o "$NR3"
    "$CC" $W $OMP "$ROOT/c/tools/wex4_verify.c" -o "$VERIFY" )
}

# ---- 1. env ---------------------------------------------------------------------------------------------------
stage_env() {
  "$PY" -I "$KIT/kit.py" env "$RESULTS" "$WORK" $( [ -d "$CHECKPOINT" ] && echo "$CHECKPOINT" )
  tools
  "$PY" -I -c "import ctypes,sys; [ctypes.CDLL(p) for p in sys.argv[1:]]" "$FAST" "$REF" "$NR3"
  ok "tools built in $TOOLS"
}

# ---- 2. fetch ---------------------------------------------------------------------------------------------------
stage_fetch() {
  if [ "$SYNTHETIC" = 1 ]; then
    # GLM-5.3-Flash's expert shapes under its names: a layer 0 without experts (the dense layers 0-2), two MXFP4
    # layers of 3 experts, and a last layer whose experts are not MXFP4 and stay ordinary tensors (layer 45, FP8)
    rm -rf "$CHECKPOINT"
    "$PY" -I "$ROOT/fixtures/make_synth_mxfp4.py" "$CHECKPOINT" --names qwen38 --layers 4 --experts 3 \
          --first-layer 1 --float-layer 3 --hidden "$SYNTH_H" --inter "$SYNTH_I" --seed "$SEED"
    ok "synthetic checkpoint in $CHECKPOINT"
    return
  fi
  mkdir -p "$CHECKPOINT"
  if "$PY" -I "$KIT/kit.py" check-checkpoint "$CHECKPOINT" > /dev/null 2>&1; then
    log "the checkpoint is complete (sizes of the manifest)"
  elif command -v hf >/dev/null 2>&1; then
    log "hf download $REPO at $REVISION into $CHECKPOINT (resumes a partial download)"
    hf download "$REPO" --revision "$REVISION" --local-dir "$CHECKPOINT"
  else
    command -v curl >/dev/null 2>&1 || die "neither hf nor curl is installed"
    log "curl, file by file (-C - resumes a partial file)"
    grep -v '^#' "$KIT/glm53-files.tsv" | cut -f1 | while read -r name; do
      curl -fL --retry 5 -C - -o "$CHECKPOINT/$name" "https://huggingface.co/$REPO/resolve/$REVISION/$name"
    done
  fi
  "$PY" -I "$KIT/kit.py" check-checkpoint "$CHECKPOINT" ${FETCH_SHA256:+--sha256}
  ok "checkpoint $CHECKPOINT matches the manifest's sizes"
}

# ---- 3. build -----------------------------------------------------------------------------------------------------
stage_build() {
  if [ -f "$CONTAINER" ] && [ "$CONTAINER" != "$WORK/glm53-wnr3.wex4" ]; then skip "--container $CONTAINER given"; return; fi
  tools
  local src; src=$(shard_list)
  [ -n "$src" ] || die "no safetensors file in $CHECKPOINT (run fetch, or give --checkpoint)"
  local out=$WORK/glm53-wnr3.wex4 done=$WORK/glm53-wnr3.wex4.done
  # resume: a container whose pack finished (its .done marker holds the root) is kept; a .partial file is what an
  # interrupted pack leaves, and pack cannot continue inside it, so it is removed and pack starts again
  if [ -f "$out" ] && [ -s "$done" ] && [ "$out" -ot "$done" ]; then
    log "the container of an earlier build is complete (root $(cat "$done")): kept; delete $out to build it again"
    echo "root $(cat "$done")" ; "$PY" -I "$KIT/kit.py" sizes "$out" "$CHECKPOINT" "$RESULTS/sizes.json" > /dev/null
    ok "$(basename "$out"), root $(cat "$done") (from an earlier build)"; return
  fi
  if [ -f "$out.partial" ]; then
    log "an interrupted pack left $(basename "$out").partial ($(du -h "$out.partial" | cut -f1)): removed, pack starts again"
    rm -f "$out.partial"
  fi
  rm -f "$out" "$done"
  local need; need=$("$PY" -I -c "import os,sys; print(int(0.9*sum(os.path.getsize(p) for p in sys.argv[1:])/1e9)+2)" $src)
  [ "$(free_gb "$WORK")" -ge "$need" ] || die "about $need GB must be free in $WORK for the container, $(free_gb "$WORK") GB are"
  # the model is deterministic (the same sample and seed give the same bytes): one trained before is reused
  if [ -s "$WORK/glm53.wnm2" ] && [ "$(wc -c < "$WORK/glm53.wnm2")" = 59408 ] && [ -s "$WORK/glm53.wnm2.args" ] &&
     [ "$(cat "$WORK/glm53.wnm2.args")" = "$TRAIN $SEED" ]; then
    log "the WNM2 model of an earlier build (sample $TRAIN, seed $SEED): reused"
  else
    rm -f "$WORK/glm53.wnm2" "$WORK/glm53.wnm2.args"
    timed train "$PY" -I "$CONV" train "$WORK/glm53.wnm2" $src --names qwen38 --nr3-dll "$NR3" \
          --sample "$TRAIN" --seed "$SEED" --threads "$THREADS"
    echo "$TRAIN $SEED" > "$WORK/glm53.wnm2.args"
  fi
  timed pack "$PY" -I "$CONV" pack "$out" $src --dll "$FAST" --experts nr3 --names qwen38 \
        --nr3-dll "$NR3" --model "$WORK/glm53.wnm2" --threads "$THREADS" | tee "$RESULTS/pack.txt"
  local root; root=$(sed -n 's/^root \([0-9a-f]\{64\}\)$/\1/p' "$RESULTS/pack.txt" | tail -1)
  [ -n "$root" ] || die "pack printed no root"
  echo "$root" > "$done"
  "$PY" -I "$KIT/kit.py" sizes "$out" "$CHECKPOINT" "$RESULTS/sizes.json"
  ok "$(basename "$out"), root $root"
}

# ---- 4. verify ------------------------------------------------------------------------------------------------------
stage_verify() {
  tools; need_container
  timed wex4.py_verify "$PY" -I "$ROOT/wex4/wex4.py" verify "$CONTAINER" --dll "$FAST" | tee "$RESULTS/verify-py.txt"
  local root; root=$(sed -n 's/^WEX4 4\.[0-9]*  root \([0-9a-f]\{64\}\)$/\1/p' "$RESULTS/verify-py.txt")
  grep -q 'every hash checked, every record decoded' "$RESULTS/verify-py.txt" || die "wex4.py verify did not finish"
  timed wex4_verify "$VERIFY" "$CONTAINER" "$THREADS" | tee "$RESULTS/verify-c.txt"
  grep -q "^COLI_WEX_ROOT=$root\$" "$RESULTS/verify-c.txt" || die "wex4_verify does not give the root $root"
  local built; built=$(root_of)
  [ -z "$built" ] || [ "$built" = "$root" ] || die "the root $root is not the one pack printed ($built)"
  [ -f "$RESULTS/sizes.json" ] || "$PY" -I "$KIT/kit.py" sizes "$CONTAINER" "$CHECKPOINT" "$RESULTS/sizes.json" > /dev/null
  ok "root $root: both verifiers"
}

# ---- 5. roundtrip ---------------------------------------------------------------------------------------------------
stage_roundtrip() {
  tools; need_container
  [ -n "$(shard_list)" ] || die "the round trip compares with the checkpoint: no safetensors file in $CHECKPOINT"
  if [ "$UNPACK_FILES" = 1 ]; then
    rm -rf "$WORK/unpacked"
    timed unpack "$PY" -I "$CONV" unpack "$CONTAINER" "$WORK/unpacked" --dll "$FAST" | tee "$RESULTS/unpack.txt"
  else
    timed unpack "$PY" -I "$CONV" unpack "$CONTAINER" "$WORK/unpacked" --dll "$FAST" --sha-only | tee "$RESULTS/unpack.txt"
  fi
  timed hash_checkpoint "$PY" -I "$KIT/kit.py" compare-sha "$CHECKPOINT" "$RESULTS/unpack.txt" "$RESULTS/roundtrip.json"
  if [ "$UNPACK_FILES" = 1 ]; then rm -rf "$WORK/unpacked"; fi
  ok "every file byte-identical (SHA-256)"
}

# ---- 6. decode ------------------------------------------------------------------------------------------------------
cuda_tools() {   # sets CUDA_HELPER / CUDA_TEST, building them with nvcc when not given
  [ -n "$CUDA_HELPER" ] && [ -n "$CUDA_TEST" ] && return 0
  local h t   # built by an earlier stage
  for h in "$TOOLS/cuda/kaira_wex_cuda.dll" "$TOOLS/cuda/kaira_wex_cuda.so"; do [ -f "$h" ] && CUDA_HELPER=${CUDA_HELPER:-$h}; done
  for t in "$TOOLS/cuda/test_wex_nr3_gpu.exe" "$TOOLS/cuda/test_wex_nr3_gpu"; do [ -f "$t" ] && CUDA_TEST=${CUDA_TEST:-$t}; done
  [ -n "$CUDA_HELPER" ] && [ -n "$CUDA_TEST" ] && return 0
  command -v nvcc >/dev/null 2>&1 || return 1
  local C=$ROOT/c
  if [ -n "$COLIBRI" ] && [ -f "$COLIBRI/tests/test_wex_nr3_gpu.cu" ]; then C=$COLIBRI; fi
  if [ "$WIN" = 0 ] && ! grep -q dlopen "$C/tests/test_wex_nr3_gpu.cu"; then
    log "the GPU test of $C is Windows-only: give --colibri (colibri's c/ has the Linux form)"; return 1
  fi
  mkdir -p "$TOOLS/cuda"
  local flags="-O3 -std=c++17 -arch=${CUDA_ARCH:-native}"
  if [ "$WIN" = 1 ]; then
    [ -n "$CUDA_HELPER" ] || nvcc $flags -shared "$C/tools/wex_cuda_bridge.cu" -o "$TOOLS/cuda/kaira_wex_cuda.dll"
    [ -n "$CUDA_TEST" ] || nvcc $flags "$C/tests/test_wex_nr3_gpu.cu" -o "$TOOLS/cuda/test_wex_nr3_gpu.exe"
    CUDA_HELPER=${CUDA_HELPER:-$TOOLS/cuda/kaira_wex_cuda.dll} CUDA_TEST=${CUDA_TEST:-$TOOLS/cuda/test_wex_nr3_gpu.exe}
  else
    [ -n "$CUDA_HELPER" ] || nvcc $flags -shared -Xcompiler=-fPIC,-fvisibility=hidden "$C/tools/wex_cuda_bridge.cu" \
                                  -o "$TOOLS/cuda/kaira_wex_cuda.so"
    [ -n "$CUDA_TEST" ] || nvcc $flags "$C/tests/test_wex_nr3_gpu.cu" -o "$TOOLS/cuda/test_wex_nr3_gpu" -ldl
    CUDA_HELPER=${CUDA_HELPER:-$TOOLS/cuda/kaira_wex_cuda.so} CUDA_TEST=${CUDA_TEST:-$TOOLS/cuda/test_wex_nr3_gpu}
  fi
}

stage_decode() {
  tools; need_container
  local args=(--ref "$REF" --fast "$FAST" --seed "$SEED")
  [ "$SAMPLE" = 0 ] || args+=(--sample "$SAMPLE")
  timed decode_cpu "$PY" -I "$KIT/kit.py" decode "$CONTAINER" "$CHECKPOINT" "$RESULTS/decode.json" "${args[@]}"
  local done="CPU reference and fast decoder"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if cuda_tools; then
      "$PY" -I "$KIT/kit.py" make-bin "$CONTAINER" "$CHECKPOINT" "$WORK/records.bin" --sample "${GPU_SAMPLE:-8}" --seed "$SEED"
      timed decode_cuda "$CUDA_TEST" "$CUDA_HELPER" "${MUTATIONS:-20}" "$WORK/records.bin" | tee "$RESULTS/decode-cuda.txt"
      grep -q 'ALL OK' "$RESULTS/decode-cuda.txt" || die "the CUDA test did not end with ALL OK"
      echo ok > "$RESULTS/decode-cuda.status"
      done="$done, CUDA helper"
    else
      echo "skipped (an NVIDIA GPU but no helper: give --cuda-helper and --cuda-test, or install nvcc)" > "$RESULTS/decode-cuda.status"
    fi
  else
    echo "skipped (no NVIDIA GPU: nvidia-smi -L found none)" > "$RESULTS/decode-cuda.status"
  fi
  if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -q deviceName && \
     command -v glslangValidator >/dev/null 2>&1; then
    MUTATIONS=${MUTATIONS_VK:-0} WNR3_VK_EVERY=${WNR3_VK_EVERY:-64} \
      timed decode_vulkan bash "$ROOT/wex4/research/vulkan-decode/run.sh" "$CONTAINER" | tee "$RESULTS/decode-vulkan.txt"
    grep -q 'vulkan-decode: all checks passed' "$RESULTS/decode-vulkan.txt" || die "the Vulkan prototype did not pass"
    echo ok > "$RESULTS/decode-vulkan.status"
    done="$done, Vulkan prototype"
  else
    echo "skipped (no Vulkan device, or no glslangValidator)" > "$RESULTS/decode-vulkan.status"
  fi
  ok "$done"
}

# ---- 7. speed -------------------------------------------------------------------------------------------------------
stage_speed() {
  tools; need_container
  # the CPU decoders on the same seeded sample, three passes, the decode alone timed (kit.py decode)
  for pass in 1 2 3; do
    "$PY" -I "$KIT/kit.py" decode "$CONTAINER" "$CHECKPOINT" "$RESULTS/speed-cpu-$pass.json" --ref "$REF" --fast "$FAST" \
          --sample "${SPEED_SAMPLE:-32}" --seed "$((SEED + pass))" | grep -E 'ms_per|mb_per|"(reference|fast)"'
  done
  if [ -n "$CUDA_HELPER$CUDA_TEST" ] || { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
    if cuda_tools && [ -f "$WORK/records.bin" ]; then
      # the helper's own timing is not exposed by its test: the whole test on the bin is timed (README says so)
      timed cuda_test_seconds "$CUDA_TEST" "$CUDA_HELPER" 0 "$WORK/records.bin" > "$RESULTS/speed-cuda.txt"
    fi
  fi
  if [ -f "$ROOT/wex4/research/vulkan-decode/out/wnr3_vk$EXE" ]; then
    WNR3_VK_SHADERS="$ROOT/wex4/research/vulkan-decode/out" WNR3_VK_EVERY=${WNR3_VK_EVERY:-64} \
      "$ROOT/wex4/research/vulkan-decode/out/wnr3_vk$EXE" bench 3 "$CONTAINER" | tee "$RESULTS/speed-vulkan.txt" || true
  fi
  ok "ms per record in speed-*.json / speed-*.txt"
}

# ---- 8. engine ------------------------------------------------------------------------------------------------------
stage_engine() {
  local bin=
  for b in "$COLIBRI/glm53$EXE" "$COLIBRI/glm53"; do [ -n "$COLIBRI" ] && [ -x "$b" ] && bin=$b && break; done
  if [ -z "$bin" ]; then skip "no glm53 binary (give --colibri, colibri's c/ with glm53 built from the GLM-5.3 WEX4 branch)"; return; fi
  need_container
  [ "$SYNTHETIC" = 0 ] || { skip "a synthetic checkpoint has no model to run"; return; }
  local ids=${IDS:-$DEFAULT_IDS} root
  root=$(sed -n 's/^COLI_WEX_ROOT=\([0-9a-f]\{64\}\)$/\1/p' "$RESULTS/verify-c.txt" 2>/dev/null | tail -1)
  [ -n "$root" ] || root=$(root_of)
  log "A: the shards; B: the container (COLI_WEX_PATH, root ${root:-not pinned}); $TOKENS greedy tokens each"
  ( unset COLI_WEX_PATH COLI_WEX_ROOT COLI_WEX_VERIFY COLI_WEX_CUDA_LIB
    timed engine_shards "$bin" --model "$CHECKPOINT" --ids "$ids" --greedy "$TOKENS" ) > "$RESULTS/engine-shards.txt" 2>&1 ||
    { tail -20 "$RESULTS/engine-shards.txt"; die "glm53 on the shards failed (engine-shards.txt)"; }
  ( export COLI_WEX_PATH=$(native "$CONTAINER") COLI_WEX_VERIFY=1; [ -z "$root" ] || export COLI_WEX_ROOT=$root
    unset COLI_WEX_CUDA_LIB
    timed engine_container "$bin" --model "$CHECKPOINT" --ids "$ids" --greedy "$TOKENS" ) > "$RESULTS/engine-container.txt" 2>&1 ||
    { tail -20 "$RESULTS/engine-container.txt"; die "glm53 on the container failed (engine-container.txt)"; }
  "$PY" -I "$KIT/kit.py" engine-ab "$RESULTS/engine-shards.txt" "$RESULTS/engine-container.txt" "$RESULTS/engine.json"
  if [ -n "$CUDA_HELPER" ] || cuda_tools 2>/dev/null; then
    ( export COLI_WEX_PATH=$(native "$CONTAINER") COLI_WEX_VERIFY=1 COLI_WEX_CUDA_LIB=$(native "$CUDA_HELPER"); [ -z "$root" ] || export COLI_WEX_ROOT=$root
      timed engine_container_cuda "$bin" --model "$CHECKPOINT" --ids "$ids" --greedy "$TOKENS" ) \
      > "$RESULTS/engine-container-cuda.txt" 2>&1 || die "glm53 on the container with the CUDA helper failed"
    "$PY" -I "$KIT/kit.py" engine-ab "$RESULTS/engine-shards.txt" "$RESULTS/engine-container-cuda.txt" "$RESULTS/engine-cuda.json"
  fi
  ok "tokens identical, shards and container"
}

# ---- optional: give the checkpoint's disk space back -----------------------------------------------------------
stage_clean() {
  [ "$DELETE_CHECKPOINT" = 1 ] || die "clean deletes the checkpoint's .safetensors files: add --delete-checkpoint to ask for it"
  "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['files'] and not d['different'] else 1)" \
        "$RESULTS/roundtrip.json" 2>/dev/null || die "no passed round trip in $RESULTS/roundtrip.json: the checkpoint stays"
  local n=0
  for f in $(shard_list); do rm -f "$f"; n=$((n + 1)); done
  ok "$n safetensors files deleted from $CHECKPOINT (unpack of the container gives them back, byte for byte)"
}

stage_report() {
  "$PY" -I "$KIT/report.py" "$RESULTS"
  ok "$RESULTS/report.md and report.json"
}

for STAGE in "${STAGES[@]}"; do
  rm -f "$RESULTS/$STAGE.status"
  echo "== stage $STAGE"
  if [ "$STAGE" = report ]; then stage_report; else "stage_$STAGE" 2>&1 | tee "$RESULTS/$STAGE.log"; fi
  [ -s "$RESULTS/$STAGE.status" ] || exit 1
done
STAGE=
echo "glm53-wex4 kit: stages ${STAGES[*]} done; results in $RESULTS"
