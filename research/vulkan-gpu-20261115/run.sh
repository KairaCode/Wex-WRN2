#!/usr/bin/env bash
# The local runs of this folder (README.md), from the repository root, on Windows 10 with MSYS2 UCRT64, an
# NVIDIA GPU with its Vulkan driver and Mesa llvmpipe (mingw-w64-ucrt-x86_64-mesa). Needs what
# wex4/research/vulkan-decode/run.sh and bench.sh need, wex4/out from `bash wex4/build_and_test.sh`, and
# about 40 minutes on a quiet machine. Every log goes to this folder.
#   bash research/vulkan-gpu-20261115/run.sh [STEP ...]     (default: all; steps machine run mutate control
#                                                           old probe bench pairs)
set -u
R=research/vulkan-gpu-20261115
D=wex4/research/vulkan-decode
A="wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4"
O=$D/out/old
export VK_LAYER_PATH=${VK_LAYER_PATH:-/ucrt64/bin}                              # the validation layer
export VK_ADD_DRIVER_FILES=${VK_ADD_DRIVER_FILES:-$(cygpath -w /ucrt64/share/vulkan/icd.d/lvp_icd.x86_64.json)}
export TMP=$PWD/$D/out TEMP=$PWD/$D/out
# The machine is shared: waits (at most 30 minutes) until no other process runs wnr3_vk; with the argument
# cpu (the runs that time the CPU decoders) also (at most 5 minutes) until three readings of the CPU load in
# a row are 20% or less. Prints what it found, with the lowest and the highest reading of the wait
others() { tasklist //FI 'IMAGENAME eq wnr3_vk.exe' //NH 2>/dev/null | grep -c -i wnr3_vk; }
load() { wmic cpu get loadpercentage 2>/dev/null | tr -dc 0-9; }
quiet() {
  local end=$(($(date +%s) + 1800)) low=0 l lo=100 hi=0 seen=""
  while [ "$(others)" != 0 ] && [ "$(date +%s)" -lt $end ]; do sleep 10; done
  end=$(($(date +%s) + 300))
  if [ "${1:-}" != cpu ]; then low=3; fi
  while [ $low -lt 3 ] && [ "$(date +%s)" -lt $end ]; do
    l=$(load); l=${l:-100}; seen=1
    if [ "$l" -lt $lo ]; then lo=$l; fi
    if [ "$l" -gt $hi ]; then hi=$l; fi
    if [ "$l" -le 20 ]; then low=$((low + 1)); else low=0; sleep 2; fi
  done
  echo "[$(date -u +%H:%M:%SZ): $(others) other wnr3_vk processes; CPU load $(load)%${seen:+, $lo% to $hi% while waiting}$([ $low -lt 3 ] && echo ', not below 20% three times in a row in 5 minutes')]"
}
step() { [ "$STEPS" != "${STEPS/ $1 /}" ]; }
STEPS=" ${*:-machine run mutate control old probe bench pairs} "

if step machine; then
  {
    echo "== GPU (nvidia-smi)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm --format=csv
    echo "== CPU"
    wmic cpu get name,numberofcores,numberoflogicalprocessors /format:list 2>/dev/null | tr -d '\r' | grep .
    echo "== OS"
    cmd //c ver | tr -d '\r' | grep .
    echo "== tools"
    gcc --version | sed -n 1p
    glslangValidator --version | sed -n 1p
    spirv-val --version | sed -n 1p
    pacman -Q mingw-w64-ucrt-x86_64-mesa mingw-w64-ucrt-x86_64-vulkan-loader mingw-w64-ucrt-x86_64-vulkan-validation-layers
  } > $R/machine.txt 2>&1
fi
if step run; then
  quiet > $R/nvidia-run.log
  WNR3_VK_DEVICE=NVIDIA WNR3_VK_VALIDATE=1 bash $D/run.sh $A >> $R/nvidia-run.log 2>&1
  echo "exit $?" >> $R/nvidia-run.log
  quiet > $R/llvmpipe-run.log
  WNR3_VK_DEVICE=llvmpipe WNR3_VK_VALIDATE=1 bash $D/run.sh $A >> $R/llvmpipe-run.log 2>&1
  echo "exit $?" >> $R/llvmpipe-run.log
fi
if step mutate; then
  quiet > $R/nvidia-mutate-1000.log
  WNR3_VK_DEVICE=NVIDIA MUTATIONS=1000 bash $D/run.sh $A >> $R/nvidia-mutate-1000.log 2>&1
  echo "exit $?" >> $R/nvidia-mutate-1000.log
fi
if step control; then
  quiet > $R/nvidia-negative-control.log
  WNR3_VK_DEVICE=NVIDIA bash $D/negative_control.sh $A >> $R/nvidia-negative-control.log 2>&1
  echo "exit $?" >> $R/nvidia-negative-control.log
fi
if step old; then
  # the shader of main before this change, on both devices; then the same with outw coherent or volatile,
  # with the direct upload and with robustBufferAccess
  mkdir -p $O
  "${PYTHON:-python}" -I $R/explain_diff.py $D/wnr3_vk.c $O/explain.c || exit 1
  gcc -std=c99 -O2 -Wall -Wextra -Wno-unused-function -I c $O/explain.c -o $O/explain -lvulkan-1 || exit 1
  git show 1af6c0d:$D/wnr3.comp > $O/main.comp
  sed 's/buffer Decoded { uint outw\[\]; }/coherent buffer Decoded { uint outw[]; }/' $O/main.comp > $O/coherent.comp
  sed 's/buffer Decoded { uint outw\[\]; }/volatile buffer Decoded { uint outw[]; }/' $O/main.comp > $O/volatile.comp
  for v in main coherent volatile; do
    mkdir -p $O/$v
    glslangValidator -V $O/$v.comp -o $O/$v/scales.spv > /dev/null && glslangValidator -V -DWEIGHTS $O/$v.comp -o $O/$v/weights.spv > /dev/null || exit 1
  done
  {
    quiet
    echo "== spirv-val of main's weights.spv"; spirv-val --target-env vulkan1.0 $O/main/weights.spv && echo "accepted"
    for run in "NVIDIA main" "NVIDIA main" "NVIDIA main WNR3_VK_STAGE=0" "NVIDIA main WNR3_VK_ROBUST=1" \
               "NVIDIA coherent" "NVIDIA volatile" "llvmpipe main"; do
      set -- $run
      echo "== device $1, shader $2${3:+, $3}"
      env WNR3_VK_DEVICE=$1 WNR3_VK_SHADERS=$O/$2 WNR3_VK_VALIDATE=1 ${3:-} $O/explain selftest 2>&1
    done
  } > $R/nvidia-main-shader.log 2>&1
fi
if step probe; then
  for dev in llvmpipe NVIDIA; do
    quiet > $R/probe-$dev.log
    WNR3_VK_DEVICE=$dev bash $D/crash_probe.sh >> $R/probe-$dev.log 2>&1
  done
fi
if step bench; then
  # 16 synthetic experts of the size of a DeepSeek V4.1 expert in one batch (limit 1 GiB), the file written
  # anew. Three rounds; in each, both shader forms alone, then both in pairs in one process. The CPU decoders
  # are timed in the first round only. Then the default form with 1 and 4 records per batch, and with the
  # direct upload (the shader reads the records where the host wrote them)
  export WNR3_VK_DEVICE=NVIDIA WNR3_VK_BATCH_BYTES=1073741824 EXPERTS=16
  rm -f $D/out/bench/big-16-5120-2304.wex4
  : > $R/bench.log
  for round in 1 2 3; do
    for form in search table pairs; do
      quiet $([ $round = 1 ] && echo cpu) >> $R/bench.log
      echo "== round $round, $form" >> $R/bench.log
      if [ $form = pairs ]; then set -- FORMS= PAIRS=1; else set -- FORMS=$form PAIRS=0; fi
      env "$@" WNR3_VK_BENCH_CPU=$([ $round = 1 ] && echo 1 || echo 0) bash $D/bench.sh 21 >> $R/bench.log 2>&1
    done
  done
  for records in 1 4; do
    quiet >> $R/bench.log
    echo "== $records records per batch, form table" >> $R/bench.log
    WNR3_VK_BENCH_CPU=0 RECORDS=$records FORMS=table PAIRS=0 bash $D/bench.sh 21 >> $R/bench.log 2>&1
  done
  quiet >> $R/bench.log
  echo "== direct upload (WNR3_VK_STAGE=0), form table" >> $R/bench.log
  WNR3_VK_BENCH_CPU=0 WNR3_VK_STAGE=0 FORMS=table PAIRS=0 bash $D/bench.sh 21 >> $R/bench.log 2>&1
fi
if step pairs; then
  # after the bench step (it uses the file it wrote): the pairs in both orders, on the 16 large records and on
  # the 48 small records of the tiny-model container, three rounds, the CPU decoders not timed. Each run
  # first times form A alone, then the pairs. bench.sh with no form and no pairs only builds
  B=$D/out/bench
  export WNR3_VK_DEVICE=NVIDIA WNR3_VK_BENCH_CPU=0
  EXPERTS=16 FORMS= PAIRS=0 bash $D/bench.sh > $R/bench-pairs.log 2>&1 || exit 1
  for round in 1 2 3; do
    for file in $B/big-16-5120-2304.wex4 wex4/out/tiny_nr3_cz.wex4; do
      for order in "search table" "table search"; do
        set -- $order
        quiet >> $R/bench-pairs.log
        echo "== round $round, $file, pairs: form A $1, form B $2" >> $R/bench-pairs.log
        WNR3_VK_BATCH_BYTES=1073741824 WNR3_VK_SHADERS=$B/$1 WNR3_VK_SHADERS_B=$B/$2 $B/wnr3_vk bench 21 $file \
            >> $R/bench-pairs.log 2>&1
      done
    done
  done
fi
# the repository root in the logs becomes <Wex-WRN2> (CONTRIBUTING.md, rule 2); pwd -W ends with a slash when
# the root is a drive
W=$(pwd -W); W=${W%/}; U=${PWD%/}
for f in $R/*.log $R/machine.txt; do
  if [ -f "$f" ]; then sed -i "s#$W/#<Wex-WRN2>/#g; s#$U/#<Wex-WRN2>/#g" "$f"; fi
done
echo "done: $STEPS"
