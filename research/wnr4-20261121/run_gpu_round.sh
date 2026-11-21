#!/usr/bin/env bash
# One GPU round under the heavy lock: packs for a G sweep, the GPU bench (word fetch x warps per block x
# rows in shared/L1) and the helper test on every pack, one Nsight Compute profile.
# LOCKSH (optional) = a shell file defining take_lock LABEL / release_lock for a machine-wide GPU lock; NCU = path of ncu (optional).
set -u
cd "$(dirname "$0")"
if [ -z "${NOLOCK:-}" ]; then   # NOLOCK=1: the caller already holds the lock
  if [ -n "${LOCKSH:-}" ]; then source "$LOCKSH"; else take_lock() { :; }; release_lock() { :; }; fi
  take_lock "wnr4 research: GPU round (packs, bench, helper, ncu; ~20 min)" || exit 3
fi
mkdir -p out/packsG
for G in 32 64 256; do
  [ -f out/packsG/G$G/q38mx.pack ] || python -I measure.py q38mx out/packG-q38mx-$G.json --train 256 --test 64 --threads 8 --G $G --pb 5 --cz 1 --pack out/packsG/G$G --pack-n 64 --pack-G $G > /dev/null 2>&1
  [ -f out/packsG/G$G/q38nv.pack ] || python -I measure.py q38nv out/packG-q38nv-$G.json --train 256 --test 64 --threads 8 --G $G --pb 5 --cz 0 --wnr3 0 --pack out/packsG/G$G --pack-n 64 --pack-G $G > /dev/null 2>&1
done
for p in q38mx q38nv v41; do ./out/bench_gpu8.exe out/packs3/$p.pack 5 > results/bench8-$p-G128.log 2>&1; done
for G in 32 64 256; do for p in q38mx q38nv; do ./out/bench_gpu8.exe out/packsG/G$G/$p.pack 5 > results/bench8-$p-G$G.log 2>&1; done; done
for p in q38mx q38nv v41; do ./out/test_helper8.exe out/packs3/$p.pack > results/helper8-$p-G128.log 2>&1; done
if [ -n "${NCU:-}" ]; then   # first decode launch of each pack (all its records); GPU clocks left alone
  for p in q38mx q38nv; do
    "$NCU" --clock-control none --kernel-name regex:wnr4_cuda_kernel --launch-count 1 --section LaunchStats --section Occupancy \
      --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
      --section InstructionStats ./out/bench_gpu8.exe out/packs3/$p.pack 1 > results/ncu8-$p.txt 2>&1
  done
fi
echo round done
