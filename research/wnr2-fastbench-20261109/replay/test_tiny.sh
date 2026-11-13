#!/bin/sh
# Tiny-fixture test of the whole replay toolchain, no heavy.lock (tens of MB of I/O, seconds of GPU):
#   sh test_tiny.sh            (ENGINE=0 skips the tiny engine capture)
# 1. make_trace.py self-test; 2. fixture: v41-tiny widened to moe_inter 1024 (records of 51 pages),
# its WEX1 and WNR2 (built once, kept in fixture/); 3. replay.exe --verify: all 48 WNR2 records
# against the WEX1, single and batch API; 4. replay of a synthetic trace in each mode;
# 5. the recorder's path on the tiny model: deepseek_v4.exe (ws-ab, GPU decode pipeline) under
# tap_capture.py + serve_bench.py with V4_DUMP_MOE, then make_trace.py and a replay of that trace
# (zero reads: the tiny model fits in RAM; this step checks capture, alignment and parsing).
set -eu
cd "$(dirname "$0")"
PY=C:/Python314/python.exe
REPO='<HOME>'/Wex-WRN2
LAB='<KAIRA>'/lab
WS=$LAB/v41-port-20261106/ws-ab
FW='<KAIRA>\lab\wnr2-fastbench-20261109\replay\fixture'
F=fixture
export PYTHONUTF8=1
"$PY" -I make_trace.py --self-test
if [ ! -f $F/m.nr2.wex ]; then
  "$PY" -I $REPO/fixtures/widen_experts.py --src $REPO/fixtures/v41-tiny --out "$FW\\m" --moe-inter 1024
  "$PY" -I $REPO/tools/repack_v4_wex.py --model-dir "$FW\\m" --output "$FW\\m.v1.wex" --format-version 1
  "$PY" -I $REPO/tools/verify_wex1.py --model-dir "$FW\\m" --wex "$FW\\m.v1.wex"
  (cd $REPO && "$PY" -I nr2/build_nr2_full.py --source "$FW\\m.v1.wex" --out "$FW\\m.nr2.wex" \
     --weight-chunk 2048 --scale-chunk 512 --verify-workers 2 --min-free-gib 1 --report-dir "$FW\\reports")
fi
./build/replay.exe --wex $F/m.nr2.wex --ref $F/m.v1.wex --verify 48
"$PY" -I make_trace.py --synth 6 8 40 --out $F/syn.trace --seed 7
for a in "--wex $F/m.v1.wex" "--wex $F/m.nr2.wex" "--wex $F/m.nr2.wex --batch" "--wex $F/m.v1.wex --batch" \
         "--wex $F/m.nr2.wex --io-only --split 4" "--wex $F/m.nr2.wex --pf-threads 0 --lanes 2" \
         "--wex $F/m.nr2.wex --batch --layer-ms 0.5 --skip 5 --tokens 30"; do
  echo "### $a"
  ./build/replay.exe $a --trace $F/syn.trace
done
if [ "${ENGINE:-1}" != 0 ]; then
  C=$F/capture
  mkdir -p $C
  rm -f $C/tiny.moe
  env -u COLI_WEX_PATH -u COLI_EXPERT_STORE -u COLI_MODEL_DIRS DSV4_CUDA=1 COLI_DSV4_DLL=$WS/dll-out/coli_cuda_dsv4_dg.dll DSV4_CUDA_VRAM_RESERVE_MB=1024 \
      DSV4_CUDA_EXPERT_MIRRORS=24 CTX=512 RAM_GB=2 V4_SKIP_COLD=0 V4_MTP=0 COLI_V4_SAVE_USAGE=0 V4_VRAM_FIRST=1 \
      V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=3 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=2 \
      V4_NEXT_PREFETCH_THREADS=2 V4_NEXT_PREFETCH_TOKSTART=1 V4_LOADER_LANES=2 OMP_NUM_THREADS=2 \
      DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 V4_DUMP_MOE="$FW\\capture\\tiny.moe" \
    "$PY" -I tap_capture.py --port 8781 --out $C/tiny.v4rt -- \
    "$PY" -I $LAB/v41-gpu-20261107/m0/serve_bench.py $WS/bin/deepseek_v4.exe "$FW\\m" \
      $LAB/v41-gpu-20261107/ab/prompts.jsonl $C/tiny 12 short1,code1,chat_it1
  "$PY" -I make_trace.py --tap $C/tiny.v4rt --moe $C/tiny.moe --log $C/tiny.stderr.log --out $C/tiny.trace
  ./build/replay.exe --wex $F/m.nr2.wex --trace $C/tiny.trace --batch
fi
echo "tiny test OK"
