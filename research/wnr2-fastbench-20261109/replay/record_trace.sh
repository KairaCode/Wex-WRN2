#!/bin/sh
# Record a V4.1 REAP decode for replay.exe, under heavy.lock.
#   DUMP=<dir outside OneDrive> [ARM=wex1|wnr2] [IDS=w0,code1,chat_it1,reason1,code2] [TOK=96] [PORT=8779] [NAME=...]
#     sh record_trace.sh
# The engine, DLLs, helper, prompts and environment are W3's (lab/v41-gpu-20261107/ab/run_w3.sh). Added:
#   V4_DUMP_MOE=$DUMP/$NAME.moe  the routed experts of every decode layer (~0.8 MB per token: it also
#                                holds each layer's input vector);
#   the routing tap (tap_capture.py) with a RESID after every token: VRAM and RAM residency at each
#   token boundary. On V4.1 the tap's FRAMEs carry no per-layer routing, hence the dump.
# make_trace.py turns them into traces/NAME.trace (+ .trace.json: counts, the engine's per-token
# MoE wait and VRAM hits for comparison). The trace holds the read set of the arm it was recorded on
# (ARM=wnr2 keeps fewer records in RAM): replay one trace on both formats to compare formats only.
set -u
cd "$(dirname "$0")"
LAB='<KAIRA>'/lab
WS=$LAB/v41-port-20261106/ws-ab
EXE=$WS/bin/deepseek_v4.exe
DLL=$WS/dll-out/coli_cuda_dsv4_dg.dll
HELPER=$LAB/v41-gpu-20261107/w1/build/kaira_wex_cuda.dll
BENCH=$LAB/v41-gpu-20261107/m0/serve_bench.py
PROMPTS=$LAB/v41-gpu-20261107/ab/prompts.jsonl
MODEL='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
PY=C:/Python314/python.exe
ARM=${ARM:-wex1}
case $ARM in
  wex1) WEX='U:\kaira_wex\v41-reap\v4_flash_experts.wex' ;;
  wnr2) WEX='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex' ;;
  *) echo "ARM must be wex1 or wnr2"; exit 2 ;;
esac
IDS=${IDS:-w0,code1,chat_it1,reason1,code2}   # no ctx2k: its 2k prefill is half of W3's time
TOK=${TOK:-96}
PORT=${PORT:-8779}
NAME=${NAME:-$ARM-$(date +%Y%m%d-%H%M)}
DUMP=${DUMP:?set DUMP to a directory outside OneDrive}
mkdir -p traces "$DUMP"
rm -f "$DUMP/$NAME.moe"   # V4_DUMP_MOE appends

. $LAB/wnr3-5090-20261107/lock.sh
take_lock "wnr2-fastbench record_trace.sh ($NAME: $IDS x $TOK tok)" || exit 3

avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
# RAM rule (user): 16 when >= 15 GB available, else 13 (W3 waits up to 10 min before settling on 13)
RAM_GB=13
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  a=$(avail)
  if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 15) }'; then RAM_GB=16; break; fi
  [ $i = 11 ] || { echo "RAM $a GB < 15 at $(date +%T), waiting"; sleep 60; }
done
for i in $(seq 1 30); do   # the engine budgets min(RAM_GB, available) at open
  a=$(avail)
  awk -v a="$a" -v r="$RAM_GB" 'BEGIN { exit !(a + 0 >= r + 0.3) }' && break
  sleep 30
done
echo "== $NAME $(date +%T): RAM $a GB available, RAM_GB=$RAM_GB, COLI_WEX_PATH=$WEX, IDS=$IDS TOK=$TOK"

export PYTHONUTF8=1 COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
COMMON="COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=4096 V4_SKIP_COLD=0 V4_MTP=0
 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=$HELPER
 DSV4_CUDA=1 COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_GPU_DIRECT=2 V4_WEX_BATCH=1 V4_GPU_HEAD=1"
{ echo "COMMON: $(echo $COMMON)"; echo "COLI_WEX_PATH=$WEX IDS=$IDS TOK=$TOK PORT=$PORT"; } > "traces/$NAME.env.txt"

env $COMMON COLI_WEX_PATH="$WEX" V4_DUMP_MOE="$DUMP/$NAME.moe" "$PY" -I tap_capture.py --port "$PORT" --out "traces/$NAME.v4rt" -- \
  "$PY" -I "$BENCH" "$EXE" "$MODEL" "$PROMPTS" "traces/$NAME" "$TOK" "$IDS"
rc=$?
echo "engine rc=$rc $(date +%T)"
grep -a "v4_route udp=\|ram_tiers\|v4_tier_prewarm\|v4_direct reads\|v4_wex_gpu_direct reads" "traces/$NAME.stderr.log" | cut -c1-200
"$PY" -I make_trace.py --tap "traces/$NAME.v4rt" --moe "$DUMP/$NAME.moe" --log "traces/$NAME.stderr.log" \
  --out "traces/$NAME.trace" || exit 1
[ $rc = 0 ] || exit $rc
