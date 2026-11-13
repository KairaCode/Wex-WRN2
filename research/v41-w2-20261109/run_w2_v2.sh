#!/bin/sh
# W2 (batched WNR2 decode per layer) REAP GPU runs. Only under heavy.lock (locked_w2.ps1).
# One fresh serve process per arm (m0/serve_bench.py), both containers on U:, the A/B's
# environment (kaira.json decode machinery + COLI_WEX_GPU_DIRECT=2), w0 + code1, chat_it1, reason1
# at TOK tokens. Arms differ only in COLI_WEX_PATH and V4_WEX_BATCH:
#   A   WEX1,  V4_WEX_BATCH=1 (a no-op on WEX1 records: the token reference)
#   R   WNR2,  V4_WEX_BATCH=0 (per-record decode, as before W2)
#   B   WNR2,  V4_WEX_BATCH=1 (batched)
# Order A R B B R: tokens/logits of every arm against A, and an RB/BR pair for the timing.
set -u
cd "$(dirname "$0")"
LAB='<KAIRA>'/lab
WS=$LAB/v41-port-20261106/ws-w
EXE=$WS/bin/deepseek_v4.exe
DLL=$LAB/v41-port-20261106/ws-integ/dll-out/coli_cuda_dsv4_dg.dll
HELPER=$LAB/v41-gpu-20261107/w1/build/kaira_wex_cuda.dll
BENCH=$LAB/v41-gpu-20261107/m0/serve_bench.py
PROMPTS=$LAB/v41-gpu-20261107/ab/prompts.jsonl
MODEL='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
WEX1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
WNR2='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
IDS=${IDS:-w0,code1,chat_it1,reason1}
TOK=${TOK:-128}
ARMS=${ARMS:-A R B B2 R2}
DUMP=${DUMP:?set DUMP to a directory outside OneDrive}
mkdir -p runs "$DUMP"

avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
# RAM rule (PLAN UPDATE 2026-11-07 21:00 / 2026-11-08): 16 when >= 15 GB available, else wait up
# to 10 min, then 13; decided once, the same in every arm
RAM_GB=13
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  a=$(avail)
  if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 15) }'; then RAM_GB=16; break; fi
  [ $i = 11 ] || { echo "RAM $a GB < 15 at $(date +%T), waiting"; sleep 60; }
done
echo "RAM: $a GB available at $(date +%T), RAM_GB=$RAM_GB for every arm"
sha256sum "$EXE" "$WS/bin-diag/deepseek_v4.exe" "$DLL" "$HELPER" > hashes.txt
cat hashes.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

export PYTHONUTF8=1 COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
COMMON="COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=4096 V4_SKIP_COLD=0 V4_MTP=0
 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=$HELPER
 DSV4_CUDA=1 COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_GPU_DIRECT=2 V4_GPU_HEAD=1"
{ echo "COMMON: $(echo $COMMON)"; echo "COLI_MODEL_DIRS=$COLI_MODEL_DIRS"; echo "IDS=$IDS TOK=$TOK ARMS=$ARMS"; } > env.txt

arm() {   # name wex batch [exe extra-env]
  name=$1 wex=$2 batch=$3 exe=${4:-$EXE} extra=${5:-}
  a=$(avail)
  echo "== $name $(date +%T) available RAM $a GB, GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader), COLI_WEX_PATH=$wex V4_WEX_BATCH=$batch"
  rm -f "$DUMP/$name.logits"
  env $COMMON $extra COLI_WEX_PATH="$wex" V4_WEX_BATCH=$batch V4_LOGITS_DUMP="$DUMP/$name.logits" \
    C:/Python314/python.exe -I "$BENCH" "$exe" "$MODEL" "$PROMPTS" "runs/$name" "$TOK" "$IDS"
  rc=$?
  echo "rc=$rc $(date +%T) logits sha256 $(sha256sum "$DUMP/$name.logits" 2>/dev/null | cut -c1-16)"
  log="runs/$name.stderr.log"
  up=$(grep -c "v4_gpu layer=[0-9]* uploaded=" "$log"); gd=$(grep -c "GPU-direct Neural-rANS reconstruction enabled" "$log")
  cont=$(grep -cF "active: $wex (" "$log"); on=$(grep -c "V4_WEX_BATCH=1: a layer's WNR2 misses" "$log")
  echo "   checks: container=$cont dense_uploads=$up gpu_direct=$gd batch_on=$on"
  grep -a "^.\{12\}wexprof\|v4_wex_gpu_direct reads" "$log"
  if [ $rc != 0 ] || [ "$cont" != 1 ] || [ "$up" -lt 40 ] || [ "$gd" -lt 1 ] || { [ "$batch" = 1 ] && [ "$on" != 1 ]; }; then
    echo "== $name INVALID: stopping"
    return 1
  fi
  sleep 10   # let the process's pinned memory go back before the next arm
}
for a in $ARMS; do
  case $a in
    A) arm w2-A-wex1 "$WEX1" 1 || break ;;
    R) arm w2-R-wnr2-record "$WNR2" 0 || break ;;
    B) arm w2-B-wnr2-batch "$WNR2" 1 || break ;;
    B2) arm w2-B2-wnr2-batch "$WNR2" 1 || break ;;
    R2) arm w2-R2-wnr2-record "$WNR2" 0 || break ;;
    # run 2 (predict before the demand batch); D: lab-only exe with the prefetch twin off
    R3) arm w2-R3-wnr2-record "$WNR2" 0 || break ;;
    B3) arm w2-B3-wnr2-batch "$WNR2" 1 || break ;;
    B4) arm w2-B4-wnr2-batch "$WNR2" 1 || break ;;
    R4) arm w2-R4-wnr2-record "$WNR2" 0 || break ;;
    D) arm w2-D-wnr2-batch-notwin "$WNR2" 1 "$WS/bin-diag/deepseek_v4.exe" W2_DIAG_NOTWIN=1 || break ;;
    D2) arm w2-D2-wnr2-batch-notwin "$WNR2" 1 "$WS/bin-diag/deepseek_v4.exe" W2_DIAG_NOTWIN=1 || break ;;
    # run 3: reserve, lanes, then the batch's reads and decode (concurrent with the lanes)
    R5) arm w2-R5-wnr2-record "$WNR2" 0 || break ;;
    B5) arm w2-B5-wnr2-batch "$WNR2" 1 || break ;;
    B6) arm w2-B6-wnr2-batch "$WNR2" 1 || break ;;
    R6) arm w2-R6-wnr2-record "$WNR2" 0 || break ;;
    # run 4: the same code, wexprof split per-record loads into demand / prefetch with read ms
    R7) arm w2-R7-wnr2-record "$WNR2" 0 || break ;;
    B7) arm w2-B7-wnr2-batch "$WNR2" 1 || break ;;
    # run 5: the batch's reads issued, then the next layer's guesses, then its wait
    R8) arm w2-R8-wnr2-record "$WNR2" 0 || break ;;
    B8) arm w2-B8-wnr2-batch "$WNR2" 1 || break ;;
    B9) arm w2-B9-wnr2-batch "$WNR2" 1 || break ;;
    R9) arm w2-R9-wnr2-record "$WNR2" 0 || break ;;
  esac
done
echo "== done $(date +%T)"
