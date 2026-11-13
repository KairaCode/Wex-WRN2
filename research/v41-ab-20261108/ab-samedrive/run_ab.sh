#!/bin/sh
# A/B same-drive WEX1 vs WNR2 on V4.1 REAP-192 (PLAN 3.2, both containers on U:), G2 review-3 build
# (the exe/DLL/helper of g2/locked_g2r3.out). Whole measurement under heavy.lock (wnr3 lock.sh protocol).
#   smoke A, smoke B (w0, 16 tokens), then ABBA: A1 B1 B2 A2 (w0 + code1, chat_it1, reason1, ctx2k at 128)
# The arms differ only in COLI_WEX_PATH. COLI_WEX_GPU_DIRECT=2 is set in both: on WEX1 records it is a
# no-op (coli_v4_expert_store_wex_entry only takes Neural-rANS entries; the V4.1 mirror guard is on anyway).
# No COLI_WEX_RAM_CACHE_GB: that cache only exists for Neural-rANS records, so it would give WNR2 extra RAM.
#   DUMP=<dir outside OneDrive> sh run_ab.sh
set -u
HERE='<KAIRA>'/lab/v41-gpu-20261107/ab-samedrive-20261108
cd "$HERE" || exit 2
. '<KAIRA>'/lab/wnr3-5090-20261107/lock.sh
take_lock "A/B same-drive WEX1 vs WNR2 (main session)" || exit 3
# release only our own lock (lock.sh's trap removes unconditionally)
trap 'head -c 40 "$LOCK" 2>/dev/null | grep -q "^A/B same-drive WEX1 vs WNR2" && rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
trap 'exit 130' INT TERM HUP

LAB='<KAIRA>'/lab
EXE=$LAB/v41-port-20261106/ws-g/g2bin/deepseek_v4-fix3.exe
DLL=$LAB/v41-port-20261106/ws-g/new-out3/coli_cuda_dsv4_dg.dll
HELPER=$LAB/v41-gpu-20261107/w1/build/kaira_wex_cuda.dll
BENCH=$LAB/v41-gpu-20261107/m0/serve_bench.py
PROMPTS=$LAB/v41-gpu-20261107/ab/prompts.jsonl
MODEL='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
WEX1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
WNR2='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
IDS=w0,code1,chat_it1,reason1,ctx2k
DUMP=${DUMP:?set DUMP to a directory outside OneDrive}
mkdir -p runs "$DUMP"

avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
a=$(avail)
# RAM rule (user, 2026-11-08): 16 when >= 15 GB available, else 13; decided once, same in every arm
if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 15) }'; then RAM_GB=16; else RAM_GB=13; fi
echo "RAM: $a GB available at $(date +%T), RAM_GB=$RAM_GB for every arm"
sha256sum "$EXE" "$DLL" "$LAB/v41-port-20261106/ws-g/new-out3/coli_cuda_dsv4.dll" "$HELPER" \
  "$LAB/v41-port-20261106/ws-g/g2bin/cudart64_13.dll" "$LAB/v41-port-20261106/ws-g/g2bin/cublasLt64_13.dll" > hashes.txt
cat hashes.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v "N/A\]$" || true

export PYTHONUTF8=1 COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
# run_g2.sh's environment (kaira.json decode machinery) + GPU-direct 2, identical in both arms
COMMON="COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=4096 V4_SKIP_COLD=0 V4_MTP=0
 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=$HELPER
 DSV4_CUDA=1 COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_GPU_DIRECT=2"
echo "COMMON: $(echo $COMMON)" > env.txt
echo "COLI_MODEL_DIRS=$COLI_MODEL_DIRS" >> env.txt
echo "arm A: COLI_WEX_PATH=$WEX1" >> env.txt
echo "arm B: COLI_WEX_PATH=$WNR2" >> env.txt

arm() {   # name max_tokens ids wex
  name=$1 max=$2 ids=$3 wex=$4
  a=$(avail)
  echo "== $name $(date +%T) available RAM $a GB, GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader), COLI_WEX_PATH=$wex"
  printf '{"run": "%s", "start": "%s", "avail_ram_gb": %s, "ram_gb": %s}\n' "$name" "$(date +%FT%T)" "$a" "$RAM_GB" > "runs/$name.meta.json"
  rm -f "$DUMP/$name.logits"
  env $COMMON COLI_WEX_PATH="$wex" V4_LOGITS_DUMP="$DUMP/$name.logits" \
    C:/Python314/python.exe -I "$BENCH" "$EXE" "$MODEL" "$PROMPTS" "runs/$name" "$max" "$ids"
  rc=$?
  echo "rc=$rc $(date +%T)"
  log="runs/$name.stderr.log"
  up=$(grep -c "v4_gpu layer=[0-9]* uploaded=" "$log"); dense=$(grep -c "fp8_block=32 dense=gpu" "$log")
  np=$(grep -c "v4_next_prefetch prefetch=1" "$log"); gd=$(grep -c "GPU-direct Neural-rANS reconstruction enabled" "$log")
  cont=$(grep -cF "active: $wex (" "$log"); ready=$(grep -c '"event": "ready"' "runs/$name.jsonl")
  echo "   checks: ready=$ready container=$cont dense_uploads=$up dense_gpu=$dense next_prefetch=$np gpu_direct=$gd"
  grep -F "[v4_wex] Container" "$log" | cut -c1-200
  if [ $rc != 0 ] || [ "$ready" != 1 ] || [ "$cont" != 1 ] || [ "$up" -lt 40 ] || [ "$dense" -lt 1 ] || [ "$np" -lt 1 ] || [ "$gd" -lt 1 ]; then
    echo "== $name INVALID: stopping the measurement"
    return 1
  fi
  sleep 10   # let the process's pinned memory go back before the next arm
}

arm ab-smoke-A-wex1 16 w0 "$WEX1" || exit 1
arm ab-smoke-B-wnr2 16 w0 "$WNR2" || exit 1
arm ab-A1-wex1 128 $IDS "$WEX1" || exit 1
arm ab-B1-wnr2 128 $IDS "$WNR2" || exit 1
arm ab-B2-wnr2 128 $IDS "$WNR2" || exit 1
arm ab-A2-wex1 128 $IDS "$WEX1" || exit 1
echo "== all runs done $(date +%T)"
