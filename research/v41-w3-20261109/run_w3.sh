#!/bin/sh
# W3 (PLAN 3.2): WEX1 vs WNR2 at equal RAM on the V4.1 GPU tiers, integrated engine built fresh
# from src in ws-ab (G1+G2+G3+P1+F1+W2). Only under heavy.lock (locked_w3.ps1).
# Both containers on U: (same drive: the M0 D:/E: confound does not apply). The arms differ only in
# COLI_WEX_PATH. One fresh serve process per run (m0/serve_bench.py): w0 warm-up + the 8 protocol
# prompts at TOK tokens. Order: smoke A, smoke B (w0, 16 tok), then ABBA ABBA (A1 B1 B2 A2 A3 B3 B4 A4).
# A file named STOP next to this script ends the sequence before the next run (ABBA once = 4 runs).
#   DUMP=<dir outside OneDrive> sh run_w3.sh
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
WEX1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
WNR2='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
IDS=${IDS:-w0,code1,code2,chat_it1,chat_it2,reason1,reason2,ctx2k,short1}
TOK=${TOK:-256}
SEQ=${SEQ:-A1 B1 B2 A2 A3 B3 B4 A4}   # sA sB: the smokes (run once, 01:50)
DUMP=${DUMP:?set DUMP to a directory outside OneDrive}
mkdir -p runs "$DUMP"
rm -f STOP

avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
# RAM rule (PLAN UPDATE 2026-11-07 21:00 / user 2026-11-08): 16 when >= 15 GB available, else wait up
# to 10 min, then 13; decided once, the same in every arm
RAM_GB=13
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  a=$(avail)
  if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 15) }'; then RAM_GB=16; break; fi
  [ $i = 11 ] || { echo "RAM $a GB < 15 at $(date +%T), waiting"; sleep 60; }
done
echo "RAM: $a GB available at $(date +%T), RAM_GB=$RAM_GB for every arm"
sha256sum "$EXE" "$DLL" "$WS/dll-out/coli_cuda_dsv4.dll" "$HELPER" "$WS/bin/cudart64_13.dll" "$WS/bin/cublasLt64_13.dll" > hashes.txt
cat hashes.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v "N/A\]\s*$" || true

export PYTHONUTF8=1 COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
# W2's environment (kaira.json decode machinery) + GPU-direct 2, batched WNR2, GPU head; F1 defaults
# (V4_NP_MIN_CONF / V4_NP_TOKEN_MB unset = off). No COLI_WEX_RAM_CACHE_GB: that cache holds only
# Neural-rANS records and comes on top of --memory-gb, so it would give the WNR2 arm 4 GB more RAM.
COMMON="COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=4096 V4_SKIP_COLD=0 V4_MTP=0
 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=$HELPER
 DSV4_CUDA=1 COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_GPU_DIRECT=2 V4_WEX_BATCH=1 V4_GPU_HEAD=1"
{ echo "COMMON: $(echo $COMMON)"; echo "COLI_MODEL_DIRS=$COLI_MODEL_DIRS"; echo "IDS=$IDS TOK=$TOK SEQ=$SEQ"
  echo "arm A: COLI_WEX_PATH=$WEX1"; echo "arm B: COLI_WEX_PATH=$WNR2"; } > env.txt

arm() {   # name max_tokens ids wex
  name=$1 max=$2 ids=$3 wex=$4
  # the engine budgets min(RAM_GB, OS available) at open (ram_tiers available=): wait until the OS
  # has RAM_GB + 0.3 GiB, so every run gets the same RAM tier (smoke A got 6 slots/layer at 15.49)
  for i in $(seq 1 30); do
    a=$(avail)
    awk -v a="$a" -v r="$RAM_GB" 'BEGIN { exit !(a + 0 >= r + 0.3) }' && break
    echo "   $name: available RAM $a GB < $RAM_GB + 0.3 at $(date +%T), waiting 30 s"; sleep 30
  done
  echo "== $name $(date +%T) available RAM $a GB, GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader), COLI_WEX_PATH=$wex"
  printf '{"run": "%s", "start": "%s", "avail_ram_gb": %s, "ram_gb": %s}\n' "$name" "$(date +%FT%T)" "$a" "$RAM_GB" > "runs/$name.meta.json"
  rm -f "$DUMP/$name.logits"
  env $COMMON COLI_WEX_PATH="$wex" V4_LOGITS_DUMP="$DUMP/$name.logits" \
    C:/Python314/python.exe -I "$BENCH" "$EXE" "$MODEL" "$PROMPTS" "runs/$name" "$max" "$ids"
  rc=$?
  echo "rc=$rc $(date +%T) logits sha256 $(sha256sum "$DUMP/$name.logits" 2>/dev/null | cut -c1-16)"
  log="runs/$name.stderr.log"
  up=$(grep -c "v4_gpu layer=[0-9]* uploaded=" "$log"); dense=$(grep -c "fp8_block=32 dense=gpu" "$log")
  np=$(grep -c "v4_next_prefetch prefetch=1" "$log"); gd=$(grep -c "GPU-direct Neural-rANS reconstruction enabled" "$log")
  cont=$(grep -cF "active: $wex (" "$log"); ready=$(grep -c '"event": "ready"' "runs/$name.jsonl")
  bat=$(grep -c "V4_WEX_BATCH=1: a layer's WNR2 misses" "$log"); dec=$(grep -c "v41_moe decode=gpu-pipeline" "$log")
  ram=$(grep -c "ram_tiers available=$RAM_GB.00GiB" "$log")
  echo "   checks: ready=$ready container=$cont dense_uploads=$up dense_gpu=$dense next_prefetch=$np gpu_direct=$gd batch=$bat moe_gpu=$dec ram_budget_full=$ram"
  grep -a "^.\{12\}wexprof\|v4_wex_gpu_direct reads\|^.\{12\}v4_direct reads\|ram_tiers\|RAM cache of compressed" "$log" | cut -c1-260
  if [ $rc != 0 ] || [ "$ready" != 1 ] || [ "$cont" != 1 ] || [ "$up" -lt 40 ] || [ "$dense" -lt 1 ] || [ "$np" -lt 1 ] \
     || [ "$gd" -lt 1 ] || [ "$bat" != 1 ] || [ "$dec" -lt 1 ]; then
    echo "== $name INVALID: stopping the measurement"
    return 1
  fi
  sleep 10   # let the process's pinned memory go back before the next run
}

for s in $SEQ; do
  [ -f STOP ] && { echo "== STOP file found before $s at $(date +%T): sequence ended"; break; }
  case $s in
    sA) arm w3-smoke-A-wex1 16 w0 "$WEX1" || exit 1 ;;
    sB) arm w3-smoke-B-wnr2 16 w0 "$WNR2" || exit 1 ;;
    A*) arm w3-$s-wex1 "$TOK" "$IDS" "$WEX1" || exit 1 ;;
    B*) arm w3-$s-wnr2 "$TOK" "$IDS" "$WNR2" || exit 1 ;;
  esac
done
echo "== done $(date +%T)"
