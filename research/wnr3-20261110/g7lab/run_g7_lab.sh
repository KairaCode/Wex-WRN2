#!/bin/bash
# G7-lab: end-to-end canonical-zero proof on V4.1 REAP without any WNR3 container. PREPARED, NOT RUN.
# The engine (worktree wnr3-20261110 + the lab switch V4_LAB_CANON_ZERO, built fresh in
# ../fresh-verify) reads the WEX1 container on U: (raw records, GPU-direct 0) and, with the switch,
# turns every FP4 -0 nibble of every expert slab into +0 right after the read: what a WNR3-CZ
# container decodes to. Per-step logits (V4_LOGITS_DUMP: every row the head computes) are compared
# byte for byte by compare_g7.py.
#   arms (ABAB + M): A1 off, B1 on, A2 off, B2 on, M1 on + DSV4_CUDA_MOE_MASKED=1
#   A1 == A2 byte-identical first (else INCONCLUSIVE), then A1 == B1 == B2 == M1.
#   M1: deepseek_v4.exe never calls dsv4_cuda_route_moe (the only caller of the masked dg_moe<256>),
#   so the masked env must change nothing; the logs must not show "DeepGEMM routed experts active".
# Waits (poll 2 min, max 3 h) for heavy.lock free and no deepseek_v4.exe, takes the lock, releases.
#   DUMP=D:/kaira_wnr3_lab/g7lab bash run_g7_lab.sh        (TOK=64 for a short pass; ARMS="A1 B1 A2" ...)
set -u
cd "$(dirname "$0")"
LAB=${KAIRA:?set KAIRA to the KAIRA tree (forward slashes)}/lab
LOCK=$LAB/v41-gpu-20261107/heavy.lock
EXE=$(pwd -W)/bin/deepseek_v4.exe
DLL=$(pwd -W)/bin/coli_cuda_dsv4_dg.dll
BENCH=$LAB/v41-gpu-20261107/m0/serve_bench.py
PROMPTS=$LAB/v41-gpu-20261107/ab/prompts.jsonl
MODEL='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
WEX1='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
IDS=${IDS:-code1,chat_it1,reason1,ctx2k}
TOK=${TOK:-256}
ARMS=${ARMS:-A1 B1 A2 B2 M1}
DUMP=${DUMP:-D:/kaira_wnr3_lab/g7lab}
mkdir -p runs "$DUMP"

deadline=$((SECONDS + 10800))
while :; do
  if ! tasklist //FI "IMAGENAME eq deepseek_v4.exe" 2>/dev/null | grep -qi deepseek_v4 &&
     ( set -o noclobber; printf '%s %s\n' "wnr3-impl-20261110 G7-lab logits (~60 min)" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; then break; fi
  [ $SECONDS -gt $deadline ] && { echo "NOT RUN: lock/engine busy for 3 h"; exit 3; }
  sleep 120
done
trap 'rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
echo "lock: taken $(date +%T)"

avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
# RAM rule (user 2026-11-08): 16 when >= 15 GB available (wait up to 10 min), else 13; same in every arm
RAM_GB=13
for i in $(seq 1 11); do
  a=$(avail)
  if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 15) }'; then RAM_GB=16; break; fi
  [ $i = 11 ] || { echo "RAM $a GB < 15 at $(date +%T), waiting"; sleep 60; }
done
echo "RAM: $a GB available, RAM_GB=$RAM_GB for every arm"
sha256sum bin/* | tee hashes.txt

export PYTHONUTF8=1 COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
# G3 arm A's configuration (WEX1 on U:, GPU-direct 0, the kaira.json decode machinery, CPU head)
COMMON="COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=4096 V4_SKIP_COLD=0 V4_MTP=0 DSV4_DECODE_PROF=1
 DSV4_CUDA=1 COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_PATH=$WEX1 COLI_WEX_GPU_DIRECT=0"
{ echo "COMMON: $(echo $COMMON)"; echo "IDS=$IDS TOK=$TOK ARMS=$ARMS"; } > env.txt

arm() {   # name extra-env...
  name=$1; shift
  for i in $(seq 1 30); do   # same RAM tier in every arm
    a=$(avail); awk -v a="$a" -v r="$RAM_GB" 'BEGIN { exit !(a + 0 >= r + 0.3) }' && break
    echo "   $name: available RAM $a GB < $RAM_GB + 0.3, waiting 30 s"; sleep 30
  done
  echo "== $name $(date +%T) RAM $a GB $*"
  rm -f "$DUMP/$name.logits"
  env $COMMON "$@" V4_LOGITS_DUMP="$DUMP/$name.logits" \
    C:/Python314/python.exe -I "$BENCH" "$EXE" "$MODEL" "$PROMPTS" "runs/$name" "$TOK" "$IDS"
  echo "rc=$? $(date +%T) logits $(sha256sum "$DUMP/$name.logits" 2>/dev/null | cut -c1-16)"
  sleep 10
}
for a in $ARMS; do
  case $a in
    A*) arm g7-$a ;;
    B*) arm g7-$a V4_LAB_CANON_ZERO=1 ;;
    M*) arm g7-$a V4_LAB_CANON_ZERO=1 DSV4_CUDA_MOE_MASKED=1 ;;
  esac
done
C:/Python314/python.exe -I compare_g7.py "$DUMP" $ARMS | tee compare.out
