#!/usr/bin/env bash
# qwen38 A B B A through verify_gateway.py new: arm A runs EXE_A, arm B runs EXE_B (same switches).
# Usage: EXE_A=..\qwen38.exe EXE_B=..\qwen38.exe ab_exe.sh NAME [words for both arms]
# Exe: REL_EXE (default lab/gpu-miss-20261112/rel/qwen38.exe, coli_cuda.dll + kaira_wex_cuda.dll beside it).
# Then: python -I '<KAIRA>'/lab/opt-integ-20261111/ab/q38_compare.py NAME
set -u
name=$1 aw=${2:-} bw=${2:-}
BASE='QT_ARENA=1 Q38_KEEP_FREE_GB=0 CUDA_RESERVE_GB=2'
VG='<KAIRA>'/lab/qwen38-ramfix-20261109/verify_gateway.py
export VERIFY_RAM_GB_MAX=13; : "${EXE_A:?}" "${EXE_B:?}"
export COLI_WEX_PATH='U:\kaira_wex\qwen38-mxfp4\q38_experts.wex' COLI_WEX_EXTRA_MIRRORS='C:\kaira_wex\qwen38-mxfp4' COLI_WEX_WEIGHTS=38,62
for x in "$EXE_A" "$EXE_B"; do [ -f "$x" ] || { echo "== no exe at $x"; exit 1; }; done
for arm in A1 B1 B2 A2; do
  case $arm in A*) words="$BASE $aw"; export VERIFY_NEW_EXE="$EXE_A" ;; *) words="$BASE $bw"; export VERIFY_NEW_EXE="$EXE_B" ;; esac
  # never start over a running KAIRA or engine (and never kill one): wait up to 30 min, then stop
  for i in $(seq 1 60); do
    tasklist | grep -qi '^KAIRA.exe\|^deepseek_v4.exe\|^qwen38.exe\|^qwen36.exe' || break
    [ $i = 60 ] && { echo "== $arm: KAIRA or an engine still running after 30 min, stopping"; exit 1; }
    sleep 30
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${used:-99999}" -le 2400 ] || { echo "== $arm: ${used} MiB of VRAM in use (> 2400), stopping"; exit 1; }
  echo "== $name $arm $(date +%T) VERIFY_SET='$words' vram0=${used}MiB"
  VERIFY_TAG="$name-$arm-" VERIFY_SET="$words" C:/Python314/python.exe -I "$VG" new || { echo "== $arm rc=$?: stopping"; exit 1; }
  sleep 20
done
echo "== done $(date +%T)"
