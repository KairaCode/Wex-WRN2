#!/usr/bin/env bash
# Qwen3.8 MXFP4 at temperature 0.6 (what the KAIRA host sends), two arms through verify_gateway_t.py 'new':
#   A = the build installed in app\colibri-q38mx (opt-q38-prefix d5e319e), B = master e245f4c (nucleus sort + one read team).
# Both arms: RAM 13 cap, the qwen38 profile as installed 11-11 18:19 (WEX1 container + C: mirror, QT_ARENA=1).
# Usage: q38_ab_temp.sh      (no arguments)      Then: python -I q38_ab_compare.py
# The harness next to this script writes verify-q38samp-A-new.json and verify-q38samp-B-new.json next to itself.
set -u
VG="$(dirname "$0")/verify_gateway_t.py"
L='<KAIRA>\lab'
A=$L'\opt-integ-20261111\rel-q38p\'                 # both end in a backslash
B=$L'\q38-speed-research-20261111\rel-e245f4c\'
[ -f "$VG" ] || { echo "== missing $VG: stopping"; exit 1; }
for d in "$A" "$B"; do
  [ -f "${d}qwen38.exe" ] || { echo "== missing ${d}qwen38.exe: stopping"; exit 1; }
  # the engine loads coli_cuda.dll from its own folder (backend_loader.c): both arms must hold arm A's two DLLs
  for f in coli_cuda.dll kaira_wex_cuda.dll; do
    cmp -s "$A$f" "$d$f" || { echo "== $d$f is missing or is not arm A's file (copy it from $A): stopping"; exit 1; }
  done
  # the CPU-only trap: a release exe names coli_cuda.dll and has no HIP loader (COLI_HIP_RUNTIME_DIR) linked in
  if [ "$(grep -a -c COLI_HIP_RUNTIME_DIR "${d}qwen38.exe")" != 0 ] || [ "$(grep -a -c coli_cuda.dll "${d}qwen38.exe")" = 0 ]; then
    echo "== ${d}qwen38.exe is not a CUDA release build: stopping"; exit 1
  fi
done
# which exe is which: only the new sampler has serve_sample_sorted
[ "$(grep -a -c serve_sample_sorted "${A}qwen38.exe")" = 0 ] && [ "$(grep -a -c serve_sample_sorted "${B}qwen38.exe")" != 0 ] || { echo "== arm A or B is not the build it should be: stopping"; exit 1; }
# the engine sizes its expert tier for the whole card: other programs' VRAM makes WDDM spill it to shared memory
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
# (today's valid runs started at 1138-1327 MiB and peaked at the card's 32 GB ceiling)
[ "${used:-99999}" -le 1600 ] || { echo "== $used MiB of VRAM in use by other programs (limit 1600): close them, stopping"; exit 1; }
rm -f "$(dirname "$0")"/verify-q38samp-*   # no stale result next to a new one
# the harness carries its own copy of the profile, without the WEX container and QT_ARENA: they come from here
unset VERIFY_CTX VERIFY_RESERVE_GB COLI_PREFIX_STORE COLI_PREFIX_STORE_MB COLI_PREFIX_STORE_BLOCK
# COLI_LOGIT_DUMP=1 (both arms): each step's top five logits, to tell a sampler difference from logit noise if the texts differ
export VERIFY_TEMP=0.6 VERIFY_RAM_GB_MAX=13 VERIFY_SET='QT_ARENA=1 COLI_LOGIT_DUMP=1'
export COLI_WEX_PATH='U:\kaira_wex\qwen38-mxfp4\q38_experts.wex' COLI_WEX_EXTRA_MIRRORS='C:\kaira_wex\qwen38-mxfp4' COLI_WEX_WEIGHTS=38,62
# B first: the first run of a session is the slow one (cold file cache), so this order cannot flatter B
for arm in B A; do
  case $arm in A) exe="${A}qwen38.exe" ;; *) exe="${B}qwen38.exe" ;; esac
  if tasklist | grep -qi '^deepseek_v4.exe\|^qwen38.exe\|^qwen36.exe'; then echo "== $arm: an engine runs, stopping"; exit 1; fi
  echo "== q38samp $arm $(date +%T) temperature $VERIFY_TEMP VERIFY_SET='$VERIFY_SET' $exe sha256 $(sha256sum "$exe" | cut -c1-16)"
  VERIFY_TAG="q38samp-$arm-" VERIFY_NEW_EXE="$exe" C:/Python314/python.exe -I "$VG" new || { echo "== $arm rc=$?: stopping"; exit 1; }
  sleep 20
done
echo "== done $(date +%T)"
