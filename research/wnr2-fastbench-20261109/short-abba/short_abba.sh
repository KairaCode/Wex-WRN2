#!/usr/bin/env bash
# Short end-to-end ABBA of two expert containers on the V4.1 REAP-192 engine, ~25 min instead of W3's ~3.5 h:
# one fresh warm serve process per run (serve_bench.py), order A1 B1 B2 A2. Prompt set (fixed, in this order,
# the prompts.jsonl order): w0 warm-up (16 tok, untimed), then code1 code2 chat_it1 chat_it2 reason1 reason2
# short1 at 64 tokens each = every W3 prompt but ctx2k (its 2k prefill measures prefill, not the container).
# Why 7 prompts at 64 and not 3 at 128: the prompts differ by SD ~3.8 points (W3), ~10x the pair noise, so the
# old code1/chat_it1/reason1 set sat 0.7-1 point away from W3's broad mean; on W3's logs cut at 64 tokens the
# 7-prompt set gives -1.89 % (pair means +-0.70) against -1.99 % at 256 (validate/cut_tokens.py).
# Engine env = ab/run_w3.sh's (ws-ab integrated build); the arms differ only in COLI_WEX_PATH and --a-env/--b-env.
#
# Guards, in this order (the log says what it waited for):
#  1. quiet machine, also under --no-lock: wait (poll 60 s, up to --quiet-max s, default 4 h) until no foreign
#     pure-compute (type C) process is on the GPU (nvidia-smi -q -d PIDS; WDDM desktop apps are C+G), GPU
#     utilization < 10 %, GPU memory.used < --vram-max-gb (default 4, env VRAM_MAX_GB; 0 or DSV4_CUDA=0: off)
#     and no cc1/gcc/make/nvcc/cl/blender or KAIRA engine (deepseek_v4, deepseek_v41, qwen38, qwen36; ours
#     included) process runs; checked again after the lock. The VRAM gate is what catches C+G apps (desktop 3D apps
#     desktop 3D apps, game editors, 3D app UIs held 29 of 32 GB on 2026-11-09): our engine fills VRAM to its reserve, so
#     another app's VRAM changes the expert tier; the reading goes to env.txt;
#  2. heavy.lock (wnr3-5090-20261107/lock.sh: waits up to 90 min); on exit the lock is removed only if the file
#     still holds the exact line take_lock wrote;
#  3. RAM rule (user): 16 only when RAM_GB + 0.3 = 16.3 GB is available (the start gate's own need, so no dead
#     band between 15 and 16.3 GB), waiting up to 10 min for it (record_trace.sh's loop), else 13; decided once,
#     same in every run, recorded in env.txt;
#  4. start gate before each run (no engine of ours running): guard 1's quiet check (VRAM included) and
#     RAM_GB + 0.3 GB available, polled every 30 s for up to 15 min; on timeout it writes OUT/INVALID and stops
#     BEFORE starting the engine (the old gate started anyway, at whatever budget the OS gave). memory.used and
#     available RAM at the start go to <run>.meta.json;
#  5. during the runs a sampler writes quiet.log every 10 s: GPU utilization (logged, not checked), memory.used,
#     available RAM and our engine's working set. Hits: a foreign process (our engine PIDs from serve_bench's
#     <run>.pid are not hits); while our engine runs, available RAM + its working set < RAM_GB - 1 (the OS
#     figure alone drops by the engine's ~15 GB: 31.2 GB machine), or memory.used > 3 GB above the run's
#     baseline, read at the first sample after its first prompt (between READY and then the engine still
#     allocates: 28.6 -> 31.9 GB in B2 of the 13 GB run). Any hit, or a failed run check, goes to OUT/INVALID
#     and stops the ABBA, and analyze_short.py then prints INVALID instead of a verdict. Limit: with VRAM full
#     (WNR2 arms ~31 of 32 GB) WDDM can page a foreign app's memory out, so it shows less than its size.
# Raw logs: OUT/<run>.{jsonl,stderr.log,meta.json,disk.txt,pid}, env.txt, hashes.txt, quiet.log, short_abba.log.
# Logits dumps (~240 MB per run) go to DUMP; analyze_short.py writes OUT/summary.json with their sha256 and
# OUT/tokens.json with the argmax token ids. On exit the dumps this run created are deleted only when
# tokens.json holds every one of them with the same sha256; otherwise they stay (the log says where). A dump
# that already exists is never deleted: the script refuses to start.
#
#   bash short_abba.sh [--ws DIR | --exe EXE | --app-engine] [--a CONTAINER] [--b CONTAINER]
#                      [--a-env 'K=V ...'] [--b-env 'K=V ...'] [--memory-gb N] [--quiet-max SECONDS] [--vram-max-gb N]
#                      [--ids code1,code2,...] [--tokens 64] [--warmup w0] [--seq "A1 B1 B2 A2"]
#                      [--model DIR] [--prompts FILE] [--out DIR] [--dump DIR|none] [--no-lock] [--check]
# CONTAINER "-" leaves COLI_WEX_PATH unset (MODEL\v4_flash_experts.wex, else safetensors). --a-env/--b-env add
# that arm's env words (no spaces inside a value). --check prints what the quiet-machine and RAM guards see now
# (and does the --app-engine copy) without waiting, locking or running. The caller's env passes through;
# DSV4_CUDA=0 runs the CPU engine and skips the GPU checks; CTX (default 4096, the v41gpu profile has 32768),
# COLI_MODEL_DIRS, COLI_DSV4_DLL and COLI_WEX_CUDA_LIB override the defaults.
#
# Mirror ABBA (the production question: WNR2 with the U:+C: mirror against WEX1 on U:), on the INSTALLED engine:
# --app-engine copies app\colibri-optimized\{deepseek_v4.exe, coli_cuda_dsv4_dg.dll, coli_cuda_dsv4.dll,
# kaira_wex_cuda.dll, cudart64_12.dll} into ../mirror-abba/bin, checks each sha256 against the app copy and runs
# from there (nothing runs from app\), with the v41gpu profile's CUDA v13.4 bin in front of PATH. Arm B is the
# mirror arm M; COLI_WEX_WEIGHTS sets the split (COLI_DISK_WEIGHTS does not: without it every read goes to U:).
# The run check then requires "Matching mirror container active", "[v4_wex] read split over 2 drives" and
# >= 10 % of the OS read bytes on each of the two drives (<run>.disk.txt):
#   bash short_abba.sh --app-engine --out ../mirror-abba/runs/$(date +%Y%m%d-%H%M%S) \
#     --a 'U:\kaira_wex\v41-reap\v4_flash_experts.wex' --b 'U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex' \
#     --b-env 'COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap COLI_WEX_WEIGHTS=38,62'
# Tiny-fixture check of this script (CPU, small, no lock; FIXTURE_COPY = a copy of v41-port-20261106/
# fixtures/v41-tiny, tiny.wex = Wex-WRN2/tools/repack_v4_wex.py --format-version 1 of it):
#   DSV4_CUDA=0 COLI_WEX_CUDA_LIB=0 CTX=512 COLI_MODEL_DIRS= OMP_NUM_THREADS=2 bash short_abba.sh --no-lock --model FIXTURE_COPY \
#     --a tiny.wex --b - --memory-gb 0.6 --tokens 16 --ids short1 --out OUTDIR --dump DUMPDIR
set -u
HERE=$(cygpath -m "$(cd "$(dirname "$0")" && pwd)")
LAB='<KAIRA>'/lab
APP='<KAIRA>'/app/colibri-optimized
PY=C:/Python314/python.exe
BENCH=$HERE/serve_bench.py   # m0/serve_bench.py + exact DATA reads (its short reads dropped reply bytes) + .pid
STAMP=$(date +%Y%m%d-%H%M%S)
WS=$LAB/v41-port-20261106/ws-ab EXE=
A='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
B='U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
AENV= BENV= RAM_GB= QUIET_MAX=14400 CHECK=0 APPENG=0 VRAM_MAX=${VRAM_MAX_GB:-4}
IDS=code1,code2,chat_it1,chat_it2,reason1,reason2,short1 TOK=64 WARM=w0 SEQ="A1 B1 B2 A2"
MODEL='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
PROMPTS=$LAB/v41-gpu-20261107/ab/prompts.jsonl
OUT=$HERE/runs/$STAMP DUMP=$(cygpath -m "${LOCALAPPDATA:?}")/Temp/short-abba/$STAMP LOCKED=1
while [ $# -gt 0 ]; do
  v=${2-}
  case $1 in
    --no-lock) LOCKED=0; shift; continue ;;
    --check) CHECK=1; shift; continue ;;
    --app-engine) APPENG=1; shift; continue ;;
    --ws) WS=$v ;; --exe) EXE=$v ;; --a) A=$v ;; --b) B=$v ;; --a-env) AENV=$v ;; --b-env) BENV=$v ;;
    --memory-gb) RAM_GB=$v ;; --quiet-max) QUIET_MAX=$v ;; --vram-max-gb) VRAM_MAX=$v ;;
    --ids) IDS=$v ;; --tokens) TOK=$v ;; --warmup) WARM=$v ;; --seq) SEQ=$v ;;
    --model) MODEL=$v ;; --prompts) PROMPTS=$v ;; --out) OUT=$v ;; --dump) DUMP=$v ;;
    *) echo "unknown argument: $1 (usage: header of $0)"; exit 2 ;;
  esac
  [ $# -ge 2 ] || { echo "$1 needs a value"; exit 2; }
  shift 2
done
EXE=${EXE:-$WS/bin/deepseek_v4.exe}
DLL=${COLI_DSV4_DLL:-$WS/dll-out/coli_cuda_dsv4_dg.dll}
HELPER=${COLI_WEX_CUDA_LIB:-$LAB/v41-gpu-20261107/w1/build/kaira_wex_cuda.dll}
CUDA=${DSV4_CUDA:-1}
VLIM=$(awk -v g="$VRAM_MAX" -v c="$CUDA" 'BEGIN { printf "%d", c == "0" ? 0 : g * 1024 }')   # MiB, 0 = no VRAM gate

if [ $APPENG = 1 ]; then   # the installed engine, from a sha256-checked copy
  BIN=$(dirname "$HERE")/mirror-abba/bin
  mkdir -p "$BIN" || exit 2
  for f in deepseek_v4.exe coli_cuda_dsv4_dg.dll coli_cuda_dsv4.dll kaira_wex_cuda.dll cudart64_12.dll; do
    [ -f "$APP/$f" ] || { echo "no $APP/$f"; exit 2; }
    s=$(sha256sum < "$APP/$f" | cut -c1-64)
    [ "$({ sha256sum < "$BIN/$f"; } 2>/dev/null | cut -c1-64)" = "$s" ] || cp "$APP/$f" "$BIN/$f" || exit 2
    [ "$(sha256sum < "$BIN/$f" | cut -c1-64)" = "$s" ] || { echo "sha256 of $BIN/$f differs from app\\$f"; exit 2; }
    echo "app-engine: $f ${s:0:16} = app copy"
  done
  EXE=$BIN/deepseek_v4.exe DLL=$BIN/coli_cuda_dsv4_dg.dll HELPER=$BIN/kaira_wex_cuda.dll
  PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64:$PATH"   # v41gpu profile's path_prepend
fi

# fail before the waits and the lock, not 20 minutes in
[ -f "$EXE" ] || { echo "no engine: $EXE"; exit 2; }
[ -f "$PROMPTS" ] || { echo "no prompts file: $PROMPTS"; exit 2; }
for c in "$A" "$B"; do [ "$c" = - ] || [ -f "$(cygpath -u "$c")" ] || { echo "no container: $c"; exit 2; }; done
for id in ${WARM//,/ } ${IDS//,/ }; do grep -q "\"id\": \"$id\"" "$PROMPTS" || { echo "prompt $id not in $PROMPTS"; exit 2; }; done
for s in $SEQ; do case $s in [AB][0-9]*) ;; *) echo "--seq items are A<n>/B<n>, got $s"; exit 2 ;; esac; done
for e in $AENV $BENV; do case $e in [A-Z_]*=*) ;; *) echo "--a-env/--b-env words are NAME=value, got $e"; exit 2 ;; esac; done
case $VRAM_MAX in ''|*[!0-9.]*) echo "--vram-max-gb / VRAM_MAX_GB is a number of GB, got '$VRAM_MAX'"; exit 2 ;; esac
if [ "$DUMP" != none ]; then   # never delete a dump this run did not write: refuse instead
  for s in $SEQ; do [ ! -e "$DUMP/$s.logits" ] || { echo "$DUMP/$s.logits already exists: refusing to start (it is left alone; pick another --dump)"; exit 2; }; done
fi

BUSY='^(cc1|cc1plus|gcc|g[+][+]|make|mingw32-make|nvcc|cl|blender|deepseek_v4|deepseek_v41|qwen38|qwen36)[.]exe$'
busy() {   # what keeps the machine from being quiet; $1 = " pid pid " of our own engines, $2 = 1: GPU utilization and VRAM gate too
  { tasklist //fo csv //nh | tr -d '\r' | awk -F'","' -v re="$BUSY" -v own="$1" '
      { n = tolower(substr($1, 2)); if (n ~ re && !index(own, " " $2 " ")) c[n]++ }
      END { for (n in c) printf "%s x%d, ", n, c[n] }'
    nvidia-smi -q -d PIDS | tr -d '\r' | awk -v own="$1" '
      /Process ID/ { pid = $NF } /^ *Type / { type = $NF }
      /^ *Name / { if (type == "C" && !index(own, " " pid " ")) { sub(/^[^:]*: /, ""); printf "GPU compute app %s (pid %s), ", $0, pid } }'
    [ "$2" = 1 ] && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d '\r' |
      awk -F', *' -v lim="$VLIM" '$1 + 0 >= 10 { printf "GPU utilization %d%%, ", $1 }
        lim > 0 && $2 + 0 >= lim { printf "GPU memory used %d MiB >= %d MiB (VRAM gate), ", $2, lim }'
  } 2>/dev/null | sed 's/, $//'
}
vram_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | head -1; }
ws_gb() {   # working set of PID $1 in GB (tasklist "Mem Usage", KB); empty when that process is not running
  [ -z "$1" ] || tasklist //fi "PID eq $1" //fo csv //nh 2>/dev/null | tr -d '\r' |
    awk -F'","' 'NF >= 5 { gsub(/[^0-9]/, "", $5); printf "%.1f", $5 / 1048576 }'
}
wait_quiet() {   # up to $1 s, polling every 60 s; logs what it waits for
  local t0=$SECONDS b
  while b=$(busy " " 1); [ -n "$b" ]; do
    [ $((SECONDS - t0)) -ge "$1" ] && { echo "quiet: still busy after $(( (SECONDS - t0) / 60 )) min: $b"; return 1; }
    echo "quiet: $(date +%T) waiting for: $b"
    sleep 60
  done
  echo "quiet: machine quiet at $(date +%T) after $(( (SECONDS - t0) / 60 )) min of waiting"
}
avail() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr , . | tr -d '\r'; }
diskbytes() { powershell -NoProfile -Command "Get-CimInstance Win32_PerfRawData_PerfDisk_LogicalDisk | ? { \$_.Name -match '^[A-Z]:\$' } | % { '{0}={1}' -f \$_.Name, \$_.DiskReadBytesPersec }" | tr -d '\r' | tr '\n' ' '; }

if [ $CHECK = 1 ]; then   # dry run: what the guards see now, no waiting
  b=$(busy " " 1); a=$(avail)
  echo "quiet-machine check $(date +%T): ${b:-quiet}"
  echo "   GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'); type C apps: $(nvidia-smi -q -d PIDS | tr -d '\r' | grep -c 'Type *: C$'); VRAM gate: memory.used < $VLIM MiB (0 = off)"
  echo "   heavy.lock: $(cat "$LAB/v41-gpu-20261107/heavy.lock" 2>/dev/null || echo free)"
  echo "RAM check $(date +%T): $a GB available; the rule picks $(awk -v a="$a" 'BEGIN { print (a + 0 >= 16.3) ? "16 now" : "13 unless >= 16.3 GB shows up within 10 min" }')"
  echo "per-drive read counters: $(diskbytes)"
  echo "engine $EXE; arm A $A ${AENV:+[$AENV]}; arm B $B ${BENV:+[$BENV]}; $IDS x $TOK tok after $WARM; $SEQ"
  exit 0
fi

mkdir -p "$OUT" || exit 2
OUT=$(cygpath -m "$OUT")
DUMP_NEW=0 DUMPS=() MINE= SAMPLER=
if [ "$DUMP" != none ]; then
  [ -d "$DUMP" ] || DUMP_NEW=1
  mkdir -p "$DUMP" && DUMP=$(cygpath -m "$DUMP") || exit 2
fi
exec > >(tee -a "$OUT/short_abba.log") 2>&1
# exit 0 only when tokens.json (argv[1]) holds every dump (argv[2:]) under its run name with the same sha256
COVERED='import hashlib, json, sys
from pathlib import Path
t = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["logits_sha256"]
sys.exit(not all(t.get(Path(p).stem) == hashlib.file_digest(open(p, "rb"), "sha256").hexdigest() for p in sys.argv[2:]))'
cleanup() {
  [ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null
  # only the dumps this run wrote, and only once tokens.json keeps their argmax ids and sha256
  if [ ${#DUMPS[@]} != 0 ]; then
    if "$PY" -I -c "$COVERED" "$OUT/tokens.json" "${DUMPS[@]}" 2>/dev/null; then
      rm -f "${DUMPS[@]}"; echo "dumps: deleted ${#DUMPS[@]} logits files (all in tokens.json with the same sha256)"
    else echo "dumps: kept ${#DUMPS[@]} logits files in $DUMP (tokens.json missing, incomplete or with another sha256)"; fi
  fi
  [ $DUMP_NEW = 0 ] || rmdir "$DUMP" 2>/dev/null
  if [ -n "$MINE" ]; then
    if [ "$(cat "$LOCK" 2>/dev/null)" = "$MINE" ]; then rm -f "$LOCK"; echo "lock: released $(date +%T)"
    else echo "lock: not ours any more, left alone ($(cat "$LOCK" 2>/dev/null))"; fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

wait_quiet "$QUIET_MAX" || exit 5
if [ $LOCKED = 1 ]; then
  . $LAB/wnr3-5090-20261107/lock.sh
  take_lock "short ABBA (wnr2-fastbench) $SEQ x $IDS at $TOK tok, ~25 min" || exit 3
  MINE=$(cat "$LOCK")   # the exact line take_lock wrote
  trap cleanup EXIT     # replaces lock.sh's unconditional rm
  echo "lock: holding '$MINE'"
  wait_quiet "$QUIET_MAX" || exit 5
fi
VRAMQ="memory.used $(vram_mib) MiB at $(date +%T), no engine running (gate < $VLIM MiB, 0 = off)"
echo "VRAM: $VRAMQ"

if [ -z "$RAM_GB" ]; then   # user's RAM rule with record_trace.sh's wait: up to 10 min for 16.3 GB (16 + the start gate's 0.3), else 13
  RAM_GB=13
  for i in 1 2 3 4 5 6 7 8 9 10 11; do
    a=$(avail)
    if awk -v a="$a" 'BEGIN { exit !(a + 0 >= 16.3) }'; then RAM_GB=16; break; fi
    [ $i = 11 ] || { echo "RAM: $a GB < 16.3 at $(date +%T), waiting"; sleep 60; }
  done
  RAMWHY="rule: $a GB available at $(date +%T) after $((i - 1)) min of waiting"
else
  RAMWHY="given with --memory-gb ($(avail) GB available at $(date +%T))"
fi
echo "RAM: RAM_GB=$RAM_GB for every run ($RAMWHY)"
for f in "$EXE" "$DLL" "$HELPER"; do [ -f "$f" ] && sha256sum "$f"; done > "$OUT/hashes.txt"
cat "$OUT/hashes.txt"

export PYTHONUTF8=1
export COLI_MODEL_DIRS=${COLI_MODEL_DIRS-'C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'}
[ -n "$COLI_MODEL_DIRS" ] || unset COLI_MODEL_DIRS
# ab/run_w3.sh's environment (W2 + GPU-direct 2, batched WNR2, GPU head; F1 defaults), identical in every run
COMMON=(COLI_V4_SAVE_USAGE=0 RAM_GB=$RAM_GB CTX=${CTX:-4096} V4_SKIP_COLD=0 V4_MTP=0
 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=$HELPER
 DSV4_CUDA=$CUDA COLI_DSV4_DLL=$DLL DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500
 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1
 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1
 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4
 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1
 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1
 COLI_WEX_GPU_DIRECT=2 V4_WEX_BATCH=1 V4_GPU_HEAD=1)
{ echo "COMMON: ${COMMON[*]}"; echo "COLI_MODEL_DIRS=${COLI_MODEL_DIRS-}"; echo "EXE=$EXE MODEL=$MODEL"
  [ $APPENG = 0 ] || echo "app-engine: copies of $APP in $BIN, sha256-checked; PATH starts with CUDA v13.4 bin/x64"
  echo "IDS=$WARM,$IDS TOK=$TOK SEQ=$SEQ DUMP=$DUMP"; echo "RAM_GB=$RAM_GB ($RAMWHY)"
  echo "VRAM gate before the lock and before each run: $VRAMQ"
  echo "sampler hits: foreign process; RAM available + engine working set < RAM_GB - 1; memory.used > run baseline + 3072 MiB"
  echo "arm A: $A ${AENV:+env $AENV}"; echo "arm B: $B ${BENV:+env $BENV}"; } > "$OUT/env.txt"

sampler() {   # every 10 s while the runs go; any hit invalidates the set
  local own b g v a ws cur base= brun=
  while :; do
    own=" $(cat "$OUT"/*.pid 2>/dev/null | tr -d '\r' | tr '\n' ' ')"
    b=$(busy "$own" 0)
    cur=$(ls -t "$OUT"/*.pid 2>/dev/null | head -1)   # the newest run's engine
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | head -1)
    v=${g#*, } a=$(avail) ws=$(ws_gb "$(cat "$cur" 2>/dev/null | tr -d '\r\n')")
    if [ -n "$ws" ]; then   # our engine runs: what it holds is RAM it has
      awk -v a="$a" -v w="$ws" -v r="$RAM_GB" 'BEGIN { exit !(a + w < r - 1) }' &&
        b="${b:+$b, }RAM available $a GB + engine working set $ws GB < RAM_GB $RAM_GB - 1"
      if [ "$CUDA" != 0 ] && [ -n "$v" ]; then
        if [ "$brun" != "$cur" ]; then   # baseline: first sample after the run's first prompt is done
          grep -q '"id"' "${cur%.pid}.jsonl" 2>/dev/null && brun=$cur base=$v
        elif awk -v v="$v" -v b="$base" 'BEGIN { exit !(v + 0 > b + 3072) }'; then
          b="${b:+$b, }GPU memory used $v MiB > $(basename "${cur%.pid}")'s baseline $base MiB + 3072"
        fi
      fi
    fi
    echo "$(date +%T) GPU ${g%%,*}%, $v MiB${ws:+ (engine $(basename "${cur%.pid}")$([ "$brun" = "$cur" ] && echo ", baseline $base MiB"))}; RAM available $a GB${ws:+ + engine working set $ws GB}${b:+ HIT: $b}" >> "$OUT/quiet.log"
    [ -z "$b" ] || echo "quiet-machine hit at $(date +%T): $b" >> "$OUT/INVALID"
    sleep 10
  done
}

arm() {   # name container extra-env
  local name=$1 wex=$2 a v b i rc bad= log=$OUT/$1.stderr.log wexenv=(COLI_WEX_PATH="$2") dumpenv=() extra=() mir
  read -ra extra <<< "$3"
  [ "$wex" = - ] && wexenv=(-u COLI_WEX_PATH)
  # start gate, no engine of ours running: the quiet check before the lock (VRAM gate included) and RAM_GB + 0.3
  # GB available (the engine budgets min(RAM_GB, OS available) at open); 15 min, then INVALID without starting
  for i in $(seq 1 31); do
    [ ! -s "$OUT/INVALID" ] || { echo "== $name not started: the set is already INVALID"; return 1; }
    a=$(avail) v=$(vram_mib) b=$(busy " " 1)
    awk -v a="$a" -v r="$RAM_GB" 'BEGIN { exit !(a + 0 >= r + 0.3) }' || b="${b:+$b, }available RAM $a GB < $RAM_GB + 0.3"
    [ -n "$b" ] || break
    if [ $i = 31 ]; then
      echo "$name: start gate still closed after 15 min at $(date +%T), engine not started: $b" >> "$OUT/INVALID"
      echo "== $name INVALID, engine not started: $b"; return 1
    fi
    echo "   $name: $(date +%T) waiting 30 s for: $b"; sleep 30
  done
  echo "== $name $(date +%T) available RAM $a GB$([ "$CUDA" = 0 ] || echo ", GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"), container $wex${3:+, env $3}"
  printf '{"run": "%s", "start": "%s", "avail_ram_gb": %s, "ram_gb": %s, "vram_used_mib": %s, "vram_gate_mib": %s, "container": "%s", "env": "%s", "model": "%s"}\n' \
    "$name" "$(date +%FT%T)" "$a" "$RAM_GB" "${v:-null}" "$VLIM" "${wex//\\/\\\\}" "${3//\\/\\\\}" "${MODEL//\\/\\\\}" > "$OUT/$name.meta.json"
  if [ "$DUMP" != none ]; then
    [ ! -e "$DUMP/$name.logits" ] || { echo "$name: $DUMP/$name.logits already exists, refused to start" >> "$OUT/INVALID"; echo "== $name: $DUMP/$name.logits already exists: refusing to start (left alone)"; return 1; }
    dumpenv=(V4_LOGITS_DUMP="$DUMP/$name.logits"); DUMPS+=("$DUMP/$name.logits")
  fi
  echo "start $(diskbytes)" > "$OUT/$name.disk.txt"
  env "${wexenv[@]}" "${COMMON[@]}" "${extra[@]}" "${dumpenv[@]}" \
    "$PY" -I "$BENCH" "$EXE" "$MODEL" "$PROMPTS" "$OUT/$name" "$TOK" "${WARM:+$WARM,}$IDS"
  rc=$?
  echo "end $(diskbytes)" >> "$OUT/$name.disk.txt"
  echo "rc=$rc $(date +%T)$([ "$DUMP" = none ] || echo " logits sha256 $(sha256sum "$DUMP/$name.logits" 2>/dev/null | cut -c1-16)")"
  [ $rc = 0 ] || bad="$bad rc=$rc"
  grep -q '"event": "ready"' "$OUT/$name.jsonl" 2>/dev/null || bad="$bad no-READY"
  [ "$wex" = - ] || grep -qF "active: $wex (" "$log" || bad="$bad container-not-active"
  if [ "$CUDA" != 0 ]; then   # run_w3.sh's checks: the GPU tiers really ran
    [ "$(grep -c 'v4_gpu layer=[0-9]* uploaded=' "$log")" -ge 40 ] || bad="$bad dense-uploads"
    grep -q 'fp8_block=32 dense=gpu' "$log" || bad="$bad dense-gpu"
    grep -q 'v4_next_prefetch prefetch=1' "$log" || bad="$bad next-prefetch"
    grep -q 'GPU-direct Neural-rANS reconstruction enabled' "$log" || bad="$bad gpu-direct"
    grep -q "V4_WEX_BATCH=1: a layer's WNR2 misses" "$log" || bad="$bad wex-batch"
    grep -q 'v41_moe decode=gpu-pipeline' "$log" || bad="$bad moe-gpu"
  fi
  mir=$(printf '%s\n' "${extra[@]}" | sed -n 's/^COLI_WEX_EXTRA_MIRRORS=//p')
  if [ -n "$mir" ]; then   # the mirror really served reads: engine lines + OS read bytes on both drives
    grep -q 'Matching mirror container active' "$log" || bad="$bad mirror-not-active"
    case " $3 " in *' COLI_WEX_WEIGHTS='*) grep -q '\[v4_wex\] read split over 2 drives' "$log" || bad="$bad no-read-split" ;; esac
    awk -v p="${wex:0:2}" -v m="${mir:0:2}" '
      { for (i = 2; i <= NF; i++) { split($i, kv, "="); v[$1, toupper(kv[1])] = kv[2] } }
      END { dp = v["end", toupper(p)] - v["start", toupper(p)]; dm = v["end", toupper(m)] - v["start", toupper(m)]
            printf "   OS read bytes: %s %.0f GiB, %s %.0f GiB\n", p, dp / 2^30, m, dm / 2^30
            exit !(dp + dm > 0 && dp >= 0.1 * (dp + dm) && dm >= 0.1 * (dp + dm)) }' "$OUT/$name.disk.txt" ||
      bad="$bad one-drive-reads"
  fi
  grep -a 'ram_tiers\|v4_wex_gpu_direct reads\|\] v4_direct reads\|read split over\|Matching mirror' "$log" | cut -c1-200
  [ -s "$OUT/INVALID" ] && bad="$bad quiet-machine-hit"
  [ -z "$bad" ] || { echo "$name:$bad" >> "$OUT/INVALID"; echo "== $name INVALID:$bad: stopping"; return 1; }
  sleep 10   # let the process's pinned memory go back before the next run
}

sampler & SAMPLER=$!
t0=$SECONDS
for s in $SEQ; do
  case $s in A*) arm "$s" "$A" "$AENV" ;; B*) arm "$s" "$B" "$BENV" ;; esac || exit 1
done
kill "$SAMPLER" 2>/dev/null; SAMPLER=
echo "== runs done $(date +%T), $(( (SECONDS - t0) / 60 )) min"
dumparg=(); [ "$DUMP" = none ] || dumparg=(--dump "$DUMP")
"$PY" -I "$HERE/analyze_short.py" "$OUT" "${dumparg[@]}" --warmup "$WARM" --json "$OUT/summary.json"
