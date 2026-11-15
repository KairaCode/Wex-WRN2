#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The round trips of this run on the experts that fetch_experts.py wrote, with the repository's tools only.
# Per checkpoint (name:profile below):
#   1. wex4_convert.py pack --experts raw --names PROFILE, wex4.py verify with the reference and with the fast
#      WNR3 DLL, c/tools/wex4_verify.c, unpack, and cmp of every rebuilt file with its source;
#   2. three WNM2 models (records.py model, the training of c/tools/build_nr3.py): on the 8 experts, on the 4 of
#      the first layer, on the 4 of the second;
#   3. pack --experts nr3 with each model (no canonical zero: byte-exact), verify, unpack and cmp;
#   4. records.py extract: the slabs, the WNR3 records and the model, the inputs of bench.sh.
# DeepSeek V4.1 also through the production path: c/tools/repack_v4_wex.py (WEX1) -> c/tools/build_nr3.py
# container (WEX3, WNR3, with and without canonical zero) -> wex4_convert.py upgrade --names deepseek -> verify
# -> downgrade, which must give the WEX3 / WEX1 file back byte for byte; and pack --experts nr3 --cz with the same
# model, whose records must be the WEX3 file's.
# Needs the DLLs of wex4/build_and_test.sh in wex4/out (bash scripts/check_all.sh builds them).
# Usage: bash research/real-experts-20261115/run.sh [DATA_DIR] [WORK_DIR]
#   defaults: data/ and out/ next to this script (both git-ignored). Last line on success: run.sh: all round trips passed
set -eu
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
D=$(cd "${1:-$HERE/data}" && pwd)
mkdir -p "${2:-$HERE/out}"
W=$(cd "${2:-$HERE/out}" && pwd)
PY=${PYTHON:-python}
CC=${CC:-gcc}
X=$ROOT/wex4/out
FAST=$X/wex4_tools.dll REF=$X/wex4_tools_ref.dll NR3=$X/wex_nr3_tools.dll
for f in "$FAST" "$REF" "$NR3"; do [ -f "$f" ] || { echo "missing $f: run bash scripts/check_all.sh first" >&2; exit 1; }; done
CONV="$PY -I $ROOT/wex4/wex4_convert.py"
RD="$PY -I $ROOT/wex4/wex4.py"
REC="$PY -I $HERE/records.py"
export TMP="$W" TEMP="$W"
"$CC" -O2 -Wall -Wextra -fopenmp "$ROOT/c/tools/wex4_verify.c" -o "$W/wex4_verify.exe"
sha() { sha256sum "$1" | cut -c1-64; }
verify() {   # wex4.py with the reference and with the fast WNR3 decoder (same digest), then the C verifier, 8 threads
  a=$($RD verify "$1" --dll "$REF")
  echo "$a" | sed 's/^/  /'
  b=$($RD verify "$1" --dll "$FAST" | grep 'content digest')
  [ "$(echo "$a" | grep 'content digest')" = "$b" ] || { echo "the two DLLs give different content digests" >&2; exit 1; }
  echo "  the fast decoder's DLL gives the same content digest"
  "$W/wex4_verify.exe" "$1" 8 | tail -1 | sed 's/^/  wex4_verify: /'
}
same_files() {   # every rebuilt file equals its source
  for s in "$@"; do cmp "$s" "$U/$(basename "$s")"; done
  echo "  cmp: $# of $# rebuilt files identical to the source files"
}
for spec in deepseek-v41:deepseek qwen3-235b:qwen3-moe minimax-m21:mixtral; do
  n=${spec%%:*} p=${spec#*:}
  S=$(ls "$D/$n"/*.experts.safetensors)
  O=$W/$n
  rm -rf "$O" && mkdir -p "$O"
  layers=$($PY -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(str(l) for c in d['checkpoints'] if c['name']==sys.argv[2] for l in c['layers']))" "$D/fetch-log.json" "$n")
  set -- $layers
  LA=$1 LB=$2
  echo "== $n (profile $p, layers $LA and $LB): $(echo $S | wc -w) source files, $(cat $S | wc -c) B"
  for s in $S; do echo "  source $(basename "$s") $(wc -c < "$s") B sha256 $(sha "$s")"; done
  echo "-- pack --experts raw --names $p"
  $CONV pack "$O/raw.wex4" $S --experts raw --names "$p" --dll "$FAST" | sed 's/^/  /'
  echo "  container $(wc -c < "$O/raw.wex4") B"
  verify "$O/raw.wex4"
  U=$O/raw-unpacked
  $CONV unpack "$O/raw.wex4" "$U" | sed 's/^/  unpack: /'
  same_files $S
  echo "-- models"
  $REC model "$NR3" "$O/all.wnm2" "$p" $S | sed 's/^/  all: /'
  $REC model "$NR3" "$O/la.wnm2" "$p" $S --layers "$LA" | sed 's/^/  la: /'
  $REC model "$NR3" "$O/lb.wnm2" "$p" $S --layers "$LB" | sed 's/^/  lb: /'
  for m in all la lb; do
    echo "-- pack --experts nr3 --names $p --model $m.wnm2"
    $CONV pack "$O/nr3-$m.wex4" $S --experts nr3 --names "$p" --nr3-dll "$NR3" --model "$O/$m.wnm2" --dll "$FAST" | sed 's/^/  /'
    echo "  container $(wc -c < "$O/nr3-$m.wex4") B"
    verify "$O/nr3-$m.wex4"
    U=$O/nr3-$m-unpacked
    $CONV unpack "$O/nr3-$m.wex4" "$U" --dll "$FAST" | sed 's/^/  unpack: /'
    same_files $S
    $REC extract "$O/nr3-$m.wex4" "$O/rec-$m" --dll "$REF" | sed 's/^/  /'
  done
  $REC extract "$O/raw.wex4" "$O/rec-raw" | tail -1 | sed 's/^/  raw records: /'
  for f in "$O"/rec-raw/*.slab; do cmp "$f" "$O/rec-all/$(basename "$f")"; done
  echo "  the slabs decoded from the WNR3 records equal the RAW records' slabs"
done

echo "== deepseek-v41 through WEX1 -> WEX3 (build_nr3.py) -> upgrade --names deepseek"
O=$W/deepseek-v41
M=$O/wex1-model
mkdir -p "$M"
cp "$D"/deepseek-v41/*.experts.safetensors "$M/"
# the geometry of the fetched part: layers 0 to 30 of which 10 and 30 hold experts 0 to 3, the V4.1 dimensions
$PY -I -c "import json,sys; c=json.load(open(sys.argv[1])); c=c.get('text_config',c); print(json.dumps({'num_hidden_layers':31,'n_routed_experts':4,'hidden_size':c['hidden_size'],'moe_intermediate_size':c['moe_intermediate_size']}))" "$D/deepseek-v41/config.json" > "$M/config.json"
cat "$M/config.json"
$PY -I "$ROOT/c/tools/repack_v4_wex.py" --model-dir "$M" --output "$O/v41.wex" |
    $PY -I -c "import json, sys; d = json.load(sys.stdin); d.pop('output'); print('  repack:', json.dumps(d))"
echo "  v41.wex $(wc -c < "$O/v41.wex") B sha256 $(sha "$O/v41.wex")"
P=10:0,10:1,10:2,10:3,30:0,30:1,30:2,30:3
for cz in "" --cz; do
  t=wnr3${cz:+-cz}
  $PY -I "$ROOT/c/tools/build_nr3.py" "$NR3" "$O/v41.wex" container "$O/$t.wex" --layers 10,30 --train "$P" $cz \
      --save-model "$O/$t.wnm2" | tail -2 | sed 's/^/  build_nr3: /'
  before=$(sha "$O/$t.wex")
  echo "  $t.wex $(wc -c < "$O/$t.wex") B sha256 $before"
  cp "$O/$t.wex" "$O/$t-up.wex4"
  $CONV upgrade "$O/$t-up.wex4" --names deepseek | sed 's/^/  upgrade: /'
  verify "$O/$t-up.wex4"
  $REC extract "$O/$t-up.wex4" "$O/rec-$t-up" --dll "$REF" | tail -1 | sed 's/^/  /'
  $CONV downgrade "$O/$t-up.wex4"
  after=$(sha "$O/$t-up.wex4")
  [ "$after" = "$before" ] || { echo "downgrade did not give $t.wex back" >&2; exit 1; }
  echo "  downgrade: sha256 $after, the WEX3 file byte for byte"
  $CONV pack "$O/pack-$t.wex4" "$D"/deepseek-v41/*.experts.safetensors --experts nr3 --names deepseek --nr3-dll "$NR3" \
      --model "$O/$t.wnm2" $cz --dll "$FAST" | sed 's/^/  pack: /'
  $CONV unpack "$O/pack-$t.wex4" "$O/pack-$t-unpacked" --dll "$FAST" | sed 's/^/  unpack: /'
  $REC extract "$O/pack-$t.wex4" "$O/rec-pack-$t" --dll "$REF" | tail -1 | sed 's/^/  /'
  $PY -I -c "
import json, sys
a, b = (json.load(open(p))['rows'] for p in sys.argv[1:3])
key = lambda rows: sorted((r['record'], r['stored'], r['sha256_stored']) for r in rows)
assert key(a) == key(b), 'the records of pack and of build_nr3 + upgrade differ'
print(f'  pack --experts nr3 {sys.argv[3]}: the same {len(a)} records as build_nr3.py + upgrade (size and SHA-256)')
" "$O/rec-$t-up/records.json" "$O/rec-pack-$t/records.json" "${cz:-without --cz}"
done
cp "$O/v41.wex" "$O/v41-up.wex4"
before=$(sha "$O/v41.wex")
$CONV upgrade "$O/v41-up.wex4" --names deepseek | sed 's/^/  upgrade WEX1: /'
verify "$O/v41-up.wex4"
$CONV downgrade "$O/v41-up.wex4"
[ "$(sha "$O/v41-up.wex4")" = "$before" ] || { echo "downgrade did not give v41.wex back" >&2; exit 1; }
echo "  downgrade: sha256 $before, the WEX1 file byte for byte"
echo "run.sh: all round trips passed"
