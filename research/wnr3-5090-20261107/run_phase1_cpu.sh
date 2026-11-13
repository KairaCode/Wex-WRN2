# Phase 1 (CPU, under heavy.lock): task 2 g0 encode/decode + task 3 sample, counts, ceilenc.
set -e
LAB='<KAIRA>'/lab/wnr3-5090-20261107
SC='<TMP>'/data
. "$LAB/lock.sh"
take_lock "wnr3-5090 phase1 CPU (g0 encode + ceil sample/encode)" || exit 3
mkdir -p "$SC"; cd "$LAB"
echo "== t2 g0 $(date +%T)"; python -I g0/t2_g0.py "$LAB/g0" "$SC/g0bins" > g0/t2_g0.log 2>&1
echo "== t3 sample $(date +%T)"; python -I t3_make_sample.py "$SC/v41-sample" "$SC/counts" > t3_make_sample.log 2>&1
echo "== mxceil $(date +%T)"; python -I team1/mxceil.py "$SC/v41-sample" "$SC/ceil-counts" > team1/mxceil.log 2>&1
echo "== ceilenc $(date +%T)"; python -I ceil-gpu/src/ceilenc.py "$SC/v41-sample" "$SC/counts" "$SC/ceil-counts" "$SC/enc" A259,D,CF,CFD,DB2,CFDB2 > ceil-gpu/ceilenc.log 2>&1
cp "$SC"/enc/encode_report*.json ceil-gpu/
echo "== done $(date +%T)"
