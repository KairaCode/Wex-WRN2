#!/bin/sh
# Encode all 121 records for one WNX config (model trained on TRAIN; TRAIN records are in-sample, used only for equality),
# 2 processes. Usage: enc_all.sh KIND BW BS NAME
cd '<TMP>'/wnr3-research/timing
K=$1; BW=$2; BS=$3; N=$4
mkdir -p wnx/$N
split -n l/2 all121.txt '<TMP>'/wnr3-research/timing/wnx/$N.part.
for p in wnx/$N.part.*; do ./bin/ff enc $K ../fastfmt/out_train.counts $BW $BS wnx/$N $(cat $p) > logs/enc_${N}_$(basename $p).txt & done
wait
