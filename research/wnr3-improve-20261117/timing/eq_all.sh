#!/bin/sh
# Byte equality on all 121 records (1 and 4 threads) for every timed variant, final binaries.
# WNX: all 121 records re-encoded here (TRAIN records in-sample; only for equality), TEST re-encodes compared
# byte for byte with the fastfmt lane's out/ files, then the local copies are deleted (disk).
cd /tmp/wnr3-research/timing
: > logs/eq_summary.txt
./eq.sh orig_a2 ref eq_ref.txt >> logs/eq_summary.txt &
./eq.sh orig_a2 w3:1,4,1 eq_orig_a2_k1x4.txt >> logs/eq_summary.txt
./eq.sh orig_a5 w3:1,4,1 eq_orig_a5_k1x4.txt >> logs/eq_summary.txt
./eq.sh new_a2 w3:3,4,1 eq_new_a2_k3x4.txt >> logs/eq_summary.txt
./eq.sh new_a5 w3:4,3,1 eq_new_a5_k4x3.txt >> logs/eq_summary.txt
wait
for cfg in "mag 2048 2048 mag_2048" "magt 2048 2048 magt_2048" "joint 2048 2048 joint_2048"; do
  set -- $cfg; N=$4
  [ -f wnx/$N/X_39_120.wnx ] && [ $(ls wnx/$N/X_*.wnx | wc -l) = 121 ] || ./enc_all.sh $1 $2 $3 $4
  echo "$N: encode round trips ok: $(grep -h -c '| yes |' logs/enc_${N}_*.txt | paste -sd+ | bc) / 121; TEST files identical to fastfmt/out: $(for f in $(cd ../fastfmt/out/$N && ls X_*); do cmp -s wnx/$N/$f ../fastfmt/out/$N/$f && echo same; done | wc -l) / 41" >> logs/eq_summary.txt
  for v in 1:1 1:2; do echo "orig_a2 $(./eq.sh orig_a2 wnx:wnx/$N:$v eq_orig_a2_${N}_$v.txt)" >> logs/eq_summary.txt; done
  for v in 2:1 2:2; do echo "orig_a5 $(./eq.sh orig_a5 wnx:wnx/$N:$v eq_orig_a5_${N}_$v.txt)" >> logs/eq_summary.txt; done
  echo "ffold_a5 $(./eq.sh ffold_a5 wnx:wnx/$N:2:2 eq_ffold_a5_${N}_2-2.txt)" >> logs/eq_summary.txt
  echo "new_a5 $(./eq.sh new_a5 wnx:wnx/$N:2:2 eq_new_a5_${N}_2-2.txt)" >> logs/eq_summary.txt
  rm -rf wnx/$N wnx/$N.part.*
done
echo DONE >> logs/eq_summary.txt
