#!/bin/sh
# Full rerun of the model lane from this directory (CPU, 2 processes; ~2.5 h on the shared 4-core Xeon).
# Needs <TMP>/wnr3-data (read only) and network for the two fetch steps (headers 7,680 x 32 B; 37 records, 586 MB).
set -e
cd '<TMP>'/wnr3-research/model
gcc -O2 -shared -fPIC -Wall -I/workspace/Wex-WRN2/c -o mlib.so mlib.c
python3 prep.py                         # 121 records: decode, coded slabs, counts, re-encode == stored
python3 fetch_headers.py                # headers.bin: anchors and SP of all 7,680 records
python3 fetch_shifted.py                # extra/: the 37 other records with anchors != (120,120,120)
python3 prep.py '<TMP>'/wnr3-research/model/extra
python3 recount.py                      # anchor-relative scale counts (+ cross-check of the counts)
python3 harness.py                      # prototype == stored records byte for byte; q_rule == library
python3 screen.py > screen.log          # ideal code lengths (bounds)
python3 scale_ctx.py > scale_ctx.log; python3 scale_alpha.py > scale_alpha.log
python3 classsets.py > classsets.log; python3 classsets.py sum 16,32 classsets_fine.json > classsets_fine.log
python3 ksets.py 14 > ksets.log         # K = 2, 4, 8 selected sets
for a in "14 8 1" "14 16 1" "14 32 1" "14 8 2" "14 8 4" "14 16 1 z8"; do python3 ksets2.py $a; done
python3 real.py prod lib_rule14_T80 lib_rule14_T40a lib_rule14_T40b lib_self P14 P14_z8 P15 P16 P16_z8 P16_srel \
  P16_z8_srel P14_srel P14_sesc P14_K2 P14_K4 P14_K8 P14_C8_K1 P14_C16_K1 P14_C32_K1 P14_C8_K2 P14_C8_K4 P14_C16_K1_z8_sesc
SET=extra python3 real.py prod P14 P14_srel P14_sesc P16_srel P14_C16_K1 P14_C16_K1_z8_sesc
SPLIT=B python3 ksets2.py 14 16 1 z8
SPLIT=B python3 real.py prod P14 P14_sesc P14_C16_K1_z8_sesc
python3 esc_test.py > esc_test.log
python3 timing.py > timing.log
python3 report.py
