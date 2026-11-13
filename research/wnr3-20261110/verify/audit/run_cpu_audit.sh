#!/bin/sh
# CPU corruption audit on real V4.1 records (audit_real.exe): WNR2 production, WNR3 non-CZ, WNR3-CZ.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
R=${R:-2000}
./audit_real.exe ../wnr2_audit.bin 0 $R 20261110 0:0:113 1:39:120
./audit_real.exe ../nr3_cz.bin 97 $R 20261110 0:0:113 40:39:120
./audit_real.exe ../nr3_noncz.bin 33 $R 20261110 0:0:113
