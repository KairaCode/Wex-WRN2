#!/bin/sh
# eq.sh BIN VARIANT LOG: equality on all 121 records (1 and 4 threads) against Z(source)
cd /tmp/wnr3-research/timing
./bin/$1 equal "$2" $(cat ids121.txt) > logs/$3 2>&1; tail -1 logs/$3
