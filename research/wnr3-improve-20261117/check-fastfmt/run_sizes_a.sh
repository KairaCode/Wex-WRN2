#!/bin/sh
cd '<TMP>'/wnr3-research/check-fastfmt
./ff enc magt  out_train.counts 2048 2048 out/magt_2048  $(cat test.txt) > logs/enc_magt_2048.txt
./ff enc magt  out_train.counts 4096 4096 out/magt_4096  $(cat test.txt) > logs/enc_magt_4096.txt
./ff enc magt  out_train.counts 2048 512  out/magt_2048_s512 $(cat test.txt) > logs/enc_magt_2048_s512.txt
./ff enc mag   out_train.counts 1024 1024 out/mag_1024   $(cat test.txt) > logs/enc_mag_1024.txt
