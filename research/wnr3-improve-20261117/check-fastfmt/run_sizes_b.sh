#!/bin/sh
cd /tmp/wnr3-research/check-fastfmt
./ff enc mag   out_train.counts 2048 2048 out/mag_2048   $(cat test.txt) > logs/enc_mag_2048.txt
./ff enc joint out_train.counts 2048 2048 out/joint_2048 $(cat test.txt) > logs/enc_joint_2048.txt
./ff enc mag   out_train.counts 4096 4096 out/mag_4096   $(cat test.txt) > logs/enc_mag_4096.txt
