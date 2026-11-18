#!/usr/bin/env python3
"""Recompute logs/vs_trainwnr3.txt: WNX padded record size against WNR3-CZ re-encoded with the
same TRAIN-trained WNM2 model (logs/wnr3_train.txt from wnr3_train_baseline.py).
Usage: python3 vs_trainwnr3.py LOGDIR   (LOGDIR holds wnr3_train.txt and the lane's enc_*.txt)
Written by the report assembler; the checker computed the same table inline."""
import re, sys, statistics as st
RAW = 18_800_640
d = sys.argv[1] if len(sys.argv) > 1 else 'logs'
base = {}
for line in open(f'{d}/wnr3_train.txt'):
    m = re.match(r'L(\d+) E(\d+) stored (\d+) wnr3-trainmodel (\d+)', line)
    if m: base[(int(m[1]), int(m[2]))] = int(m[4])
for cfg in ['mag_2048', 'magt_2048', 'magt_2048_s512', 'magt_4096', 'mag_4096', 'joint_2048', 'mag_1024']:
    rows = []
    for line in open(f'{d}/enc_{cfg}.txt'):
        m = re.match(r'\| L(\d+) E(\d+) \| (\d+) \| (\d+) \| (\d+) \|', line)
        if m:
            k = (int(m[1]), int(m[2])); rows.append((k, (int(m[5]) - base[k]) / RAW * 100))
    v = [x for _, x in rows]; vn = [x for k, x in rows if k != (39, 120)]
    print(f'{cfg}: vs WNR3 re-encoded with TRAIN model: {st.mean(v):+.4f}% of raw (n={len(v)}), '
          f'without L39E120 {st.mean(vn):+.4f}%, min {min(v):+.4f} max {max(v):+.4f}')
