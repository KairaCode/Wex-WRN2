#!/usr/bin/env python3
"""Copy scripts and small text results of every lane and check into this folder.
Run from anywhere: python3 <TMP>/wnr3-research/report/collect.py"""
import os, shutil, filecmp
R = '<TMP>/wnr3-research'; DST = os.path.join(R, 'report')
SRC_EXT = {'.py', '.c', '.h', '.sh', '.inc', '.patch', '.md'}
RES_EXT = {'.json', '.jsonl', '.txt', '.log', '.md', '.out'}
SKIP_DIRS = {'__pycache__', 'data', 'coded', 'tables', 'adj', 'bin', 'extra', 'counts', 'scales', 'repo_fcd', 'wnx'}
SKIP_FILES = {'wex_fast_decode.orig.h', 'wex_fast_decode.k4v1.h', 'wex_fast_decode.k4v2.h'}
MAXF = 400_000
PAIRS = [('floor', None), ('check-floor', 'floor'), ('overhead', None), ('check-overhead', 'overhead'),
         ('model', None), ('check-model', 'model'), ('fastfmt', None), ('check-fastfmt', 'fastfmt'),
         ('fastdec', None), ('check-fastdec', 'fastdec'), ('timing', None)]
for d, lane in PAIRS:
    src = os.path.join(R, d)
    for root, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        if d == 'fastdec' and rel.startswith('out'):
            continue  # binaries and generated negative-control sources
        if d == 'check-fastdec' and rel.startswith('out'):
            continue
        for f in files:
            p = os.path.join(root, f); ext = os.path.splitext(f)[1]
            if f in SKIP_FILES or os.path.getsize(p) > MAXF:
                continue
            is_src = ext in SRC_EXT
            is_res = ext in RES_EXT and (rel != '.' or ext in {'.txt', '.json', '.log', '.md'})
            if not (is_src or is_res):
                continue
            if lane:  # check dirs: skip unchanged copies of the lane's files
                q = os.path.join(R, lane, rel, f)
                if os.path.exists(q) and filecmp.cmp(p, q, shallow=False):
                    continue
            out = os.path.join(DST, d, rel, f)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(p, out)
