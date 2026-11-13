"""Merge team3 Engram samples + team7 per-segment samples into one row set.

Usage: python -I prep.py TEAM3_ENGRAM_DIR TEAM7_SEG_DIR OUT_NPZ
Rows: W [R,256] u8 (E4M3), S [R,8] u8 (E8M0); meta: layer, seg (0..23), chunk id, global row id.
Duplicate (layer,row) pairs (overlapping samples) are dropped.
"""
import json, os, sys
import numpy as np

t3, t7, out = sys.argv[1:4]
seg = json.load(open(os.path.join(t7, 'segments.json')))
bounds = {int(L): np.array([s['row0'] for s in v] + [v[-1]['row0'] + v[-1]['prime']]) for L, v in seg.items()}


def segof(L, r):
    return np.searchsorted(bounds[L], r, side='right') - 1

Ws, Ss, Ls, Rs, Cs, Src = [], [], [], [], [], []
cid = 0
m3 = json.load(open(os.path.join(t3, 'manifest.json')))['items']
for it in m3:
    if not it['name'].endswith('weight'):
        continue
    L = int(it['name'].split('.')[1]); r0, r1 = it['rows']
    st = [x for x in m3 if x['name'] == it['name'].replace('weight', 'scale') and x['rows'] == it['rows'] and x['tag'].startswith('run16k') == it['tag'].startswith('run16k')][0]
    W = np.fromfile(os.path.join(t3, it['tag'] + '.bin'), np.uint8).reshape(-1, 256)
    S = np.fromfile(os.path.join(t3, st['tag'] + '.bin'), np.uint8).reshape(-1, 8)
    rows = np.arange(r0, r1)
    if it['tag'].startswith('run16k'):  # split the 16k run into 16 chunks of 1024 for the chunk-level split
        for k in range(0, len(rows), 1024):
            Ws.append(W[k:k + 1024]); Ss.append(S[k:k + 1024]); Ls.append(np.full(1024, L)); Rs.append(rows[k:k + 1024]); Cs.append(np.full(1024, cid)); Src.append(np.full(1024, 1)); cid += 1
    else:
        Ws.append(W); Ss.append(S); Ls.append(np.full(len(rows), L)); Rs.append(rows); Cs.append(np.full(len(rows), cid)); Src.append(np.full(len(rows), 0)); cid += 1
m7 = json.load(open(os.path.join(t7, 'manifest.json')))['items']
for it in m7:
    W = np.fromfile(os.path.join(t7, it['tag'] + '.w.bin'), np.uint8).reshape(-1, 256)
    S = np.fromfile(os.path.join(t7, it['tag'] + '.s.bin'), np.uint8).reshape(-1, 8)
    rows = np.arange(it['row0'], it['row0'] + it['rows'])
    Ws.append(W); Ss.append(S); Ls.append(np.full(len(rows), it['layer'])); Rs.append(rows); Cs.append(np.full(len(rows), cid)); Src.append(np.full(len(rows), 2)); cid += 1
W = np.concatenate(Ws); S = np.concatenate(Ss); L = np.concatenate(Ls); R = np.concatenate(Rs); C = np.concatenate(Cs); SRC = np.concatenate(Src)
key = L.astype(np.int64) * (1 << 40) + R
_, first = np.unique(key, return_index=True)
first.sort()
W, S, L, R, C, SRC = W[first], S[first], L[first], R[first], C[first], SRC[first]
SEG = np.array([segof(int(l), int(r)) for l, r in zip(L, R)]) if len(L) < 0 else np.empty(len(L), np.int64)
for Lv in (1, 14):
    m = L == Lv
    SEG[m] = np.searchsorted(bounds[Lv], R[m], side='right') - 1
np.savez(out, W=W, S=S, L=L, R=R, C=C, SRC=SRC, SEG=SEG)
print('rows', len(W), 'chunks', len(np.unique(C)), 'dropped dups', len(key) - len(first),
      'per layer', {int(v): int((L == v).sum()) for v in (1, 14)}, 'segments covered', len(np.unique(L * 100 + SEG)))
