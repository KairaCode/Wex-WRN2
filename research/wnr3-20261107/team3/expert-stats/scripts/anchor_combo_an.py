"""Anchor fix on top of team 1's richer contexts (a) c_bm2h, c_bm2p2b and (c) c_z, with/without K clustered models.

Usage: python -I anchor_combo_an.py SAMPLE_DIR COUNTS_DIR COUNTS15_DIR TEAM1_COMBO_DIR OUT_JSON
Held-out protocol and accounting identical to team 1 mxcombo_an.py (split = index % 2, both ways; side bits of the
block-max flag added with their held-out cost; flush 0.015625 bit/w; clusters pooled over matrix types, EM with
layer-band init, encoder picks the best model per matrix).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, codelen, cost_bits, load_all

FL = 0.015625
sample_dir, counts_dir, c15_dir, combo_dir, out_json = sys.argv[1:6]
ex = load_all(sample_dir, counts_dir, keys=('s_o0', 'meta'))
n = len(ex)
units = [(i, m) for i in range(n) for m in range(3)]
nu = len(units)
split = np.array([ex[i]['split'] for i, m in units])
mtype = np.array([m for i, m in units])
layer = np.array([ex[i]['layer'] for i, m in units])
H = np.stack([ex[i]['s_o0'][m] for i, m in units])
MED = np.array([int(ex[i]['meta'][m, 2]) for i, m in units])
A_pair = (H[:, :-1] + H[:, 1:]).argmax(1)
z15 = [np.load(os.path.join(c15_dir, e['file'].replace('.bin', '.npz'))) for e in ex]
assert all((z['med'] == ex[i]['meta'][:, 2]).all() for i, z in enumerate(z15))
B = {k: np.stack([z15[i][k][m] for i, m in units]).astype(np.int64) for k in ('b15h', 'b15p', 'b15z', 'side15')}


def remap(key, d):
    X = B[key]
    X = X.reshape(nu, 15, -1, X.shape[-1]) if key != 'side15' else X.reshape(nu, 15, 1, 2)
    out = np.zeros((nu, 7) + X.shape[2:], np.int64)
    for j in range(15):
        nb = np.clip(j - 7 - d, -3, 3) + 3
        out[np.arange(nu), nb] += X[:, j]     # (unit, bucket) pairs are unique within one call
    return out.reshape(nu, -1, X.shape[-1])


# validation against team 1's counts (d = 0 must reproduce c_bm2h / c_bm2p2b / c_z / bm2_side exactly)
co = [np.load(os.path.join(combo_dir, e['file'].replace('.bin', '.npz'))) for e in ex]
d0 = np.zeros(nu, int)
val = {}
for k15, kc in [('b15h', 'c_bm2h'), ('b15p', 'c_bm2p2b'), ('b15z', 'c_z'), ('side15', 'bm2_side')]:
    ref = np.stack([co[i][kc][m] for i, m in units]).astype(np.int64).reshape(nu, -1, 2 if kc == 'bm2_side' else 16)
    val[kc] = bool((remap(k15, d0) == ref).all())
print('validation vs team-1 counts:', val, flush=True)


def bpw(bits):
    return bits / (nu * NW) + FL


def ho_per_type(C):
    ins = ho = 0.0
    for m in range(3):
        g = mtype == m
        c = C[g].sum(0); a = C[g & (split == 0)].sum(0); b = C[g & (split == 1)].sum(0)
        ins += cost_bits(c, quantize(c)); ho += cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))
    return ins, ho


def clusters(C, K, side_ho):
    tb = 0.0
    for tr in (0, 1):
        U = C[split == tr]; lay = layer[split == tr]
        assign = lay * K // 40
        for _ in range(15):
            Lm = np.stack([codelen(U[assign == k].sum(0) if (assign == k).any() else np.zeros(U.shape[1:], np.int64)) for k in range(K)])
            new = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm).argmin(1)
            if (new == assign).all():
                break
            assign = new
        T = C[split != tr].astype(np.float64)
        tb += np.einsum('ucs,kcs->uk', T, Lm).min(1).sum()
    return bpw(tb + side_ho)


out = {'validation': val}
for aname, d in [('median (WNR2)', d0), ('densest pair lower', A_pair - MED)]:
    S = remap('side15', d)
    si, sh = ho_per_type(S)
    r = {'side_bits_per_w': si / (nu * NW)}
    for cname, key, side in [('c_bm2h (a)', 'b15h', True), ('c_bm2p2b', 'b15p', True), ('c_z (c, value-exact)', 'b15z', True)]:
        C = remap(key, d)
        i, h = ho_per_type(C)
        r[cname] = {'insample': bpw(i + si), 'heldout': bpw(h + sh)}
        if key in ('b15h', 'b15z'):
            r[cname]['clusters_pooled_heldout'] = {K: clusters(C, K, sh) for K in (1, 2, 4)}
    out[aname] = r
    print(aname, json.dumps(r, indent=1), flush=True)
json.dump(out, open(out_json, 'w'), indent=1)
