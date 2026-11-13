"""(1) srel anchor experiments: recompute the WNR2 base-context counts with srel = clamp(s - anchor, -3, 3)
for other per-matrix anchors (exactly, from the w_srel15 counts, whose +-7 clamp makes any |d|<=3 shift exact).
(2) K clustered models: EM (as team 1) vs assignment by a cheap per-matrix statistic, held-out.

Usage: python -I anchor_clusters.py SAMPLE_DIR COUNTS_DIR OUT_JSON
Held-out protocol = team 1 (mxcombo_an.py): split = manifest index % 2; fit on one split, cost on the other, both ways, summed.
bit/w include the 4-B flush per 2048-weight stream (0.015625 bit/w), like team 1's 3.7826.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, P, quantize, codelen, cost_bits, load_all

FL = 0.015625
sample_dir, counts_dir, out_json = sys.argv[1:4]
ex = load_all(sample_dir, counts_dir, keys=('w_base', 'w_srel15', 's_o0', 'meta', 'bm_side'))
n = len(ex)
units = [(i, m) for i in range(n) for m in range(3)]
nu = len(units)
split = np.array([ex[i]['split'] for i, m in units])
mtype = np.array([m for i, m in units])
layer = np.array([ex[i]['layer'] for i, m in units])
WB = np.stack([ex[i]['w_base'][m] for i, m in units])            # [u,567,16]
W15 = np.stack([ex[i]['w_srel15'][m] for i, m in units]).reshape(nu, 15, 81, 16)
H = np.stack([ex[i]['s_o0'][m] for i, m in units])               # [u,256]
MED = np.array([int(ex[i]['meta'][m, 2]) for i, m in units])


def remap(d):
    """counts [u,567,16] for anchor = med + d (d: int array [u])."""
    out = np.zeros((nu, 7, 81, 16), np.int64)
    for j in range(15):
        nb = np.clip(j - 7 - d, -3, 3) + 3
        out[np.arange(nu), nb] += W15[:, j]
    return out.reshape(nu, 567, 16)


assert (remap(np.zeros(nu, int)) == WB).all(), 'w_srel15 does not reproduce w_base'

# ---------- per-unit scale statistics ----------
v = np.arange(256)
tot = H.sum(1)
pair = H[:, :-1] + H[:, 1:]
A_pair = pair.argmax(1)                                   # lower value of the densest adjacent pair
mean = (H * v).sum(1) / tot
A_floor = np.floor(mean).astype(int)
flip = H[np.arange(nu), MED - 1] > H[np.arange(nu), MED + 1]
A_flipfix = MED - flip.astype(int)
p_up = H[np.arange(nu), A_pair + 1] / tot                 # share at lower+1 ("phase" of the matrix)
std = np.sqrt((H * (v - mean[:, None]) ** 2).sum(1) / tot)
skew = (H * (v - mean[:, None]) ** 3).sum(1) / tot / std ** 3
zero = np.array([(ex[i]['w_base'][m][:, 0].sum() + ex[i]['w_base'][m][:, 8].sum()) / NW for i, m in units])
zneg = np.array([ex[i]['w_base'][m][:, 8].sum() / NW for i, m in units])
bm7 = np.array([ex[i]['bm_side'][m][:, 7].sum() / ex[i]['bm_side'][m].sum() for i, m in units])
h0 = np.array([-(c / c.sum() * np.log2(np.where(c > 0, c / c.sum(), 1))).sum() for c in WB.sum(1)])
out = {'anchor_census': {
    'pair_eq_med': int((A_pair == MED).sum()), 'pair_eq_med_minus1': int((A_pair == MED - 1).sum()),
    'pair_other': int(((A_pair != MED) & (A_pair != MED - 1)).sum()),
    'floor_mean_eq_pair': int((A_floor == A_pair).sum()), 'flipfix_eq_pair': int((A_flipfix == A_pair).sum()),
    'flipped_units': int(flip.sum()), 'units': nu}}


def bpw_of(bits):
    return bits / (nu * NW) + FL


def ev_types(C, per_type=True):
    """in-sample and held-out bit/w for a shared model per matrix type (production) or pooled over types."""
    groups = [mtype == m for m in range(3)] if per_type else [np.ones(nu, bool)]
    ins = ho = 0.0
    for g in groups:
        c = C[g].sum(0)
        ins += cost_bits(c, quantize(c))
        a = C[g & (split == 0)].sum(0); b = C[g & (split == 1)].sum(0)
        ho += cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))
    return bpw_of(ins), bpw_of(ho)


res = {}
anchors = {'median (WNR2)': MED, 'densest pair lower': A_pair, 'floor(mean)': A_floor, 'flip-fixed median': A_flipfix,
           'median-1 always': MED - 1}
CNT = {}
for name, A in anchors.items():
    C = remap(A - MED)
    CNT[name] = C
    i3, h3 = ev_types(C, True)
    i1, h1 = ev_types(C, False)
    res[name] = {'per_type_insample': i3, 'per_type_heldout': h3, 'pooled_insample': i1, 'pooled_heldout': h1}

# encoder-chosen anchor offset d in {-1,0,+1} around the densest-pair anchor (2 bits per matrix), alternating fit
Cs = {d: remap(A_pair + d - MED) for d in (-1, 0, 1)}
choice = np.zeros(nu, int)
for it in range(6):
    C = np.stack([Cs[d][k] for k, d in zip(range(nu), choice)])
    Lm = [codelen(C[mtype == m].sum(0)) for m in range(3)]
    costs = np.stack([np.array([(Cs[d][k] * Lm[mtype[k]]).sum() for k in range(nu)]) for d in (-1, 0, 1)], 1)
    new = np.array([-1, 0, 1])[costs.argmin(1)]
    if (new == choice).all():
        break
    choice = new
res['encoder-chosen offset around densest pair (in-sample)'] = {'per_type_insample': bpw_of(costs.min(1).sum()),
                                                               'offsets_used': {int(d): int((choice == d).sum()) for d in (-1, 0, 1)}}
out['anchors'] = res


# ---------- clusters ----------
def em_fit(U, init, K, iters=15):
    assign = init.copy()
    for _ in range(iters):
        Lm = np.stack([codelen(U[assign == k].sum(0) if (assign == k).any() else np.zeros(U.shape[1:], np.int64)) for k in range(K)])
        new = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm).argmin(1)
        if (new == assign).all():
            break
        assign = new
    return Lm


def held_out(C, label_fn, K, mode):
    """label_fn(train_mask, test_mask) -> (train_labels, test_labels) in 0..K-1.
    mode 'assigned': test unit uses its label's model; 'best': encoder picks the cheapest of the K models;
    'em': EM on train starting from train labels, then best pick."""
    tb = 0.0
    for tr in (0, 1):
        trm = split == tr; tem = ~trm
        ltr, lte = label_fn(trm, tem)
        U = C[trm]
        if mode == 'em':
            Lm = em_fit(U, ltr, K)
        else:
            Lm = np.stack([codelen(U[ltr == k].sum(0) if (ltr == k).any() else np.zeros(U.shape[1:], np.int64)) for k in range(K)])
        T = C[tem].astype(np.float64)
        cost = np.einsum('ucs,kcs->uk', T, Lm)
        tb += cost[np.arange(len(lte)), lte].sum() if mode == 'assigned' else cost.min(1).sum()
    return bpw_of(tb)


def by_stat(x, K):
    """quantile bins of statistic x, thresholds from the training split only."""
    def f(trm, tem):
        q = np.quantile(x[trm], np.linspace(0, 1, K + 1)[1:-1]) if K > 1 else np.array([])
        return np.searchsorted(q, x[trm], side='right'), np.searchsorted(q, x[tem], side='right')
    return f


def fixed(lab):
    return lambda trm, tem: (lab[trm], lab[tem])


cl = {}
for cname in ['median (WNR2)', 'densest pair lower']:
    C = CNT[cname]
    r = {}
    r['K1 pooled'] = held_out(C, fixed(np.zeros(nu, int)), 1, 'assigned')
    r['K3 by matrix type (production)'] = held_out(C, fixed(mtype), 3, 'assigned')
    for K in (2, 4):
        r[f'K{K} EM (layer-band init, best pick)'] = held_out(C, fixed(layer * K // 40), K, 'em')
    r['K2 by flip flag, assigned'] = held_out(C, fixed(flip.astype(int)), 2, 'assigned')
    r['K6 by type x flip flag, assigned'] = held_out(C, fixed(mtype * 2 + flip), 6, 'assigned')
    r['K2 by skew sign, assigned'] = held_out(C, fixed((skew < 0).astype(int)), 2, 'assigned')
    for sname, x in [('p_up (share at anchor+1)', p_up), ('s_std', std), ('zero frac', zero), ('-0 frac', zneg),
                     ('blk_max7', bm7), ('order-0 entropy', h0), ('layer', layer.astype(float)), ('median scale', MED.astype(float))]:
        for K in (2, 4):
            r[f'K{K} by {sname} quantiles, assigned'] = held_out(C, by_stat(x, K), K, 'assigned')
            r[f'K{K} by {sname} quantiles, best pick'] = held_out(C, by_stat(x, K), K, 'best')
    for K in (2, 4):
        r[f'K{K*3} by type x p_up quantiles, assigned'] = held_out(
            C, lambda trm, tem, K=K: tuple(mtype[mm] * K + lab for mm, lab in zip((trm, tem), by_stat(p_up, K)(trm, tem))), 3 * K, 'assigned')
    cl[cname] = r
out['clusters_heldout_bit_per_w'] = cl

# EM composition (in-sample, all units) on the original counts
comp = {}
for K in (2, 4):
    U = WB
    assign = layer * K // 40
    for _ in range(15):
        Lm = np.stack([codelen(U[assign == k].sum(0)) if (assign == k).any() else np.full((567, 16), 4.0) for k in range(K)])
        new = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm).argmin(1)
        if (new == assign).all():
            break
        assign = new
    comp[K] = [{'cluster': k, 'units': int((assign == k).sum()), 'flipped': int((flip & (assign == k)).sum()),
                'by_type': [int(((mtype == m) & (assign == k)).sum()) for m in range(3)],
                'p_up_mean': round(float(p_up[assign == k].mean()), 4) if (assign == k).any() else None,
                'p_up_range': [round(float(p_up[assign == k].min()), 3), round(float(p_up[assign == k].max()), 3)] if (assign == k).any() else None,
                'layers_37_39': int(((layer >= 37) & (assign == k)).sum())} for k in range(K)]
out['em_composition_original_counts'] = comp
out['p_up_vs_flip'] = {'p_up_flipped': [round(float(x), 3) for x in sorted(p_up[flip])],
                       'p_up_unflipped_max': float(p_up[~flip].max()), 'p_up_unflipped_min': float(p_up[~flip].min())}
json.dump(out, open(out_json, 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
