"""Held-out ideal code lengths of static context models for Engram rows (chunk-level even/odd split).

Usage: python -I models.py ROWS_NPZ OUT_JSON
Every model is one shared static table for BOTH tables (layers 1 and 14), trained on even chunks, scored on odd
chunks and vice versa (add-0.5 smoothing).  Bits are reported per row (256 weight bytes + 8 scale bytes).
"""
import json, sys
import numpy as np

d = np.load(sys.argv[1])
W, S, L, C, SEG = d['W'], d['S'], d['L'], d['C'], d['SEG']
R = len(W)
tr = (C % 2) == 0
te = ~tr

sgn = (W >> 7).astype(np.int64); e = ((W >> 3) & 15).astype(np.int64); m = (W & 7).astype(np.int64)
em = (W & 0x7F).astype(np.int64); by = W.astype(np.int64)
eg = e.reshape(R, 8, 32)
pe = np.empty_like(eg); pe[..., 0] = 16; pe[..., 1:] = eg[..., :-1]
ppe = np.empty_like(eg); ppe[..., :2] = 16; ppe[..., 2:] = eg[..., :-2]
cm = np.maximum.accumulate(eg, axis=2)
mx = np.empty_like(eg); mx[..., 0] = 16; mx[..., 1:] = cm[..., :-1]
sc = np.clip(S.astype(np.int64) - 117, 0, 4)  # 118..121 -> 1..4, others clamp
scb = np.broadcast_to(sc.reshape(R, 8, 1), (R, 8, 32))
pos = np.broadcast_to(np.arange(32).reshape(1, 1, 32), (R, 8, 32))
first = (pos == 0).astype(np.int64)
ng = np.broadcast_to((SEG // 8).reshape(R, 1, 1), (R, 8, 32)).astype(np.int64)
lay = np.broadcast_to((L == 14).astype(np.int64).reshape(R, 1, 1), (R, 8, 32))
# remaining-in-block flag: has the block max (>=14) appeared yet?  (block max exp is always 14 or 15)
seen14 = (mx >= 14) & (mx != 16)
seen14 = seen14.astype(np.int64)
nleft = 31 - pos  # elements left after this one


def heldout_row_bits(ctx, sym, nsym, nctx):
    """Per-row held-out bits: model trained on the other half (add-0.5)."""
    ctx = ctx.reshape(R, -1); sym = sym.reshape(R, -1)
    bits = np.zeros(R)
    for a, b in ((tr, te), (te, tr)):
        cnt = np.bincount((ctx[a] * nsym + sym[a]).ravel(), minlength=nctx * nsym).reshape(nctx, nsym) + 0.5
        p = cnt / cnt.sum(1, keepdims=True)
        bits[b] = -np.log2(p[ctx[b], sym[b]]).sum(1)
    return bits


def insample_bits(ctx, sym, nsym, nctx):
    cnt = np.bincount((ctx.ravel() * nsym + sym.ravel()), minlength=nctx * nsym).reshape(nctx, nsym).astype(float)
    t = cnt.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pp = np.where(cnt > 0, cnt / np.where(t == 0, 1, t), 1)
    return float(-(cnt * np.log2(pp)).sum()) / R

out = {'rows': int(R), 'train_rows': int(tr.sum()), 'test_rows': int(te.sum())}
# ---- scales
s = S.astype(np.int64)
left = np.concatenate([np.full((R, 1), 256), s[:, :-1]], 1)
sc_models = {'S_order0': (np.zeros_like(s), 1), 'S_left': (left, 257),
             'S_left_pos': (left * 8 + np.arange(8), 257 * 8)}
res = {}
for k, (cx, n) in sc_models.items():
    b = heldout_row_bits(cx, s, 256, n)
    res[k] = {'heldout_bits_row': float(b.mean()), 'insample_bits_row': insample_bits(cx, s, 256, n), 'contexts': n}
out['scale_models'] = res
best_sc_bits = heldout_row_bits(left, s, 256, 257)

# ---- weight models.  kind: 'exp' (exp coded, sign+mant raw 4 bits), 'em7' (sign raw), 'byte' (all coded)
ctxs = {
    'order0': (np.zeros_like(eg), 1),
    'pe': (pe, 17),
    'sc_pe': (scb * 17 + pe, 85),
    'pe_mx': (pe * 17 + mx, 289),
    'sc_pe_mx': ((scb * 17 + pe) * 17 + mx, 85 * 17),
    'pe_ppe': (pe * 17 + ppe, 289),
    'sc_pe_ppe': ((scb * 17 + pe) * 17 + ppe, 85 * 17),
    'sc_pe_seen14': ((scb * 17 + pe) * 2 + seen14, 170),
    'sc_pe_seen14_last': (((scb * 17 + pe) * 2 + seen14) * 2 + (nleft == 0), 340),
    'sc_pe_ng': ((scb * 17 + pe) * 3 + ng, 255),
    'sc_pe_layer': ((scb * 17 + pe) * 2 + lay, 170),
}
wres = {}
for kind, sym, nsym, raw in (('exp', e, 16, 4), ('em7', em, 128, 1), ('byte', by, 256, 0)):
    for k, (cx, n) in ctxs.items():
        if kind == 'byte' and n > 300:
            continue
        b = heldout_row_bits(cx, sym.reshape(R, 8, 32), nsym, n)
        wres[f'{kind}|{k}'] = {'heldout_coded_bits_row': float(b.mean()), 'raw_bits_row': raw * 256,
                               'heldout_total_bytes_row': float((b.mean() + raw * 256) / 8),
                               'insample_coded_bits_row': insample_bits(cx, sym.reshape(R, 8, 32), nsym, n), 'contexts': n}
        print(kind, k, round(wres[f'{kind}|{k}']['heldout_total_bytes_row'], 3), flush=True)
out['weight_models'] = wres
# mantissa | exponent per exponent value (is raw mantissa wasting anything?)
mres = {}
for ev in range(16):
    mk = e == ev
    c = np.bincount(m[mk], minlength=8).astype(float)
    p = c / max(c.sum(), 1)
    H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    mres[ev] = {'count': int(c.sum()), 'H_mant': H, 'waste_bits_if_raw_total_per_row': float((3 - H) * c.sum() / R)}
out['mantissa_given_exp'] = mres
# sign | (exp, mant)
c = np.bincount((em.ravel() * 2 + sgn.ravel()), minlength=256).reshape(128, 2).astype(float)
pp = c / np.maximum(c.sum(1, keepdims=True), 1)
with np.errstate(divide='ignore', invalid='ignore'):
    Hs = -(np.where(c > 0, c * np.log2(np.where(pp > 0, pp, 1)), 0)).sum() / c.sum()
out['H_sign_given_em7_insample'] = float(Hs)

# per segment / ngram size / layer: held-out bytes/row of exp|sc_pe + S_left
b_exp = heldout_row_bits(scb * 17 + pe, eg, 16, 85)
tot = (b_exp + best_sc_bits) / 8 + 128
out['per_ngram_bytes_row_exp_sc_pe'] = {f'L{lv}_ng{g + 2}': float(tot[(L == lv) & (SEG // 8 == g)].mean()) for lv in (1, 14) for g in range(3)}
out['per_segment_bytes_row'] = {f'L{lv}_s{sg}': float(tot[(L == lv) & (SEG == sg)].mean()) for lv in (1, 14) for sg in range(24) if ((L == lv) & (SEG == sg)).any()}
qs = [0, 0.01, 0.1, 1, 10, 50, 90, 99, 99.9, 99.99, 100]
out['ideal_row_bytes_exp_sc_pe_percentiles'] = dict(zip([str(q) for q in qs], np.percentile(tot, qs).tolist()))
# segment-level duplicate / similarity checks
out['dup_rows'] = int(R - len(np.unique(W, axis=0)))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('weight_models', 'per_segment_bytes_row')}, indent=1))
