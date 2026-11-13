"""Entropy of real Engram table rows (FP8 E4M3 [rows, 256] + E8M0 scale [rows, 8]).

Usage: python -I engram_an.py DATA_DIR OUT_JSON
Reads DATA_DIR/manifest.json written by fetch_tensors.py (96 random 128-row chunks per table + one 16384-row run).
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entlib import *

d = sys.argv[1]
man = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
items = man['items']


def load(tag):
    return np.fromfile(os.path.join(d, tag + '.bin'), dtype=np.uint8)

tables = {}
for L in (1, 14):
    ws, ss, starts = [], [], []
    for it in items:
        if it['tag'].startswith('run16k'):
            continue
        if it['name'] == f'layers.{L}.engram.embed.weight':
            r0 = it['rows'][0]
            sc = [x for x in items if x['name'] == f'layers.{L}.engram.embed.scale' and x['rows'][0] == r0 and not x['tag'].startswith('run16k')][0]
            ws.append(load(it['tag']).reshape(-1, 256)); ss.append(load(sc['tag']).reshape(-1, 8)); starts.append(r0)
    tables[L] = (np.concatenate(ws), np.concatenate(ss), starts)
run_w = [x for x in items if x['tag'].startswith('run16k') and x['name'].endswith('weight')][0]
run_s = [x for x in items if x['tag'].startswith('run16k') and x['name'].endswith('scale')][0]
RW = load(run_w['tag']).reshape(-1, 256); RS = load(run_s['tag']).reshape(-1, 8)

out = {'source': {'repo': man['engram_repo'], 'revision': man['engram_revision'],
                  'chunks_per_table': 96, 'rows_per_chunk': 128, 'run_rows': int(RW.shape[0]), 'run_start': run_w['rows'][0]}}


def contexts(W, S):
    """Per-element context arrays for W [R,256] uint8 with S [R,8] (one E8M0 per 32 along the row)."""
    R = W.shape[0]
    s, e, m = fp8_fields(W)
    g = W.reshape(R, 8, 32)
    eg = e.reshape(R, 8, 32).astype(np.int64)
    pe = np.empty_like(eg); pe[..., 0] = 16; pe[..., 1:] = eg[..., :-1]
    cm = np.maximum.accumulate(eg, axis=2)
    mx = np.empty_like(eg); mx[..., 0] = 16; mx[..., 1:] = cm[..., :-1]
    bmax = cm[..., -1]  # final group max exponent [R, 8]
    col = np.broadcast_to(np.arange(256).reshape(1, 8, 32), (R, 8, 32))
    sc = np.broadcast_to(S.reshape(R, 8, 1).astype(np.int64), (R, 8, 32))
    return {'sym': g.astype(np.int64), 'pe': pe, 'mx': mx, 'col': col, 'sc': sc, 'bmax': bmax}


def model_costs(Wa, Sa, Wb, Sb):
    """Return {model: (in-sample bits on a+b, held-out bits a->b + b->a, n)} for several static context models."""
    A, B = contexts(Wa, Sa), contexts(Wb, Sb)
    smin = int(min(Sa.min(), Sb.min()))
    nsc = int(max(Sa.max(), Sb.max())) - smin + 1
    defs = {
        'order0': (lambda C: np.zeros_like(C['sym']), 1),
        'scale': (lambda C: C['sc'] - smin, nsc),
        'column': (lambda C: C['col'], 256),
        'prev_exp': (lambda C: C['pe'], 17),
        'scale_prevexp': (lambda C: (C['sc'] - smin) * 17 + C['pe'], nsc * 17),
        'scale_prevexp_runmax': (lambda C: ((C['sc'] - smin) * 17 + C['pe']) * 17 + C['mx'], nsc * 289),
        'column_prevexp': (lambda C: C['col'] * 17 + C['pe'], 256 * 17),
    }
    res = {}
    for k, (f, n) in defs.items():
        ca = ctx_counts(f(A), A['sym'], n, 256)
        cb = ctx_counts(f(B), B['sym'], n, 256)
        res[k] = {'insample_bits_per_byte': ent_bits(ca + cb) / (Wa.size + Wb.size),
                  'heldout_bits_per_byte': (xent_bits(cb, ca) + xent_bits(ca, cb)) / (Wa.size + Wb.size),
                  'contexts': n}
    return res


for L, (W, S, starts) in tables.items():
    r = {}
    n = W.size
    r['rows'] = int(W.shape[0])
    r['decompose_order0'] = fp8_decompose(W)
    r['zero_codes_frac'] = float(np.isin(W, E4M3_ZERO_CODES).mean())
    r['neg_zero_frac'] = float((W == 0x80).mean())
    r['nan_codes'] = int(np.isin(W, (0x7F, 0xFF)).sum())
    r['all_zero_rows'] = int((~np.isin(W, E4M3_ZERO_CODES)).sum(1).__eq__(0).sum())
    u = np.unique(W, axis=0)
    r['duplicate_rows'] = int(W.shape[0] - u.shape[0])
    e = (W >> 3) & 15
    r['exp_hist'] = bc(e, 16).tolist()
    bmax = e.reshape(-1, 8, 32).max(2)
    r['group_max_exp_hist'] = bc(bmax, 16).tolist()
    bmaxbyte = (W & 0x7F).reshape(-1, 8, 32).max(2)
    r['group_max_abs_code_top'] = collections.Counter(bmaxbyte.ravel().tolist()).most_common(6)
    # scales
    r['scale_hist_top'] = collections.Counter(S.ravel().tolist()).most_common(12)
    r['scale_H_order0_bits'] = ent_bits(bc(S, 256)) / S.size
    left = np.concatenate([np.full((S.shape[0], 1), 256), S[:, :-1].astype(np.int64)], 1)
    r['scale_H_given_left_bits'] = ent_bits(ctx_counts(left, S, 257, 256)) / S.size
    r['scale_H_given_group_index_bits'] = ent_bits(ctx_counts(np.broadcast_to(np.arange(8), S.shape), S, 8, 256)) / S.size
    # split chunks even/odd for held-out
    ch = W.reshape(-1, 128, 256); chs = S.reshape(-1, 128, 8)
    Wa, Wb = ch[0::2].reshape(-1, 256), ch[1::2].reshape(-1, 256)
    Sa, Sb = chs[0::2].reshape(-1, 8), chs[1::2].reshape(-1, 8)
    r['models'] = model_costs(Wa, Sa, Wb, Sb)
    r['generic_on_random_chunks'] = generic(W.tobytes())
    r['generic_on_scales'] = generic(S.tobytes())
    out[f'table_L{L}'] = r

# cross table: model of L1 applied to L14 (and back), the "one shared model" setting
W1, S1, _ = tables[1]; W14, S14, _ = tables[14]
out['cross_table'] = model_costs(W1, S1, W14, S14)
out['run16k'] = {'decompose_order0': fp8_decompose(RW), 'generic_weights': generic(RW.tobytes()),
                 'generic_scales': generic(RS.tobytes()), 'all_zero_rows': int((~np.isin(RW, E4M3_ZERO_CODES)).sum(1).__eq__(0).sum()),
                 'duplicate_rows': int(RW.shape[0] - np.unique(RW, axis=0).shape[0])}
# adjacent-row correlation inside the run (same bucket range): MI of byte with the byte above
up = RW[:-1].astype(np.int64); cur = RW[1:].astype(np.int64)
out['run16k']['H_byte_order0'] = ent_bits(bc(cur, 256)) / cur.size
out['run16k']['H_byte_given_byte_above_insample'] = ent_bits(ctx_counts(up, cur, 256, 256)) / cur.size

# per-row code length under the cross-table model (scale_prevexp_runmax trained on L1, applied to L14 rows)
A = contexts(W1, S1); B = contexts(W14, S14)
smin = int(min(S1.min(), S14.min())); nsc = int(max(S1.max(), S14.max())) - smin + 1
f = lambda C: ((C['sc'] - smin) * 17 + C['pe']) * 17 + C['mx']
ca = ctx_counts(f(A), A['sym'], nsc * 289, 256) + 0.5
p = ca / ca.sum(1, keepdims=True)
bits = -np.log2(p[f(B), B['sym']])  # [R, 8, 32]
row_bits = bits.reshape(W14.shape[0], -1).sum(1)
sc_p = (bc(S1, 256) + 0.5); sc_p = sc_p / sc_p.sum()
row_bits_sc = -np.log2(sc_p[S14]).sum(1)
tot = (row_bits + row_bits_sc) / 8
qs = np.percentile(tot, [0, 1, 10, 50, 90, 99, 100])
out['per_row_bytes_L14_under_L1_model'] = {'raw_bytes': 264, 'percentiles_0_1_10_50_90_99_100': qs.tolist(),
                                           'mean': float(tot.mean()),
                                           'frac_rows_fit_in_256B': float((tot <= 256).mean()),
                                           'frac_rows_fit_in_240B': float((tot <= 240).mean()),
                                           'frac_rows_fit_in_224B': float((tot <= 224).mean()),
                                           'frac_rows_ideal_le_220B': float((tot <= 220).mean()),
                                           'frac_rows_ideal_le_216B': float((tot <= 216).mean()),
                                           'frac_rows_ideal_le_212B': float((tot <= 212).mean())}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
