"""Entropy of FP8 E4M3 dense weights with 32x32 UE8M0 block scales (shared expert, attention, engram wkv).

Usage: python -I fp8_an.py FP8_DATA_DIR OUT_JSON
Static context models per tensor TYPE (e.g. shared_experts.w1), costs:
  in-sample (all layers pooled) and held-out leave-one-layer-out (train on the other layers of that type).
"""
import json, os, sys, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entlib import *

d, outp = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
by = {it['name']: it for it in man['items']}

MODELS = {  # name: (n_contexts, function(ctx dict) -> int64 ctx)
    'order0': 1,
    'srel': 7,
    'prev_exp': 17,
    'srel_prevexp': 7 * 17,
    'srel_prevexp_runmax': 7 * 17 * 17,
    'srel_prevexp_upexp': 7 * 17 * 17,
}


def counts_for(W, S):
    """W uint8 [R, C], S uint8 [ceil(R/32), ceil(C/32)] -> {model: counts [nctx, 256]} plus extras."""
    R, C = W.shape
    med = int(np.sort(S.ravel().astype(np.int16))[(S.size - 1) // 2])
    acc = {k: np.zeros((n, 256), np.int64) for k, n in MODELS.items()}
    ncol = C // 32
    assert C % 32 == 0
    bmax_hist = np.zeros(16, np.int64)
    rowseg_max_hist = np.zeros(16, np.int64)
    for r0 in range(0, R, 2048):
        r1 = min(R, r0 + 2048)
        w = W[r0:r1]
        n = r1 - r0
        e = ((w >> 3) & 15).astype(np.int32)
        sym = w.astype(np.int32)
        sc = S[np.arange(r0, r1) // 32][:, np.arange(C) // 32].astype(np.int32)
        srel = np.clip(sc - med, -3, 3) + 3
        eg = e.reshape(n, ncol, 32)
        pe = np.empty_like(eg); pe[..., 0] = 16; pe[..., 1:] = eg[..., :-1]
        cm = np.maximum.accumulate(eg, axis=2)
        mx = np.empty_like(eg); mx[..., 0] = 16; mx[..., 1:] = cm[..., :-1]
        rowseg_max_hist += bc(cm[..., -1], 16)
        pe = pe.reshape(n, C); mx = mx.reshape(n, C)
        up = np.empty_like(e)
        if r0 == 0:
            up[0] = 16
        else:
            up[0] = ((W[r0 - 1] >> 3) & 15)
        up[1:] = e[:-1]
        # row 0 of every 32-row block band has no 'up' inside the block: keep the real upper row anyway (it is decoded earlier)
        ctxs = {
            'order0': np.zeros_like(sym),
            'srel': srel,
            'prev_exp': pe,
            'srel_prevexp': srel * 17 + pe,
            'srel_prevexp_runmax': (srel * 17 + pe) * 17 + mx,
            'srel_prevexp_upexp': (srel * 17 + pe) * 17 + up,
        }
        for k, c in ctxs.items():
            acc[k] += ctx_counts(c, sym, MODELS[k], 256)
    # true 32x32 block max exponent
    Rb, Cb = -(-R // 32), C // 32
    e_all = ((W >> 3) & 15)
    pad = np.zeros((Rb * 32, C), np.uint8); pad[:R] = e_all
    bmax = pad.reshape(Rb, 32, Cb, 32).max(axis=(1, 3))
    bmax_hist += bc(bmax, 16)
    return acc, {'median_scale': med, 'block32x32_max_exp_hist': bmax_hist.tolist(),
                 'rowseg32_max_exp_hist': rowseg_max_hist.tolist()}


groups = collections.defaultdict(list)  # type -> [(layer, name)]
for name, it in by.items():
    if not name.endswith('.weight'):
        continue
    m = re.match(r'layers\.(\d+)\.(.*)\.weight$', name)
    groups[m.group(2)].append((int(m.group(1)), name))

out = {}
for typ, lst in sorted(groups.items()):
    per_layer = {}
    extras = {}
    nbytes = {}
    sc_counts = {}
    dec = {}
    gen = {}
    for L, name in sorted(lst):
        wi, si = by[name], by[name.replace('.weight', '.scale')]
        W = np.fromfile(os.path.join(d, wi['tag'] + '.bin'), np.uint8).reshape(wi['shape'])
        S = np.fromfile(os.path.join(d, si['tag'] + '.bin'), np.uint8).reshape(si['shape'])
        assert wi['dtype'] == 'F8_E4M3' and si['dtype'] == 'F8_E8M0'
        per_layer[L], extras[L] = counts_for(W, S)
        nbytes[L] = W.size
        sc_counts[L] = bc(S, 256)
        dec[L] = fp8_decompose(W)
        dec[L]['neg_zero_frac'] = float((W == 0x80).mean()); dec[L]['pos_zero_frac'] = float((W == 0x00).mean())
        sl = W.ravel()[: 8 << 20].tobytes()
        gen[L] = generic(sl)
        gen[L]['slice_bytes'] = len(sl)
        extras[L]['scale_shape'] = list(S.shape)
        extras[L]['scale_hist_top'] = collections.Counter(S.ravel().tolist()).most_common(6)
    layers = sorted(per_layer)
    tot_bytes = sum(nbytes.values())
    r = {'layers': layers, 'bytes_per_layer': nbytes[layers[0]], 'shape': by[lst[0][1]]['shape'],
         'decompose_order0': dec, 'generic_8MB_slice': gen, 'extras': extras, 'models': {}}
    for k in MODELS:
        allc = sum(per_layer[L][k] for L in layers)
        ins = ent_bits(allc) / tot_bytes
        if len(layers) > 1:
            ho = sum(xent_bits(per_layer[L][k], allc - per_layer[L][k]) for L in layers) / tot_bytes
        else:
            ho = None
        r['models'][k] = {'insample_bits_per_byte': ins, 'heldout_leave_one_layer_bits_per_byte': ho}
    sall = sum(sc_counts.values())
    r['scales_order0_bits_per_scale_insample'] = ent_bits(sall) / sall.sum()
    out[typ] = r
    print(typ, layers, {k: (round(v['insample_bits_per_byte'], 4), None if v['heldout_leave_one_layer_bits_per_byte'] is None else round(v['heldout_leave_one_layer_bits_per_byte'], 4)) for k, v in r['models'].items()}, flush=True)

json.dump(out, open(outp, 'w'), indent=1, default=int)
