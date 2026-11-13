"""Transfer held-out of the ceil-form gain on DeepSeek checkpoints (team4/ceil-origin).

Usage: python -I transfer_ho.py V41CEIL_DIR OUT_JSON SET=DIR [SET=DIR ...]
Model: WNR2 base context (srel7 x pm9 x mx9) x 16, one model PER MATRIX TYPE (w1, w2, w3) as in WNR2, plus order-0
scales per matrix type, trained on ALL 160 V4.1-REAP sample experts (counts from mxceil.py npz in V41CEIL_DIR:
w_orig/w_ceil [3,567,16], s_orig/s_ceil [3,256]); the ceil model is trained on the transformed sample (median recomputed).
Test: every expert in each DIR (manifest.json from fetch_upstream.py or fetch_other.py); experts flagged
'identical_to_sample' / kind=='kept' are reported separately (they are IN the training set).
Bits/weight include a 3.5 B flush per 2048-weight stream (same in both forms).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp

FLUSH_BITS_PER_W = 3.5 * 8 / 2048


def load_items(d):
    man = json.load(open(os.path.join(d, 'manifest.json')))
    for e in man['experts']:
        raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
        off, parts = 0, {}
        for t in e['tensors']:
            if 'role' in t:
                key = (t['role'], t['part'])
            else:
                w, p = t['name'].split('.')
                key = (w, 'weight' if p == 'weight' else 'scale')
            parts[key] = (raw[off:off + t['bytes']], t['shape'])
            off += t['bytes']
        assert off == raw.size
        yield e, parts


def main():
    cdir, out_json = sys.argv[1], sys.argv[2]
    sets = [a.split('=', 1) for a in sys.argv[3:]]
    tr = {k: 0 for k in ['w_orig', 'w_ceil', 's_orig', 's_ceil']}
    n_tr = 0
    for f in sorted(os.listdir(cdir)):
        if f.endswith('.npz'):
            z = np.load(os.path.join(cdir, f))
            for k in tr:
                tr[k] = tr[k] + z[k].astype(np.int64)
            n_tr += 1
    q = {k: mxfp.quantize(v) for k, v in tr.items()}  # [3, 567, 16] / [3, 256]
    res = {'train_experts': n_tr, 'flush_B_per_stream': 3.5, 'sets': {}}
    for name, d in sets:
        rows = []
        agg = {'held': np.zeros(6), 'insample': np.zeros(6)}  # bits w_orig, w_ceil, s_orig, s_ceil, nW, nS
        for e, parts in load_items(d):
            v = np.zeros(6)
            own = np.zeros(4)
            for m, w in enumerate(['w1', 'w2', 'w3']):
                wraw, wshape = parts[(w, 'weight')]
                sraw, _ = parts[(w, 'scale')]
                R = wshape[0]
                sym = mxfp.unpack(wraw.reshape(R, -1))
                sc = sraw.reshape(R, -1)
                tsym, tsc, even, bmax = mxfp.forward(sym, sc)
                assert int((bmax < 6).sum()) == 0
                co = mxfp.ctx_counts(sym, sc)
                cc = mxfp.ctx_counts(tsym, tsc.astype(np.int16))
                so = np.bincount(sc.ravel(), minlength=256)[:256]
                scc = np.bincount(tsc.ravel().astype(np.int64), minlength=256)[:256]
                v[0] += mxfp.cost_bits(co, q['w_orig'][m]); v[1] += mxfp.cost_bits(cc, q['w_ceil'][m])
                v[2] += mxfp.cost_bits(so, q['s_orig'][m]); v[3] += mxfp.cost_bits(scc, q['s_ceil'][m])
                v[4] += sym.size; v[5] += sc.size
                own[0] += mxfp.cost_bits(co, mxfp.quantize(co)); own[1] += mxfp.cost_bits(cc, mxfp.quantize(cc))
                own[2] += mxfp.cost_bits(so, mxfp.quantize(so)); own[3] += mxfp.cost_bits(scc, mxfp.quantize(scc))
            insample = bool(e.get('identical_to_sample')) or e.get('kind') == 'kept'
            agg['insample' if insample else 'held'] += v
            nW, nS = v[4], v[5]
            r = {'tag': e.get('tag') or e['file'], 'kind': e.get('kind'), 'in_training_set': insample,
                 'w_orig_bpw': v[0] / nW + FLUSH_BITS_PER_W, 'w_ceil_bpw': v[1] / nW + FLUSH_BITS_PER_W,
                 's_orig_bps': v[2] / nS, 's_ceil_bps': v[3] / nS,
                 'own_model_w_orig_bpw': own[0] / nW + FLUSH_BITS_PER_W, 'own_model_w_ceil_bpw': own[1] / nW + FLUSH_BITS_PER_W}
            r['w_delta_pct'] = 100 * (r['w_ceil_bpw'] - r['w_orig_bpw']) / r['w_orig_bpw']
            r['own_w_delta_pct'] = 100 * (r['own_model_w_ceil_bpw'] - r['own_model_w_orig_bpw']) / r['own_model_w_orig_bpw']
            raw_bits = 4 * nW + 8 * nS
            r['record_delta_pct_of_raw'] = 100 * ((v[1] - v[0]) + (v[3] - v[2])) / raw_bits
            rows.append(r)
            print(name, r['tag'], round(r['w_orig_bpw'], 4), round(r['w_ceil_bpw'], 4), round(r['w_delta_pct'], 3),
                  'scales', round(r['s_orig_bps'], 3), round(r['s_ceil_bps'], 3), flush=True)
        out = {'experts': rows}
        for k, v in agg.items():
            if v[4] == 0:
                continue
            nW, nS = v[4], v[5]
            wo = v[0] / nW + FLUSH_BITS_PER_W; wc = v[1] / nW + FLUSH_BITS_PER_W
            out[k] = {'n_weights': int(nW), 'w_orig_bpw': wo, 'w_ceil_bpw': wc, 'w_delta_pct': 100 * (wc - wo) / wo,
                      's_orig_bps': v[2] / nS, 's_ceil_bps': v[3] / nS,
                      'record_delta_pct_of_raw': 100 * ((v[1] - v[0]) + (v[3] - v[2])) / (4 * nW + 8 * nS),
                      'ratio_orig_est': (wo * nW + v[2]) / (4 * nW + 8 * nS),
                      'ratio_ceil_est': (wc * nW + v[3]) / (4 * nW + 8 * nS)}
        res['sets'][name] = out
    json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps({s: {k: v for k, v in o.items() if k != 'experts'} for s, o in res['sets'].items()}, indent=1))


if __name__ == '__main__':
    main()
