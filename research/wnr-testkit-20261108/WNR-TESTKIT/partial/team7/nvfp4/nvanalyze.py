"""Pass 2: held-out code lengths of WNR-style static-context models on NVFP4 counts, scale entropy, block-max and
zero-sign statistics, and an estimated WNR-NV record size.

Usage: python -I nvanalyze.py COUNTS_DIR OUT_JSON
Model = one shared static model per matrix type (w1, w2, w3), 14-bit CDFs quantized like wex_nr2_quantize.
held-out = experts split by item index parity; train on one half, cost on the other, both ways, summed.
Record estimate per item (inferred structure, WNR3-like): 64 B header + 8 B raw (tensor scale + input scale) per matrix
+ payload bits/8 + per stream 3.5 B flush + 0.70 B coded length table + 0.07 B hash;
weight streams of 2048 codes, scale streams of 1024 scales (16,384 weights, as WNR2's 512 x 32).
"""
import importlib.util, glob, json, os, sys
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)

WKEYS = ['w_o0', 'w_b1', 'w_b2', 'w_b4', 'w_b8', 'w_b1p', 'w_b2p', 'w_b4p', 'w_b8p', 'w_s15', 'w_b2pm2', 'w_b2lm', 'w_b2ls',
         'w_b2par', 'v_o0', 'v_b2', 'v_b2p']
SKEYS = ['s_o0', 's_left', 's_up', 's_lu', 's_par']
PER_STREAM = 3.5 + 0.70 + 0.07


def main():
    cdir, out_json = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(cdir, '*.npz')))
    ts = json.load(open(os.path.join(cdir, 'tensor_scales.json')))
    raw_by_tag = {x['tag']: x['raw_bytes'] for x in ts['items']}
    items = []
    for i, f in enumerate(files):
        z = np.load(f)
        tag = os.path.basename(f)[:-4]
        items.append({'tag': tag, 'split': i % 2, **{k: z[k] for k in z.files}})
    n = len(items)
    meta = np.stack([it['meta'] for it in items])  # [n, 3, 8]
    Wm = (meta[:, :, 0] * meta[:, :, 1]).astype(np.int64)  # weights per matrix
    MN = ['w1', 'w2', 'w3'][:meta.shape[1]]
    nmat = meta.shape[1]
    W = int(Wm.sum())
    S = W // 16
    wstreams = int(np.ceil(Wm / 2048).sum())
    sstreams = int(np.ceil(Wm / 16 / 1024).sum())

    def tot(key, sel=lambda it: True):
        return sum(it[key].astype(np.int64) for it in items if sel(it))

    def ev(key):
        a = tot(key, lambda it: it['split'] == 0)
        b = tot(key, lambda it: it['split'] == 1)
        al = a + b
        ins = nc.cost_bits(al, nc.quantize(al))
        ho = nc.cost_bits(b, nc.quantize(a)) + nc.cost_bits(a, nc.quantize(b))
        return ins, ho, nc.ideal_bits(al)

    out = {'counts_dir': cdir, 'repo': ts['repo'], 'revision': ts['revision'], 'items': n, 'weights': W, 'scales': S,
           'weight_streams': wstreams, 'scale_streams': sstreams,
           'raw_bytes_total': int(sum(raw_by_tag[it['tag']] for it in items))}
    wres = {}
    for k in WKEYS:
        if k not in items[0]:
            continue
        ins, ho, idl = ev(k)
        wres[k] = {'insample_bit_per_w': ins / W, 'heldout_bit_per_w': ho / W, 'ideal_insample_bit_per_w': idl / W,
                   'rows': int(items[0][k].shape[1])}
    out['weights_payload_bit_per_w (no flush)'] = wres
    # per matrix, base b2 held-out
    pm = {}
    for m, nm in enumerate(MN):
        a = tot('w_b2', lambda it: it['split'] == 0)[m]; b = tot('w_b2', lambda it: it['split'] == 1)[m]
        pm[nm] = (nc.cost_bits(b, nc.quantize(a)) + nc.cost_bits(a, nc.quantize(b))) / int(Wm[:, m].sum())
    out['w_b2_heldout_per_matrix_bit_per_w'] = pm
    # cost by srel bin for b2 (in-sample), share of weights per bin
    al = tot('w_b2')
    q = nc.quantize(al)
    lb = -np.log2(q / nc.P)
    byb = {}
    for sbin in range(8):
        rows = slice(sbin * 81, (sbin + 1) * 81)
        cnt = int(al[:, rows].sum())
        bits = float((al[:, rows] * lb[:, rows]).sum())
        byb[f'srel{sbin - 3 if sbin < 7 else "far"}'] = {'weights_share': cnt / W, 'bit_per_w_in_bin': bits / max(cnt, 1)}
    out['w_b2_by_srel_bin_insample'] = byb
    sres = {}
    for k in SKEYS:
        ins, ho, idl = ev(k)
        sres[k] = {'insample_bit_per_scale': ins / S, 'heldout_bit_per_scale': ho / S, 'ideal_insample': idl / S}
    out['scales_payload_bit_per_scale (no flush)'] = sres
    # scale byte histogram (top) and distinct
    so = tot('s_o0').sum(0)
    nzb = np.nonzero(so)[0]
    out['scale_bytes'] = {'distinct': int(nzb.size), 'min': int(nzb.min()), 'max': int(nzb.max()),
                          'pow2_share': float(so[(np.arange(256) & 7) == 0].sum() / so.sum()),
                          'top10': [(int(b), round(float(so[b] / so.sum()), 5), float(nc.e4m3_value(b))) for b in np.argsort(-so)[:10]],
                          'order0_entropy_bits': nc.ideal_bits(so) / so.sum()}
    # block max
    bmx = tot('bmax')  # [3, 256, 8]
    h = bmx.sum(1)
    out['blockmax16_hist_per_matrix'] = {nm: (h[m] / h[m].sum()).round(6).tolist() for m, nm in enumerate(MN)}
    hz = bmx.sum((0, 1))
    out['blockmax16_hist_all'] = (hz / hz.sum()).round(6).tolist()
    # live blocks only (scale value > 0 and max > 0)
    sv = nc.e4m3_value(np.arange(256))
    livemask = np.nan_to_num(sv) > 0
    hl = bmx[:, livemask, :].sum((0, 1)); hl[0] = 0
    out['blockmax16_hist_live'] = (hl / hl.sum()).round(6).tolist()
    n13 = tot('n13').sum(0)
    mm7 = tot('magmax7').sum(0)
    nb7 = int(n13.sum())
    # non-max positions of max-7 blocks: remove one 7 per block
    nonmax = mm7.astype(np.float64).copy(); nonmax[7] -= nb7
    q13 = (nonmax[1] + nonmax[3]) / nonmax.sum()
    out['doubled16'] = {'max7_blocks': nb7, 'no_1_or_3': int(n13[0]), 'share_of_max7_blocks': n13[0] / nb7,
                        'share_of_all_blocks': n13[0] / int(hz.sum()),
                        'q13_nonmax': q13, 'iid_expected_share_of_max7': (1 - q13) ** 15,
                        'n13_hist': n13.tolist()}
    zs = tot('zsgn').sum(0)  # [2 live/dead, 2 (+0,-0)]
    out['zero_sign'] = {'live_plus0': int(zs[0, 0]), 'live_minus0': int(zs[0, 1]), 'dead_plus0': int(zs[1, 0]),
                        'dead_minus0': int(zs[1, 1]),
                        'minus0_share_of_weights': float(zs[:, 1].sum() / W),
                        'minus0_share_of_zeros_live': float(zs[0, 1] / max(zs[0].sum(), 1))}
    out['dead'] = {'zero_value_scale_blocks_share': float(meta[:, :, 4].sum() / (W / 16)),
                   'max0_blocks_share': float(meta[:, :, 7].sum() / (W / 16)),
                   'per_matrix_zero_scale_share': (meta[:, :, 4].sum(0) / (Wm.sum(0) / 16)).round(5).tolist(),
                   'per_matrix_max0_share': (meta[:, :, 7].sum(0) / (Wm.sum(0) / 16)).round(5).tolist()}
    out['anchor_bytes_by_matrix'] = {nm: np.unique(meta[:, m, 2], return_counts=True)[0].tolist() for m, nm in enumerate(MN)}

    # record estimates (held-out)
    raw = out['raw_bytes_total'] / n
    over = 64 + nmat * 8 + PER_STREAM * (wstreams + sstreams) / n

    def rec(wk, sk):
        return over + (wres[wk]['heldout_bit_per_w'] * W + sres[sk]['heldout_bit_per_scale'] * S) / 8 / n

    best_w = min((k for k in wres if k.startswith('w_') and k != 'w_o0'), key=lambda k: wres[k]['heldout_bit_per_w'])
    best_wv = min((k for k in wres if k.startswith('v_') and k != 'v_o0'), key=lambda k: wres[k]['heldout_bit_per_w'])
    best_s = min(sres, key=lambda k: sres[k]['heldout_bit_per_scale'])
    R = {}
    for name, wk, sk in [('order0 both', 'w_o0', 's_o0'), ('WNR-style b2 + scale o0', 'w_b2', 's_o0'),
                         ('b2 + best scale ctx', 'w_b2', best_s), (f'best byte-exact ({best_w}, {best_s})', best_w, best_s),
                         (f'value-exact ({best_wv}, {best_s})', best_wv, best_s)]:
        b = rec(wk, sk)
        R[name] = {'record_bytes': b, 'ratio_vs_raw': b / raw, 'bit_per_w_total': b * 8 / (W / n)}
    out['record_estimate_heldout'] = {'raw_bytes_per_item': raw, 'overhead_bytes_per_item': over, 'variants': R}
    json.dump(out, open(out_json, 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:9000])


if __name__ == '__main__':
    main()
