"""Per-expert / per-matrix WNR2 compressibility from the pass-1 counts.

Usage: python -I countstats.py SAMPLE_DIR COUNTS_DIR KEEPSET_JSON OUT_JSON
All bit/w figures are PAYLOAD bits / weights (no 4-B stream flush) unless the key ends in _fl
(+0.015625 bit/w flush, comparable with the 3.7821 / 3.7826 figures of team 1).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import (P, NW, FLUSH_W_PER_MATRIX, quantize, codelen, cost_bits, ideal_bits, load_all,
                    spearman, pearson)

FL = FLUSH_W_PER_MATRIX / NW  # 0.015625


def scale_hist_stats(h, med):
    v = np.arange(256)
    n = h.sum()
    p = h / n
    mean = (p * v).sum()
    std = np.sqrt((p * (v - mean) ** 2).sum())
    cdf = np.cumsum(h)
    q = lambda f: int(np.searchsorted(cdf, f * n, side='left'))
    nz = np.nonzero(h)[0]
    return {'s_med': int(med), 's_mean': float(mean), 's_std': float(std), 's_iqr': q(0.75) - q(0.25),
            's_p1': q(0.01), 's_p99': q(0.99), 's_range99': q(0.99) - q(0.01), 's_min': int(nz[0]), 's_max': int(nz[-1]),
            's_clamp_lo': float(h[:max(int(med) - 3, 0)].sum() / n), 's_clamp_hi': float(h[int(med) + 4:].sum() / n),
            's_skew': float((p * (v - mean) ** 3).sum() / std ** 3)}


def main():
    sample_dir, counts_dir, keep_json, out_json = sys.argv[1:5]
    ex = load_all(sample_dir, counts_dir)
    keep = json.load(open(keep_json, encoding='utf-8'))['keep']
    n = len(ex)
    WB = np.stack([e['w_base'] for e in ex])            # [n,3,567,16]
    SO = np.stack([e['s_o0'] for e in ex])              # [n,3,256]
    lay = np.array([e['layer'] for e in ex])
    tot = WB.sum(0)                                     # [3,567,16]
    Lsh = np.stack([codelen(tot[m]) for m in range(3)])  # per-type shared model
    Lpool = codelen(tot.sum(0))                         # one model for all matrix types
    Ssh = np.stack([codelen(SO[:, m].sum(0)) for m in range(3)])
    units = []
    for i, e in enumerate(ex):
        for m in range(3):
            c = WB[i, m]
            same = [j for j in range(n) if lay[j] == lay[i] and j != i]
            r = {'i': i, 'file': e['file'], 'layer': int(e['layer']), 'expert': int(e['expert']),
                 'orig_id': int(keep[str(e['layer'])][e['expert']]), 'm': m}
            r['bpw_shared'] = float((c * Lsh[m]).sum() / NW)
            r['bpw_pooled'] = float((c * Lpool).sum() / NW)
            r['bpw_shared_loo'] = cost_bits(c, quantize(tot[m] - c)) / NW
            r['bpw_layer_loo'] = cost_bits(c, quantize(WB[same, m].sum(0))) / NW
            r['bpw_own'] = cost_bits(c, quantize(c)) / NW
            r['bpw_own_ideal'] = ideal_bits(c) / NW
            r['excess_shared_vs_own'] = r['bpw_shared'] - r['bpw_own']
            sym = c.sum(0)
            r['h0_own'] = ideal_bits(sym) / NW                     # order-0 entropy of this matrix
            mag = (sym[:8] + sym[8:]) / NW
            r['mag_hist'] = mag.round(6).tolist()
            r['zpos'] = float(sym[0] / NW)
            r['zneg'] = float(e['meta'][m, 3] / NW)
            assert e['meta'][m, 3] == sym[8]
            r['zero'] = r['zpos'] + r['zneg']
            r['neg_share_nonzero'] = float(sym[9:].sum() / (sym[1:8].sum() + sym[9:].sum()))
            vals = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
            r['mean_absval_code'] = float((mag * vals).sum())
            r['frac_mag67'] = float(mag[6] + mag[7])
            r['frac_mag_le1'] = float(mag[:3].sum())
            srel = c.reshape(7, 81, 16).sum((1, 2)) / NW
            r['srel_hist'] = srel.round(6).tolist()
            # payload bits by srel bucket under the shared model (bit/w within that bucket)
            cb = (c * Lsh[m]).reshape(7, 81, 16).sum((1, 2))
            cw = c.reshape(7, 81, 16).sum((1, 2))
            r['bpw_by_srel'] = [float(cb[k] / cw[k]) if cw[k] else None for k in range(7)]
            r.update(scale_hist_stats(SO[i, m], e['meta'][m, 2]))
            r['sbits_shared'] = float((SO[i, m] * Ssh[m]).sum() / SO[i, m].sum())
            r['sbits_own'] = cost_bits(SO[i, m], quantize(SO[i, m])) / SO[i, m].sum()
            bm = e['bm_side'][m]
            r['blk_max7'] = float(bm[:, 7].sum() / bm.sum())
            # block max == 7 fraction by srel bucket
            r['blk_max7_by_srel'] = [float(bm[k, 7] / bm[k].sum()) if bm[k].sum() else None for k in range(7)]
            units.append(r)
    # population: bit/w per srel bucket and srel shares
    pop = {}
    for m in range(3):
        c = tot[m]
        cb = (c * Lsh[m]).reshape(7, 81, 16).sum((1, 2)); cw = c.reshape(7, 81, 16).sum((1, 2))
        pop[f'm{m}_bpw_by_srel'] = (cb / cw).round(4).tolist()
        pop[f'm{m}_srel_share'] = (cw / cw.sum()).round(5).tolist()
        pop[f'm{m}_bpw_shared'] = float(cb.sum() / cw.sum())
    json.dump({'units': units, 'pop': pop}, open(out_json, 'w'), indent=0)
    print('units', len(units))
    print(json.dumps(pop, indent=0))


if __name__ == '__main__':
    main()
