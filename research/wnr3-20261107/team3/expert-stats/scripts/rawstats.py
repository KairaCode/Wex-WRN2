"""Per-expert, per-matrix statistics from the raw MXFP4 data (scales + dequantized weights).

Usage: python -I rawstats.py SAMPLE_DIR OUT_JSON
Reads each L??_E???.bin (w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight, per manifest.json).
Per matrix:
  scale: median, mean, std, p1/p5/p95/p99, min, max, distinct values, frac |s-med|>3 (srel clamp),
         two-way decomposition s_ij = g + r_i + c_j + e_ij: var of row effect, col effect, residual (share of total)
  weights: kurtosis of dequantized values (whole matrix), kurtosis of code values (scale removed),
           frac of blocks whose max magnitude code is 7 (6.0), mean number of max-magnitude ties per block,
           frac of all-zero rows, rows whose scales are all equal
"""
import json, os, sys, time
import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def kurt(x):
    x = x.astype(np.float64, copy=False)
    m = x.mean()
    d = x - m
    v = (d * d).mean()
    return float((d ** 4).mean() / (v * v)) if v > 0 else float('nan')


def mstats(wb, sc):
    R = wb.shape[0]
    C = wb.shape[1] * 2
    nb = C // 32
    s = sc.astype(np.float64)
    med = float(np.sort(sc.ravel().astype(np.int16))[(sc.size - 1) // 2])
    q = np.percentile(s, [1, 5, 25, 75, 95, 99])
    g = s.mean()
    r = s.mean(1, keepdims=True) - g
    c = s.mean(0, keepdims=True) - g
    e = s - g - r - c
    vt = s.var()
    out = {
        'rows': R, 'cols': C,
        's_med': med, 's_mean': float(g), 's_std': float(s.std()),
        's_p1': q[0], 's_p5': q[1], 's_p25': q[2], 's_p75': q[3], 's_p95': q[4], 's_p99': q[5],
        's_min': float(s.min()), 's_max': float(s.max()), 's_distinct': int(np.unique(sc).size),
        's_clamp_lo': float((s - med < -3).mean()), 's_clamp_hi': float((s - med > 3).mean()),
        's_var_row': float(r.var()), 's_var_col': float(c.var()), 's_var_res': float(e.var()), 's_var_tot': float(vt),
        's_rows_const': float((sc.max(1) == sc.min(1)).mean()),
    }
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    bmax = mag.max(2)
    out['blk_max7'] = float((bmax == 7).mean())
    out['blk_max_lt6'] = float((bmax < 6).mean())
    out['blk_ties'] = float((mag == bmax[..., None]).sum(2).mean())
    code = E2M1[blk]  # [R, nb, 32]
    out['kurt_code'] = kurt(code.ravel())
    deq = code * np.exp2(s - 127.0)[..., None]
    out['kurt_deq'] = kurt(deq.ravel())
    out['deq_std'] = float(deq.std())
    out['rows_allzero'] = float(((mag == 0).all(axis=(1, 2))).mean())
    # share of |value| mass in the top 1% of blocks by scale (heaviness of block-scale tail)
    out['mean_abs_code'] = float(np.abs(code).mean())
    return out


def main():
    sample_dir, out_json = sys.argv[1:3]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    res = []
    t0 = time.time()
    for k, ex in enumerate(man['experts']):
        raw = np.fromfile(os.path.join(sample_dir, ex['file']), dtype=np.uint8)
        t, off = {}, 0
        for tt in ex['tensors']:
            t[tt['name']] = (raw[off:off + tt['bytes']], tt['shape'])
            off += tt['bytes']
        assert off == raw.size
        row = {'file': ex['file'], 'layer': ex['layer'], 'expert': ex['expert'], 'm': []}
        for w in ['w1', 'w2', 'w3']:
            wr, ws = t[w + '.weight']
            sr, ss = t[w + '.scale']
            row['m'].append(mstats(wr.reshape(ws[0], -1), sr.reshape(ss[0], -1)))
        res.append(row)
        if k % 20 == 0:
            print(k, ex['file'], f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(out_json, 'w'), indent=0)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
