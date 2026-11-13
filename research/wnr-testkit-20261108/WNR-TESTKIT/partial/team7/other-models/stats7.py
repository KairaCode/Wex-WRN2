"""Per-matrix sufficient statistics for WNR2 / WNR3 model experiments (team7/other-models).

Usage: python -I stats7.py DATA_DIR OUT_DIR [LABEL]
Reads DATA_DIR/manifest.json (formats of fetch_other.py / fetch7.py / v41-sample / team4-data extra) and writes one
OUT_DIR/<tag>.npz per item plus OUT_DIR/index.json. Nothing in DATA_DIR is executed or imported.

Record = 3 matrices m = 0 gate (w1), 1 down (w2), 2 up (w3). gpt-oss fused gate_up_proj is split by rows: even rows ->
gate (m0), odd rows -> up (m2) (HF GptOssExperts: gate = gate_up[..., ::2], up = gate_up[..., 1::2]); the script also
records a diagnostic (scale means of even/odd rows vs first/second half) so the split can be checked.

Per matrix m (arrays in the npz):
  m{m}_shape          [R, C] FP4 weights
  m{m}_bmax           [8]   block max magnitude histogram (stored form)
  m{m}_mask           [256] histogram of the per-block set of magnitudes used (bit a = magnitude a present)
  m{m}_sym            [16]  nibble histogram (code 8 = -0)
  m{m}_sc_p           [256] stored scale histogram
  m{m}_Cp, m{m}_smin_p   weight counts by ABSOLUTE stored scale: [s - smin, pm*9+mx (81), nibble (16)]
  m{m}_sc_c, m{m}_Cc, m{m}_smin_c   same in ceil form (only if the per-matrix gate passes: no stored block with max 5)
  m{m}_scal           [even, blocks, max5, max_lt6, even_s255, max6_no13, gate_ok]
Context of a nibble: pm = previous magnitude in the block (8 = none), mx = running max before it (8 = none);
the WNR2 context is srel(anchor)*81 + pm*9 + mx with srel = clamp(s - anchor, -3, 3) + 3.
"""
import json, os, sys, time
import numpy as np

HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
ROLE_M = {'w1': 0, 'w2': 1, 'w3': 2, 'gate_proj': 0, 'down_proj': 1, 'up_proj': 2, 'down_proj_gptoss': 1}


def unpack(wb):
    R = wb.shape[0]
    sym = np.empty((R, wb.shape[1] * 2), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    return sym


def counts_by_scale(sym, sc):
    """sym uint8 [R, C], sc uint8/int16 [R, C/32] -> (smin, C[ns, 81, 16] int64)."""
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    s = np.asarray(sc).astype(np.int32)
    smin, smax = int(s.min()), int(s.max())
    ns = smax - smin + 1
    idx = ((s - smin)[..., None] * 81 + pm.astype(np.int32) * 9 + mx) * 16 + blk
    return smin, np.bincount(idx.ravel(), minlength=ns * 81 * 16).reshape(ns, 81, 16).astype(np.int64)


def matrix_stats(sym, sc, pre):
    R, C = sym.shape
    nb = C // 32
    assert sc.shape == (R, nb), (sym.shape, sc.shape)
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    bmax = mag.max(2)
    mask = np.bitwise_or.reduce(np.left_shift(np.uint8(1), mag), axis=2)
    has13 = (mask & 0b1010) != 0
    even = (bmax == 7) & ~has13
    out = {}
    out[pre + 'shape'] = np.array([R, C])
    out[pre + 'bmax'] = np.bincount(bmax.ravel(), minlength=8)[:8]
    out[pre + 'mask'] = np.bincount(mask.ravel(), minlength=256)[:256]
    out[pre + 'sym'] = np.bincount(sym.ravel(), minlength=16)[:16]
    out[pre + 'sc_p'] = np.bincount(sc.ravel(), minlength=256)[:256]
    smin, Cp = counts_by_scale(sym, sc)
    out[pre + 'smin_p'] = np.array(smin); out[pre + 'Cp'] = Cp
    max5 = int((bmax == 5).sum())
    gate_ok = int(max5 == 0)
    e255 = int((even & (sc == 255)).sum())
    out[pre + 'scal'] = np.array([int(even.sum()), int(bmax.size), max5, int((bmax < 6).sum()), e255,
                                  int(((bmax == 6) & ~has13).sum()), gate_ok])
    if gate_ok:
        tr = even & (sc <= 254)
        tblk = np.where(tr[..., None], (blk & 8) | HALF[mag], blk)
        tsc = np.where(tr, sc.astype(np.int16) + 1, sc.astype(np.int16))
        # round trip check of the inverse (decoder rule: coded max == 5 -> double, scale - 1)
        tmag = tblk & 7
        tmax = tmag.max(2)
        sel = tmax == 5
        DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)
        iblk = np.where(sel[..., None], (tblk & 8) | DOUBLE[tmag], tblk)
        isc = np.where(sel, tsc - 1, tsc)
        assert np.array_equal(iblk, blk) and np.array_equal(isc, sc.astype(np.int16)), 'ceil round trip failed'
        tsym = tblk.reshape(R, C)
        out[pre + 'sc_c'] = np.bincount(tsc.ravel(), minlength=256)[:256]
        smin_c, Cc = counts_by_scale(tsym, tsc)
        out[pre + 'smin_c'] = np.array(smin_c); out[pre + 'Cc'] = Cc
    return out


def load_parts(d, e):
    raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
    off, parts = 0, {}
    for t in e['tensors']:
        if 'role' in t:
            key = (t['role'], t['part'])
        else:
            w, p = t['name'].split('.')
            key = (w, 'weight' if p == 'weight' else 'scale')
        parts[key] = (raw[off:off + t['bytes']], list(t['shape']))
        off += t['bytes']
    assert off == raw.size, (e['file'], off, raw.size)
    return parts


def main():
    d, od = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(d.rstrip('\\/'))
    os.makedirs(od, exist_ok=True)
    man = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
    index = {'label': label, 'data_dir': d, 'repo': man.get('repo'), 'revision': man.get('revision'), 'items': []}
    t0 = time.time()
    for i, e in enumerate(man['experts']):
        tag = e.get('tag') or os.path.splitext(e['file'])[0]
        outp = os.path.join(od, tag + '.npz')
        parts = load_parts(d, e)
        roles = sorted({k[0] for k in parts})
        mats = {}
        diag = {}
        for role in roles:
            wraw, wshape = parts[(role, 'weight')]
            sraw, sshape = parts[(role, 'scale')]
            R = wshape[0]
            sym = unpack(wraw.reshape(R, -1))
            sc = sraw.reshape(R, -1)
            if role == 'gate_up_proj':
                se, so = sc[0::2].astype(np.float64), sc[1::2].astype(np.float64)
                sa, sb = sc[:R // 2].astype(np.float64), sc[R // 2:].astype(np.float64)
                diag = {'even_rows_mean_scale': se.mean(), 'odd_rows_mean_scale': so.mean(),
                        'first_half_mean_scale': sa.mean(), 'second_half_mean_scale': sb.mean(),
                        'even_odd_absdiff_rowmean': float(np.abs(se.mean(1) - so.mean(1)).mean()),
                        'half_absdiff_rowmean': float(np.abs(sa.mean(1) - sb.mean(1)).mean())}
                mats[0] = (sym[0::2], sc[0::2])
                mats[2] = (sym[1::2], sc[1::2])
            else:
                mats[ROLE_M[role]] = (sym, sc)
        assert sorted(mats) == [0, 1, 2], (tag, roles)
        out = {}
        for m in range(3):
            out.update(matrix_stats(mats[m][0], mats[m][1], f'm{m}_'))
        np.savez_compressed(outp, **out)
        it = {'tag': tag, 'layer': e.get('layer'), 'expert': e.get('expert'), 'kind': e.get('kind'), 'npz': os.path.basename(outp),
              'raw_bytes': int(sum(v[0].size for v in parts.values())), 'gate_up_diag': diag,
              'shapes': [out[f'm{m}_shape'].tolist() for m in range(3)],
              'even_pct': [round(100 * out[f'm{m}_scal'][0] / out[f'm{m}_scal'][1], 4) for m in range(3)],
              'gate_ok': [int(out[f'm{m}_scal'][6]) for m in range(3)]}
        index['items'].append(it)
        print(label, i, tag, it['shapes'], it['even_pct'], it['gate_ok'], diag and {k: round(v, 3) for k, v in diag.items()},
              round(time.time() - t0, 1), 's', flush=True)
    json.dump(index, open(os.path.join(od, 'index.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
