"""ceil-refuter: per-expert invariant checks of the ceil-form transform (independent packed-byte implementation)
and context counts for the original form, side-info alternatives and the ceil form.

Usage: python -I refute_count.py DATA_DIR OUT_DIR GROUPING
  GROUPING = 'sample'  -> group = manifest index % 4 (each group = one expert per layer)
           = 'extra'   -> group 4 = routed (kind 'layers'), group 5 = MTP (kind 'mtp')
Writes OUT_DIR/stats_<GROUPING>.json (per expert/matrix checks) and OUT_DIR/counts_<GROUPING>.npz
(group sums of every count array, leading axis = group 0..5). Resumable (checkpoint every 10 experts).

Transform T (byte level): a block (16 packed bytes = 32 E2M1 nibbles) whose max magnitude is 7 and that has no
magnitude 1 or 3 ('even') gets every magnitude halved (0,2,4,5,6,7 -> 0,1,2,3,4,5; sign bit 8 kept), scale + 1.
Inverse: every block whose max magnitude is 5 is doubled back, scale - 1.

Count keys (per group, per matrix type m): weights [6, 3, rows, 16]
  o_base      (srel7, pm9, mx9) = 567                            original form, WNR2 today
  o_pm2b      base x B6[pm2] = 3402                              B6 = {none, 0, 1, 2, 3, 4-7}
  o_ev        ev2 x base = 1134                                  side: 'even' flag per block (all blocks)
  o_bm7h      (bm7, half) x base = 2268                          side: bm7 per block (WNR3 feature a)
  o_cls       cls3 x base = 1701                                 side: class per block 0 max6, 1 max7 odd, 2 even
  o_clsh      (cls3, half) x base = 3402
  o_bm7h_pm2b (bm7, half) x base x B6 = 13608                    team-2 audit best non-ceil (-0.478%)
  o_clsh_pm2b (cls3, half) x base x B6 = 20412
  c_base, c_pm2b, c_bm3 (bm3 = ceil max - 5), c_bm3h, c_bm3h_pm2b   same on the ceil form (srel vs ceil median)
  side counts [6, 3, 7, k]: o_side_bm7, o_side_ev, o_side_cls (given srel of the original form), c_side_bm3 (srel ceil)
  scale counts: s_o [6,3,256], s_c, s_o_cls [6,3,3,256] (scale given class), s_c_bm3 [6,3,3,256]
"""
import hashlib, json, os, sys, time
import numpy as np

B6 = np.array([1, 2, 3, 4, 5, 5, 5, 5, 0], np.int32)  # index = pm2 (8 = none)
NG = 6

# ---- packed-byte LUTs (independent of mxceil.py / ceilform.py, which work on unpacked nibbles) ----
_lo = np.arange(256) & 15
_hi = np.arange(256) >> 4
LUT_MAX = np.maximum(_lo & 7, _hi & 7).astype(np.uint8)
LUT_ODD = (np.isin(_lo & 7, [1, 3]) | np.isin(_hi & 7, [1, 3])).astype(np.uint8)
_H = np.array([0, 0, 1, 0, 2, 3, 4, 5])  # 1, 3 never occur in even blocks
_D = np.array([0, 2, 4, 5, 6, 7, 6, 7])  # 6, 7 never occur in max-5 blocks
LUT_HALF = (((_lo & 8) | _H[_lo & 7]) | (((_hi & 8) | _H[_hi & 7]) << 4)).astype(np.uint8)
LUT_DBL = (((_lo & 8) | _D[_lo & 7]) | (((_hi & 8) | _D[_hi & 7]) << 4)).astype(np.uint8)
LUT_Z = ((np.where(_lo == 8, 0, _lo)) | (np.where(_hi == 8, 0, _hi) << 4)).astype(np.uint8)
LUT_NZ = (_lo == 8).astype(np.int64) + (_hi == 8)


def T_fwd(blocks, sc):
    """blocks uint8 [Nb, 16], sc uint8 [Nb] -> (tw uint8, ts int16, even bool)."""
    bmax = LUT_MAX[blocks].max(1)
    odd = LUT_ODD[blocks].any(1)
    even = (bmax == 7) & ~odd
    tw = blocks.copy()
    tw[even] = LUT_HALF[blocks[even]]
    ts = sc.astype(np.int16) + even
    return tw, ts, even, bmax


def T_inv(tw, ts):
    dec = LUT_MAX[tw].max(1) == 5
    iw = tw.copy()
    iw[dec] = LUT_DBL[tw[dec]]
    return iw, ts - dec, dec


def lower_median(v):
    f = np.sort(v.ravel())
    return int(f[(f.size - 1) // 2])


def unpack(blocks):
    nb = blocks.shape[0]
    sym = np.empty((nb, 32), np.uint8)
    sym[:, 0::2] = blocks & 15
    sym[:, 1::2] = blocks >> 4
    return sym


def ctx_arrays(sym, scale, med):
    mag = (sym & 7).astype(np.int32)
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    pm2 = np.empty_like(mag); pm2[:, :2] = 8; pm2[:, 2:] = mag[:, :-2]
    cm = np.maximum.accumulate(mag, axis=1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    srel = (np.clip(scale.astype(np.int32) - med, -3, 3) + 3)
    base = (srel[:, None] * 9 + pm) * 9 + mx
    return base, B6[pm2], cm[:, -1], srel


def cnt(ctx, sym32, rows):
    return np.bincount((ctx * 16 + sym32).ravel(), minlength=rows * 16)[:rows * 16].reshape(rows, 16)


ROWS = {'o_base': 567, 'o_pm2b': 3402, 'o_ev': 1134, 'o_bm7h': 2268, 'o_cls': 1701, 'o_clsh': 3402,
        'o_bm7h_pm2b': 13608, 'o_clsh_pm2b': 20412,
        'c_base': 567, 'c_pm2b': 3402, 'c_bm3': 1701, 'c_bm3h': 3402, 'c_bm3h_pm2b': 20412}
SIDE = {'o_side_bm7': 2, 'o_side_ev': 2, 'o_side_cls': 3, 'c_side_bm3': 3}


def new_acc():
    a = {k: np.zeros((NG, 3, r, 16), np.int64) for k, r in ROWS.items()}
    a.update({k: np.zeros((NG, 3, 7, n), np.int64) for k, n in SIDE.items()})
    a['s_o'] = np.zeros((NG, 3, 256), np.int64); a['s_c'] = np.zeros((NG, 3, 256), np.int64)
    a['s_o_cls'] = np.zeros((NG, 3, 3, 256), np.int64); a['s_c_bm3'] = np.zeros((NG, 3, 3, 256), np.int64)
    return a


def do_matrix(wraw, sraw, R, acc, g, m):
    blocks = wraw.reshape(-1, 16)
    nb = blocks.shape[0]
    sc = sraw.reshape(-1)
    assert sc.size == nb, (sc.size, nb)
    st = {}
    tw, ts, even, bmax = T_fwd(blocks, sc)
    st['bmax_hist'] = np.bincount(bmax, minlength=8).tolist()
    st['blocks'] = int(nb)
    st['allzero'] = int((bmax == 0).sum())
    st['max_lt6'] = int((bmax < 6).sum())
    st['stored_max5'] = int((bmax == 5).sum())
    st['even'] = int(even.sum())
    odd = LUT_ODD[blocks].any(1)
    st['max6_no13'] = int(((bmax == 6) & ~odd).sum())
    st['max7'] = int((bmax == 7).sum())
    st['scale_min'] = int(sc.min()); st['scale_max'] = int(sc.max())
    st['scale_n0'] = int((sc == 0).sum()); st['scale_n254'] = int((sc == 254).sum()); st['scale_n255'] = int((sc == 255).sum())
    st['scale_hist_top'] = {int(v): int(c) for v, c in zip(*np.unique(sc, return_counts=True))}
    st['even_scale_max'] = int(sc[even].max()) if even.any() else -1
    st['even_at_254'] = int((even & (sc == 254)).sum()); st['even_at_255_overflow'] = int((even & (sc == 255)).sum())
    st['ts_max'] = int(ts.max())
    # round trip
    iw, isc, dec = T_inv(tw, ts)
    st['rt_ok'] = bool(np.array_equal(iw, blocks) and np.array_equal(isc, sc.astype(np.int16)))
    st['inv_selects_exactly_even'] = bool(np.array_equal(dec, even))
    # canonical zero interplay: Z(T(x)) == T(Z(x)) and T^-1(Z(T(x))) == Z(x)
    zb = LUT_Z[blocks]
    tz, tzs, evz, _ = T_fwd(zb, sc)
    zt = LUT_Z[tw]
    st['z_commute'] = bool(np.array_equal(tz, zt) and np.array_equal(tzs, ts) and np.array_equal(evz, even))
    izw, izs, _ = T_inv(zt, ts)
    st['z_rt_ok'] = bool(np.array_equal(izw, zb) and np.array_equal(izs, sc.astype(np.int16)))
    st['neg0'] = int(LUT_NZ[blocks].sum())
    # sign plane unchanged by T
    st['sign_plane_kept'] = bool(np.array_equal(tw & 0x88, blocks & 0x88))
    st['sha256'] = hashlib.sha256(wraw.tobytes() + sraw.tobytes()).hexdigest()
    med_o = lower_median(sc); med_c = lower_median(ts)
    st['med_o'] = med_o; st['med_c'] = med_c
    if not st['rt_ok'] or st['even_at_255_overflow'] or st['stored_max5']:
        return st  # do not count a matrix the transform cannot handle
    # ---- counts ----
    sym = unpack(blocks)
    s32 = sym.astype(np.int32)
    base, b6, cmax, srel = ctx_arrays(sym, sc, med_o)
    bm7 = (cmax == 7).astype(np.int32)
    cls = np.where(cmax == 7, np.where(even, 2, 1), 0).astype(np.int32)
    ev = even.astype(np.int32)
    half = (np.arange(32, dtype=np.int32) >= 16)[None, :].astype(np.int32)
    acc['o_base'][g, m] += cnt(base, s32, 567)
    acc['o_pm2b'][g, m] += cnt(base * 6 + b6, s32, 3402)
    acc['o_ev'][g, m] += cnt(ev[:, None] * 567 + base, s32, 1134)
    bh = (bm7[:, None] * 2 + half) * 567 + base
    acc['o_bm7h'][g, m] += cnt(bh, s32, 2268)
    acc['o_bm7h_pm2b'][g, m] += cnt(bh * 6 + b6, s32, 13608)
    acc['o_cls'][g, m] += cnt(cls[:, None] * 567 + base, s32, 1701)
    ch = (cls[:, None] * 2 + half) * 567 + base
    acc['o_clsh'][g, m] += cnt(ch, s32, 3402)
    acc['o_clsh_pm2b'][g, m] += cnt(ch * 6 + b6, s32, 20412)
    acc['o_side_bm7'][g, m] += np.bincount(srel * 2 + bm7, minlength=14).reshape(7, 2)
    acc['o_side_ev'][g, m] += np.bincount(srel * 2 + ev, minlength=14).reshape(7, 2)
    acc['o_side_cls'][g, m] += np.bincount(srel * 3 + cls, minlength=21).reshape(7, 3)
    acc['s_o'][g, m] += np.bincount(sc, minlength=256)
    acc['s_o_cls'][g, m] += np.bincount(cls * 256 + sc, minlength=768).reshape(3, 256)
    del base, b6, bh, ch
    # ceil form
    tsym = unpack(tw)
    t32 = tsym.astype(np.int32)
    tsc = ts.astype(np.int32)
    base, b6, cmax_c, srel_c = ctx_arrays(tsym, tsc, med_c)
    bm3 = (cmax_c - 5).astype(np.int32)
    assert bm3.min() >= 0 and bm3.max() <= 2
    acc['c_base'][g, m] += cnt(base, t32, 567)
    acc['c_pm2b'][g, m] += cnt(base * 6 + b6, t32, 3402)
    acc['c_bm3'][g, m] += cnt(bm3[:, None] * 567 + base, t32, 1701)
    ch = (bm3[:, None] * 2 + half) * 567 + base
    acc['c_bm3h'][g, m] += cnt(ch, t32, 3402)
    acc['c_bm3h_pm2b'][g, m] += cnt(ch * 6 + b6, t32, 20412)
    acc['c_side_bm3'][g, m] += np.bincount(srel_c * 3 + bm3, minlength=21).reshape(7, 3)
    acc['s_c'][g, m] += np.bincount(tsc, minlength=256)[:256]
    acc['s_c_bm3'][g, m] += np.bincount(bm3 * 256 + tsc, minlength=768)[:768].reshape(3, 256)
    st['counted'] = True
    return st


def main():
    data_dir, out_dir, grouping = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(data_dir, 'manifest.json')))
    sp = os.path.join(out_dir, f'stats_{grouping}.json')
    cp = os.path.join(out_dir, f'counts_{grouping}.npz')
    stats = json.load(open(sp)) if os.path.exists(sp) and os.path.exists(cp) else {}
    acc = new_acc()
    if stats:
        z = np.load(cp)
        for k in acc:
            acc[k] = z[k].astype(np.int64)
    t0 = time.time()
    todo = [(i, e) for i, e in enumerate(man['experts']) if e['file'] not in stats]
    for n, (i, e) in enumerate(todo):
        if grouping == 'sample':
            g = i % 4
        else:
            g = 4 if e.get('kind', 'layers') == 'layers' else 5
        raw = np.fromfile(os.path.join(data_dir, e['file']), dtype=np.uint8)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == e['sha256'], e['file']
        parts, off = {}, 0
        for t in e['tensors']:
            parts[t['name']] = (raw[off:off + t['bytes']], t['shape'], t['dtype'])
            off += t['bytes']
        assert off == raw.size
        est = {'index': i, 'group': g, 'layer': e['layer'], 'expert': e['expert'], 'kind': e.get('kind', 'layers'), 'm': []}
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, wdt = parts[f'{w}.weight']
            sraw, sshape, sdt = parts[f'{w}.scale']
            assert wdt == 'I8' and sdt == 'F8_E8M0', (wdt, sdt)
            assert sshape[1] * 16 == wshape[1], (sshape, wshape)  # complete 32-blocks only
            est['m'].append(do_matrix(wraw, sraw, wshape[0], acc, g, m))
        stats[e['file']] = est
        bad = [(e['file'], m, s) for m, s in enumerate(est['m']) if not s.get('counted')]
        print(f"{n + 1}/{len(todo)} {e['file']} even% {100 * sum(s['even'] for s in est['m']) / sum(s['blocks'] for s in est['m']):.2f} "
              f"lt6 {sum(s['max_lt6'] for s in est['m'])} rt {all(s['rt_ok'] for s in est['m'])} "
              f"smax {max(s['scale_max'] for s in est['m'])} {'BAD ' + str(bad) if bad else ''} {time.time() - t0:.0f}s", flush=True)
        if (n + 1) % 10 == 0 or n + 1 == len(todo):
            np.savez_compressed(cp + '.tmp.npz', **acc)
            json.dump(stats, open(sp + '.tmp', 'w'))
            os.replace(cp + '.tmp.npz', cp)
            os.replace(sp + '.tmp', sp)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
