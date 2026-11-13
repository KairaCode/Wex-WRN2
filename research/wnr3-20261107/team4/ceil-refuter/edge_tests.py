"""Edge cases of the ceil-form bijection on synthetic blocks + a smoke run of do_matrix on one real matrix.
Usage: python -I edge_tests.py SAMPLE_DIR OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refute_count as rc


def pack(nibbles):
    n = np.asarray(nibbles, np.uint8).reshape(-1, 32)
    return (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)


def case(name, nibbles, scale):
    b = pack(nibbles)
    s = np.array([scale], np.uint8) if scale <= 255 else None
    tw, ts, even, bmax = rc.T_fwd(b, s)
    out = {'case': name, 'stored_max': int(bmax[0]), 'even': bool(even[0]), 't_scale': int(ts[0]),
           'scale_fits_u8': bool(ts[0] <= 255)}
    iw, isc, dec = rc.T_inv(tw, ts)
    out['rt_ok'] = bool(np.array_equal(iw, b) and np.array_equal(isc, s.astype(np.int16)))
    out['t_nibbles'] = rc.unpack(tw)[0].tolist()[:8]
    # canonical zero on top: Z(T(x)) then inverse -> Z(x)?
    zt = rc.LUT_Z[tw]
    izw, izs, _ = rc.T_inv(zt, ts)
    out['z_rt_ok'] = bool(np.array_equal(izw, rc.LUT_Z[b]) and np.array_equal(izs, s.astype(np.int16)))
    return out


def main():
    sample_dir, out_json = sys.argv[1:3]
    rng = np.random.default_rng(1)
    res = []
    res.append(case('all +0', [0] * 32, 120))
    res.append(case('all -0', [8] * 32, 120))
    res.append(case('only +-0 and +-6 (codes 0,8,7,15)', [0, 8, 7, 15] * 8, 120))
    res.append(case('only +-0 and +-4 (codes 0,8,6,14)', [0, 8, 6, 14] * 8, 120))
    res.append(case('single 6.0, rest 0', [7] + [0] * 31, 120))
    res.append(case('even block scale 0', [7, 2, 4, 5, 6, 0] * 5 + [0, 0], 0))
    res.append(case('even block scale 253', [7, 2, 4, 5, 6, 0] * 5 + [0, 0], 253))
    res.append(case('even block scale 254 -> 255 (E8M0 NaN byte as coded symbol)', [7, 2, 4, 5, 6, 0] * 5 + [0, 0], 254))
    res.append(case('even block scale 255 (NaN) -> 256 overflow', [7, 2, 4, 5, 6, 0] * 5 + [0, 0], 255))
    res.append(case('non-even max7 (has 1)', [7, 1] + [0] * 30, 120))
    res.append(case('stored max 5 block (3.0) = NOT ceil-safe', [5, 4, 2, 1] + [0] * 28, 120))
    res.append(case('stored max 4 block (2.0)', [4, 2, 1] + [0] * 29, 120))
    res.append(case('stored max 3 block', [3, 1] + [0] * 30, 120))
    res.append(case('random max6 block', list(rng.integers(0, 7, 31)) + [6], 120))
    # collision: is there any canonical (stored) block equal to the halved image of an even block? Images have max 5,
    # stored blocks have max 6/7 (invariant), so T is injective iff no stored block has max 5. Exhaustive check on the
    # magnitude alphabet of one nibble position class is impossible (16^32); argue by max instead and verify on random:
    nrand = 200000
    nib = rng.integers(0, 16, (nrand, 32)).astype(np.uint8)
    # force some even blocks: map magnitudes 1->2, 3->5 with prob 0.5 and put a 7 in
    sel = rng.random(nrand) < 0.5
    mag = nib & 7; sg = nib & 8
    mag2 = np.where(sel[:, None], np.array([0, 2, 2, 5, 4, 5, 6, 7])[mag], mag)
    mag2[sel, 0] = 7
    nib = (mag2 | sg).astype(np.uint8)
    b = pack(nib.ravel())
    s = rng.integers(100, 140, nrand).astype(np.uint8)
    tw, ts, even, bmax = rc.T_fwd(b, s)
    ok_blocks = bmax != 5
    iw, isc, dec = rc.T_inv(tw, ts)
    good = np.all(iw == b, 1) & (isc == s)
    res.append({'case': f'random {nrand} blocks (half forced even)', 'even': int(even.sum()), 'stored_max5': int((~ok_blocks).sum()),
                'rt_fail_on_max5_blocks': int((~good & ~ok_blocks).sum()), 'rt_fail_on_other_blocks': int((~good & ok_blocks).sum())})
    # smoke: one real matrix
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    e = man['experts'][0]
    raw = np.fromfile(os.path.join(sample_dir, e['file']), dtype=np.uint8)
    off = 0; parts = {}
    for t in e['tensors']:
        parts[t['name']] = raw[off:off + t['bytes']]; off += t['bytes']
    acc = rc.new_acc()
    st = rc.do_matrix(parts['w2.weight'], parts['w2.scale'], 5120, acc, 0, 1)
    st.pop('scale_hist_top')
    res.append({'case': 'smoke ' + e['file'] + ' w2', **st,
                'o_base_total': int(acc['o_base'][0, 1].sum()), 'c_base_total': int(acc['c_base'][0, 1].sum()),
                'o_clsh_pm2b_total': int(acc['o_clsh_pm2b'][0, 1].sum())})
    json.dump(res, open(out_json, 'w'), indent=1)
    for r in res:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
