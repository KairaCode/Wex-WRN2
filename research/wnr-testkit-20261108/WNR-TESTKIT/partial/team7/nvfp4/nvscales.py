"""E4M3 block-scale entropy with richer neighbour contexts, plus 2D-tile (16x16) replication tests.

Usage: python -I nvscales.py FAM_DIR OUT_JSON
Contexts (shared static model per matrix type, 14-bit CDFs, held-out by item parity):
  o0       order-0 byte
  left     exact left byte (256 = row start)
  lx       exact left byte x clamp(up - left, -2, 2) (+ none) -> 257 x 6 rows
  med      LOCO-I MED predictor p = med(left, up, left + up - upleft); symbol = (s - p) mod 256;
           context = |left - up| bucket {0,1,2,3-4,5-8,9+} x (p - anchor) bucket {<-8,-8..-3,-2..2,3..8,>8} (+ edge row)
  par      parity of block index x left byte (pairs of 16-blocks forming a 32-group)
Tile tests: P(s == up) for rows r % 16 != 0 vs r % 16 == 0; P(s == left) for even->odd vs odd->even block pairs.
"""
import importlib.util, json, os, sys
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)


def bc(idx, n):
    return np.bincount(idx.ravel(), minlength=n)[:n]


def ab(x, edges):
    return np.searchsorted(np.array(edges), x, side='right')


def counts(sb):
    s = sb.astype(np.int32)
    R, B = s.shape
    sv = nc.e4m3_value(sb)
    live = sv > 0
    ls = np.sort(s[live]) if live.any() else np.sort(s.ravel())
    anchor = int(ls[(ls.size - 1) // 2])
    left = np.full_like(s, 256); left[:, 1:] = s[:, :-1]
    up = np.full_like(s, 256); up[1:] = s[:-1]
    ul = np.full_like(s, 256); ul[1:, 1:] = s[:-1, :-1]
    r = {}
    r['o0'] = bc(s, 256)[None, :]
    r['left'] = bc(left * 256 + s, 257 * 256).reshape(257, 256)
    du = np.where((up == 256) | (left == 256), 5, np.clip(up - left, -2, 2) + 2)
    r['lx'] = bc((left * 6 + du) * 256 + s, 257 * 6 * 256).reshape(257 * 6, 256)
    lf = np.where(left == 256, up, left); uf = np.where(up == 256, lf, up); ulf = np.where(ul == 256, uf, ul)
    lf = np.where(lf == 256, anchor, lf); uf = np.where(uf == 256, anchor, uf); ulf = np.where(ulf == 256, anchor, ulf)
    g = lf + uf - ulf
    p = np.maximum(np.minimum(lf, uf), np.minimum(np.maximum(lf, uf), g))
    res = (s - p) % 256
    act = ab(np.abs(lf - uf), [0, 1, 2, 4, 8])  # 0:0 1:1 2:2 3:3-4 4:5-8 5:9+
    pc = ab(p - anchor, [-9, -3, 2, 8])
    edge = (left == 256) | (up == 256)
    ctx = np.where(edge, 30, act * 5 + pc)
    r['med'] = bc(ctx * 256 + res, 31 * 256).reshape(31, 256)
    par = np.broadcast_to((np.arange(B) & 1)[None, :], (R, B))
    r['par'] = bc((par * 257 + left) * 256 + s, 514 * 256).reshape(514, 256)
    rows = np.arange(R)
    in_tile = (rows % 16 != 0) & (rows > 0)
    at_b = (rows % 16 == 0) & (rows > 0)
    eq_up = (s[1:] == s[:-1])
    t = {'eq_up_in_tile': float(eq_up[in_tile[1:]].mean()), 'eq_up_tile_boundary': float(eq_up[at_b[1:]].mean()) if at_b.any() else None,
         'eq_left_even_to_odd': float((s[:, 1::2] == s[:, 0::2]).mean()),
         'eq_left_odd_to_even': float((s[:, 2::2] == s[:, 1:-1:2]).mean()) if B > 2 else None}
    return r, t, anchor


def main():
    d, out = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(d, 'manifest.json')))
    per = []
    tiles = []
    for i, it in enumerate(man['experts']):
        acc = {}
        for m, (role, codes, sb, ts, isc, nb) in enumerate(nc.matrices(d, it)):
            r, t, a = counts(sb)
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
            tiles.append({'tag': it['tag'], 'role': role, 'anchor': a, **t})
        per.append({'split': i % 2, **{k: np.stack(v).astype(np.int64) for k, v in acc.items()}})
    S = int(sum(x['o0'].sum() for x in per))
    res = {'repo': man['repo'], 'items': len(per), 'scales': S}
    for k in ['o0', 'left', 'lx', 'med', 'par']:
        a = sum(x[k] for x in per if x['split'] == 0)
        b = sum(x[k] for x in per if x['split'] == 1)
        al = a + b
        res[k] = {'insample_bit_per_scale': nc.cost_bits(al, nc.quantize(al)) / S,
                  'heldout_bit_per_scale': (nc.cost_bits(b, nc.quantize(a)) + nc.cost_bits(a, nc.quantize(b))) / S,
                  'ideal_insample': nc.ideal_bits(al) / S, 'rows_per_matrix': int(al.shape[1])}
    keys = ['eq_up_in_tile', 'eq_up_tile_boundary', 'eq_left_even_to_odd', 'eq_left_odd_to_even']
    res['tile_tests_mean'] = {k: float(np.mean([t[k] for t in tiles if t[k] is not None])) for k in keys}
    res['tile_tests_first'] = tiles[:6]
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
