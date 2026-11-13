"""team5 / conf-encoder: all single-bit flips of the first N weight streams of one record: count structurally accepted
flips (exact length, final state == x0) and check each against the H32 rule (h23 of the decoded coded bytes with the
stream's ident must equal the original h23). Usage: python -I signflip_all.py RECORDS_DIR NSTREAMS OUT_JSON"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W


def main():
    rdir, ns, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    idx = json.load(open(os.path.join(rdir, 'index.json')))
    row = idx['records'][0]
    model = W.parse_wnm3(open(os.path.join(rdir, idx['model_file']), 'rb').read())
    rec = open(os.path.join(rdir, row['record_file']), 'rb').read()
    slab, info = W.decode_record(rec, model, row['entry_flags'], tuple(row['bind']), [tuple(s) for s in row['shapes_symbols']])
    H = W.unpack_header(rec[:64])
    tab = W.RecTables(model, info['set_id'])
    SS, wbeg = info['SS'], info['wbeg']
    res = []
    for j in [int(x) for x in np.linspace(0, info['WS'] - 1, ns).round()]:
        m = int(np.searchsorted(np.cumsum(info['nws']), j, side='right'))
        o, n = int(info['woffs'][j]), int(info['wlens'][j])
        sb = np.frombuffer(rec, np.uint8, n, wbeg + o)
        nb = 8 * n
        lane = n + 2 * W.CW + 64
        pl = np.zeros(nb * lane, np.uint8)
        pl.reshape(nb, lane)[:, :n] = sb
        bit = np.arange(nb)
        pl.reshape(nb, lane)[bit, bit // 8] ^= (1 << (bit % 8)).astype(np.uint8)
        offs = np.arange(nb, dtype=np.int64) * lane
        wT, x, ptr, ok = W.decode_weights(pl, offs, np.repeat(info['srelT'][:, j:j + 1], nb, 1),
                                          np.full(nb, m * W.FULLCTX, np.int64), tab)
        orig = info['coded_weights'][j]
        want = int(info['wantW'][j])
        acc = np.flatnonzero(ok & (ptr == offs + n) & (x == want))
        cw = wT.T[acc]
        diff = cw != orig[None, :]
        one_sign = (diff.sum(1) == 1) & (((cw ^ orig[None, :]) == 8) | ~diff).all(1)
        cb = np.ascontiguousarray(cw[:, 0::2] | (cw[:, 1::2] << 4))
        ident = W.stream_idents(H['model_id'], H['owner_layer'], H['owner_expert'], np.full(acc.size, SS + j),
                                np.full(acc.size, (H['ceil_mask'] >> m) & 1), H['flags'], np.full(acc.size, m),
                                np.full(acc.size, info['set_id'][m]), np.full(acc.size, info['anchor'][m]))
        h = (W.stream_h32(ident, cb) & W.U32(W.HMASK)).astype(np.int64) + W.L if acc.size else np.zeros(0)
        hash_accepts = int((h == want).sum())
        res.append({'g': SS + j, 'm': m, 'bytes': n, 'flips': nb, 'struct_accepted': int(acc.size),
                    'struct_accepted_one_sign_flip': int(one_sign.sum()), 'hash_accepted': hash_accepts})
        print(res[-1], flush=True)
    tot = {k: sum(r[k] for r in res) for k in ('flips', 'struct_accepted', 'struct_accepted_one_sign_flip', 'hash_accepted')}
    tot['struct_accept_rate'] = tot['struct_accepted'] / tot['flips']
    by_m = {f'w{m+1}': {k: sum(r[k] for r in res if r['m'] == m) for k in ('flips', 'struct_accepted', 'hash_accepted')} for m in range(3)}
    json.dump({'record': row['record_file'], 'streams': res, 'total': tot, 'by_matrix': by_m}, open(out, 'w'), indent=1)
    print(tot, by_m)


if __name__ == '__main__':
    main()
