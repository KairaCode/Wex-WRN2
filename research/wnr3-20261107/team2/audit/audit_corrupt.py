"""team2/audit: how well today's WNR2 structural rule (initial state in [L,256L), exact length, final state == L)
detects damage, on many real V4.1 weight streams (vectorized decoder, audit_core).

Usage: python -I audit_corrupt.py SAMPLE_DIR RANS_DIR OUT_JSON [streams_per_expert] [experts]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import cum_of, encode_vec, decode_vec, inv_table, L
from audit_count import read_expert, unpack, lower_median, block_contexts


def streams_of(tens, m, wf, wc):
    w = ('w1', 'w2', 'w3')[m]
    wb, _ = tens[f'{w}.weight']; sc, _ = tens[f'{w}.scale']
    med, _ = lower_median(sc)
    srel = (np.clip(sc.astype(np.int32) - med, -3, 3) + 3).astype(np.int32)
    sym = unpack(wb).reshape(-1, 32)
    ctx, _, _ = block_contexts(sym, srel)
    ns = sym.size // 2048
    return sym.reshape(ns, 2048), ctx.reshape(ns, 2048).astype(np.int64), srel.reshape(ns, 64)


def main():
    sample_dir, rans_dir, out_json = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    nex = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    md = np.load(os.path.join(rans_dir, 'model.npz'))
    wf = md['wf']; wc = cum_of(wf)
    invs = [inv_table(wf[m]) for m in range(3)]
    rng = np.random.default_rng(20261107)
    picks = np.linspace(3, len(man['experts']) - 1, nex).astype(int)
    out = {'experts': [man['experts'][i]['file'] for i in picks], 'streams_per_expert': per}
    for m in (0, 1):
        S, CT, SR, SR_next = [], [], [], []
        for i in picks:
            tens, _ = read_expert(sample_dir, man['experts'][i])
            sym, ctx, srel = streams_of(tens, m, wf, wc)
            k = rng.choice(sym.shape[0] - 1, per, replace=False)
            S.append(sym[k]); CT.append(ctx[k]); SR.append(srel[k]); SR_next.append(srel[k + 1])
        sym = np.concatenate(S).astype(np.int64); ctx = np.concatenate(CT); srel = np.concatenate(SR); srel_n = np.concatenate(SR_next)
        N = sym.shape[0]
        F = wf[m][ctx, sym]; C = wc[m][ctx, sym]
        lens, _, buf = encode_vec(F, C, want_bytes=True)
        Wd = buf.shape[1]
        ar = np.arange(N)
        res = {}
        t0 = time.time()

        def run(name, b, ln, sr=srel, tab=m, expect_same=None):
            v, ds, _ = decode_vec(b, ln, 2048, sr, wf[tab], wc[tab], invs[tab])
            same = (ds == sym).all(1)
            res[name] = {'trials': int(N), 'accepted': int(v.sum()), 'detected_frac': float(1 - v.mean()),
                         'accepted_with_wrong_symbols': int((v & ~same).sum())}

        run('clean (must all be accepted)', buf, lens)
        # A: one random bit anywhere in the stream (state bytes included)
        b = buf.copy(); j = (rng.random(N) * lens).astype(np.int64); b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
        run('A bitflip anywhere', b, lens)
        b = buf.copy(); j = lens - 1; b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
        run('B bitflip in last byte', b, lens)
        b = buf.copy(); j = rng.integers(0, 4, N); b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
        run('C bitflip in 4-B initial state', b, lens)
        b = buf.copy(); j = (rng.random(N) * (lens - 1)).astype(np.int64)
        for d in (0, 1):
            b[ar, j + d] = rng.integers(0, 256, N).astype(np.uint8)
        changed = (b != buf).any(1)
        run('D random 2-byte overwrite', b, lens)
        res['D random 2-byte overwrite']['unchanged_trials'] = int((~changed).sum())
        b = buf.copy(); j = 4 + (rng.random(N) * (lens - 4)).astype(np.int64)
        b[np.arange(Wd)[None, :] >= j[:, None]] = 0
        b[np.arange(Wd)[None, :] >= lens[:, None]] = 0
        res_z = (b != buf).any(1)
        run('E zero-fill from random offset to end (torn write)', b, lens)
        res['E zero-fill from random offset to end (torn write)']['unchanged_trials'] = int((~res_z).sum())
        run('F length - 1 (truncated)', buf, lens - 1)
        b = buf.copy(); b[ar, lens] = rng.integers(0, 256, N).astype(np.uint8)
        run('G length + 1 (one extra byte)', b, lens + 1)
        run('H valid stream decoded at the next index (next stream srel, own length)', buf, lens, sr=srel_n)
        same_srel = (srel == srel_n).all(1)
        res['H valid stream decoded at the next index (next stream srel, own length)']['identical_srel_pairs'] = int(same_srel.sum())
        run('I valid stream at wrong index, IDENTICAL contexts (rans_check.py test)', buf, lens)
        run('J valid stream decoded with another matrix type model (w%d -> w3 model)' % (m + 1), buf, lens, tab=2)
        res['_seconds'] = time.time() - t0
        out[f'matrix_w{m + 1}'] = res
        print(json.dumps(res, indent=1), flush=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
