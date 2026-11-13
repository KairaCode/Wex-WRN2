"""Checks the lock-step NumPy rANS against a scalar copy of rans_check.py's encode/decode (the exact WNR2 rule).

Usage: python -I test_rans.py SAMPLE_DIR NPZ_DIR
1. Real streams (expert index 17, w1 and w2): the vectorised encoder must give the same bytes as the scalar
   reference encoder, and the vectorised decoder must give back the symbols with final state L (or L + h).
2. Adversarial streams (uniform random symbols under a skewed model, many 2-byte renormalisations).
3. Length-table pack/unpack round trip incl. extreme residuals.
"""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W

L, PB, P = W.L, W.PB, W.P


# ---- scalar reference: copied logic of wnr2 tools rans_check.py (encode / decode), not imported
def ref_encode(syms, ctxs, freq, cum, x0):
    x = x0
    out = bytearray()
    for s, c in zip(reversed(syms), reversed(ctxs)):
        f = int(freq[c, s]); st = int(cum[c, s])
        x_max = ((L >> PB) << 8) * f
        while x >= x_max:
            out.append(x & 255)
            x >>= 8
        x = ((x // f) << PB) + (x % f) + st
    out += bytes([(x >> 0) & 255, (x >> 8) & 255, (x >> 16) & 255, (x >> 24) & 255])
    out.reverse()
    return bytes(out)


def ref_decode(buf, ctx_fn, n, freq, cum):
    x = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
    pos = 4
    syms = []
    for i in range(n):
        c = ctx_fn(i, syms)
        slot = x & (P - 1)
        row = cum[c]
        s = int(np.searchsorted(row, slot, side='right') - 1)
        x = int(freq[c, s]) * (x >> PB) + slot - int(row[s])
        while x < L:
            if pos >= len(buf):
                return None, None
            x = (x << 8) | buf[pos]
            pos += 1
        syms.append(s)
    if pos != len(buf):
        return None, None
    return syms, x


def main():
    sample_dir, npz_dir = sys.argv[1:3]
    man = W.load_manifest(sample_dir)
    tot = None
    for e in man['experts']:
        z = np.load(os.path.join(npz_dir, e['file'].replace('.bin', '.npz')))
        tot = z['w_base'].astype(np.int64) if tot is None else tot + z['w_base']
    freq_all = W.quantize(tot)
    res = {}
    e = man['experts'][17]
    raw, mats = W.load_expert(sample_dir, e)
    rng = random.Random(7)
    for m in (0, 1):
        f = W.block_features(mats[m]['sym'], mats[m]['sc'])
        freq = freq_all[m].astype(np.int64)
        cum = np.zeros((W.NCTX, 17), np.int64); cum[:, 1:] = np.cumsum(freq, 1)
        nstr = f['srel'].size // W.BPS
        picks = np.array(sorted(rng.sample(range(nstr), 48)))
        blk = f['blk'].reshape(nstr, W.BPS * 32)[picks].astype(np.int64)
        ctx = f['base'].reshape(nstr, W.BPS * 32)[picks].astype(np.int64)
        srel = f['srel'].reshape(nstr, W.BPS)[picks]
        hseed = np.array([rng.randrange(1 << 23) for _ in picks])
        for tag, x0 in (('x0=L', np.full(picks.size, L)), ('x0=L+h', L + hseed)):
            FT = freq[ctx, blk].T.copy(); CT = cum[ctx, blk].T.copy()
            pay, lens = W.rans_encode_lockstep(FT, CT, x0)
            offs = np.zeros(picks.size, np.int64); offs[1:] = np.cumsum(lens)[:-1]
            same = 0
            for i in range(picks.size):
                rb = ref_encode(blk[i].tolist(), ctx[i].tolist(), freq, cum, int(x0[i]))
                same += rb == pay[offs[i]:offs[i] + lens[i]].tobytes()
            # vectorised decode (context computed by the decoder itself)
            tab = W.Tables(np.concatenate([freq_all[k] for k in range(3)]), np.full((3, 256), 64))
            pl = np.concatenate([pay, np.zeros(5000, np.uint8)])
            out, ok = W.decode_weights_lockstep(pl, offs, lens, srel.T.copy(), np.full(picks.size, m * W.NCTX),
                                                 np.zeros(picks.size, np.int64), tab, False,
                                                 None)
            dec_ok = bool((out.T.astype(np.int64) == blk).all())
            # decoder final state == L check only applies to x0 = L
            fin_ok = bool(ok.all()) if tag == 'x0=L' else bool((~ok).all())
            # scalar decode of the first 4 vectorised streams
            sc_ok = 0
            for i in range(4):
                def ctx_fn(t, dec, i=i):
                    if t % 32 == 0:
                        return (int(srel[i, t // 32]) * 9 + 8) * 9 + 8
                    mm = [d & 7 for d in dec[t - (t % 32):t]]
                    return (int(srel[i, t // 32]) * 9 + mm[-1]) * 9 + max(mm)
                d, xf = ref_decode(pay[offs[i]:offs[i] + lens[i]].tobytes(), ctx_fn, 2048, freq, cum)
                sc_ok += (d == blk[i].tolist()) and xf == int(x0[i])
            res[f'm{m} {tag}'] = {'streams': int(picks.size), 'bytes_equal_to_scalar_ref': int(same),
                                  'vector_decode_symbols_equal': dec_ok,
                                  'final_state_rule_as_expected': fin_ok, 'scalar_decode_of_vector_bytes_ok': f'{sc_ok}/4',
                                  'mean_len': float(lens.mean())}
    # adversarial: uniform symbols, skewed model (one symbol gets most of the mass)
    nrng = np.random.default_rng(1)
    cnt = np.ones((3 * W.NCTX, 16), np.int64); cnt[:, 0] = 10 ** 6
    tabx = W.Tables(W.quantize(cnt), np.full((3, 256), 64))
    S = 40
    blk = nrng.integers(0, 16, (S, 2048))
    # contexts as the decoder will compute them, srel random
    srel = nrng.integers(0, 7, (S, 64))
    b3 = blk.reshape(S, 64, 32)
    mag = b3 & 7
    cm = np.maximum.accumulate(mag, 2)
    pm = np.concatenate([np.full((S, 64, 1), 8), mag[..., :-1]], 2)
    mx = np.concatenate([np.full((S, 64, 1), 8), cm[..., :-1]], 2)
    ctx = ((srel[..., None] * 9 + pm) * 9 + mx).reshape(S, 2048)
    FT = tabx.wf[ctx, blk].T.copy(); CT = tabx.wc[ctx, blk].T.copy()
    pay, lens = W.rans_encode_lockstep(FT, CT, np.full(S, L))
    offs = np.zeros(S, np.int64); offs[1:] = np.cumsum(lens)[:-1]
    same = sum(ref_encode(blk[i].tolist(), ctx[i].tolist(), tabx.wf, np.concatenate([tabx.wc, np.full((tabx.wc.shape[0], 1), P)], 1), L)
               == pay[offs[i]:offs[i] + lens[i]].tobytes() for i in range(8))
    pl = np.concatenate([pay, np.zeros(5000, np.uint8)])
    out, ok = W.decode_weights_lockstep(pl, offs, lens, srel.T.copy(), np.zeros(S, np.int64), np.zeros(S, np.int64), tabx, False)
    res['adversarial'] = {'streams': S, 'mean_len': float(lens.mean()), 'max_len': int(lens.max()), 'bound_2n+8': 2 * 2048 + 8,
                          'bytes_equal_to_scalar_ref(first 8)': int(same), 'decode_ok': bool(ok.all() and (out.T == blk).all())}
    # length table pack/unpack
    segs = [nrng.integers(-40, 40, 720), nrng.integers(-3, 3, 23), np.array([0] * 31 + [65535]), nrng.integers(-200, 200, 5760),
            np.zeros(64, np.int64)]
    buf, nb = W.pack_groups(segs)
    back, used = W.unpack_groups(buf, [s.size for s in segs])
    res['length_pack'] = {'roundtrip': all((a == b).all() for a, b in zip(segs, back)), 'bytes': len(buf), 'used': used}
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
