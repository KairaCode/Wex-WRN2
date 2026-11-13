"""ans-prior-art: anatomy of single-bit flips that today's WNR2 checks accept silently.
For each accepted case: original vs decoded symbol, their frequencies in that context, and the CDF distance.
Hypothesis: s and s^8 (same magnitude, opposite sign) often have equal frequency, and if their CDF starts differ
by a power of two 2^b, a flip that lands in bit b of the slot turns s into s^8 and leaves the state unchanged.

Usage: python -I corrupt2.py SAMPLE_DIR COUNTS_DIR OUT_JSON [trials] [streams]
"""
import json, os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flushx import quantize, load_expert, matrix_syms_ctx, Model, enc_core, dec_core, L, W_CHUNK


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    nstr = int(sys.argv[5]) if len(sys.argv) > 5 else 240
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    cw = None
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        cw = z['w_base'].astype(np.int64) if cw is None else cw + z['w_base']
    freq = quantize(cw)
    models = [Model(freq[mi]) for mi in range(3)]
    # static property of the model: how many (context, s) pairs have f[s] == f[s^8] and a power-of-two CDF distance
    cum = np.zeros((3, 567, 17), np.int64); cum[..., 1:] = np.cumsum(freq, -1)
    eq = pow2 = 0
    for mi in range(3):
        for c in range(567):
            for s in range(8):
                if freq[mi, c, s] == freq[mi, c, s + 8]:
                    eq += 1
                    dist = int(cum[mi, c, s + 8] - cum[mi, c, s])
                    pow2 += dist > 0 and (dist & (dist - 1)) == 0
    rng = random.Random(4242)
    streams = []
    for rec in rng.sample(range(len(man['experts'])), 4):
        e = man['experts'][rec]
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for mi, w in enumerate(['w1', 'w2', 'w3']):
            sym, ctx, srel = matrix_syms_ctx(t[f'{w}.weight'][0], t[f'{w}.weight'][1], t[f'{w}.scale'][0])
            ns = sym.size // W_CHUNK
            for si in rng.sample(range(ns), nstr // 12):
                a = si * W_CHUNK
                S = (rec, mi, si, sym[a:a + W_CHUNK].tolist(), ctx[a:a + W_CHUNK].tolist(), srel[a:a + W_CHUNK].tolist())
                streams.append((S, enc_core(S[3], S[4], models[mi], L)))
    acc = []
    reached = 0
    for tr in range(trials):
        S, (words, xf) = streams[rng.randrange(len(streams))]
        m = models[S[1]]
        buf = bytearray(xf.to_bytes(4, 'big') + bytes(words))
        p = rng.randrange(len(buf) * 8); buf[p // 8] ^= 1 << (p % 8)
        xs = int.from_bytes(buf[:4], 'big')
        d = None if not (L <= xs < 256 * L) else dec_core(list(buf[4:]), xs, S[5], W_CHUNK, m)
        if d is None:
            continue
        reached += 1
        if d[1] != L:
            continue
        diff = [i for i in range(W_CHUNK) if d[0][i] != S[3][i]]
        rows = []
        for i in diff[:3]:
            c = S[4][i]; so = S[3][i]; sw = d[0][i]
            rows.append({'pos': i, 'orig': so, 'decoded': sw, 'sign_flip': (so ^ 8) == sw,
                         'f_orig': m.f[c][so], 'f_dec': m.f[c][sw], 'cdf_dist': m.c[c][sw] - m.c[c][so]})
        acc.append({'byte': p // 8, 'bit': p % 8, 'n_wrong': len(diff), 'wrong': rows})
    out = {'trials': trials, 'reached_final_check': reached, 'accepted': len(acc),
           'accepted_all_single_sign_flips': all(a['n_wrong'] == 1 and a['wrong'][0]['sign_flip'] for a in acc),
           'model_pairs_f_s_eq_f_s^8': eq, 'of_which_pow2_cdf_distance': pow2, 'model_pairs_total': 3 * 567 * 8,
           'cases': acc}
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'cases'}, indent=1))
    for a in acc[:20]:
        print(a)


if __name__ == '__main__':
    main()
