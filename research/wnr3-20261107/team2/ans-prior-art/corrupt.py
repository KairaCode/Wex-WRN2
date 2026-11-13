"""ans-prior-art: how often does a corrupted WNR2 stream pass today's structural checks (initial state in
[L, 256L), exact length, final state == L)? And with x0 = L + h23(decoded symbols, index, record)?

Usage: python -I corrupt.py SAMPLE_DIR COUNTS_DIR OUT_JSON [trials_per_kind] [streams]
Kinds: 1-bit flip anywhere in [state|payload]; random byte replacement; 4-byte zeroed burst.
For every accepted (= silently wrong) decode we record where the damage was and how many symbols differ.
"""
import json, os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flushx import quantize, load_expert, matrix_syms_ctx, Model, enc_core, dec_core, h_bits, L, W_CHUNK


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    nstr = int(sys.argv[5]) if len(sys.argv) > 5 else 240
    t0 = time.time()
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    cw = None
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        cw = z['w_base'].astype(np.int64) if cw is None else cw + z['w_base']
    freq = quantize(cw)
    models = [Model(freq[mi]) for mi in range(3)]
    rng = random.Random(777)
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
                m = models[mi]
                wL, xL = enc_core(S[3], S[4], m, L)
                x0h = L + h_bits(S[3], si, rec, 23)
                wH, xH = enc_core(S[3], S[4], m, x0h)
                streams.append((S, (wL, xL), (wH, xH)))
    print(len(streams), 'streams', f'{time.time() - t0:.0f}s', flush=True)
    out = {}
    for kind in ('bitflip', 'byte', 'zero4'):
        for rule in ('today_x0_L', 'hash_x0_L_plus_h23'):
            struct = reached = acc = 0
            examples = []
            for tr in range(trials):
                S, encL, encH = streams[rng.randrange(len(streams))]
                words, xf = encL if rule == 'today_x0_L' else encH
                m = models[S[1]]
                buf = bytearray(xf.to_bytes(4, 'big') + bytes(words))
                if kind == 'bitflip':
                    p = rng.randrange(len(buf) * 8); buf[p // 8] ^= 1 << (p % 8); where = p // 8
                elif kind == 'byte':
                    where = rng.randrange(len(buf)); old = buf[where]
                    buf[where] = (old + 1 + rng.randrange(255)) & 255
                else:
                    where = rng.randrange(len(buf) - 3)
                    if bytes(buf[where:where + 4]) == b'\0\0\0\0':
                        buf[where] = 1
                    buf[where:where + 4] = b'\0\0\0\0'
                xs = int.from_bytes(buf[:4], 'big')
                d = None if not (L <= xs < 256 * L) else dec_core(list(buf[4:]), xs, S[5], W_CHUNK, m)
                if d is None:
                    struct += 1
                    continue
                reached += 1
                exp = L if rule == 'today_x0_L' else L + h_bits(d[0], S[2], S[0], 23)
                if d[1] == exp:
                    acc += 1
                    diff = [i for i in range(W_CHUNK) if d[0][i] != S[3][i]]
                    if len(examples) < 12:
                        examples.append({'damaged_byte': where, 'stream_bytes': len(buf), 'n_wrong_symbols': len(diff),
                                         'first_wrong': diff[0] if diff else None, 'last_wrong': diff[-1] if diff else None})
            out[f'{kind}/{rule}'] = {'trials': trials, 'caught_structural': struct, 'reached_final_check': reached,
                                     'ACCEPTED_SILENTLY': acc, 'examples': examples}
            print(kind, rule, {k: v for k, v in out[f'{kind}/{rule}'].items() if k != 'examples'},
                  examples[:4], f'{time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
