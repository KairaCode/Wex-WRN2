"""team2/audit: verify with the SCALAR decoder that some single-bit flips are accepted by today's rule
(final state == L, exact length) while decoding wrong symbols; characterise them; check the x0 = L + h rule.

Usage: python -I audit_resync.py SAMPLE_DIR RANS_DIR OUT_JSON [expert_index] [matrix] [streams]
"""
import json, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import cum_of, encode_vec, decode_vec, inv_table, scalar_decode, L
from audit_count import read_expert
from audit_corrupt import streams_of


def h23(syms, idx):
    return zlib.crc32(bytes(int(s) for s in syms) + int(idx).to_bytes(4, 'little')) & ((1 << 23) - 1)


def main():
    sample_dir, rans_dir, out_json = sys.argv[1:4]
    ei = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    m = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    per = int(sys.argv[6]) if len(sys.argv) > 6 else 5760
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    md = np.load(os.path.join(rans_dir, 'model.npz'))
    wf = md['wf']; wc = cum_of(wf); inv = inv_table(wf[m])
    tens, _ = read_expert(sample_dir, man['experts'][ei])
    sym, ctx, srel = streams_of(tens, m, wf, wc)
    sym = sym[:per].astype(np.int64); ctx = ctx[:per]; srel = srel[:per]
    N = sym.shape[0]
    F = wf[m][ctx, sym]; C = wc[m][ctx, sym]
    lens, _, buf = encode_vec(F, C, want_bytes=True)
    rng = np.random.default_rng(99)
    ar = np.arange(N)
    out = {'expert': man['experts'][ei]['file'], 'matrix': m, 'streams': int(N)}
    rows = []
    reps = 4
    acc_total = 0
    for rep in range(reps):
        b = buf.copy()
        j = (rng.random(N) * lens).astype(np.int64)
        bit = rng.integers(0, 8, N)
        b[ar, j] ^= (1 << bit).astype(np.uint8)
        v, ds, xf = decode_vec(b, lens, 2048, srel, wf[m], wc[m], inv)
        acc = np.nonzero(v)[0]
        acc_total += acc.size
        for k in acc:
            bb = bytes(b[k, :lens[k]])
            d = scalar_decode(bb, 2048, srel[k], wf[m].tolist(), cum_of(wf[m]).tolist())
            diff = np.nonzero(np.asarray(d) != sym[k])[0] if d is not None else None
            # what the hash rule would see: encoder used x0 = L + h(orig); decoder recomputes h from decoded symbols
            rows.append({'stream': int(k), 'byte': int(j[k]), 'len': int(lens[k]), 'bit': int(bit[k]),
                         'scalar_accepts': d is not None,
                         'n_symbols_differ': None if diff is None else int(diff.size),
                         'first_diff': None if diff is None or diff.size == 0 else int(diff[0]),
                         'last_diff': None if diff is None or diff.size == 0 else int(diff[-1]),
                         'hash_rule_detects': None if d is None else bool(h23(d, k) != h23(sym[k], k))})
    out['trials'] = int(N * reps)
    out['accepted_by_today_rule'] = int(acc_total)
    out['accept_rate'] = acc_total / (N * reps)
    out['scalar_decoder_agrees'] = int(sum(r['scalar_accepts'] for r in rows))
    nd = [r['n_symbols_differ'] for r in rows if r['n_symbols_differ'] is not None]
    out['symbols_differ_hist'] = {str(k): int(v) for k, v in zip(*np.unique(nd, return_counts=True))} if nd else {}
    out['hash_rule_detects_all_accepted'] = bool(all(r['hash_rule_detects'] for r in rows if r['hash_rule_detects'] is not None))
    out['byte_pos_frac_of_len'] = [round(r['byte'] / r['len'], 3) for r in rows][:60]
    out['examples'] = rows[:25]
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1))
    for r in rows[:12]:
        print(r)


if __name__ == '__main__':
    main()
