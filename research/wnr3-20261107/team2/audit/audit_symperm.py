"""team2/audit: silent sign-flip acceptance vs CDF symbol order.
Baseline order: codes 0..15 (sign = bit 3, so +x and -x sit P/2 apart in a sign-symmetric row).
Interleaved order: coded label = 2*magnitude + sign (+0,-0,+0.5,-0.5,...).
Single-bit flips anywhere in the stream; acceptance by today's rule (exact length, final state == L).

Usage: python -I audit_symperm.py SAMPLE_DIR RANS_DIR OUT_JSON expert_index [expert_index ...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import quantize, cum_of, encode_vec, decode_vec, inv_table, PB
from audit_count import read_expert
from audit_corrupt import streams_of


def main():
    sample_dir, rans_dir, out_json = sys.argv[1:4]
    eis = [int(v) for v in sys.argv[4:]] or [50]
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    md = np.load(os.path.join(rans_dir, 'model.npz'))
    base = md['base']
    perm = np.array([2 * (s & 7) + (s >> 3) for s in range(16)])  # code -> label
    out = {}
    rng = np.random.default_rng(11)
    for order in ('codes 0..15 (production)', 'interleaved 2*mag+sign'):
        if order.startswith('codes'):
            wf = quantize(base); mag = None; lab = np.arange(16)
        else:
            cp = np.zeros_like(base); cp[..., perm] = base
            wf = quantize(cp); mag = np.arange(16) >> 1; lab = perm
        wc = cum_of(wf)
        sym_eq = 0
        acc = tot = 0; bits = 0.0; nw = 0
        accC = totC = 0
        for ei in eis:
            tens, _ = read_expert(sample_dir, man['experts'][ei])
            for m in range(3):
                sym, ctx, srel = streams_of(tens, m, wf, wc)
                L_ = lab[sym.astype(np.int64)]
                F = wf[m][ctx, L_]; C = wc[m][ctx, L_]
                bits += float((PB - np.log2(F.astype(np.float64))).sum()); nw += F.size
                lens, _, buf = encode_vec(F, C, want_bytes=True)
                inv = inv_table(wf[m])
                N = F.shape[0]; ar = np.arange(N)
                for rep in range(2):
                    b = buf.copy(); j = (rng.random(N) * lens).astype(np.int64)
                    b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
                    v, ds, _ = decode_vec(b, lens, 2048, srel, wf[m], wc[m], inv, mag)
                    acc += int(v.sum()); tot += N
                b = buf.copy(); j = rng.integers(0, 4, N)
                b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
                v, _, _ = decode_vec(b, lens, 2048, srel, wf[m], wc[m], inv, mag)
                accC += int(v.sum()); totC += N
            sym_eq += int(sum(((wf[m][:, :8] == wf[m][:, 8:]) if order.startswith('codes') else (wf[m][:, 0::2] == wf[m][:, 1::2])).sum() for m in range(3)))
        out[order] = {'experts': [man['experts'][e]['file'] for e in eis], 'bitflip_trials': tot, 'accepted': acc,
                      'accept_rate': acc / tot, 'state_flip_trials': totC, 'state_flip_accepted': accC,
                      'payload_bit_per_w_noflush': bits / nw, 'rows_x_mag_with_f(+x)==f(-x)': sym_eq}
        print(order, out[order], flush=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
