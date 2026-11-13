"""Ideal (unquantized) conditional entropy of the FP4 sign given (magnitude, 567-context), from the
team-1 pass-1 counts (w_base, summed over the 160-expert sample). Usage: python -I signent.py COUNTS_DIR SAMPLE_DIR"""
import json, os, sys
import numpy as np

counts_dir, sample_dir = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
wb = None
for e in man['experts']:
    z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
    a = z['w_base'].astype(np.int64)
    wb = a if wb is None else wb + a


def H(c):
    c = c.astype(np.float64)
    t = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / np.where(t == 0, 1, t), 1)
    return float(-(c * np.log2(p)).sum())


N = wb.sum()
h16 = H(wb)                                   # H(sym | ctx)
mag = wb[..., :8] + wb[..., 8:]
h8 = H(mag)                                   # H(mag | ctx)
pairs = np.stack([wb[..., :8], wb[..., 8:]], -1)  # [3,567,8,2] sign | (mag, ctx)
hs = H(pairs)                                 # H(sign | mag, ctx)
out = {'weights': int(N), 'H_sym_given_ctx_bits_per_w': h16 / N, 'H_mag_given_ctx_bits_per_w': h8 / N,
       'H_sign_given_mag_ctx_bits_per_w': hs / N, 'check_sum': (h8 + hs) / N,
       'sign_bypass_cost_bits_per_w': (h8 + N - h16) / N,
       'sign_bypass_cost_pct_of_weight_bits': 100 * (h8 + N - h16) / h16,
       'P_negative_by_magnitude': (wb[..., 8:].sum((0, 1)) / mag.sum((0, 1))).round(6).tolist()}
print(json.dumps(out, indent=1))
