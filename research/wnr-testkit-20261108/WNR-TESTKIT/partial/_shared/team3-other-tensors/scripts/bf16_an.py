"""BF16 entropy (embed/head row slices, one router) - order-0 symbol and field decomposition.
Usage: python -I bf16_an.py BF16_DATA_DIR OUT_JSON"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entlib import *

d, outp = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
grp = collections.defaultdict(list)
for it in man['items']:
    grp[it['name']].append(it)
out = {}
for name, its in grp.items():
    v = np.concatenate([np.fromfile(os.path.join(d, it['tag'] + '.bin'), np.uint16) for it in its])
    sign = v >> 15; exp = (v >> 7) & 255; mant = v & 127
    n = v.size
    r = {'elements': int(n), 'slices': [it['rows'] for it in its],
         'H_16bit_symbol_order0': ent_bits(bc(v, 65536)) / n,
         'H_sign': ent_bits(bc(sign, 2)) / n, 'H_exp': ent_bits(bc(exp, 256)) / n,
         'H_mant_given_exp': ent_bits(ctx_counts(exp, mant, 256, 128)) / n,
         'H_sign_given_exp': ent_bits(ctx_counts(exp, sign, 256, 2)) / n,
         'distinct_exponents': int((bc(exp, 256) > 0).sum()),
         'generic': generic(v[: 4 << 20].tobytes())}
    # exponent given previous exponent in the row (rows are 5120 wide)
    cols = 5120 if name != 'layers.20.ffn.gate.weight' else 5120
    e2 = exp.reshape(-1, cols).astype(np.int64)
    pe = np.concatenate([np.full((e2.shape[0], 1), 256), e2[:, :-1]], 1)
    r['H_exp_given_prev_exp'] = ent_bits(ctx_counts(pe, e2, 257, 256)) / n
    r['bits_per_element_exp_o0_plus_raw_rest'] = r['H_exp'] + 8
    r['bits_per_element_best_simple'] = r['H_exp_given_prev_exp'] + r['H_mant_given_exp'] + r['H_sign_given_exp']
    out[name] = r
json.dump(out, open(outp, 'w'), indent=1)
print(json.dumps(out, indent=1))
