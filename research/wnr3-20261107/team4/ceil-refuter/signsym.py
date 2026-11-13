"""Mechanism of the silent sign-flip hole (team 2 audit: accepted flips have f[s] == f[s^8] and CDF starts 8192 apart):
usage-weighted share of coded symbols whose context row has that property, original vs ceil form tables.
Usage: python -I -B signsym.py COUNTS_SAMPLE_NPZ OUT_JSON"""
import json, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from audit_core_t2copy import quantize, cum_of

z = np.load(sys.argv[1])
out = {}
for form, key in (('orig', 'o_base'), ('ceil', 'c_base')):
    f = quantize(z[key][[0, 2]].sum(0))  # [3, 567, 16]
    c = cum_of(f)[..., :16]
    te = z[key][[1, 3]].sum(0).astype(np.float64)
    s = np.arange(16)
    prop = (f == f[..., s ^ 8]) & (np.abs(c - c[..., s ^ 8]) == 8192)
    rows_half = (c[..., 8] == 8192)  # positive half of the row sums to exactly 8192
    r = {}
    for m in range(3):
        r[f'w{m + 1}'] = {'usage_share_symbols_with_property': float((te[m] * prop[m]).sum() / te[m].sum()),
                          'usage_share_rows_positive_half_8192': float((te[m].sum(1) * rows_half[m]).sum() / te[m].sum()),
                          'rows_with_positive_half_8192': int(rows_half[m].sum())}
    out[form] = r
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
