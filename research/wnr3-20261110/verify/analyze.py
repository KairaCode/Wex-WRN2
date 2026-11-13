"""Sizes (WNR3 CZ / non-CZ vs WNR2 production vs raw) and an independent value-exact CPU decode check.
usage: python -I analyze.py TOOLS.dll WEX1   -> analyze.json
Value-exact: every record of the CZ bin is decoded again (wex_nr3_decode) and compared with the WEX1 slab
read fresh from U: : scales equal; weight nibbles equal except WEX1 0x8 (-0) -> 0x0 (+0); the number of
such nibbles equals the encoder's cz_count. Non-CZ: decode == WEX1 bytes."""
import ctypes as C
import concurrent.futures as cf
import json
import struct
import sys
import numpy as np
import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'nr3'))  # build_nr3.py (KAIRA tools/)
import build_nr3 as B  # noqa: E402

tools, src = B.Tools(sys.argv[1]), B.Wex1(sys.argv[2])
tools.lib.wex_nr3_decode.argtypes = [C.c_void_p] * 3 + [C.c_uint32, C.c_uint32, C.c_void_p]
sel = json.load(open('select.json'))
test = {tuple(p) for p in sel['test']}
tail = {tuple(p) for p in sel['tail']}


def records(path):
    with open(path, 'rb') as f:
        mb = struct.unpack('<I', f.read(4))[0]
        blob = f.read(mb)
        n = struct.unpack('<I', f.read(4))[0]
        for _ in range(n):
            eb = f.read(72)
            rb = struct.unpack('<I', f.read(4))[0]
            f.seek(rb, 1)
            total = struct.unpack_from('<QIII', eb)[1]
            yield blob, eb, f.read(total)


def check(blob, eb, rec, row, cz):
    L, E = row['layer'], row['expert']
    _, wex1 = src.slab(L, E)
    out = C.create_string_buffer(len(wex1))
    if tools.lib.wex_nr3_decode(blob, eb, rec, L, E, out) != 0:
        return False, 'decode rejected'
    a = np.frombuffer(wex1, np.uint8)
    d = np.frombuffer(out.raw, np.uint8)
    e = B.ENTRY.unpack(eb)
    sb = e[2]   # scale_bytes
    if not np.array_equal(a[:sb], d[:sb]):
        return False, 'scales differ'
    lo_a, hi_a = a[sb:] & 15, a[sb:] >> 4
    lo_d, hi_d = d[sb:] & 15, d[sb:] >> 4
    if not cz:
        return bool(np.array_equal(a, d)), 'bytes'
    neg = int((lo_a == 8).sum() + (hi_a == 8).sum())
    ok = (np.array_equal(np.where(lo_a == 8, 0, lo_a), lo_d) and np.array_equal(np.where(hi_a == 8, 0, hi_a), hi_d)
          and neg == row['cz_count'])
    return bool(ok), neg


res = {}
for name, cz in (('nr3_noncz', False), ('nr3_cz', True)):
    rows = json.load(open(name + '.bin.json'))['rows']
    with cf.ThreadPoolExecutor(8) as pool:
        chk = list(pool.map(lambda x: check(*x[0], x[1], cz), zip(records(name + '.bin'), rows)))
    res[name] = {'rows': rows, 'check': chk}

out, per = {}, []
for i, (r2, r3) in enumerate(zip(res['nr3_noncz']['rows'], res['nr3_cz']['rows'])):
    key = (r3['layer'], r3['expert'])
    w2 = sel['wnr2'][f'{key[0]}:{key[1]}']
    per.append({'layer': key[0], 'expert': key[1], 'set': 'test' if key in test else 'tail',
                'in_tail64': key in tail, 'raw': r3['raw'], 'wnr2_total': w2['total'], 'wnr2_used': w2['used'],
                'wnr2_median': w2['median'], 'nr3_total': r2['total'], 'nr3_used': r2['used'],
                'nr3cz_total': r3['total'], 'nr3cz_used': r3['used'], 'cz_count': r3['cz_count'],
                'anchors': r3['anchors'], 'ceil_mask': r3['ceil_mask'],
                'cpu_exact_noncz': res['nr3_noncz']['check'][i][0], 'cpu_value_exact_cz': res['nr3_cz']['check'][i][0]})


def summ(rows):
    raw = rows[0]['raw']
    m = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    return {'n': len(rows), 'raw': raw,
            'wnr2_mean': m('wnr2_total'), 'nr3_mean': m('nr3_total'), 'nr3cz_mean': m('nr3cz_total'),
            'wnr2_ratio': m('wnr2_total') / raw, 'nr3_ratio': m('nr3_total') / raw, 'nr3cz_ratio': m('nr3cz_total') / raw,
            'nr3cz_vs_wnr2': m('nr3cz_total') / m('wnr2_total'), 'nr3_vs_wnr2': m('nr3_total') / m('wnr2_total'),
            'cz_gain_pct_raw': (m('nr3_total') - m('nr3cz_total')) / raw * 100,
            'nr3cz_used_ratio': m('nr3cz_used') / raw,
            'max_wnr2': max(r['wnr2_total'] for r in rows), 'max_nr3': max(r['nr3_total'] for r in rows),
            'max_nr3cz': max(r['nr3cz_total'] for r in rows),
            'argmax_nr3cz': max(rows, key=lambda r: r['nr3cz_total'])[('layer')] * 1000 + max(rows, key=lambda r: r['nr3cz_total'])['expert'],
            'nr3cz_worse_than_wnr2': sum(r['nr3cz_total'] > r['wnr2_total'] for r in rows),
            'cpu_exact_noncz': sum(r['cpu_exact_noncz'] for r in rows),
            'cpu_value_exact_cz': sum(r['cpu_value_exact_cz'] for r in rows)}


out['test40'] = summ([r for r in per if r['set'] == 'test'])
out['tail64'] = summ([r for r in per if r['in_tail64']])
out['all'] = summ(per)
out['models'] = {'noncz': json.load(open('nr3_noncz.bin.json'))['summary'], 'cz': json.load(open('nr3_cz.bin.json'))['summary']}
out['rows'] = per
json.dump(out, open('analyze.json', 'w'), indent=1)
for k in ('test40', 'tail64', 'all'):
    print(k, json.dumps(out[k]))
