# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Follow-up checks after analyze.py.

  python -I extra.py DATA_DIR OUT.json
  python -I extra.py nz5 DATA_DIR OUT.json      (item 4 only)

1. DeepSeek V4.1: is the Quark MXFP4 export Z(original checkpoint) byte for byte? How many nibbles of the NVFP4
   export equal those of the Quark export, how many NVFP4 scale pairs (blocks 2k, 2k+1 of a row) are equal, and
   how many dequantized NVFP4 values equal the dequantized original values?
2. NVFP4 nibble contexts beyond WNR3: the position in the block while no maximum magnitude 7 has been seen
   (pos bucket 0..3: positions 0-11, 12-13, 14, 15; bucket 0 once a 7 was seen), in-sample entropies.
3. NVFP4 scale contexts: previous scale in the row together with the parity of the position in the row.
4. The NVFP4_NZ5 export of GLM-5.3-Flash (glm-nz5 of fetch.py, csf0_v1 scales: a base byte per row and a code
   4-bit offset per 32 values, two per byte): share of nibble code 8 (+5 there, -0 in E2M1), nibble entropies
   with the WNR3 contexts (class = the group's 4-bit offset) and with a 9-level magnitude rank that puts +5
   between 4 and 6;
   nibble order checked by correlating the unscaled NZ5 values with the unscaled values of glm-mxfp4.
numpy and the standard library.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import fetch  # noqa: E402


def deq_mx(z):
    q = A.nibbles(z['wbytes'])
    return A.E2M1[q] * np.repeat(2.0 ** (z['scales'].astype(np.float64) - 127), 32)


def deq_nv(z):
    q = A.nibbles(z['wbytes'])
    return A.E2M1[q] * np.repeat(A.e4m3(z['scales']) * z['g2'], 16)


def ds_checks(data):
    nat = {(L, e): mm for L, e, _, mm in A.experts('ds-mxfp4', data)}
    qk = {(L, e): mm for L, e, _, mm in A.experts('ds-quark', data)}
    nv = {(L, e): mm for L, e, _, mm in A.experts('ds-nvfp4', data)}
    tz = bytes(((b & 0xF0) if b & 15 == 8 else b) for b in range(256))
    tz = np.frombuffer(bytes(((x & 0x0F) if x >> 4 == 8 else x) for x in tz), dtype=np.uint8)
    r = {'quark_equals_z_native': 0, 'experts': len(nat), 'nibbles': 0, 'nv_nibbles_equal_quark': 0,
         'nv_scale_pairs': 0, 'nv_scale_pairs_equal': 0, 'values': 0, 'nv_values_equal_native': 0,
         'nv_scale_codes': {}, 'nv_mantissa_nonzero': 0}
    for k in nat:
        if all(np.array_equal(tz[a['wbytes']], b['wbytes']) and np.array_equal(a['scales'], b['scales'])
               for a, b in zip(nat[k], qk[k])):
            r['quark_equals_z_native'] += 1
        for a, b in zip(qk[k], nv[k]):
            qa, qb = A.nibbles(a['wbytes']), A.nibbles(b['wbytes'])
            r['nibbles'] += len(qa)
            r['nv_nibbles_equal_quark'] += int((qa == qb).sum())
            s = b['scales'].reshape(b['rows'], -1)
            r['nv_scale_pairs'] += s.size // 2
            r['nv_scale_pairs_equal'] += int((s[:, 0::2] == s[:, 1::2]).sum())
            r['nv_mantissa_nonzero'] += int((s & 7).astype(bool).sum())
            for c, n in zip(*np.unique(s, return_counts=True)):
                r['nv_scale_codes'][int(c)] = r['nv_scale_codes'].get(int(c), 0) + int(n)
        for a, b in zip(nat[k], nv[k]):
            va, vb = deq_mx(a), deq_nv(b)
            r['values'] += len(va)
            r['nv_values_equal_native'] += int((va == vb).sum())
    r['nv_scale_codes'] = {f'0x{c:02x}': n for c, n in sorted(r['nv_scale_codes'].items())}
    return r


def ctx_checks(name, data):
    acc = A.Acc()
    stats = {'scale_mantissa_hist': np.zeros(8, np.int64), 'scale_exp_hist': np.zeros(16, np.int64)}
    n = ns = 0
    for L, e, fmt, mm in A.experts(name, data):
        for m, x in enumerate(mm):
            q = A.nibbles(x['wbytes'])
            q = np.where(q == 8, 0, q)   # in the CZ domain
            s = x['scales'].astype(np.int64)
            B = x['B']
            cls, nc = A.classes(fmt, x['scales'])['exponent srel7' if fmt == 'nvfp4' else 'srel7 (WNR3)']
            cb = np.repeat(cls, B)
            pm, mx = A.wnr_ctx(q, B)
            pos = np.tile(np.arange(B), len(q) // B)
            seen7 = np.zeros_like(q, dtype=bool).reshape(-1, B)
            mag = (q & 7).reshape(-1, B)
            seen7[:, 1:] = np.maximum.accumulate(mag == 7, axis=1)[:, :-1]
            seen7 = seen7.reshape(-1)
            pb = np.where(seen7, 0, np.select([pos >= B - 1, pos >= B - 2, pos >= B - 4], [3, 2, 1], 0))
            base = (cb * 9 + pm) * 9 + mx
            acc.add(('wnr', m), base, q, nc * 81, 16)
            acc.add(('wnr+pos', m), base * 4 + pb, q, nc * 81 * 4, 16)
            s2 = s.reshape(x['rows'], -1)
            prev = np.full_like(s2, 256)
            prev[:, 1:] = s2[:, :-1]
            par = np.tile(np.arange(s2.shape[1]) % 2, (s2.shape[0], 1))
            acc.add(('s_left', m), prev.reshape(-1), s2.reshape(-1), 257, 256)
            acc.add(('s_left_parity', m), (prev * 2 + par).reshape(-1), s2.reshape(-1), 514, 256)
            up = np.full_like(s2, 256)
            up[1:, :] = s2[:-1, :]
            lb = np.clip(prev - A.rule_g(s.reshape(-1)), -8, 8) + 8
            lb[prev == 256] = 17
            ub = np.clip(up - A.rule_g(s.reshape(-1)), -8, 8) + 8
            ub[up == 256] = 17
            acc.add(('s_left_up_buckets', m), (lb * 18 + ub).reshape(-1), s2.reshape(-1), 18 * 18, 256)
            n += len(q)
            ns += s.size
            if fmt == 'nvfp4':
                stats['scale_mantissa_hist'] += np.bincount(s & 7, minlength=8)
                stats['scale_exp_hist'] += np.bincount((s >> 3) & 15, minlength=16)
    tot = lambda k: sum(A.ent(v) for kk, v in acc.t.items() if kk[0] == k)
    return {'checkpoint': name, 'nibble_bits_wnr3_ctx_cz': tot('wnr') / n, 'nibble_bits_wnr3_ctx_plus_pos_cz': tot('wnr+pos') / n,
            'scale_bits_left': tot('s_left') / ns, 'scale_bits_left_parity': tot('s_left_parity') / ns,
            'scale_bits_left_up_buckets': tot('s_left_up_buckets') / ns,
            'scale_mantissa_hist': stats['scale_mantissa_hist'].tolist(), 'scale_exp_hist': stats['scale_exp_hist'].tolist()}


NZ5 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 5, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)
RANK9 = np.array([0, 1, 2, 3, 4, 5, 6, 8, 7, 1, 2, 3, 4, 5, 6, 8])   # magnitude rank with +5 between 4 and 6


def nz5_checks(data):
    """NVFP4_NZ5 export of GLM-5.3-Flash: how often code 8 (+5) occurs, and the nibble entropies."""
    pre, mats = fetch.CHECKPOINTS['glm-nz5'][3], fetch.CHECKPOINTS['glm-nz5'][4]
    t = A.st_read(os.path.join(data, 'glm-nz5', 'experts.safetensors'))
    mx = {(L, e): mm for L, e, _, mm in A.experts('glm-mxfp4', data)}
    acc = A.Acc()
    r = {'nibbles': 0, 'code8': 0, 'weights_bytes': 0, 'csf0_codes_bytes': 0, 'csf0_bases_bytes': 0, 'scalars_bytes': 0,
         'corr_low_first': [], 'corr_high_first': []}
    for (L, e), mm in sorted(mx.items()):
        for m, name in enumerate(mats):
            p = pre.format(L=L, e=e) + name + '.'
            w = t[p + 'weight'][2]
            codes, bases = t[p + 'weight_scale.csf0_codes'], t[p + 'weight_scale.csf0_bases'][2]
            r['weights_bytes'] += len(w)
            r['csf0_codes_bytes'] += len(codes[2])
            r['csf0_bases_bytes'] += len(bases)
            r['scalars_bytes'] += 8
            q = A.nibbles(w)
            r['nibbles'] += len(q)
            r['code8'] += int((q == 8).sum())
            if L == min(k[0] for k in mx) and len(r['corr_low_first']) < 3:
                ref = A.E2M1[A.nibbles(mm[m]['wbytes'])][:1 << 20]
                for hf, key in ((False, 'corr_low_first'), (True, 'corr_high_first')):
                    r[key].append(float(np.corrcoef(NZ5[A.nibbles(w, hf)][:1 << 20], ref)[0, 1]))
            mag = (q & 7)
            pm, mxx = A.wnr_ctx(q, 32)
            rk = RANK9[q].reshape(-1, 32)
            pm9 = np.full_like(rk, 9)
            pm9[:, 1:] = rk[:, :-1]
            mx9 = np.full_like(rk, 9)
            mx9[:, 1:] = np.maximum.accumulate(rk, axis=1)[:, :-1]
            off = A.nibbles(codes[2]).astype(np.int64)   # 4-bit offset per group of 32, even group in the low nibble
            assert len(off) * 32 == len(q)
            cls = np.repeat(off, 32)
            acc.add(('q0', m), np.zeros(len(q), np.int64), q, 1, 16)
            acc.add(('wnr', m), (cls * 9 + pm) * 9 + mxx, q, 16 * 81, 16)
            acc.add(('wnr9', m), (cls * 10 + pm9.reshape(-1)) * 10 + mx9.reshape(-1), q, 16 * 100, 16)
            c2 = off.reshape(codes[1][0], -1)
            prev = np.full_like(c2, 16)
            prev[:, 1:] = c2[:, :-1]
            acc.add(('s0', m), np.zeros(c2.size, np.int64), c2.reshape(-1), 1, 16)
            acc.add(('s1', m), prev.reshape(-1), c2.reshape(-1), 17, 16)
            acc.add(('b0', m), np.zeros(len(bases), np.int64), bases.astype(np.int64), 1, 256)
            del mag
    tot = lambda k: sum(A.ent(v) for kk, v in acc.t.items() if kk[0] == k)
    n = r['nibbles']
    raw = r['weights_bytes'] + r['csf0_codes_bytes'] + r['csf0_bases_bytes'] + r['scalars_bytes']
    r.update(code8_fraction=r['code8'] / n, raw_total=raw, nibble_bits_order0=tot('q0') / n,
             nibble_bits_wnr3_ctx=tot('wnr') / n, nibble_bits_ctx_rank9=tot('wnr9') / n,
             csf0_offset_bits_order0=tot('s0') / (2 * r['csf0_codes_bytes']),
             csf0_offset_bits_left=tot('s1') / (2 * r['csf0_codes_bytes']),
             csf0_base_bits_order0=tot('b0') / r['csf0_bases_bytes'])
    r['estimated_fraction_of_raw'] = (min(tot('wnr'), tot('wnr9')) / 8 + min(tot('s0'), tot('s1')) / 8 + tot('b0') / 8 +
                                      r['scalars_bytes']) / raw
    return r


def main(argv):
    if argv[1] == 'nz5':
        r = nz5_checks(argv[2])
        print(json.dumps(r))
        json.dump(r, open(argv[3], 'w', encoding='utf-8'), indent=1)
        return 0
    data, out = argv[1], argv[2]
    res = {'deepseek': ds_checks(data)}
    print(json.dumps(res['deepseek']), flush=True)
    res['contexts'] = []
    for name in ('ds-nvfp4', 'qwen-nvfp4', 'glm-nvfp4', 'ds-quark', 'qwen-mxfp4', 'glm-mxfp4'):
        r = ctx_checks(name, data)
        print(json.dumps(r), flush=True)
        res['contexts'].append(r)
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
