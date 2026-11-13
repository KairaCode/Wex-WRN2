"""Paired comparison of the signplane-z records with team 4's held-out g0 records (same 40 experts, same anchor 120).
Bootstrap: 10,000 expert resamples, seed 20261107 (as team 4 summarize_ceil.py), 95% percentile CIs.
Usage: python -I summarize_z.py OUT_JSON Z_JSONL TEAM4_G0_JSONL
"""
import json, sys
import numpy as np

RAW = 18_800_640
W_PER_REC = 3 * 2304 * 5120
NWS = 17280
B = 10000


def boot(x, rng):
    x = np.asarray(x, np.float64)
    idx = rng.integers(0, x.size, (B, x.size))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    out_json, zj, t4j = sys.argv[1:4]
    rng = np.random.default_rng(20261107)
    rows = [json.loads(l) for l in open(zj) if l.strip()]
    t4 = [json.loads(l) for l in open(t4j) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r['config'], {})[r['idx']] = r
    for r in t4:
        if r['config'] in ('fgcde', 'fgde', 'wnr2', 'fgce', 'fgcd'):
            by.setdefault('t4_' + r['config'], {})[r['idx']] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    res = {'n_experts': len(common), 'experts': common, 'configs': {}, 'pairs': {}}
    for c, d in by.items():
        rr = [d[i] for i in common]
        tot = np.array([r['sizes']['total'] for r in rr], np.float64)
        used = np.array([r['sizes']['used'] for r in rr], np.float64)
        ent = {'mean_total': float(tot.mean()), 'ratio_total': float(tot.mean() / RAW),
               'mean_used': float(used.mean()), 'ratio_used': float(used.mean() / RAW),
               'mean_parts': {k: float(np.mean([r['sizes'][k] for r in rr])) for k in
                              ('length_table', 'scale_payload', 'weight_payload', 'padding')},
               'weight_bits_per_w (payload incl. flush, planes)': float(np.mean([r['sizes']['weight_payload'] for r in rr]) * 8 / W_PER_REC)}
        if 'verify' in rr[0]:
            v = rr[0]['verify']
            if 'equal_to_canonical_source' in v:
                ent['decode_check'] = f"{sum(r['verify']['equal_to_canonical_source'] for r in rr)}/{len(rr)} equal to canonical source"
                ent['decode_errors'] = [r['verify']['decode_error'] for r in rr if r['verify']['decode_error']]
                ent['differing_bytes_eq_neg0_bytes'] = f"{sum(r['verify']['bytes_differing_from_source'] == r['verify']['expected_differing_bytes (bytes holding a -0 nibble)'] for r in rr)}/{len(rr)}"
                ent['source_sha256_ok'] = f"{sum(r['source_sha256_ok'] for r in rr)}/{len(rr)}"
                ent['anchors_all_120'] = all(r['anchors'] == [120, 120, 120] for r in rr)
                ent['ceil_applied'] = f"{sum(bool(r['ceil_applied']) for r in rr)}/{len(rr)}"
            else:
                ent['decode_check'] = f"{sum(r['verify']['equal_to_reference'] for r in rr)}/{len(rr)} (team 4)"
        if rr[0].get('z'):
            zz = {k: float(np.mean([r['z'][k] for r in rr])) for k in rr[0]['z']}
            zz['pad_bytes_per_stream'] = zz['pad_bits'] / 8 / NWS
            zz['sign_bits_per_w'] = zz['sign_bits'] / W_PER_REC
            zz['mag_rans_bits_per_w (incl. flush/hash state)'] = zz['rans_bytes'] * 8 / W_PER_REC
            zz['est_mag_bits_per_w'] = zz['est_mag_bits'] / W_PER_REC
            ent['z'] = zz
            ent['len_resid_std_w'] = float(np.mean([r['len_resid_std'][1] for r in rr]))
        res['configs'][c] = ent
    pairs = [('z1de vs t4_fgcde', 'z1de', 't4_fgcde'), ('z2de vs t4_fgcde', 'z2de', 't4_fgcde'),
             ('z1de vs t4_fgde', 'z1de', 't4_fgde'), ('z2de vs t4_fgde', 'z2de', 't4_fgde'),
             ('z2de vs z1de', 'z2de', 'z1de'), ('z1de vs t4_wnr2', 'z1de', 't4_wnr2'), ('z2de vs t4_wnr2', 'z2de', 't4_wnr2'),
             ('t4_fgcde vs t4_fgde', 't4_fgcde', 't4_fgde')]
    for name, a, b in pairs:
        if a not in by or b not in by:
            continue
        for kind in ('total', 'used'):
            d = np.array([by[a][i]['sizes'][kind] - by[b][i]['sizes'][kind] for i in common], np.float64)
            ci = boot(d, rng)
            res['pairs'].setdefault(name, {})[kind] = {
                'mean_delta_bytes': float(d.mean()), 'CI95_bytes': ci,
                'mean_delta_pct_of_raw': float(100 * d.mean() / RAW), 'CI95_pct_of_raw': [100 * v / RAW for v in ci],
                'median_delta_bytes': float(np.median(d)), 'experts_smaller': int((d < 0).sum()),
                'range_bytes': [float(d.min()), float(d.max())]}
        # weight payload component
        dw = np.array([by[a][i]['sizes']['weight_payload'] - by[b][i]['sizes']['weight_payload'] for i in common], np.float64)
        dl = np.array([by[a][i]['sizes']['length_table'] - by[b][i]['sizes']['length_table'] for i in common], np.float64)
        ds = np.array([by[a][i]['sizes']['scale_payload'] - by[b][i]['sizes']['scale_payload'] for i in common], np.float64)
        res['pairs'][name]['components_mean_delta_bytes'] = {'weight_payload': float(dw.mean()), 'weight_payload_CI95': boot(dw, rng),
                                                             'length_table': float(dl.mean()), 'scale_payload': float(ds.mean())}
    json.dump(res, open(out_json, 'w'), indent=1)
    for c, e in res['configs'].items():
        print(f"{c:10s} total {e['mean_total']:12.1f} ratio {e['ratio_total']:.5f} used {e['mean_used']:12.1f} ratio_used {e['ratio_used']:.5f}",
              e.get('decode_check', ''), e['mean_parts'])
    for n, p in res['pairs'].items():
        print(f"{n:22s} total {p['total']['mean_delta_bytes']:+9.1f} B CI {p['total']['CI95_bytes'][0]:+.1f}..{p['total']['CI95_bytes'][1]:+.1f} "
              f"({p['total']['mean_delta_pct_of_raw']:+.4f}% raw) | used {p['used']['mean_delta_bytes']:+9.1f} CI "
              f"{p['used']['CI95_bytes'][0]:+.1f}..{p['used']['CI95_bytes'][1]:+.1f} ({p['used']['mean_delta_pct_of_raw']:+.4f}%) "
              f"smaller {p['used']['experts_smaller']}/{len(common)} | comp {p['components_mean_delta_bytes']}")


if __name__ == '__main__':
    main()
