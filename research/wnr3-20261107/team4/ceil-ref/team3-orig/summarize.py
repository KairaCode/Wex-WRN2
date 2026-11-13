"""Summarise run_records.py output: real record sizes, deltas vs WNR2 with expert-bootstrap 95% intervals, timings.

Usage: python -I summarize.py OUT_JSON JSONL [JSONL ...]
"""
import json, sys
import numpy as np

RAW = 18_800_640
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
B = 10000

# estimates from measured-v41.md (held-out, % of weight bits vs WNR2 3.7826 bit/w), for the side-by-side table
DOC = {'a': '-0.213% (block-max side bit + half, 803 rows)', 'aK2': '-0.34% (K=2 on top of bm+half: 3.7696)',
       'aK4': '-0.35% (K=4 on top of bm+half: 3.7692)', 'c': '-3.08% (WNR2z in-sample, base contexts)',
       'aK4c': '-3.38% (bm + pm2 contexts + 4 models + canonical zero, 3.6544)',
       'd': '-0.148% of raw record (length table 38.9 KB -> ~11.0 KB, entropy estimate)',
       'e': '+0.007% bytes (+0.067 B/stream)'}


def boot(x, rng):
    x = np.asarray(x, np.float64)
    n = x.size
    idx = rng.integers(0, n, (B, n))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    out_json = sys.argv[1]
    rows = []
    for p in sys.argv[2:]:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    rng = np.random.default_rng(20261107)
    res = {}
    for models in sorted({r['models'] for r in rows}):
        R = [r for r in rows if r['models'] == models]
        cfgs = list(dict.fromkeys(r['config'] for r in R))
        by = {c: {r['idx']: r for r in R if r['config'] == c} for c in cfgs}
        if 'wnr2' not in by:
            continue
        common = sorted(set.intersection(*[set(v) for v in by.values()]))
        base = by['wnr2']
        tab = {}
        for c in cfgs:
            rr = [by[c][i] for i in common]
            tot = np.array([r['sizes']['total'] for r in rr], np.float64)
            used = np.array([r['sizes']['used'] for r in rr], np.float64)
            wp = np.array([r['sizes']['weight_payload'] for r in rr], np.float64)
            sp = np.array([r['sizes']['scale_payload'] for r in rr], np.float64)
            lt = np.array([r['sizes']['length_table'] for r in rr], np.float64)
            pad = np.array([r['sizes']['padding'] for r in rr], np.float64)
            estw = np.array([r['est']['est_weight_payload'] for r in rr])
            ests = np.array([r['est']['est_scale_payload'] for r in rr])
            btot = np.array([base[i]['sizes']['total'] for i in common], np.float64)
            bused = np.array([base[i]['sizes']['used'] for i in common], np.float64)
            bwp = np.array([base[i]['sizes']['weight_payload'] for i in common], np.float64)
            d_tot = tot - btot
            d_used = used - bused
            d_wp_pct = 100 * (wp - bwp) / bwp
            enc = np.array([r['timing']['enc_total_s'] for r in rr])
            dec = np.array([r['timing']['dec_total_s'] for r in rr])
            decw = np.array([r['timing']['dec_rans_weights_s'] for r in rr])
            encw = np.array([r['timing']['enc_rans_weights_s'] for r in rr])
            ok = sum(r['verify']['equal_to_reference'] for r in rr)
            bx = sum(r['verify']['byte_exact_vs_source'] for r in rr)
            tab[c] = {
                'n_experts': len(rr), 'verified': f"{ok}/{len(rr)} ({rr[0]['verify']['mode']})", 'byte_exact_vs_source': f'{bx}/{len(rr)}',
                'mean_total_bytes': float(tot.mean()), 'ratio_of_raw': float(tot.mean() / RAW),
                'mean_used_bytes (unpadded)': float(used.mean()),
                'mean_parts': {'header': rr[0]['sizes']['header'], 'length_table': float(lt.mean()), 'scale_payload': float(sp.mean()),
                               'weight_payload': float(wp.mean()), 'padding': float(pad.mean())},
                'weight_bits_per_w (payload incl. flush and side bits)': float(wp.mean() * 8 / W_PER_REC),
                'scale_bits_per_scale (incl. flush)': float(sp.mean() * 8 / S_PER_REC),
                'estimate_from_code_lengths': {'weight_payload': float(estw.mean()), 'scale_payload': float(ests.mean()),
                                               'real_minus_est_weight_bytes': float((wp - estw).mean()),
                                               'real_minus_est_weight_pct': float(100 * ((wp - estw) / estw).mean())},
                'delta_total_vs_wnr2_bytes': float(d_tot.mean()), 'delta_total_vs_wnr2_bytes_CI95': boot(d_tot, rng),
                'delta_total_pct_of_wnr2_record': float(100 * d_tot.mean() / btot.mean()),
                'delta_total_pct_of_wnr2_record_CI95': [100 * v / btot.mean() for v in boot(d_tot, rng)],
                'delta_total_pct_of_raw': float(100 * d_tot.mean() / RAW),
                'delta_used_vs_wnr2_bytes': float(d_used.mean()), 'delta_used_vs_wnr2_bytes_CI95': boot(d_used, rng),
                'delta_weight_payload_pct': float(d_wp_pct.mean()), 'delta_weight_payload_pct_CI95': boot(d_wp_pct, rng),
                'doc_estimate': DOC.get(c),
                'model_blob_bytes': rr[0]['model_blob_bytes'],
                'timing_s': {'enc_total_median': float(np.median(enc)), 'enc_total_mean': float(enc.mean()),
                             'enc_rans_weights_median': float(np.median(encw)),
                             'dec_total_median': float(np.median(dec)), 'dec_total_mean': float(dec.mean()),
                             'dec_rans_weights_median': float(np.median(decw))},
                'model_choice_hist': (np.bincount(np.array([r['ks'] for r in rr]).ravel(), minlength=max(1, rr[0]['K'])).tolist()
                                      if rr[0]['K'] else None),
                'max_weight_stream_bytes': int(max(r['max_weight_stream'] for r in rr)),
            }
            if rr[0].get('len_bits') is not None:
                tab[c]['length_table_detail'] = {'mean_bytes': float(lt.mean()),
                                                 'resid_std_scale_streams': float(np.mean([r['len_resid_std'][0] for r in rr])),
                                                 'resid_std_weight_streams': float(np.mean([r['len_resid_std'][1] for r in rr])),
                                                 'saving_vs_u16_table_pct_of_raw': float(100 * (38880 - lt.mean()) / RAW)}
            cor = [r['corruption'] for r in rr if 'corruption' in r]
            if cor:
                tab[c]['corruption_tests'] = cor
        # pairwise feature deltas (same experts), with CI
        pairs = [('d vs w3', 'd', 'w3'), ('e vs w3', 'e', 'w3'), ('w3 vs wnr2', 'w3', 'wnr2'),
                 ('aK4d vs aK4', 'aK4d', 'aK4'), ('aK4de vs aK4d', 'aK4de', 'aK4d'),
                 ('aK4cd vs aK4c', 'aK4cd', 'aK4c'), ('aK4ce vs aK4c', 'aK4ce', 'aK4c'), ('aK4cde vs aK4cd', 'aK4cde', 'aK4cd'),
                 ('c vs w3', 'c', 'w3'), ('a vs w3', 'a', 'w3'), ('aK4 vs a', 'aK4', 'a'), ('aK2 vs a', 'aK2', 'a'),
                 ('aK4c vs aK4', 'aK4c', 'aK4')]
        pw = {}
        for name, x, y in pairs:
            if x in by and y in by:
                dt = np.array([by[x][i]['sizes']['total'] - by[y][i]['sizes']['total'] for i in common], np.float64)
                du = np.array([by[x][i]['sizes']['used'] - by[y][i]['sizes']['used'] for i in common], np.float64)
                pw[name] = {'delta_total_bytes': float(dt.mean()), 'CI95': boot(dt, rng), 'delta_used_bytes': float(du.mean()),
                            'used_CI95': boot(du, rng), 'delta_used_pct_of_raw': float(100 * du.mean() / RAW)}
        res[models] = {'experts': common, 'configs': tab, 'pairwise': pw}
    json.dump(res, open(out_json, 'w'), indent=1)
    for models, v in res.items():
        print('==', models, 'experts', len(v['experts']))
        for c, t in v['configs'].items():
            print(f"{c:8s} total {t['mean_total_bytes']:12.1f}  ratio {t['ratio_of_raw']:.5f}  d% {t['delta_total_pct_of_wnr2_record']:+.3f} "
                  f"CI {t['delta_total_pct_of_wnr2_record_CI95'][0]:+.3f}..{t['delta_total_pct_of_wnr2_record_CI95'][1]:+.3f}  "
                  f"wbits {t['weight_bits_per_w (payload incl. flush and side bits)']:.4f}  ok {t['verified']}  "
                  f"enc {t['timing_s']['enc_total_median']:.2f}s dec {t['timing_s']['dec_total_median']:.2f}s")
        for k, p in v['pairwise'].items():
            print(f"  {k:18s} used {p['delta_used_bytes']:+10.1f} B  CI {p['used_CI95'][0]:+.1f}..{p['used_CI95'][1]:+.1f}  ({p['delta_used_pct_of_raw']:+.4f}% raw)")


if __name__ == '__main__':
    main()
