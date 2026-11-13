"""Summarise run_ceil.py output: real record sizes, deltas vs WNR2 with 10,000-resample expert-bootstrap 95% CIs,
median expert deltas, bit/w, bit/scale, decode checks, entropy-estimate cross-check (team4 / ceil-ref).

Usage: python -I summarize_ceil.py OUT_JSON TRAIN_SUMMARY_JSON JSONL [JSONL ...]
"""
import json, sys
import numpy as np

RAW = 18_800_640
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
NSTREAMS = 17280 + 2160
B = 10000
MODEL_OF = {'wnr2': 'wnr2_a0_K0_c0', 'f': 'f_a0_K0_c0', 'fg': 'fg_a0_K0_c0', 'fgd': 'fg_a0_K0_c0', 'fge': 'fg_a0_K0_c0',
            'fgde': 'fg_a0_K0_c0', 'fgt': 'fg_a0_K0_c0', 'fgdt': 'fg_a0_K0_c0', 'fga': 'fg_a1_K0_c0', 'fgad': 'fg_a1_K0_c0',
            'fgaK2': 'fg_a1_K2_c0', 'fgaK2d': 'fg_a1_K2_c0', 'fgc': 'fg_a0_K0_c1', 'fgcd': 'fg_a0_K0_c1',
            'fgce': 'fg_a0_K0_c1', 'fgcde': 'fg_a0_K0_c1', 'fgaK2ce': 'fg_a1_K2_c1', 'fgaK2cde': 'fg_a1_K2_c1',
            'fgct': 'fg_a0_K0_c1', 'fgcdt': 'fg_a0_K0_c1'}
PAIRS = [('f vs wnr2', 'f', 'wnr2'), ('fg vs f (g on ceil-form)', 'fg', 'f'), ('fgd vs fg (d)', 'fgd', 'fg'),
         ('fge vs fg (e)', 'fge', 'fg'), ('fgde vs fgd (e on d)', 'fgde', 'fgd'), ('fgt vs fg (tail seeding)', 'fgt', 'fg'),
         ('fgdt vs fgd (tail seeding on d)', 'fgdt', 'fgd'), ('fgdt vs fgde (tail instead of hash)', 'fgdt', 'fgde'),
         ('fga vs fg (a, 3-valued)', 'fga', 'fg'), ('fgad vs fgd', 'fgad', 'fgd'), ('fgaK2 vs fga (b K=2)', 'fgaK2', 'fga'),
         ('fgc vs fg (c)', 'fgc', 'fg'), ('fgcde vs fgde (c on d+e)', 'fgcde', 'fgde'),
         ('fgaK2cde vs fgcde (a+b on c+d+e)', 'fgaK2cde', 'fgcde'), ('fgcdt vs fgcd (tail on c+d)', 'fgcdt', 'fgcd'),
         ('fgcdt vs fgcde (tail instead of hash, value-exact)', 'fgcdt', 'fgcde'), ('fgcd vs fgd (c on d)', 'fgcd', 'fgd')]


def boot(x, rng):
    x = np.asarray(x, np.float64)
    n = x.size
    idx = rng.integers(0, n, (B, n))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    out_json, train_json = sys.argv[1:3]
    train = json.load(open(train_json))
    rows = []
    for p in sys.argv[3:]:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    rng = np.random.default_rng(20261107)
    cfgs = list(dict.fromkeys(r['config'] for r in rows))
    by = {c: {r['idx']: r for r in rows if r['config'] == c} for c in cfgs}
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    base = by['wnr2']
    btot = np.array([base[i]['sizes']['total'] for i in common], np.float64)
    bused = np.array([base[i]['sizes']['used'] for i in common], np.float64)

    def est_total(r):
        return r['sizes']['header'] + r['sizes']['length_table'] + r['est']['est_weight_payload'] + r['est']['est_scale_payload']

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
        sideb = np.array([r['est']['side_bits'] for r in rr])
        d_tot = tot - btot
        d_used = used - bused
        ok = sum(r['verify']['equal_to_reference'] for r in rr)
        bx = sum(r['verify']['byte_exact_vs_source'] for r in rr)
        derr = [r['verify']['decode_error'] for r in rr if r['verify']['decode_error']]
        fb = [r['file'] for r in rr if r['ceil'] is not None and r['ceil']['fallback']]
        enc = np.array([r['timing']['enc_total_s'] for r in rr])
        dec = np.array([r['timing'].get('dec_total_s', np.nan) for r in rr])
        mk = MODEL_OF.get(c)
        tr = train.get(mk, {}) if mk else {}
        tab[c] = {
            'n_experts': len(rr),
            'decode_check': f"{ok}/{len(rr)} equal to reference ({rr[0]['verify']['mode']})",
            'byte_exact_vs_source': f'{bx}/{len(rr)}', 'decode_errors': derr,
            'source_sha256_ok': f"{sum(r['source_sha256_ok'] for r in rr)}/{len(rr)}",
            'ceil_fallback_records': fb if rr[0]['ceil'] is not None else 'n/a (no F)',
            'record_flags_seen': sorted({r['rec_flags'] for r in rr}),
            'mean_total_bytes': float(tot.mean()), 'ratio_of_raw': float(tot.mean() / RAW),
            'mean_used_bytes_unpadded': float(used.mean()), 'ratio_used_of_raw': float(used.mean() / RAW),
            'mean_parts': {'header': rr[0]['sizes']['header'], 'length_table': float(lt.mean()), 'scale_payload': float(sp.mean()),
                           'weight_payload': float(wp.mean()), 'padding': float(pad.mean())},
            'bit_per_w (weight payload incl. flush and side symbols)': float(wp.mean() * 8 / W_PER_REC),
            'side_symbol_bit_per_w (est)': float(sideb.mean() / W_PER_REC),
            'bit_per_scale (scale payload incl. flush)': float(sp.mean() * 8 / S_PER_REC),
            'all_in_bit_per_w (used bytes)': float(used.mean() * 8 / W_PER_REC),
            'delta_total_vs_wnr2_bytes': float(d_tot.mean()), 'delta_total_vs_wnr2_bytes_CI95': boot(d_tot, rng),
            'delta_total_pct_of_raw': float(100 * d_tot.mean() / RAW),
            'delta_total_pct_of_raw_CI95': [100 * v / RAW for v in boot(d_tot, rng)],
            'delta_total_pct_of_wnr2_record': float(100 * d_tot.mean() / btot.mean()),
            'median_expert_delta_total_bytes': float(np.median(d_tot)),
            'median_expert_delta_total_pct_of_raw': float(100 * np.median(d_tot) / RAW),
            'delta_used_vs_wnr2_bytes': float(d_used.mean()), 'delta_used_vs_wnr2_bytes_CI95': boot(d_used, rng),
            'delta_used_pct_of_raw': float(100 * d_used.mean() / RAW),
            'median_expert_delta_used_bytes': float(np.median(d_used)),
            'delta_used_range_over_experts': [float(d_used.min()), float(d_used.max())],
            'estimate_from_code_lengths': {'est_total_unpadded (4 B flush/stream)': float(np.mean([est_total(r) for r in rr])),
                                           'real_minus_est_weight_payload_bytes': float((wp - estw).mean()),
                                           'real_minus_est_scale_payload_bytes': float((sp - ests).mean()),
                                           'real_minus_est_per_stream_bytes': float(((wp - estw) + (sp - ests)).mean() / NSTREAMS)},
            'model': mk, 'model_blob_bytes': rr[0]['model_blob_bytes'],
            'model_rows_stored': tr.get('rows_stored'), 'model_rows_with_training_counts': tr.get('rows_with_training_counts'),
            'model_compacted_bytes_est': (None if not tr else tr['weight_cdf_bytes_compacted (used rows only)'] + tr['side_cdf_bytes']
                                          + tr['scale_cdf_bytes'] + tr['etab_bytes'] + 32),
            'model_choice_hist': (np.bincount(np.array([r['ks'] for r in rr]).ravel(), minlength=max(1, rr[0]['K'])).tolist()
                                  if rr[0]['K'] else None),
            'max_weight_stream_bytes': int(max(r['max_weight_stream'] for r in rr)),
            'mean_weight_stream_bytes': float(np.mean([r['mean_weight_stream'] for r in rr])),
            'timing_s': {'enc_total_median': float(np.median(enc)), 'dec_total_median': float(np.nanmedian(dec))},
        }
        if rr[0].get('len_bits') is not None:
            tab[c]['length_table_detail'] = {'mean_bytes': float(lt.mean()),
                                             'resid_std_scale_streams': float(np.mean([r['len_resid_std'][0] for r in rr])),
                                             'resid_std_weight_streams': float(np.mean([r['len_resid_std'][1] for r in rr])),
                                             'saving_vs_u16_table_pct_of_raw': float(100 * (38880 - lt.mean()) / RAW)}
    pw = {}
    for name, x, y in PAIRS:
        if x in by and y in by:
            dt = np.array([by[x][i]['sizes']['total'] - by[y][i]['sizes']['total'] for i in common], np.float64)
            du = np.array([by[x][i]['sizes']['used'] - by[y][i]['sizes']['used'] for i in common], np.float64)
            pw[name] = {'delta_total_bytes': float(dt.mean()), 'total_CI95': boot(dt, rng), 'delta_used_bytes': float(du.mean()),
                        'used_CI95': boot(du, rng), 'delta_used_pct_of_raw': float(100 * du.mean() / RAW),
                        'median_used_bytes': float(np.median(du)), 'experts_improved_used': int((du < 0).sum())}
    # entropy-estimate cross-check for f
    xc = {}
    if 'f' in by:
        ef = np.array([est_total(by['f'][i]) - est_total(base[i]) for i in common])
        du = np.array([by['f'][i]['sizes']['used'] - base[i]['sizes']['used'] for i in common], np.float64)
        dw = np.array([by['f'][i]['sizes']['weight_payload'] - base[i]['sizes']['weight_payload'] for i in common], np.float64)
        ds = np.array([by['f'][i]['sizes']['scale_payload'] - base[i]['sizes']['scale_payload'] for i in common], np.float64)
        estw_d = np.array([by['f'][i]['est']['est_weight_payload'] - base[i]['est']['est_weight_payload'] for i in common])
        ests_d = np.array([by['f'][i]['est']['est_scale_payload'] - base[i]['est']['est_scale_payload'] for i in common])
        xc = {'real_used_delta_bytes': float(du.mean()), 'real_used_delta_pct_of_raw': float(100 * du.mean() / RAW),
              'est_delta_same_models_same_experts_bytes': float(ef.mean()),
              'est_delta_pct_of_raw': float(100 * ef.mean() / RAW),
              'real_minus_est_delta_bytes': float(du.mean() - ef.mean()),
              'real_minus_est_delta_CI95': boot(du - ef, rng),
              'weight_payload_delta_real_vs_est': [float(dw.mean()), float(estw_d.mean())],
              'scale_payload_delta_real_vs_est': [float(ds.mean()), float(ests_d.mean())],
              'quoted_entropy_estimate (mxceil_an, two-way half split, 160 experts)': {'bytes': -339507.8, 'pct_of_raw': -1.8058},
              'real_minus_quoted_pct_points': float(100 * du.mean() / RAW + 1.8058),
              'flush_note': 'est assumes 4 B flush per stream; real-minus-est per stream for wnr2 and f in the config table'}
    res = {'experts': common, 'n_experts': len(common), 'raw_record_bytes': RAW, 'configs': tab, 'pairwise': pw,
           'f_entropy_crosscheck': xc}
    json.dump(res, open(out_json, 'w'), indent=1)
    print('experts', len(common))
    for c, t in tab.items():
        print(f"{c:9s} total {t['mean_total_bytes']:12.1f} ratio {t['ratio_of_raw']:.5f} d%raw {t['delta_total_pct_of_raw']:+.3f} "
              f"CI {t['delta_total_pct_of_raw_CI95'][0]:+.3f}..{t['delta_total_pct_of_raw_CI95'][1]:+.3f} med {t['median_expert_delta_total_pct_of_raw']:+.3f} "
              f"bpw {t['bit_per_w (weight payload incl. flush and side symbols)']:.4f} bps {t['bit_per_scale (scale payload incl. flush)']:.4f} "
              f"{t['decode_check']} fb {t['ceil_fallback_records']} blob {t['model_blob_bytes']}")
    for k, p in pw.items():
        print(f"  {k:38s} used {p['delta_used_bytes']:+10.1f} B CI {p['used_CI95'][0]:+.1f}..{p['used_CI95'][1]:+.1f} ({p['delta_used_pct_of_raw']:+.4f}% raw)")
    print(json.dumps(xc, indent=1))


if __name__ == '__main__':
    main()
