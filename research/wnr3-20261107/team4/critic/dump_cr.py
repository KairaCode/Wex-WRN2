import json, sys
RAW = 18800640
NW = 35389440
NS = 1105920
for fn in [r'<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary.json',
           r'<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary_g0.json']:
    d = json.load(open(fn))
    print('=====', fn, 'n_experts', d.get('n_experts'), 'raw', d.get('raw_record_bytes'))
    print('experts', d['experts'][:5], '...', len(d['experts']))
    for k, c in d['configs'].items():
        mp = c['mean_parts']
        tot = c['mean_total_bytes']
        s = sum(mp.values())
        bw = mp['weight_payload'] * 8 / NW
        bs = mp['scale_payload'] * 8 / NS
        print(f"{k:9s} tot {tot:14.1f} ratio {c['ratio_of_raw']:.5f} chk {tot/RAW:.5f} sumparts {s:12.1f} "
              f"bw {c['bit_per_w (weight payload incl. flush and side symbols)']:.4f}/{bw:.4f} "
              f"bs {c['bit_per_scale (scale payload incl. flush)']:.4f}/{bs:.4f} "
              f"LT {mp['length_table']:.1f} SP {mp['scale_payload']:.1f} WP {mp['weight_payload']:.1f} pad {mp['padding']:.1f} hdr {mp['header']} "
              f"dpct {c.get('delta_total_pct_of_raw')} CI {c.get('delta_total_pct_of_raw_CI95')} med {c.get('median_expert_delta_total_pct_of_raw')} "
              f"dec {c['decode_check']} bx {c['byte_exact_vs_source']} fb {c['ceil_fallback_records']} flags {c['record_flags_seen']} "
              f"blob {c.get('model_blob_bytes')} rows {c.get('model_rows_with_training_counts')} comp {c.get('model_compacted_bytes_est')} "
              f"t {c.get('timing_s')} side {c.get('side_symbol_bit_per_w (est)')} hist {c.get('model_choice_hist')}")
    print('--- pairwise')
    print(json.dumps(d['pairwise'], indent=0)[:6000])
    print('--- crosscheck')
    print(json.dumps(d['f_entropy_crosscheck'], indent=0)[:3000])
