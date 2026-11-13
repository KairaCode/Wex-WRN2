import json, sys
s = json.load(open(sys.argv[1]))
for c, t in s['configs'].items():
    e = t['estimate_from_code_lengths']
    print(c, 'used', round(t['mean_used_bytes_unpadded'], 1), 'ratio_used', round(t['ratio_used_of_raw'], 5),
          'd_used%', round(t['delta_used_pct_of_raw'], 4), 'med_used_B', t['median_expert_delta_used_bytes'],
          'CI_tot_B', [round(x) for x in t['delta_total_vs_wnr2_bytes_CI95']], 'd_tot_B', round(t['delta_total_vs_wnr2_bytes'], 1),
          'med_tot_B', t['median_expert_delta_total_bytes'],
          'parts', {k: round(v, 1) for k, v in t['mean_parts'].items()},
          'side_bpw', round(t['side_symbol_bit_per_w (est)'], 5), 'allin_bpw', round(t['all_in_bit_per_w (used bytes)'], 4),
          'real-est/stream', round(e['real_minus_est_per_stream_bytes'], 3),
          'blob', t['model_blob_bytes'], 'rows', t['model_rows_stored'], t['model_rows_with_training_counts'],
          'compact', t['model_compacted_bytes_est'], 'khist', t['model_choice_hist'],
          'lt', t.get('length_table_detail'), 'time', t['timing_s'], 'maxws', t['max_weight_stream_bytes'])
