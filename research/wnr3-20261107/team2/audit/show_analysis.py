import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
sps = ['S1_manifest_parity (theirs)', 'S2_layer_parity', 'S3_slot_in_layer_4fold', 'S4_layer_blocks_4fold (0-9,10-19,20-29,30-39)']
for k, r in d['variants'].items():
    s = f"{k:14s} rows {r['rows']:6d} used {r['rows_used_max']:5d} ideal {r['ideal_in_bit_per_w']:.5f} in {r['in_bit_per_w']:.6f} ({r['in_delta_pct']:+.3f}%)"
    for sp in sps:
        s += f" | {sp[:2]} {r['ho_' + sp + '_bit_per_w']:.5f} {r['ho_' + sp + '_delta_pct']:+.3f}%"
    print(s)
for k in ['per_expert_bit_per_w', 'scales', 'per_expert_scale_bits']:
    print(k, d[k])
