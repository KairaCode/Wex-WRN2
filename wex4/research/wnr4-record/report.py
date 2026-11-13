# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Reduce the experiment outputs to results.json and print the tables of RESULTS.md.

  python -I report.py out/proto.json out/container_scan.json out/layer_model.json results.json [CACHE_DIR]

Savings are unpadded record bytes against the production WNR3-CZ record of the same expert
(candidate records carry a 40-B header, 8 B more than WNR3). "container" figures marked
extrapolated are sample mean x 7,680; the ones marked exact come from the scan of every record.
"""
import glob, json, os, struct, sys
import numpy as np

RAW = 18800640
RECORDS = 7680


def stats(v):
    v = np.asarray(v, np.float64)
    return {'mean': round(float(v.mean()), 1), 'min': int(v.min()), 'max': int(v.max()),
            'pct_of_raw_mean': round(float(v.mean()) / RAW * 100, 4)}


def length_alternatives(cache):
    """Other ways to code the 2048:512 length table of the cached production records (bytes per record)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fetch import ENTRY, lenc_group_bytes
    table = open(os.path.join(cache, 'entries.bin'), 'rb').read()
    out = {}
    for path in sorted(glob.glob(os.path.join(cache, 'rec_*.bin'))):
        layer, expert = map(int, os.path.basename(path)[4:-4].split('_'))
        e = ENTRY.unpack_from(table, (layer * 192 + expert) * ENTRY.size)
        head = open(path, 'rb').read(32 + 2 * 19440)
        wchunk, schunk = struct.unpack_from('<2I', head, 16)
        parts = [(e[4 + 2 * m] * e[5 + 2 * m] + schunk - 1) // schunk for m in range(3)] + \
                [(e[10 + 2 * m] * e[11 + 2 * m] * 2 + wchunk - 1) // wchunk for m in range(3)]
        lengths = np.frombuffer(head, '<u2', sum(parts), 32).astype(np.int64)
        row = {'u16': 2 * len(lengths)}
        for g in (16, 32, 64, 128):
            row[f'group{g}_min_width'] = lenc_group_bytes(lengths, g)
        fixed = h0 = hd = at = 0
        for n in parts:   # one (kind, matrix) part at a time
            part = lengths[at:at + n]; at += n
            fixed += 3 + (n * int(np.ceil(np.log2(part.max() - part.min() + 1))) + 7) // 8
            for key, sym in (('h0', part - int(np.median(part))), ('hd', np.diff(part, prepend=part[0]))):
                p = np.unique(sym, return_counts=True)[1] / n
                bits = float(-(p * np.log2(p)).sum()) * n / 8
                if key == 'h0': h0 += bits
                else: hd += bits
        row['fixed_width_per_matrix'] = fixed
        row['entropy_bound_len_minus_median'] = h0      # ideal static code per matrix, table cost not counted
        row['entropy_bound_delta_previous'] = hd
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return {k: round(float(np.mean(v)), 1) for k, v in out.items()} | {'records': len(next(iter(out.values())))}


def main():
    proto = json.load(open(sys.argv[1]))
    scan = json.load(open(sys.argv[2]))
    layer = json.load(open(sys.argv[3]))
    cache = sys.argv[5] if len(sys.argv) > 5 else None
    ex = proto['experts']
    cfg = np.array([e['cfg'] for e in ex], np.int64)          # [expert][config][column]
    used0 = cfg[:, 0, 1]
    assert all(e['harness_ok'] for e in ex) and (cfg[:, 0, 1] == [e['used_bytes'] for e in ex]).all()
    rows = np.array(scan['rows'], np.int64)
    total, used = rows[:, 2], rows[:, 3]
    res = {
        'data': 'DeepSeek V4.1 Flash REAP-192 MXFP4, production WNR3-CZ container, production WNM2 model',
        'raw_record_bytes': RAW,
        'container_exact': {
            'records': len(rows), 'record_bytes_sum': int(total.sum()), 'used_bytes_sum': int(used.sum()),
            'mean_used_bytes': round(float(used.mean()), 1), 'mean_used_of_raw': round(float(used.mean()) / RAW, 5),
            'padding_bytes_sum': int((total - used).sum()),
            'saved_by_alignment': {str(a): int(total.sum() - ((used + a - 1) // a * a).sum()) for a in (512, 64, 1)},
            'length_table_u16_bytes_per_record': 38880,
            'lenc_group32_bytes_per_record': stats(rows[:, 7]), 'lenc_group64_bytes_per_record': stats(rows[:, 8]),
            'lenc_group32_saved_sum': int((2 * rows[:, 4] - rows[:, 7]).sum()),
        },
        'sample': {
            'experts': [f"{e['layer']}:{e['expert']}" for e in ex], 'count': len(ex),
            'harness_byte_exact': sum(e['harness_ok'] for e in ex),
            'mean_used_bytes': round(float(used0.mean()), 1), 'mean_used_of_raw': round(float(used0.mean()) / RAW, 5),
            'reference_decode_s_mean': round(float(np.mean([e['ref_decode_s'] for e in ex])), 3),
            'offsets_table_host_us': {'u16_prefix_sum': round(float(np.mean([e['offsets_u16_us'] for e in ex])), 1),
                                      'lenc_unpack_and_prefix_sum': round(float(np.mean([e['offsets_lenc_us'] for e in ex])), 1)},
            'zero_magnitude_share': round(float(np.mean([e['zero_magnitudes'] / e['symbols'] for e in ex])), 5),
        },
        'candidates': [],
    }
    # where the bytes of a production record go (sample mean)
    ideal = np.array([(e['ideal_weight_bits'] + e['ideal_scale_bits']) / 8 for e in ex])
    payload = cfg[:, 0, 3] + cfg[:, 0, 4]
    res['sample']['overhead_per_stream_bytes'] = {
        'length_table': 2.0, 'payload_minus_model_code_length': round(float(((payload - ideal) / cfg[:, 0, 5]).mean()), 3),
        'streams_with_3_byte_state_share': round(float((cfg[:, 2, 6] / cfg[:, 2, 5]).mean()), 4)}
    res['sample']['bits_per_weight_model'] = round(float(np.mean([e['ideal_weight_bits'] / e['symbols'] for e in ex])), 4)
    res['sample']['bits_per_scale_model'] = round(float(np.mean([e['ideal_scale_bits'] / e['scales'] for e in ex])), 4)
    print('| candidate | weight:scale chunk | streams | round trip | saved B/record mean (min..max) | % of raw | container, extrapolated |')
    print('|---|---|---|---|---|---|---|')
    for i, c in enumerate(proto['configs']):
        saved = used0 - cfg[:, i, 1]
        s = stats(saved)
        s.update(name=c['name'], config=c, streams=int(cfg[0, i, 5]), round_trip=f"{int(cfg[:, i, 0].sum())}/{len(ex)}",
                 length_section_bytes_mean=round(float(cfg[:, i, 2].mean()), 1),
                 bit_flip_rejected=f"{int(cfg[:, i, 7].sum())}/{len(ex)}" if c['lenc'] and c['sd'] and c['wchunk'] == 2048 else None,
                 container_saved_bytes_extrapolated=int(round(float(saved.mean()) * RECORDS)))
        res['candidates'].append(s)
        print(f"| {c['name']} | {c['wchunk']}:{c['schunk']} | {s['streams']} | {s['round_trip']} | "
              f"{s['mean']:,.0f} ({s['min']:,}..{s['max']:,}) | {s['pct_of_raw_mean']:.3f} | "
              f"{s['container_saved_bytes_extrapolated'] / 1e6:,.1f} MB |")
    # sample vs whole container, on the one candidate that is exact for both
    res['sample']['lenc_group32_saved_mean_sample_vs_container'] = [
        round(float((cfg[:, 0, 2] - cfg[:, 1, 2]).mean()), 1), round(float((2 * rows[:, 4] - rows[:, 7]).mean()), 1)]
    lr = layer['rows']
    res['layer_model'] = {
        'train_experts_per_layer': layer['train_experts_per_layer'], 'layers': layer['layers'], 'rows': lr,
        'layer_model_saved_bytes': stats([r['prod_used'] - r['layer_used'] for r in lr]),
        'pooled_model_saved_bytes': stats([r['prod_used'] - r['pooled_used'] for r in lr])}
    if cache:
        res['length_table_alternatives_bytes_per_record'] = length_alternatives(cache)
    json.dump(res, open(sys.argv[4], 'w'), indent=1)
    for k in ('container_exact', 'sample', 'layer_model', 'length_table_alternatives_bytes_per_record'):
        if k in res:
            print(k, json.dumps({a: b for a, b in res[k].items() if a not in ('rows', 'experts')}, indent=1))


if __name__ == '__main__':
    main()
