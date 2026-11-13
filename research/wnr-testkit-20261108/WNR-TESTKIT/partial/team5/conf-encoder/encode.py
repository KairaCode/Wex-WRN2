"""team5 / conf-encoder: encode the 40 held-out experts (CR protocol: odd manifest indices, one per layer) into
standalone WNR3 v1-default records + JSON index; CPU-verify every record (S:441) before writing it.
Usage: python -I encode.py SAMPLE_DIR BLOB OUT_DIR CR_HELDOUT_JSONL [--override-anchor-stop]
"""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W

RAW = 18800640


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    override = '--override-anchor-stop' in sys.argv
    sample_dir, blob_path, out_dir, cr_jsonl = args[:4]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    blob = open(blob_path, 'rb').read()
    model = W.parse_wnm3(blob)
    mid = model['model_id']
    cr = {}
    for l in open(cr_jsonl):
        r = json.loads(l)
        if r['config'] == 'fgde':
            cr[r['idx']] = r
    idxs = sorted(cr)                                   # the 40 CR test experts
    assert len(idxs) == 40 and all(i % 2 for i in idxs)
    # ---------------- anchor pre-pass with set-level stop (S:353-357)
    t0 = time.time()
    pre = []
    for i in idxs:
        e = man['experts'][i]
        raw, mats = W.load_expert(os.path.join(sample_dir, e['file']), e)
        syms, scs, set_ids, cm, st = W.choose_and_transform(mats, model)
        pre.append({'idx': i, 'set_id': set_ids, 'pair': [W.densest_pair(scs[m]) for m in range(3)]})
    stop, pre_rep = False, {}
    for k in range(model['K']):
        for t in range(3):
            vals = [p['pair'][t] for p in pre if p['set_id'][t] == k]
            if not vals:
                continue
            a0 = model['sets'][k]['anchor0'][t]
            mode = int(np.bincount(vals).argmax())
            nd = sum(v != a0 for v in vals)
            frac = nd / len(vals)
            cond = mode != a0 or frac > 0.01
            stop |= cond
            pre_rep[f'set{k}_w{t+1}'] = {'matrices': len(vals), 'anchor0': a0, 'mode_densest_pair': mode,
                                          'differ': nd, 'differ_frac': frac, 'stop_condition': bool(cond),
                                          'differing': [(man['experts'][p['idx']]['file'], p['pair'][t]) for p in pre
                                                        if p['set_id'][t] == k and p['pair'][t] != a0]}
    pre_rep['stop'] = bool(stop)
    pre_rep['override_used'] = bool(stop and override)
    pre_rep['seconds'] = time.time() - t0
    print(json.dumps(pre_rep, indent=1), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(pre_rep, open(os.path.join(out_dir, '..', 'anchor_prepass.json'), 'w'), indent=1)
    if stop and not override:
        print('SET-LEVEL ANCHOR STOP: nothing written (rerun with --override-anchor-stop to record an override)')
        sys.exit(2)
    # ---------------- encode + verify
    desc, summ = W.wnd3_descriptor(blob)
    open(os.path.join(out_dir, 'wnd3_descriptor.bin'), 'wb').write(desc)
    rows = []
    rep_f = open(os.path.join(out_dir, '..', 'build_report.jsonl'), 'w')
    for i in idxs:
        e = man['experts'][i]
        path = os.path.join(sample_dir, e['file'])
        src = open(path, 'rb').read()
        src_sha = hashlib.sha256(src).hexdigest()
        raw, mats = W.load_expert(path, e)
        shapes = [(mt['R'], mt['C']) for mt in mats]
        t1 = time.time()
        rec, rep = W.encode_record(mats, model, mid, e['layer'], e['expert'])
        t2 = time.time()
        if rec and len(rec) >= W.align4096(RAW):
            raise SystemExit('record not smaller than raw: would be stored raw (flags 0)')
        slab, dinfo = W.decode_record(rec, model, rep['entry_flags'], (e['layer'], e['expert']), shapes)
        t3 = time.time()
        dec_sha = hashlib.sha256(slab).hexdigest()
        ok = slab == src
        for m in range(3):                                     # S:441 anchor equality in the CPU verify
            assert dinfo['anchor'][m] == model['sets'][dinfo['set_id'][m]]['anchor0'][m]
        if not ok:
            raise SystemExit(f'VERIFY FAILED {e["file"]}')
        name = e['file'].replace('.bin', '.wnr3')
        open(os.path.join(out_dir, name), 'wb').write(rec)
        c = cr[i]
        row = {'idx': i, 'layer': e['layer'], 'expert': e['expert'], 'source_file': e['file'],
               'source_sha256': src_sha, 'manifest_sha256': e.get('sha256'),
               'source_sha256_matches_manifest': src_sha == e.get('sha256'),
               'record_file': name, 'record_bytes': len(rec), 'record_sha256': hashlib.sha256(rec).hexdigest(),
               'expected_decoded_sha256': dec_sha, 'decoded_equals_source': bool(ok),
               'entry_flags': rep['entry_flags'], 'entry_flags_hex': f"0x{rep['entry_flags']:04X}",
               'record_flags': rep['flags'], 'ceil_mask': rep['ceil_mask'], 'set_id': rep['set_id'],
               'anchor': rep['anchor'], 'bind': [e['layer'], e['expert']], 'shapes_symbols': shapes,
               'header_check_hex': f"0x{rep['header_check']:08X}", 'sizes': rep['sizes'],
               'cr_fgde_g0': {'total': c['sizes']['total'], 'used': c['sizes']['used'],
                              'length_table': c['sizes']['length_table'], 'scale_payload': c['sizes']['scale_payload'],
                              'weight_payload': c['sizes']['weight_payload'], 'header': c['sizes']['header']},
               'enc_s': round(t2 - t1, 2), 'dec_s': round(t3 - t2, 2)}
        rows.append(row)
        rep_f.write(json.dumps({'idx': i, 'file': e['file'], **{k: v for k, v in rep.items()}}) + '\n')
        rep_f.flush()
        print(e['file'], len(rec), c['sizes']['total'], len(rec) - c['sizes']['total'], ok,
              f"LS {rep['sizes']['LS']} vs {c['sizes']['length_table']}", f'enc {t2-t1:.1f}s dec {t3-t2:.1f}s', flush=True)
    rep_f.close()
    tot = np.array([r['record_bytes'] for r in rows], np.float64)
    crt = np.array([r['cr_fgde_g0']['total'] for r in rows], np.float64)
    used = np.array([r['sizes']['used'] for r in rows], np.float64)
    crused = np.array([r['cr_fgde_g0']['used'] for r in rows], np.float64)
    d = tot - crt
    rng = np.random.default_rng(20261107)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)])
    du = used - crused
    bsu = np.array([du[rng.integers(0, du.size, du.size)].mean() for _ in range(10000)])
    summary = {
        'n_records': len(rows), 'mean_total': tot.mean(), 'ratio_of_raw': tot.mean() / RAW,
        'cr_fgde_g0_mean_total': crt.mean(), 'cr_ratio_of_raw': crt.mean() / RAW,
        'delta_mean_B': d.mean(), 'delta_pp_of_raw': 100 * d.mean() / RAW,
        'delta_CI95_B': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        'target_cr_plus_2452': crt.mean() + 2452, 'delta_vs_target_pp': 100 * (tot.mean() - crt.mean() - 2452) / RAW,
        'T9_pass_vs_cr_fgde_g0_pm0.02pp': bool(abs(100 * d.mean() / RAW) <= 0.02),
        'T9_pass_vs_cr_plus_2452_pm0.02pp': bool(abs(100 * (tot.mean() - crt.mean() - 2452) / RAW) <= 0.02),
        'mean_used': used.mean(), 'cr_mean_used': crused.mean(), 'delta_used_mean_B': du.mean(),
        'delta_used_CI95_B': [float(np.percentile(bsu, 2.5)), float(np.percentile(bsu, 97.5))],
        'mean_LS': float(np.mean([r['sizes']['LS'] for r in rows])),
        'cr_mean_length_table': float(np.mean([r['cr_fgde_g0']['length_table'] for r in rows])),
        'mean_SP': float(np.mean([r['sizes']['SP'] for r in rows])),
        'cr_mean_SP': float(np.mean([r['cr_fgde_g0']['scale_payload'] for r in rows])),
        'mean_WP': float(np.mean([r['sizes']['WP'] for r in rows])),
        'cr_mean_WP': float(np.mean([r['cr_fgde_g0']['weight_payload'] for r in rows])),
        'mean_padding': float(np.mean([r['sizes']['padding'] for r in rows])),
        'max_record': int(tot.max()), 'max_over_mean': float(tot.max() / tot.mean()),
        'all_byte_exact': all(r['decoded_equals_source'] for r in rows),
        'all_source_sha_match_manifest': all(r['source_sha256_matches_manifest'] for r in rows),
        'flags_seen': sorted({r['record_flags'] for r in rows}), 'ceil_masks_seen': sorted({r['ceil_mask'] for r in rows}),
    }
    index = {'format': 'WNR3 DRAFT 0.2.1 v1 default profile (CEIL|LENC|HASH, g0), team5/conf-encoder',
             'spec': 'docs/WNR3-DRAFT-v2.1.md',
             'model_file': os.path.basename(blob_path), 'model_bytes': len(blob),
             'model_sha256': hashlib.sha256(blob).hexdigest(), 'model_id': mid, 'model_id_hex': f'0x{mid:08X}',
             'model_summary': summ, 'model_summary_hex': f'0x{summ:08X}',
             'container_header_flags': W.HDR_FLAGS_NR3, 'table_offset': 4096 + W.align4096(len(blob)),
             'wnd3_descriptor_file': 'wnd3_descriptor.bin', 'wnd3_descriptor_hex': desc.hex(),
             'anchor_prepass': {k: v for k, v in pre_rep.items()},
             'chunks': [W.CW, W.CS], 'raw_record_bytes': RAW, 'summary': summary, 'records': rows}
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
