"""End-to-end real rANS check of ceil-form with team 3's vectorised WNR2 reference coder (COPY: wnr_common_t3copy.py).

Usage: python -I -B e2e_real.py COUNTS_SAMPLE_NPZ SAMPLE_DIR EXTRA_DIR OUT_JSON
Models (shared, one per matrix type, 567 contexts + order-0 scales) are trained on the even-indexed sample experts
(groups 0, 2 of refute_count.py) in two forms: original and ceil-form. Each test expert (held-out: odd-indexed
sample experts i%4==1, extra routed experts, MTP experts) is encoded as
  orig : WNR2 record of the original bytes with the original-trained model
  ceil : WNR2-layout record of T(x) with the ceil-trained model
  ceilE: WNR3-layout record of T(x) with the per-stream hash (x0 = L + h23 over the CODED = transformed symbols)
then decoded with the reference decoder; the decoded slab must equal T(x); T^-1(slab) must equal the expert file
byte for byte (= the WEX1 slab layout: scales w1..w3 then packed weights w1..w3).
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr_common_t3copy as wc
import refute_count as rc


def mats_from_bytes(scs, wbs, shapes):
    mats = []
    for m in range(3):
        R, C = shapes[m]
        w2 = wbs[m].reshape(R, C // 2)
        sym = np.empty((R, C), np.uint8)
        sym[:, 0::2] = w2 & 15
        sym[:, 1::2] = w2 >> 4
        mats.append({'sym': sym.ravel(), 'sc': np.ascontiguousarray(scs[m]), 'R': R, 'C': C})
    return mats


def main():
    cnp, sample_dir, extra_dir, out_json = sys.argv[1:5]
    z = np.load(cnp)
    tr = [0, 2]
    tabs = {}
    for form, wk, sk in [('orig', 'o_base', 's_o'), ('ceil', 'c_base', 's_c')]:
        wf = wc.quantize(z[wk][tr].sum(0)).reshape(3 * 567, 16)
        sf = wc.quantize(z[sk][tr].sum(0))
        tabs[form] = wc.Tables(wf, sf)
    man_s = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    man_e = json.load(open(os.path.join(extra_dir, 'manifest.json')))
    tests = []
    for i, e in enumerate(man_s['experts']):
        if i % 4 == 1 and e['layer'] in (0, 5, 10, 15, 20, 25, 30, 35, 39):
            tests.append((sample_dir, e, 'sample_odd'))
    seen = set()
    for e in man_e['experts']:
        if e['kind'] == 'layers' and e['layer'] in (2, 9, 17, 24, 31, 38) and e['layer'] not in seen:
            tests.append((extra_dir, e, 'extra_routed')); seen.add(e['layer'])
    for e in man_e['experts']:
        if e['kind'] == 'mtp' and e['file'] in ('M00_E%03d.bin' % e['expert'], 'M02_E%03d.bin' % e['expert']):
            if not any(t[1]['kind'] == 'mtp' and t[1]['layer'] == e['layer'] for t in tests if 'kind' in t[1]):
                tests.append((extra_dir, e, 'mtp'))
    res = {'models': {f: tabs[f].model_id for f in tabs}, 'records': []}
    t0 = time.time()
    for ddir, e, kind in tests:
        raw = np.fromfile(os.path.join(ddir, e['file']), dtype=np.uint8)
        off, parts = 0, {}
        for t in e['tensors']:
            parts[t['name']] = (raw[off:off + t['bytes']], t['shape']); off += t['bytes']
        shapes = [(parts[f'w{k}.weight'][1][0], parts[f'w{k}.weight'][1][1] * 2) for k in (1, 2, 3)]
        scs = [parts[f'w{k}.scale'][0] for k in (1, 2, 3)]
        wbs = [parts[f'w{k}.weight'][0] for k in (1, 2, 3)]
        tscs, twbs = [], []
        for m in range(3):
            tw, ts, even, _ = rc.T_fwd(wbs[m].reshape(-1, 16), scs[m])
            assert ts.max() <= 255
            tscs.append(ts.astype(np.uint8)); twbs.append(tw.ravel())
        traw = np.concatenate(tscs + twbs)
        row = {'file': e['file'], 'set': kind, 'layer': e['layer'], 'expert': e['expert']}
        for name, form, cfg, scs_, wbs_, ref in [
                ('orig', 'orig', wc.Config('wnr2', wnr2=True), scs, wbs, raw),
                ('ceil', 'ceil', wc.Config('wnr2', wnr2=True), tscs, twbs, traw),
                ('ceilE', 'ceil', wc.Config('e', e=True), tscs, twbs, traw)]:
            mats = mats_from_bytes(scs_, wbs_, shapes)
            rec, info, _, _ = wc.encode_record(mats, tabs[form], cfg, e['layer'], e['expert'])
            slab, dinfo = wc.decode_record(rec, tabs[form], e['layer'], e['expert'], shapes)
            ok_coded = bool(np.array_equal(slab, ref))
            r = {'total': info['total'], 'used': info['used'], 'scale_payload': info['scale_payload'],
                 'weight_payload': info['weight_payload'], 'decoded_equals_coded_form': ok_coded}
            if form == 'ceil':
                # inverse T on the decoded slab, then byte-compare with the original expert bytes
                nsc = [s.size for s in scs]
                p = 0; osc = []; ow = []
                dsc = [slab[sum(nsc[:m]):sum(nsc[:m + 1])] for m in range(3)]
                base = sum(nsc)
                dw = []
                for m in range(3):
                    n = wbs[m].size
                    dw.append(slab[base:base + n]); base += n
                for m in range(3):
                    iw, isc, _ = rc.T_inv(dw[m].reshape(-1, 16), dsc[m])
                    osc.append(isc.astype(np.uint8)); ow.append(iw.ravel())
                r['inverse_equals_original_file'] = bool(np.array_equal(np.concatenate(osc + ow), raw))
            row[name] = r
        row['ceil_vs_orig_total_pct_of_raw'] = 100 * (row['ceil']['total'] - row['orig']['total']) / raw.size
        row['ceil_vs_orig_used_pct_of_raw'] = 100 * (row['ceil']['used'] - row['orig']['used']) / raw.size
        row['hash_cost_bytes'] = row['ceilE']['used'] - row['ceil']['used'] - 16  # WNR3 header is 48 B vs 32 B
        res['records'].append(row)
        print(json.dumps({k: v for k, v in row.items()}), f'{time.time() - t0:.0f}s', flush=True)
        json.dump(res, open(out_json, 'w'), indent=1)
    recs = res['records']
    res['summary'] = {
        'n': len(recs), 'all_decoded_ok': all(r[k]['decoded_equals_coded_form'] for r in recs for k in ('orig', 'ceil', 'ceilE')),
        'all_inverse_byte_exact': all(r[k]['inverse_equals_original_file'] for r in recs for k in ('ceil', 'ceilE')),
        'mean_used_pct_by_set': {s: float(np.mean([r['ceil_vs_orig_used_pct_of_raw'] for r in recs if r['set'] == s])) for s in {r['set'] for r in recs}},
        'mean_ratio_orig_total': float(np.mean([r['orig']['total'] / 18800640 for r in recs])),
        'mean_ratio_ceil_total': float(np.mean([r['ceil']['total'] / 18800640 for r in recs])),
        'mean_hash_cost_bytes': float(np.mean([r['hash_cost_bytes'] for r in recs])),
    }
    json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps(res['summary'], indent=1))


if __name__ == '__main__':
    main()
