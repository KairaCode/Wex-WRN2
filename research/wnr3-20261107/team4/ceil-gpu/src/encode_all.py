"""Encode real V4.1 expert matrices into every prototype variant.

Usage: python -I encode_all.py SAMPLE_DIR COUNTS_DIR OUT_DIR NEXPERTS [variants]
Writes OUT_DIR/common.npz (raw weights, scales, medians, matrix types), OUT_DIR/var_<V>.npz
(payload, lengths, signs, tables) and OUT_DIR/encode_report.json.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr2fmt as W

VARS_ALL = ['A', 'A259', 'B1', 'B2', 'C', 'CB2', 'D', 'DB2']
MODEL_OF = {'A': 'A', 'A259': 'A259', 'B1': 'A', 'B2': 'A', 'C': 'C', 'CB2': 'C', 'D': 'D', 'DB2': 'D',
            'B2s': 'A259', 'DB2s': 'D'}


def main():
    sample_dir, counts_dir, out_dir, nexp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    variants = sys.argv[5].split(',') if len(sys.argv) > 5 else VARS_ALL
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    models = W.build_models(counts_dir, sample_dir)
    rep = {'model_build_s': time.time() - t0}
    # model in-sample costs over the 160-expert sample (bits per weight, payload only, no flush)
    cnt = models['_counts']
    nW = cnt['A'].sum()
    rep['model_insample_bits_per_w'] = {
        'A_567ctx': W.cost_bits(cnt['A'], models['A']['freq']) / nW,
        'D_mag_only_plus_raw_sign': (W.cost_bits(cnt['D'], models['D']['freq']) + nW) / nW,
        'C_bm7_half_plus_side_bit': (W.cost_bits(cnt['C'], models['C']['freq']) +
                                     W.cost_bits(cnt['Cbit'], models['C']['bitfreq'])) / nW,
    }
    print(json.dumps(rep, indent=1), flush=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    # spread over layers: experts 17, 0, 40, 80, 120, 159, 60, 100, ...
    pick = [17, 0, 40, 80, 120, 159, 60, 100, 20, 140, 5, 90][:nexp]
    mats = []
    for ei in pick:
        e = man['experts'][ei]
        t = W.load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{w}.weight']; sraw, _, _ = t[f'{w}.scale']
            R = wshape[0]
            mats.append({'name': f"{e['file'][:-4]}.{w}", 'mtype': m,
                         'wb': wraw.reshape(R, -1), 'sc': sraw.reshape(R, -1)})
    rep['matrices'] = [m['name'] for m in mats]
    common_raw, common_sc, meds, mtypes = [], [], [], []
    mds = []
    for mt in mats:
        md = W.matrix_data(mt['wb'], mt['sc'])
        mds.append(md)
        common_raw.append(md['raw']); common_sc.append(md['scales']); meds.append(md['med']); mtypes.append(mt['mtype'])
    spm = mds[0]['nstreams']
    assert all(md['nstreams'] == spm for md in mds)
    np.savez(os.path.join(out_dir, 'common.npz'), raw=np.concatenate(common_raw), scales=np.concatenate(common_sc),
             med=np.array(meds, np.uint8), mtype=np.array(mtypes, np.uint8), spm=np.int64(spm),
             names=np.array([m['name'] for m in mats]))
    rep['streams_per_matrix'] = spm
    sizes = {}
    payA = None
    for var in variants:
        tv = time.time()
        pays, lens, signs = [], [], []
        for g in range(0, len(mats), 3):   # one expert (3 matrices) per vectorised encode
            Fs, STs = [], []
            for mt, md in zip(mats[g:g + 3], mds[g:g + 3]):
                if var in ('B2s', 'DB2s'):
                    # two SEPARATE sub-streams per thread: even blocks -> sub-stream 2s, odd blocks -> 2s+1
                    F, ST, sid, K, extra = W.variant_steps(md, 'A' if var == 'B2s' else 'D', mt['mtype'], models)
                    S_ = F.shape[0]
                    F = F.reshape(S_, 32, 2, 32).transpose(0, 2, 1, 3).reshape(2 * S_, 1024)
                    ST = ST.reshape(S_, 32, 2, 32).transpose(0, 2, 1, 3).reshape(2 * S_, 1024)
                    sid = np.zeros(1024, np.int64); K = 1
                else:
                    F, ST, sid, K, extra = W.variant_steps(md, var, mt['mtype'], models)
                Fs.append(F.astype(np.int32)); STs.append(ST.astype(np.int32))
                if 'signs' in extra:
                    signs.append(extra['signs'])
            pay, ln = W.rans_encode_vec(np.concatenate(Fs), np.concatenate(STs), sid, K)
            del Fs, STs
            assert ln.min() >= 4 * K and ln.max() <= 65535, (var, ln.min(), ln.max())
            pays.append(pay); lens.append(ln)
            mi = g; mt = mats[g]; md = mds[g]
            if var == 'A' and mi == 0:
                # scalar reference check on 12 streams
                fr = models['A']['freq'][mt['mtype']]
                cum = np.zeros((567, 17), np.int64); cum[:, 1:] = np.cumsum(fr, 1)
                rows = ((md['srel'][:, None] * 9 + md['pm']) * 9 + md['mx']).reshape(spm, -1)
                syms = md['blk'].reshape(spm, -1)
                offs = np.concatenate([[0], np.cumsum(ln)])
                for s in list(range(6)) + list(range(spm - 6, spm)):
                    ref = W.encode_ref(syms[s].tolist(), rows[s].tolist(), fr, cum)
                    assert ref == pay[offs[s]:offs[s + 1]].tobytes(), ('scalar mismatch', s)
                rep['scalar_reference_check'] = '12/12 streams byte-identical (matrix 0)'
        pay = np.concatenate(pays); ln = np.concatenate(lens)
        if var == 'A':
            payA = pay
        if var == 'A259' and payA is not None:
            rep['A259_bytes_identical_to_A'] = bool(pay.size == payA.size and (pay == payA).all())
        mod = models[MODEL_OF[var]]
        tabs = np.stack([W.table_u16(mod, m) for m in range(3)])  # [3, rows, alpha]
        save = {'payload': pay, 'lengths': ln.astype(np.uint16), 'tables': tabs}
        if var in ('C', 'CB2'):
            save['bittab'] = np.stack([mod['bitfreq'][m][:, 0] for m in range(3)]).astype(np.uint16)  # c1 per srel
        if signs:
            save['signs'] = np.concatenate(signs)
        np.savez(os.path.join(out_dir, f'var_{var}.npz'), **save)
        nW = len(mats) * spm * 2048
        sign_bytes = sum(s.size * 4 for s in signs)
        tot = pay.size + 2 * ln.size + sign_bytes
        sizes[var] = {'payload_bytes': int(pay.size), 'length_table_bytes': int(2 * ln.size), 'sign_plane_bytes': int(sign_bytes),
                      'total_bytes': int(tot), 'bits_per_weight_total': 8 * tot / nW,
                      'bits_per_weight_rans_only': 8 * pay.size / nW,
                      'mean_stream_bytes': float(ln.mean()), 'max_stream_bytes': int(ln.max()),
                      'encode_s': time.time() - tv}
        print(var, json.dumps(sizes[var]), flush=True)
    base = sizes.get('A', {}).get('total_bytes')
    if base:
        for v in sizes:
            sizes[v]['delta_vs_A_pct'] = 100 * (sizes[v]['total_bytes'] - base) / base
    rep['sizes'] = sizes
    rep['total_s'] = time.time() - t0
    rname = 'encode_report.json' if variants == VARS_ALL else 'encode_report_' + '_'.join(variants) + '.json'
    json.dump(rep, open(os.path.join(out_dir, rname), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
