"""Portable copy of team 4 ceil-gpu/src/ceilenc.py (encoder of the 8 GPU test experts), as a library.

Changes vs the original (logic unchanged):
  * the models come from the kit file ref/model_counts.npz (the sums over the 160 sample experts that ceilenc.py
    built from the team-1 count dirs; tables derived from it were checked equal to team 4's var_*.npz tables);
  * all six weight variants (A259, D, CF, CFD, DB2, CFDB2) are encoded in one pass (team 4 ran DB2/CFDB2 separately
    with identical code);
  * the per-expert ceil-count cross-check uses the sha256 of team 4's w_ceil counts (ref/t4enc_expected.json);
  * every written array is compared with the sha256 of team 4's encoder output (reproducibility check).
Output format = team 4's enc dir (common.npz, var_*.npz, scl_*.npz, encode_report.json), so team 5's gpu-v1 bench can
use it as its T4_ENC_DIR.
"""
import hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr2fmt as W

PICK = [17, 0, 40, 80, 120, 159, 60, 100]   # team 2 encode_all.py:36 with NEXPERTS = 8
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)
SPS = 512   # scales per scale stream
ALL_VARIANTS = ('A259', 'D', 'CF', 'CFD', 'DB2', 'CFDB2')


def unpack(wb):
    R = wb.shape[0]
    sym = np.empty((R, wb.shape[1] * 2), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    return sym


def pack(sym):
    return (sym[:, 0::2] | (sym[:, 1::2] << 4)).astype(np.uint8)


def ceil_forward(wb, sc):
    sym = unpack(wb)
    R, C = sym.shape
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    bmax = mag.max(1)
    has13 = ((mag == 1) | (mag == 3)).any(1)
    even = (bmax == 7) & ~has13
    tb = np.where(even[:, None], (blk & 8) | HALF[mag], blk).astype(np.uint8)
    s = sc.ravel().astype(np.int16)
    ts = s + even
    stats = {'blocks': int(even.size), 'even': int(even.sum()), 'max_below_6': int((bmax < 6).sum()),
             'scale_overflow': int((ts > 254).sum()), 'max_hist': np.bincount(bmax, minlength=8).tolist()}
    return pack(tb.reshape(R, C)), ts.astype(np.uint8).reshape(sc.shape), even, stats


def ceil_inverse(twb, tsc):
    sym = unpack(twb)
    R, C = sym.shape
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    odd = mag.max(1) == 5
    ob = np.where(odd[:, None], (blk & 8) | DOUBLE[mag], blk).astype(np.uint8)
    os_ = tsc.ravel().astype(np.int16) - odd
    return pack(ob.reshape(R, C)), os_.astype(np.uint8).reshape(tsc.shape)


def ctx_counts(wb, sc):
    md = W.matrix_data(wb, sc)
    rows = (md['srel'][:, None] * 9 + md['pm']) * 9 + md['mx']
    return np.bincount((rows * 16 + md['blk']).ravel(), minlength=567 * 16).reshape(567, 16)


def models_from_counts(wb):
    """wb int64 [3, 567, 16] -> {'A259': ..., 'D': ...} (same construction as wnr2fmt.build_models)."""
    mp = W.ctx567_to_259()
    assert wb[:, mp < 0].sum() == 0, 'unreachable 567-row with counts'
    fA = W.quantize(wb)
    f259 = np.full((3, 259, 16), W.P // 16, np.int64)
    f259[:, mp[mp >= 0]] = fA[:, mp >= 0]
    c259 = np.zeros((3, 259, 16), np.int64)
    c259[:, mp[mp >= 0]] = wb[:, mp >= 0]
    mag = wb[..., :8] + wb[..., 8:]
    m259 = np.zeros((3, 259, 8), np.int64)
    m259[:, mp[mp >= 0]] = mag[:, mp >= 0]
    return {'A259': {'rows': 259, 'alpha': 16, 'freq': f259, 'counts': c259},
            'D': {'rows': 259, 'alpha': 8, 'freq': W.quantize(m259), 'counts': m259}}


def scale_model(cnt):
    f = W.quantize(cnt)   # [3, 256], every symbol >= 1 (WNR2 rule)
    cum = np.zeros((3, 257), np.int64); cum[:, 1:] = np.cumsum(f, 1)
    lut = np.zeros((3, 16384), np.uint8)
    for m in range(3):
        lut[m] = np.repeat(np.arange(256), f[m]).astype(np.uint8)
    return f, cum, lut


def load_models(counts_npz):
    z = np.load(counts_npz)
    wb_o, wb_c, s_o, s_c = (z[k].astype(np.int64) for k in ('w_orig', 'w_ceil', 's_orig', 's_ceil'))
    MO = models_from_counts(wb_o)
    MC = models_from_counts(wb_c)
    return {'wb_o': wb_o, 'wb_c': wb_c, 's_o': s_o, 's_c': s_c, 'MO': MO, 'MC': MC,
            'so': scale_model(s_o), 'sc': scale_model(s_c)}


def variant_table(M):
    return {'A259': (False, M['MO']['A259'], 'A259'), 'D': (False, M['MO']['D'], 'D'),
            'CF': (True, M['MC']['A259'], 'A259'), 'CFD': (True, M['MC']['D'], 'D'),
            'DB2': (False, M['MO']['D'], 'DB2'), 'CFDB2': (True, M['MC']['D'], 'DB2')}


def encode_matrix(wb, sc, m, M, variants, scalar_check=False):
    """Encode one matrix (packed weights wb [R, C/2], scales sc [R, C/32], type m) -> dict of per-variant
    (payload, lengths, signs or None) + scale streams + transform data. Used by encode_all and by smoke checks."""
    VT = variant_table(M)
    twb, tsc, even, st = ceil_forward(wb, sc)
    iwb, isc = ceil_inverse(twb, tsc)
    rt_ok = bool(np.array_equal(iwb, wb) and np.array_equal(isc, sc))
    mdo = W.matrix_data(wb, sc); mdc = W.matrix_data(twb, tsc)
    out = {'twb': twb, 'tsc': tsc, 'stats': st, 'cpu_roundtrip_ok': rt_ok, 'mdo_med': mdo['med'], 'mdc_med': mdc['med'],
           'spm': mdo['nstreams'], 'own_ceil_counts': ctx_counts(twb, tsc), 'var': {}, 'scl': {}}
    sfo, scumo, _ = M['so']; sfc, scumc, _ = M['sc']
    for key, s_, f_, c_ in (('orig', sc, sfo, scumo), ('ceil', tsc, sfc, scumc)):
        s1 = s_.ravel().astype(np.int64).reshape(-1, SPS)
        F = f_[m][s1]; ST = c_[m][s1]
        pay, ln = W.rans_encode_vec(F, ST, np.zeros(SPS, np.int64), 1)
        if scalar_check:
            offs = np.concatenate([[0], np.cumsum(ln)])
            for si in (0, 1, s1.shape[0] - 1):
                ref = W.encode_ref(s1[si].tolist(), [m] * SPS, f_, c_)
                assert ref == pay[offs[si]:offs[si + 1]].tobytes(), ('scale scalar mismatch', key, si)
        out['scl'][key] = (pay, ln)
    for v in variants:
        isceil, model, kind = VT[v]
        md = mdc if isceil else mdo
        mods = {'A259': model, 'D': model}   # variant_steps looks the model up under 'A259' or 'D'
        F, ST, sid, K, extra = W.variant_steps(md, kind, m, mods)
        pay, ln = W.rans_encode_vec(F.astype(np.int32), ST.astype(np.int32), sid, K)
        del F, ST
        assert ln.min() >= 4 and ln.max() <= 65535
        out['var'][v] = (pay, ln, extra.get('signs'))
    return out


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def encode_all(man, paths, counts_npz, out_dir, log, expected=None, variants=ALL_VARIANTS, pick=PICK):
    """Encode the 8 experts into out_dir (team-4 enc format). Returns the report dict."""
    import gpukit_common as K
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    M = load_models(counts_npz)
    VT = variant_table(M)
    nW = M['wb_o'].sum(); nS = M['s_o'].sum()
    sfo, scumo, sluto = M['so']; sfc, scumc, slutc = M['sc']
    rep = {'pick': list(pick), 'repo': man.get('repo'), 'revision': man.get('revision'), 'variants': list(variants),
           'model_counts_file_sha256': K.sha256_file(counts_npz)}
    rep['model_insample'] = {
        'A259_bits_per_w': W.cost_bits(M['MO']['A259']['counts'], M['MO']['A259']['freq']) / nW,
        'D_bits_per_w_incl_raw_sign': (W.cost_bits(M['MO']['D']['counts'], M['MO']['D']['freq']) + nW) / nW,
        'CF_bits_per_w': W.cost_bits(M['MC']['A259']['counts'], M['MC']['A259']['freq']) / nW,
        'CFD_bits_per_w_incl_raw_sign': (W.cost_bits(M['MC']['D']['counts'], M['MC']['D']['freq']) + nW) / nW,
        'scale_orig_bits_per_scale': W.cost_bits(M['s_o'], sfo) / nS,
        'scale_ceil_bits_per_scale': W.cost_bits(M['s_c'], sfc) / nS,
        'n_weights': int(nW), 'n_scales': int(nS)}
    acc = {v: {'pay': [], 'len': [], 'signs': []} for v in variants}
    sacc = {k: {'pay': [], 'len': []} for k in ('orig', 'ceil')}
    raw_all, sco_all, scc_all, med_o, med_c, mtypes, names = [], [], [], [], [], [], []
    tstats = {'blocks': 0, 'even': 0, 'max_below_6': 0, 'scale_overflow': 0, 'max_hist': np.zeros(8, np.int64),
              'cpu_roundtrip_fail_matrices': 0, 'own_counts_eq_mxceil_counts_experts': 0, 'sha256_ok': 0}
    exp_wc = (expected or {}).get('pick_w_ceil_counts_sha256', {})
    spm = None
    for ei in pick:
        e = man['experts'][ei]
        tstats['sha256_ok'] += int(K.sha256_file(paths[e['file']]) == e['sha256'])
        t = K.load_record(paths[e['file']], e)
        own = np.zeros((3, 567, 16), np.int64)
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{w}.weight']; sraw, _, _ = t[f'{w}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1); sc = sraw.reshape(R, -1)
            r = encode_matrix(wb, sc, m, M, variants, scalar_check=(ei == pick[0] and m == 0))
            for k in ('blocks', 'even', 'max_below_6', 'scale_overflow'):
                tstats[k] += r['stats'][k]
            tstats['max_hist'] += np.array(r['stats']['max_hist'])
            tstats['cpu_roundtrip_fail_matrices'] += int(not r['cpu_roundtrip_ok'])
            own[m] = r['own_ceil_counts']
            raw_all.append(wb.ravel().copy()); sco_all.append(sc.ravel().copy()); scc_all.append(r['tsc'].ravel().copy())
            med_o.append(r['mdo_med']); med_c.append(r['mdc_med']); mtypes.append(m); names.append(f"{e['file'][:-4]}.{w}")
            spm = r['spm']
            for key in ('orig', 'ceil'):
                sacc[key]['pay'].append(r['scl'][key][0]); sacc[key]['len'].append(r['scl'][key][1])
            for v in variants:
                pay, ln, sg = r['var'][v]
                acc[v]['pay'].append(pay); acc[v]['len'].append(ln)
                if sg is not None:
                    acc[v]['signs'].append(sg)
            del r
        if e['file'] in exp_wc:
            tstats['own_counts_eq_mxceil_counts_experts'] += int(_sha(own.astype(np.int64)) == exp_wc[e['file']])
        log('encoded', e['file'], 'even%%=%.3f' % (100 * tstats['even'] / tstats['blocks']), 'elapsed %.1f s' % (time.time() - t0))
    nmat = len(mtypes)
    np.savez(os.path.join(out_dir, 'common.npz'), raw=np.concatenate(raw_all), scales=np.concatenate(sco_all),
             scales_ceil=np.concatenate(scc_all), med=np.array(med_o, np.uint8), med_ceil=np.array(med_c, np.uint8),
             mtype=np.array(mtypes, np.uint8), spm=np.int64(spm), sspm=np.int64(368640 // SPS), names=np.array(names))
    del raw_all, sco_all, scc_all
    tstats['max_hist'] = tstats['max_hist'].tolist()
    tstats['even_pct'] = 100 * tstats['even'] / tstats['blocks']
    rep['transform'] = tstats
    rep['medians'] = {'orig': [int(x) for x in med_o], 'ceil': [int(x) for x in med_c]}
    sizes = {}
    nWs = nmat * spm * 2048
    nSc = nmat * 368640
    for key, (c_, l_) in (('orig', (scumo, sluto)), ('ceil', (scumc, slutc))):
        pay = np.concatenate(sacc[key]['pay']); ln = np.concatenate(sacc[key]['len'])
        np.savez(os.path.join(out_dir, f'scl_{key}.npz'), payload=pay, lengths=ln.astype(np.uint16),
                 cum=c_.astype(np.uint16), lut=l_)
        sizes[f'scales_{key}'] = {'payload_bytes': int(pay.size), 'length_table_bytes': int(2 * ln.size),
                                  'bits_per_scale_payload': 8 * pay.size / nSc,
                                  'bits_per_scale_incl_lengths': 8 * (pay.size + 2 * ln.size) / nSc,
                                  'streams': int(ln.size), 'mean_stream_bytes': float(ln.mean()), 'max_stream_bytes': int(ln.max())}
    for v in variants:
        isceil, model, kind = VT[v]
        pay = np.concatenate(acc[v]['pay']); ln = np.concatenate(acc[v]['len'])
        save = {'payload': pay, 'lengths': ln.astype(np.uint16),
                'tables': np.stack([np.cumsum(model['freq'][m], 1).astype(np.uint16) for m in range(3)])}
        sign_bytes = 0
        if acc[v]['signs']:
            save['signs'] = np.concatenate(acc[v]['signs']); sign_bytes = save['signs'].size * 4
        np.savez(os.path.join(out_dir, f'var_{v}.npz'), **save)
        acc[v] = None
        sk = 'scales_ceil' if isceil else 'scales_orig'
        scb = sizes[sk]['payload_bytes'] + sizes[sk]['length_table_bytes']
        wt = pay.size + 2 * ln.size + sign_bytes
        sizes[v] = {'weight_payload_bytes': int(pay.size), 'weight_length_table_bytes': int(2 * ln.size),
                    'sign_plane_bytes': int(sign_bytes), 'weight_total_bytes': int(wt),
                    'scale_streams_bytes': int(scb), 'record_total_bytes_all8': int(wt + scb),
                    'per_record_bytes': (wt + scb) / (nmat // 3),
                    'ratio_of_raw_record_18800640': (wt + scb) / (nmat // 3) / 18800640,
                    'weight_bits_per_w_rans_only': 8 * pay.size / nWs, 'weight_bits_per_w_total': 8 * wt / nWs,
                    'mean_stream_bytes': float(ln.mean()), 'max_stream_bytes': int(ln.max())}
        del pay, ln, save
    for v in variants:
        if 'A259' in sizes:
            sizes[v]['delta_vs_A259_pct'] = 100 * (sizes[v]['record_total_bytes_all8'] / sizes['A259']['record_total_bytes_all8'] - 1)
    rep['sizes'] = sizes
    # ---------------- reproducibility vs team 4's encoder output (sha256 of every saved array)
    if expected:
        cmp = {}
        for f, keys in expected['files'].items():
            p = os.path.join(out_dir, f)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            cmp[f] = {k: (k in z.files and _sha(z[k]) == keys[k]['sha256']) for k in keys}
        rep['eq_team4_encoder_output'] = cmp
        rep['eq_team4_encoder_output_all'] = bool(cmp) and all(all(d.values()) for d in cmp.values())
        es = expected.get('sizes', {})
        rep['eq_team4_sizes'] = {v: (v in es and es[v]['record_total_bytes_all8'] == sizes[v]['record_total_bytes_all8'])
                                 for v in variants}
    rep['total_s'] = time.time() - t0
    rep['complete'] = True
    with open(os.path.join(out_dir, 'encode_report.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    return rep
