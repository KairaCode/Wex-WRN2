"""team4/ceil-gpu: encode the 8 team-2 V4.1 experts as baseline and ceil-form WNR2 streams.

Usage: python -I ceilenc.py SAMPLE_DIR COUNTS_DIR CEIL_COUNTS_DIR OUT_DIR

Models (in-sample over all 160 sampled experts, as team 2 did):
  A259 : WNR2 base contexts (srel7, pm, mx) compacted to 259 rows, 16-symbol nibble CDFs, ORIGINAL data
         (counts = sum of w_base in COUNTS_DIR, team-1 mxcount.py)
  D    : same contexts, 8-symbol magnitude CDFs + raw sign plane (team 2 variant D), original data
  CF   : A259 layout retrained on CEIL-FORM data (counts = sum of w_ceil in CEIL_COUNTS_DIR, team-1 mxceil.py;
         median recomputed on the transformed scales)
  CFD  : CF + sign bypass (8-symbol magnitude CDFs on ceil-form data + raw sign plane)
  scales: order-0 256-symbol CDF per matrix type (s_orig / s_ceil of CEIL_COUNTS_DIR), 512 scales per stream
Ceil-form transform T (own implementation from the definition): block with max magnitude 7 and no magnitude
1 or 3 -> magnitudes 0,2,4,5,6,7 -> 0,1,2,3,4,5 (sign bit 8 kept) and scale + 1.  Inverse: decoded block max 5
-> doubled back, scale - 1.
Writes OUT_DIR/common.npz, var_{A259,D,CF,CFD}.npz (payload, lengths, tables, [signs]),
scl_{orig,ceil}.npz (payload, lengths, cum, lut) and encode_report.json.
"""
import hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr2fmt as W

PICK = [17, 0, 40, 80, 120, 159, 60, 100]   # team 2 encode_all.py:36 with NEXPERTS = 8
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)
SPS = 512   # scales per scale stream


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


def main():
    sample_dir, counts_dir, ceil_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    rep = {'pick': PICK, 'repo': man.get('repo'), 'revision': man.get('revision')}
    # ---------------- models
    wb_o = np.zeros((3, 567, 16), np.int64); wb_c = np.zeros_like(wb_o); wb_o2 = np.zeros_like(wb_o)
    s_o = np.zeros((3, 256), np.int64); s_c = np.zeros_like(s_o)
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        wb_o += z['w_base']
        y = np.load(os.path.join(ceil_dir, e['file'].replace('.bin', '.npz')))
        wb_c += y['w_ceil']; wb_o2 += y['w_orig']; s_o += y['s_orig']; s_c += y['s_ceil']
    rep['counts_consistency_w_base_eq_mxceil_w_orig'] = bool((wb_o == wb_o2).all())
    MO = models_from_counts(wb_o)
    MC = models_from_counts(wb_c)
    sfo, scumo, sluto = scale_model(s_o)
    sfc, scumc, slutc = scale_model(s_c)
    nW = wb_o.sum(); nS = s_o.sum()
    rep['model_insample'] = {
        'A259_bits_per_w': W.cost_bits(MO['A259']['counts'], MO['A259']['freq']) / nW,
        'D_bits_per_w_incl_raw_sign': (W.cost_bits(MO['D']['counts'], MO['D']['freq']) + nW) / nW,
        'CF_bits_per_w': W.cost_bits(MC['A259']['counts'], MC['A259']['freq']) / nW,
        'CFD_bits_per_w_incl_raw_sign': (W.cost_bits(MC['D']['counts'], MC['D']['freq']) + nW) / nW,
        'scale_orig_bits_per_scale': W.cost_bits(s_o, sfo) / nS,
        'scale_ceil_bits_per_scale': W.cost_bits(s_c, sfc) / nS,
        'n_weights': int(nW), 'n_scales': int(nS)}
    rep['model_table_bytes'] = {'A259_or_CF_u16': 3 * 259 * 16 * 2, 'D_or_CFD_u16': 3 * 259 * 8 * 2,
                                'scale_cum_u16': 3 * 257 * 2, 'scale_lut_u8': 3 * 16384}
    print(json.dumps(rep, indent=1), flush=True)
    VARS = {'A259': (False, MO['A259'], 'A259'), 'D': (False, MO['D'], 'D'),
            'CF': (True, MC['A259'], 'A259'), 'CFD': (True, MC['D'], 'D'),
            'DB2': (False, MO['D'], 'DB2'), 'CFDB2': (True, MC['D'], 'DB2')}
    only = sys.argv[5].split(',') if len(sys.argv) > 5 else None
    VARS = {k: v for k, v in VARS.items() if (k in only if only else k in ('A259', 'D', 'CF', 'CFD'))}
    acc = {v: {'pay': [], 'len': [], 'signs': []} for v in VARS}
    sacc = {k: {'pay': [], 'len': []} for k in ('orig', 'ceil')}
    raw_all, sco_all, scc_all, med_o, med_c, mtypes, names = [], [], [], [], [], [], []
    tstats = {'blocks': 0, 'even': 0, 'max_below_6': 0, 'scale_overflow': 0, 'max_hist': np.zeros(8, np.int64),
              'cpu_roundtrip_fail_matrices': 0, 'own_counts_eq_mxceil_counts_experts': 0, 'sha256_ok': 0}
    enc_s = {v: 0.0 for v in VARS}
    spm = None
    for ei in PICK:
        e = man['experts'][ei]
        fpath = os.path.join(sample_dir, e['file'])
        if 'sha256' in e:
            h = hashlib.sha256(open(fpath, 'rb').read()).hexdigest()
            tstats['sha256_ok'] += int(h == e['sha256'])
        t = W.load_expert(fpath, e['tensors'])
        y = np.load(os.path.join(ceil_dir, e['file'].replace('.bin', '.npz')))
        mo, mc = [], []
        own = np.zeros((3, 567, 16), np.int64)
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{w}.weight']; sraw, _, _ = t[f'{w}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1); sc = sraw.reshape(R, -1)
            twb, tsc, even, st = ceil_forward(wb, sc)
            for k in ('blocks', 'even', 'max_below_6', 'scale_overflow'):
                tstats[k] += st[k]
            tstats['max_hist'] += np.array(st['max_hist'])
            iwb, isc = ceil_inverse(twb, tsc)
            tstats['cpu_roundtrip_fail_matrices'] += int(not (np.array_equal(iwb, wb) and np.array_equal(isc, sc)))
            own[m] = ctx_counts(twb, tsc)
            mdo = W.matrix_data(wb, sc); mdc = W.matrix_data(twb, tsc)
            mo.append((m, mdo)); mc.append((m, mdc))
            raw_all.append(wb.ravel().copy()); sco_all.append(sc.ravel().copy()); scc_all.append(tsc.ravel().copy())
            med_o.append(mdo['med']); med_c.append(mdc['med']); mtypes.append(m); names.append(f"{e['file'][:-4]}.{w}")
            spm = mdo['nstreams']
            # scale streams (order-0), 512 per stream
            for key, s_, f_, c_ in (('orig', sc, sfo, scumo), ('ceil', tsc, sfc, scumc)):
                s1 = s_.ravel().astype(np.int64).reshape(-1, SPS)
                F = f_[m][s1]; ST = c_[m][s1]
                pay, ln = W.rans_encode_vec(F, ST, np.zeros(SPS, np.int64), 1)
                sacc[key]['pay'].append(pay); sacc[key]['len'].append(ln)
                if m == 0 and ei == PICK[0]:
                    offs = np.concatenate([[0], np.cumsum(ln)])
                    for si in (0, 1, 2, s1.shape[0] - 1):
                        ref = W.encode_ref(s1[si].tolist(), [m] * SPS, f_, c_)
                        assert ref == pay[offs[si]:offs[si + 1]].tobytes(), ('scale scalar mismatch', key, si)
        tstats['own_counts_eq_mxceil_counts_experts'] += int((own == y['w_ceil']).all())
        for v, (isceil, model, kind) in VARS.items():
            tv = time.time()
            Fs, STs = [], []
            mods = {'A259': model, 'D': model}   # variant_steps looks the model up under 'A259' or 'D'
            for m, md in (mc if isceil else mo):
                F, ST, sid, K, extra = W.variant_steps(md, kind, m, mods)
                Fs.append(F.astype(np.int32)); STs.append(ST.astype(np.int32))
                if 'signs' in extra:
                    acc[v]['signs'].append(extra['signs'])
            pay, ln = W.rans_encode_vec(np.concatenate(Fs), np.concatenate(STs), sid, K)
            del Fs, STs
            assert ln.min() >= 4 and ln.max() <= 65535
            if ei == PICK[0] and v in ('CF', 'CFD'):
                # scalar reference check (rans_check.py rule) on 8 streams of matrix 0
                md = mc[0][1]
                fr = model['freq'][0]
                cum = np.zeros((fr.shape[0], fr.shape[1] + 1), np.int64); cum[:, 1:] = np.cumsum(fr, 1)
                rows = (md['srel'][:, None] * 37 + W.tri(md['pm'], md['mx'])).reshape(spm, -1)
                syms = (md['blk'] if v == 'CF' else md['mag']).reshape(spm, -1)
                offs = np.concatenate([[0], np.cumsum(ln)])
                for s in (0, 1, 2, 3, spm - 4, spm - 3, spm - 2, spm - 1):
                    ref = W.encode_ref(syms[s].tolist(), rows[s].tolist(), fr, cum)
                    assert ref == pay[offs[s]:offs[s + 1]].tobytes(), ('scalar mismatch', v, s)
                rep[f'scalar_reference_check_{v}'] = '8/8 streams byte-identical (matrix 0)'
            acc[v]['pay'].append(pay); acc[v]['len'].append(ln)
            enc_s[v] += time.time() - tv
        print(e['file'], 'even%', round(100 * tstats['even'] / tstats['blocks'], 3), 'elapsed', round(time.time() - t0, 1),
              flush=True)
        del mo, mc
    nmat = len(mtypes)
    np.savez(os.path.join(out_dir, 'common.npz'), raw=np.concatenate(raw_all), scales=np.concatenate(sco_all),
             scales_ceil=np.concatenate(scc_all), med=np.array(med_o, np.uint8), med_ceil=np.array(med_c, np.uint8),
             mtype=np.array(mtypes, np.uint8), spm=np.int64(spm), sspm=np.int64(368640 // SPS), names=np.array(names))
    tstats['max_hist'] = tstats['max_hist'].tolist()
    tstats['even_pct'] = 100 * tstats['even'] / tstats['blocks']
    rep['transform'] = tstats
    rep['medians'] = {'orig': med_o, 'ceil': med_c}
    sizes = {}
    nWs = nmat * spm * 2048
    nSc = nmat * 368640
    for key, (f_, c_, l_) in (('orig', (sfo, scumo, sluto)), ('ceil', (sfc, scumc, slutc))):
        pay = np.concatenate(sacc[key]['pay']); ln = np.concatenate(sacc[key]['len'])
        np.savez(os.path.join(out_dir, f'scl_{key}.npz'), payload=pay, lengths=ln.astype(np.uint16),
                 cum=c_.astype(np.uint16), lut=l_)
        sizes[f'scales_{key}'] = {'payload_bytes': int(pay.size), 'length_table_bytes': int(2 * ln.size),
                                  'bits_per_scale_payload': 8 * pay.size / nSc,
                                  'bits_per_scale_incl_lengths': 8 * (pay.size + 2 * ln.size) / nSc,
                                  'streams': int(ln.size), 'mean_stream_bytes': float(ln.mean()), 'max_stream_bytes': int(ln.max())}
    for v, (isceil, model, kind) in VARS.items():
        pay = np.concatenate(acc[v]['pay']); ln = np.concatenate(acc[v]['len'])
        save = {'payload': pay, 'lengths': ln.astype(np.uint16),
                'tables': np.stack([np.cumsum(model['freq'][m], 1).astype(np.uint16) for m in range(3)])}
        sign_bytes = 0
        if acc[v]['signs']:
            save['signs'] = np.concatenate(acc[v]['signs']); sign_bytes = save['signs'].size * 4
        np.savez(os.path.join(out_dir, f'var_{v}.npz'), **save)
        sk = 'scales_ceil' if isceil else 'scales_orig'
        scb = sizes[sk]['payload_bytes'] + sizes[sk]['length_table_bytes']
        wt = pay.size + 2 * ln.size + sign_bytes
        sizes[v] = {'weight_payload_bytes': int(pay.size), 'weight_length_table_bytes': int(2 * ln.size),
                    'sign_plane_bytes': int(sign_bytes), 'weight_total_bytes': int(wt),
                    'scale_streams_bytes': int(scb), 'record_total_bytes_all8': int(wt + scb),
                    'per_record_bytes': (wt + scb) / (nmat // 3),
                    'ratio_of_raw_record_18800640': (wt + scb) / (nmat // 3) / 18800640,
                    'weight_bits_per_w_rans_only': 8 * pay.size / nWs, 'weight_bits_per_w_total': 8 * wt / nWs,
                    'mean_stream_bytes': float(ln.mean()), 'max_stream_bytes': int(ln.max()), 'encode_s': enc_s[v]}
    for v in VARS:
        bk = {'D': 'D', 'CFD': 'D', 'DB2': 'DB2', 'CFDB2': 'DB2'}.get(v, 'A259')
        if bk in sizes:
            sizes[v]['delta_vs_same_alphabet_baseline_pct'] = 100 * (sizes[v]['record_total_bytes_all8'] / sizes[bk]['record_total_bytes_all8'] - 1)
        if 'A259' in sizes:
            sizes[v]['delta_vs_A259_pct'] = 100 * (sizes[v]['record_total_bytes_all8'] / sizes['A259']['record_total_bytes_all8'] - 1)
            sizes[v]['delta_vs_A259_pct_of_raw'] = 100 * (sizes[v]['record_total_bytes_all8'] - sizes['A259']['record_total_bytes_all8']) / (18800640 * (nmat // 3))
    rep['sizes'] = sizes
    rep['total_s'] = time.time() - t0
    rname = 'encode_report.json' if not only else 'encode_report_' + '_'.join(VARS) + '.json'
    json.dump(rep, open(os.path.join(out_dir, rname), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
