"""wnr2_g0: plain WNR2 records whose srel anchor (header median[m] byte) is set by an encoder-side rule (CPU, numpy).

Rules (anchor used for training AND written in the record; the decoder only reads the byte):
  med   true lower median of the matrix scales (WNR2 today, wex_nr2_median)
  c120  constant 120 (WNR2-g0, spec 5.3 / T14)
  pair  lower value of the densest adjacent scale pair, per matrix, on the plain scales
One WNM2 model per rule, trained on the 80 even-indexed sample experts (anchor rule applied to the training matrices
too). The 40 held-out experts are encoded as ordinary WNR2 records (32-B header, u16 lengths, 2048:512 chunks, x0 = L)
and decoded by two UNMODIFIED plain WNR2 decode paths, with tables rebuilt from the 59,408-B WNM2 blob read back from
disk (model id = FNV-1a of the blob):
  D1 = lib/wnr_common.decode_record (team 4 ceil-ref copy)   D2 = lib/wnr_common_t3.decode_record (team 3 copy)
During decoding the anchor functions of both modules are replaced by functions that raise, so a decoder that used an
anchor rule would abort the run. For records whose stored anchor differs from the lower median (anchor mismatch):
  R1 header patched to the lower median of the true scales (= what a median re-deriving decoder would do), D1 + D2
  R2 per-stream classification with the re-derived srel (structural fail / silent wrong / correct)
Reference: team 5 wnr2-g0 results (ref/wnr2_g0_ref.json: sizes and record sha256 per expert and rule) and its models
(ref/team5_wnr2g0_models_even80.npz); 'med' must also equal team 4 ceil-ref's WNR2 model.

Run (PowerShell, from the kit root):
  python -I tests\\cpu\\wnr2_g0.py --quick      # reference models, the 3 mismatch experts + 1, exact-match check
  python -I tests\\cpu\\wnr2_g0.py              # train per rule on 80, all 40 test experts
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'wnr2_g0', 'order': 21, 'needs': ['data'], 'quick_minutes': 3, 'full_minutes': 18,
            'description': 'WNR2 with encoder-side anchor rules med / c120 / pair, two unmodified decoders (CPU)'}
import hashlib
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import numpy as np  # noqa: E402
import kitcommon as K  # noqa: E402
import wnr_common as W  # noqa: E402
import wnr_common_t3 as T3  # noqa: E402

NAME = 'wnr2_g0'
RULES = ('med', 'c120', 'pair')
MISMATCH_FILES = ('L24_E058.bin', 'L38_E005.bin', 'L39_E156.bin')   # test experts whose lower median != 120 somewhere
ORIG = {mod: {n: getattr(mod, n) for n in ('prepare', 'lower_median', 'anchor_pair', 'block_features') if hasattr(mod, n)}
        for mod in (W, T3)}


def poison():
    def bad(*a, **k):
        raise RuntimeError('decoder used an anchor rule')
    for mod, d in ORIG.items():
        for n in d:
            setattr(mod, n, bad)


def unpoison():
    for mod, d in ORIG.items():
        for n, f in d.items():
            setattr(mod, n, f)


def parse_wnm2(blob):
    """WNM2 blob -> (wf [1701,16], scf [3,256]); validates header and every CDF like wex_nr2_model_open."""
    assert len(blob) == 59408
    hdr = struct.unpack_from('<8I', blob, 0)
    assert hdr == (W.WNM2_MAGIC, 1, 14, 567, 3, 0, 0, 0), hdr
    wcdf = np.frombuffer(blob, '<u2', 3 * 567 * 17, 32).astype(np.int64).reshape(3 * 567, 17)
    scdf = np.frombuffer(blob, '<u2', 3 * 257, 32 + 57834).astype(np.int64).reshape(3, 257)
    for c in (wcdf, scdf):
        assert (c[:, 0] == 0).all() and (c[:, -1] == W.P).all() and (np.diff(c, axis=1) >= 1).all()
    return np.diff(wcdf, axis=1), np.diff(scdf, axis=1)


_BLOB = {}


def blob_tables(path):
    if path not in _BLOB:
        blob = open(path, 'rb').read()
        wf, scf = parse_wnm2(blob)
        t1, t3 = W.Tables(wf, scf), T3.Tables(wf, scf)
        assert t1.blob == blob and t3.blob == blob
        mid = W.fnv1a32(blob) or 1
        assert t1.model_id == mid and t3.model_id == mid
        _BLOB[path] = (t1, t3, mid, hashlib.sha256(blob).hexdigest())
    return _BLOB[path]


def try_decode(dec, rec, tab, e, shapes, raw):
    t0 = time.perf_counter()
    try:
        slab, _ = dec.decode_record(rec, tab, e['layer'], e['expert'], shapes)
        err = None
    except dec.DecodeError as ex:
        slab, err = None, str(ex)
    dt = time.perf_counter() - t0
    exact = slab is not None and slab.size == raw.size and bool((slab == raw).all())
    ndiff = None if slab is None or slab.size != raw.size else int((slab != raw).sum())
    return {'error': err, 'byte_exact': bool(exact), 'bytes_differing': ndiff, 'seconds': round(dt, 3)}


def per_stream_rederive(rec, tab, mats, true_meds, stored):
    """R2: decode every weight stream with srel from the re-derived (lower median) anchors and classify it."""
    r = np.frombuffer(rec, np.uint8)
    _, mid, spb, wpb, wch, sch, m0, m1, m2, z0, z1 = struct.unpack_from('<6I3BBI', r, 0)
    assert [m0, m1, m2] == stored
    nsym = [mt['R'] * mt['C'] for mt in mats]
    nws = [n // W.W_CHUNK for n in nsym]
    nss = [n // 32 // W.S_CHUNK for n in nsym]
    SS, WS = sum(nss), sum(nws)
    lt = np.frombuffer(r, '<u2', SS + WS, 32).astype(np.int64)
    wlens = lt[SS:]
    wbeg = 32 + 2 * (SS + WS) + spb
    pad = np.zeros(2 * (W.W_CHUNK + W.BPS) + 16, np.uint8)
    pl_w = np.concatenate([r[wbeg:wbeg + wpb], pad])
    woffs = np.zeros(WS, np.int64)
    woffs[1:] = np.cumsum(wlens)[:-1]
    srelT = np.empty((W.BPS, WS), np.int64)
    srel_same = np.empty(WS, bool)
    col = 0
    for m in range(3):
        d = mats[m]['sc'].astype(np.int16)
        s_new = (np.clip(d - true_meds[m], -3, 3) + 3).reshape(nws[m], W.BPS)
        s_old = (np.clip(d - stored[m], -3, 3) + 3).reshape(nws[m], W.BPS)
        srelT[:, col:col + nws[m]] = s_new.T
        srel_same[col:col + nws[m]] = (s_new == s_old).all(1)
        col += nws[m]
    wmat = np.repeat(np.arange(3), nws)
    ctxoff = wmat.astype(np.int64) * W.NCTX
    wT, wok = W.decode_weights_lockstep(pl_w, woffs, wlens, srelT, ctxoff, ctxoff * 0, tab, False)
    truth = np.concatenate([mt['sym'].reshape(-1, W.W_CHUNK) for mt in mats]).T
    correct = (wT == truth).all(0)
    out = {}
    for m in range(3):
        sel = wmat == m
        out[W.MAT[m]] = {'anchor_stored': stored[m], 'anchor_rederived': true_meds[m], 'streams': int(sel.sum()),
                         'srel_unchanged_streams': int((sel & srel_same).sum()),
                         'structural_ok': int((sel & wok).sum()), 'structural_fail': int((sel & ~wok).sum()),
                         'ok_and_correct': int((sel & wok & correct).sum()),
                         'SILENT_ok_but_wrong': int((sel & wok & ~correct).sum()),
                         'fail_but_correct': int((sel & ~wok & correct).sum())}
    out['record_rejected'] = bool((~wok).any())
    out['record_silent_wrong'] = bool(wok.all() and not correct.all())
    return out


def job_fn(job):
    """Worker: one (expert, rule) record: encode with the rule's anchors, decode D1/D2, re-derive tests."""
    W.set_chunks(2048, 512)
    e, rule = job['entry'], job['rule']
    raw, mats, sha_ok = K.load_checked(job['data_dir'], e)
    shapes = [(mt['R'], mt['C']) for mt in mats]
    true_meds = [W.lower_median(mt['sc']) for mt in mats]
    t1, t3, mid, bsha = blob_tables(job['blob'])
    etab = K.tables(rule)                                   # encoder: trainer output in memory
    assert etab.model_id == mid and etab is not t1
    cfg = W.Config('wnr2', wnr2=True)
    t0 = time.perf_counter()
    rec, info, st, bi = K.encode_with_anchors(mats, etab, cfg, e['layer'], e['expert'], rule)
    tenc = time.perf_counter() - t0
    hdr = struct.unpack_from('<6I3BBI', rec, 0)
    stored = list(hdr[6:9])
    want = [K.anchor_value(rule, mt['sc']) for mt in mats]
    poison()
    try:
        d1 = try_decode(W, rec, t1, e, shapes, raw)
        d2 = try_decode(T3, rec, t3, e, shapes, raw)
    finally:
        unpoison()
    row = {'idx': job['idx'], 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'rule': rule,
           'model_id': mid, 'model_blob_sha256': bsha, 'source_sha256_ok': sha_ok,
           'header_ok': bool(hdr[0] == W.WNR2_MAGIC and hdr[1] == mid and stored == want),
           'anchors_stored': stored, 'lower_median': true_meds,
           'anchor_differs_from_median': [int(x != y) for x, y in zip(stored, true_meds)],
           'sizes': {'header': info['header'], 'length_table': info['length_table'], 'scale_payload': info['scale_payload'],
                     'weight_payload': info['weight_payload'], 'used': info['used'], 'padding': info['padding'],
                     'total': info['total']},
           'record_sha256': hashlib.sha256(rec).hexdigest(), 'encode_s': round(tenc, 3),
           'decode_D1_ceilref': d1, 'decode_D2_team3': d2}
    if any(row['anchor_differs_from_median']):
        rec2 = bytearray(rec)
        rec2[24:27] = bytes(true_meds)
        rec2 = bytes(rec2)
        poison()
        try:
            row['rederive_R1_D1'] = try_decode(W, rec2, t1, e, shapes, raw)
            row['rederive_R1_D2'] = try_decode(T3, rec2, t3, e, shapes, raw)
        finally:
            unpoison()
        row['rederive_R2_streams'] = per_stream_rederive(rec, t1, mats, true_meds, stored)
    return row


def main():
    p = K.base_parser(NAME, __file__, __doc__.split('\n')[0])
    p.add_argument('--models', choices=('train', 'ref'), default=None,
                   help='train = per rule on the 80 even experts (default full); ref = shipped team 5 models (default quick)')
    p.add_argument('--rules', default=','.join(RULES))
    a = K.finish_args(p.parse_args(), NAME)
    with K.Run(NAME, a) as R:
        run(R, a)
    sys.exit(R.exit_code())


def run(R, a):
    sample = K.find_sample(a, R)
    man = K.load_manifest(sample)
    rules = a.rules.split(',')
    test = K.test_indices()
    if a.quick:
        mis = [i for i in test if man['experts'][i]['file'] in MISMATCH_FILES]
        test = [test[0]] + mis
    if a.limit_test:
        test = test[:a.limit_test]
    mode = a.models or ('ref' if a.quick else 'train')
    ref = K.load_ref('wnr2_g0_ref.json')
    ref5 = K.load_ref('team5_wnr2g0_models_even80.npz')
    ref4 = K.load_ref('ceilref_models_even80.npz')
    models, full_train = {}, False
    if mode == 'train':
        tr = K.train_indices()
        if a.limit_train:
            tr = K.spread(tr, a.limit_train)
        full_train = len(tr) == 80
        cnt = K.ensure_counts(R, [('S', sample, i, man['experts'][i]) for i in tr], a.cache, a.jobs)
        cl = [cnt[k] for k in sorted(cnt)]
        R.check('train_sources_sha256', all(c['sha_ok'] for c in cl), f'{sum(c["sha_ok"] for c in cl)}/{len(cl)}')
        cmp = {}
        for rule in rules:
            m = K.train_model(cl, 'o', rule)
            models[rule] = K.model_arrays(m, wnr2=True)
            cmp[rule] = K.compare_model(m, ref5, rule)
            if rule == 'med':
                cmp['med_vs_ceilref_wnr2'] = K.compare_model(m, ref4, 'wnr2_a0_K0_c0')
            R.log(f'model {rule}: anchor census {m["census"]}')
        R.check('models_equal_reference', all(v is True for d in cmp.values() for v in d.values()) if full_train else None,
                cmp if full_train else f'skipped: trained on {len(tr)} experts')
    else:
        for rule in rules:
            models[rule] = K.model_arrays({'wf': ref5[rule + '_wf'], 'scf': ref5[rule + '_scf'], 'etab': None}, wnr2=True)
    blobs = {}
    for rule in rules:
        t = W.Tables(models[rule]['wf'], models[rule]['scf'])
        assert len(t.blob) == 59408
        path = os.path.join(a.out, f'{rule}.wnm2')
        open(path, 'wb').write(t.blob)
        blobs[rule] = path
        R.log(f'model {rule}: id 0x{t.model_id:08x}, blob {path}')
    R.summary['models'] = {'mode': mode, 'model_ids': {r: int(W.fnv1a32(open(blobs[r], 'rb').read()) or 1) for r in rules}}
    R.check('model_ids_equal_team5', all(R.summary['models']['model_ids'][r] == ref['model_ids'][r] for r in rules
                                         if r in ref['model_ids']) if (mode == 'ref' or full_train) else None,
            {'here': R.summary['models']['model_ids'], 'ref': ref['model_ids']})
    jobs = [{'data_dir': sample, 'idx': i, 'entry': man['experts'][i], 'rule': r, 'blob': blobs[r]} for i in test for r in rules]
    R.log(f'{len(test)} experts x {len(rules)} rules = {len(jobs)} records on {a.jobs} workers')
    rows = []
    for job, row in K.run_jobs(job_fn, jobs, a.jobs, R, 'record', K.init_models, (models,)):
        rows.append(row)
        R.log(f"{row['file']} {row['rule']:5s} anchors {row['anchors_stored']} med {row['lower_median']} total {row['sizes']['total']} "
              f"D1 {row['decode_D1_ceilref']['byte_exact']} D2 {row['decode_D2_team3']['byte_exact']}"
              + (f" R1 {row['rederive_R1_D1']['error'] or ('exact' if row['rederive_R1_D1']['byte_exact'] else 'WRONG OUTPUT')}"
                 if 'rederive_R1_D1' in row else ''))
    rows.sort(key=lambda r: (r['idx'], rules.index(r['rule'])))
    K.write_jsonl(os.path.join(a.out, 'records.jsonl'), rows)
    summarize(R, rows, rules, ref, mode, full_train)


def summarize(R, rows, rules, ref, mode, full_train):
    by = {r: {x['idx']: x for x in rows if x['rule'] == r} for r in rules}
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    rng = np.random.default_rng(K.BOOT_SEED)
    out = {}
    for r in rules:
        rr = [by[r][i] for i in common]
        tot = np.array([x['sizes']['total'] for x in rr], float)
        used = np.array([x['sizes']['used'] for x in rr], float)
        dif = [x for x in rr if any(x['anchor_differs_from_median'])]
        t = {'n': len(rr), 'mean_total_B': float(tot.mean()), 'ratio_total_to_raw': float(tot.mean() / K.RAW),
             'mean_used_B': float(used.mean()), 'ratio_used_to_raw': float(used.mean() / K.RAW),
             'mean_weight_payload_B': float(np.mean([x['sizes']['weight_payload'] for x in rr])),
             'mean_scale_payload_B': float(np.mean([x['sizes']['scale_payload'] for x in rr])),
             'bit_per_w_weight_payload': float(np.mean([x['sizes']['weight_payload'] for x in rr]) * 8 / K.W_PER_REC),
             'matrices_anchor_differs_from_lower_median': int(sum(sum(x['anchor_differs_from_median']) for x in rr)),
             'matrices_total': 3 * len(rr), 'records_with_any_differing_anchor': len(dif),
             'anchor_values_stored': {str(v): int(sum(x['anchors_stored'].count(v) for x in rr))
                                      for v in sorted({v for x in rr for v in x['anchors_stored']})},
             'decode_D1_exact': f"{sum(x['decode_D1_ceilref']['byte_exact'] for x in rr)}/{len(rr)}",
             'decode_D2_exact': f"{sum(x['decode_D2_team3']['byte_exact'] for x in rr)}/{len(rr)}"}
        if dif:
            t['rederive_R1'] = {
                'D1': {'rejected': sum(x['rederive_R1_D1']['error'] is not None for x in dif),
                       'silent_wrong': sum(x['rederive_R1_D1']['error'] is None and not x['rederive_R1_D1']['byte_exact'] for x in dif),
                       'exact': sum(x['rederive_R1_D1']['byte_exact'] for x in dif)},
                'D2': {'rejected': sum(x['rederive_R1_D2']['error'] is not None for x in dif),
                       'silent_wrong': sum(x['rederive_R1_D2']['error'] is None and not x['rederive_R1_D2']['byte_exact'] for x in dif),
                       'exact': sum(x['rederive_R1_D2']['byte_exact'] for x in dif)}}
            t['rederive_R2_streams_total'] = {k: int(sum(x['rederive_R2_streams'][m][k] for x in dif for m in W.MAT))
                                             for k in ('streams', 'srel_unchanged_streams', 'structural_fail',
                                                       'SILENT_ok_but_wrong', 'ok_and_correct')}
        if r != 'med' and 'med' in by:
            dt = [by[r][i]['sizes']['total'] - by['med'][i]['sizes']['total'] for i in common]
            du = [by[r][i]['sizes']['used'] - by['med'][i]['sizes']['used'] for i in common]
            t['vs_med_total'] = K.describe_delta(dt, rng)
            t['vs_med_used'] = K.describe_delta(du, rng)
            dix = {x['idx'] for x in dif}
            t['split'] = {'experts_anchor_equal_to_median': {'n': len(common) - len(dix),
                                                             'mean_delta_used_B': float(np.mean([du[k] for k, i in enumerate(common) if i not in dix])) if len(dix) < len(common) else None},
                          'experts_anchor_differs': [{'file': by[r][i]['file'], 'anchors': by[r][i]['anchors_stored'],
                                                      'median': by[r][i]['lower_median'], 'delta_total_B': dt[k],
                                                      'delta_pct_raw': 100 * dt[k] / K.RAW}
                                                     for k, i in enumerate(common) if i in dix],
                          'share_of_total_gain_from_differing_experts':
                              float(sum(dt[k] for k, i in enumerate(common) if i in dix) / sum(dt)) if sum(dt) else None}
            t['records_identical_to_med_record'] = int(sum(by[r][i]['record_sha256'] == by['med'][i]['record_sha256'] for i in common))
        out[r] = t
        R.log(f"{r:5s} total {t['mean_total_B']:12.1f} ratio {t['ratio_total_to_raw']:.5f} anchors!=med {t['matrices_anchor_differs_from_lower_median']}"
              f" D1 {t['decode_D1_exact']} D2 {t['decode_D2_exact']}"
              + (f" vs med {t['vs_med_total']['mean_pct_of_raw']:+.4f}% CI {t['vs_med_total']['CI95_pct_of_raw'][0]:+.4f}..{t['vs_med_total']['CI95_pct_of_raw'][1]:+.4f}"
                 if 'vs_med_total' in t else ''))
    R.summary['experts'] = common
    R.summary['rules'] = out
    R.check('sources_sha256', all(x['source_sha256_ok'] for x in rows))
    R.check('headers_carry_rule_anchors', all(x['header_ok'] for x in rows))
    bad = [(x['file'], x['rule']) for x in rows if not (x['decode_D1_ceilref']['byte_exact'] and x['decode_D2_team3']['byte_exact'])]
    R.check('plain_wnr2_decoders_byte_exact', not bad, bad[:20])
    silent = [(x['file'], x['rule']) for x in rows if 'rederive_R1_D1' in x and
              any(x[k]['error'] is None and not x[k]['byte_exact'] for k in ('rederive_R1_D1', 'rederive_R1_D2'))]
    R.check('rederive_never_silent_wrong (R1)', not silent, silent, hard=False)
    if mode == 'ref' or full_train:
        mism, n = [], 0
        for x in rows:
            k = f"{x['idx']}:{x['rule']}"
            if k in ref['rows']:
                n += 1
                q = ref['rows'][k]
                if q['record_sha256'] != x['record_sha256'] or q['total'] != x['sizes']['total']:
                    mism.append({'key': k, 'ref_total': q['total'], 'here_total': x['sizes']['total']})
        R.check('records_identical_to_team5 (sha256)', n > 0 and not mism, {'compared': n, 'mismatch': mism[:20]})
    R.summary['reference'] = {'source': ref['source'], 'summary_rules': ref['summary_rules']}
    parts = []
    for r in rules:
        if 'vs_med_total' in out[r]:
            parts.append(f"{r} {out[r]['vs_med_total']['mean_pct_of_raw']:+.3f}%raw vs med "
                         f"({out[r]['matrices_anchor_differs_from_lower_median']} anchor mismatches)")
    R.result(f'{len(common)} experts; ' + '; '.join(parts))


if __name__ == '__main__':
    main()
