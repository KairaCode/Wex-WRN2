"""heldout_configs: real-record held-out table of the WNR2 / WNR3-candidate configurations (CPU, numpy).

Re-runs team 4 ceil-ref's held-out protocol: models trained on the 80 even-indexed experts of the 160-expert V4.1
sample, real records written for the 40 held-out experts team 3 used (one per layer), every record decoded back and
compared with the source file (byte-exact; canonical-zero configs value-exact: nibble 8 (-0) -> 0).

Configurations (names as in the WNR3 draft; g0 = srel anchor constant 120 written in the median[m] byte):
  wnr2    plain WNR2 (32-B header, lower-median anchors, u16 length table)
  f       ceil form, lower median of the transformed scales
  fg0     f + g0                fg0d   + coded length table (LENC, d)
  fg0de   fg0d + 23-bit hash in the start state (e)
  fg0dt   fg0d + raw tail seeding (t)
  fg0c    fg0 + canonical zero (c, value-exact)      fg0cde  fg0c + d + e
  (side products, also decoded: fg0e, fg0t, fg0cd, fg0ce)
Statistics: mean record sizes, bit/w, deltas vs WNR2 and pairwise deltas with 10,000-resample expert-bootstrap 95% CIs.
Reference values: team 4 ceil-ref results/heldout_g0.jsonl + heldout_summary_g0.json (ref/heldout_ref.json) and its
models (ref/ceilref_models_even80.npz). The coder is deterministic: with the same models every record size must match.

Run (PowerShell, from the kit root):
  python -I tests\\cpu\\heldout_configs.py --quick      # reference models, 4 test experts, exact-match check
  python -I tests\\cpu\\heldout_configs.py              # train on 80, all 40 test experts
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'heldout_configs', 'order': 20, 'needs': ['data'], 'quick_minutes': 3, 'full_minutes': 35,
            'description': 'held-out real-record table of the WNR2 / WNR3 configs (CPU)'}
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import numpy as np  # noqa: E402
import kitcommon as K  # noqa: E402

NAME = 'heldout_configs'
MODELS = {  # name: (form, rule, c, etab, wnr2, reference key)
    'wnr2': ('o', 'med', False, False, True, 'wnr2_a0_K0_c0'),
    'f': ('c', 'med', False, True, False, 'f_a0_K0_c0'),
    'fg0': ('c', 'c120', False, True, False, 'fg_a0_K0_c0'),
    'fg0c': ('c', 'c120', True, True, False, 'fg_a0_K0_c1'),
}
ITEMS = [('wnr2', 'wnr2'), ('f', 'f'), ('fg0', 'fg0'), ('fg0e', 'fg0'), ('fg0t', 'fg0'), ('fg0c', 'fg0c'), ('fg0ce', 'fg0c')]
MAIN = ['wnr2', 'f', 'fg0', 'fg0d', 'fg0de', 'fg0dt', 'fg0c', 'fg0cde']
PAIRS = [('f vs wnr2', 'f', 'wnr2'), ('fg0 vs f (g0 on ceil form)', 'fg0', 'f'), ('fg0d vs fg0 (d)', 'fg0d', 'fg0'),
         ('fg0e vs fg0 (e)', 'fg0e', 'fg0'), ('fg0de vs fg0d (e on d)', 'fg0de', 'fg0d'),
         ('fg0t vs fg0 (t)', 'fg0t', 'fg0'), ('fg0dt vs fg0d (t on d)', 'fg0dt', 'fg0d'),
         ('fg0dt vs fg0de (tail instead of hash)', 'fg0dt', 'fg0de'), ('fg0c vs fg0 (c)', 'fg0c', 'fg0'),
         ('fg0cd vs fg0d (c on d)', 'fg0cd', 'fg0d'), ('fg0cde vs fg0de (c on d+e)', 'fg0cde', 'fg0de'),
         ('fg0 vs wnr2', 'fg0', 'wnr2'), ('fg0de vs wnr2', 'fg0de', 'wnr2'), ('fg0dt vs wnr2', 'fg0dt', 'wnr2'),
         ('fg0cde vs wnr2', 'fg0cde', 'wnr2')]


def main():
    p = K.base_parser(NAME, __file__, __doc__.split('\n')[0])
    p.add_argument('--models', choices=('train', 'ref'), default=None,
                   help='train = train on the 80 even experts here (default full); ref = shipped ceil-ref models (default quick)')
    p.add_argument('--configs', default=None, help='comma list of payload configs to run (default all: ' +
                   ','.join(c for c, _ in ITEMS) + '; D siblings are built automatically)')
    a = K.finish_args(p.parse_args(), NAME)
    with K.Run(NAME, a) as R:
        run(R, a)
    sys.exit(R.exit_code())


def run(R, a):
    sample = K.find_sample(a, R)
    man = K.load_manifest(sample)
    R.log('sample', sample, 'experts', len(man['experts']))
    test = K.test_indices()
    if a.quick:
        test = K.spread(test, 4)
    if a.limit_test:
        test = test[:a.limit_test]
    mode = a.models or ('ref' if a.quick else 'train')
    ref_npz = K.load_ref('ceilref_models_even80.npz')
    ref = K.load_ref('heldout_ref.json')
    R.check('test_set_matches_reference', sorted(ref['experts']) == K.test_indices(),
            {'n_test_run': len(test), 'test_run': test})
    # ---- models
    models = {}
    full_train = False
    if mode == 'train':
        tr = K.train_indices()
        if a.limit_train:
            tr = K.spread(tr, a.limit_train)
        full_train = len(tr) == 80
        cnt = K.ensure_counts(R, [('S', sample, i, man['experts'][i]) for i in tr], a.cache, a.jobs)
        cl = [cnt[k] for k in sorted(cnt)]
        R.check('train_sources_sha256', all(c['sha_ok'] for c in cl), f'{sum(c["sha_ok"] for c in cl)}/{len(cl)}')
        R.check('train_ceil_applicable', all(c['ceil_ok'] for c in cl), hard=True)
        cmp = {}
        for name, (form, rule, c, et, w2, rk) in MODELS.items():
            m = K.train_model(cl, form, rule, c=c, etab=et)
            models[name] = K.model_arrays(m, wnr2=w2)
            cmp[name] = K.compare_model(m, ref_npz, rk)
            R.log(f'model {name}: trained on {len(cl)} experts, anchor census {m["census"]}, vs ceil-ref {cmp[name]}')
        eq = all(v is True for d in cmp.values() for v in d.values())
        R.check('models_equal_ceilref_even80', eq if full_train else None,
                cmp if full_train else f'skipped: trained on {len(tr)} experts, not the 80 of the reference')
        np.savez_compressed(os.path.join(a.out, 'models_trained.npz'),
                            **{f'{n}_{k}': v for n, d in models.items() for k, v in d.items() if v is not None and k != 'wnr2'})
    else:
        for name, (form, rule, c, et, w2, rk) in MODELS.items():
            models[name] = K.model_arrays(K.model_from_ref(ref_npz, rk), wnr2=w2)
        R.log('models: shipped reference models (team 4 ceil-ref even.npz)')
    ids = {n: K.model_id_of(m) for n, m in models.items()}
    R.summary['models'] = {'mode': mode, 'model_id_and_blob_bytes': ids}
    # ---- jobs: one per (expert, payload config)
    items = ITEMS
    if a.configs:
        want = a.configs.split(',')
        items = [x for x in ITEMS if x[0] in want]
    jobs = [{'data_dir': sample, 'idx': i, 'entry': man['experts'][i], 'chunk': (2048, 512),
             'items': [{'config': c, 'model': mname}]} for i in test for c, mname in items]
    R.log(f'encode/decode: {len(test)} experts x {len(items)} payload configs = {len(jobs)} jobs on {a.jobs} workers')
    rows = []
    for job, res in K.run_jobs(K.encode_job, jobs, a.jobs, R, 'record', K.init_models, (models,)):
        for r in res['rows']:
            rows.append(r)
            R.log(f"{r['file']} {r['config']:7s} total {r['sizes']['total']} used {r['sizes']['used']} "
                  f"ok {r['verify']['equal_to_reference']} err {r['verify']['decode_error']} "
                  f"enc {r['timing'].get('enc_total_s')}s dec {r['timing'].get('dec_total_s')}s")
    rows.sort(key=lambda r: (r['idx'], r['config']))
    K.write_jsonl(os.path.join(a.out, 'records.jsonl'), rows)
    summarize(R, a, rows, ref, mode, full_train)


def summarize(R, a, rows, ref, mode, full_train):
    cfgs = sorted({r['config'] for r in rows}, key=lambda c: ((MAIN + [c]).index(c), c))
    by = {c: {r['idx']: r for r in rows if r['config'] == c} for c in cfgs}
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    rng = np.random.default_rng(K.BOOT_SEED)
    tab = {}
    for c in cfgs:
        tab[c] = K.config_table([by[c][i] for i in common])
        if 'wnr2' in by and c != 'wnr2':
            dt = [by[c][i]['sizes']['total'] - by['wnr2'][i]['sizes']['total'] for i in common]
            du = [by[c][i]['sizes']['used'] - by['wnr2'][i]['sizes']['used'] for i in common]
            tab[c]['delta_total_vs_wnr2'] = K.describe_delta(dt, rng)
            tab[c]['delta_used_vs_wnr2'] = K.describe_delta(du, rng)
    pw = {}
    for nm, x, y in PAIRS:
        if x in by and y in by:
            du = [by[x][i]['sizes']['used'] - by[y][i]['sizes']['used'] for i in common]
            dt = [by[x][i]['sizes']['total'] - by[y][i]['sizes']['total'] for i in common]
            pw[nm] = {'used': K.describe_delta(du, rng), 'total': K.describe_delta(dt, rng)}
    R.summary['experts'] = common
    R.summary['configs'] = tab
    R.summary['pairwise'] = pw
    for c in cfgs:
        t = tab[c]
        d = t.get('delta_total_vs_wnr2')
        R.log(f"{c:7s} total {t['mean_total_bytes']:12.1f} ratio {t['ratio_of_raw']:.5f} bpw {t['bit_per_w_weight_payload']:.4f} "
              f"{t['decode_check']}" + (f" d%raw {d['mean_pct_of_raw']:+.3f} CI {d['CI95_pct_of_raw'][0]:+.3f}..{d['CI95_pct_of_raw'][1]:+.3f}" if d else ''))
    # ---- checks
    R.check('sources_sha256', all(r['source_sha256_ok'] for r in rows), f"{sum(r['source_sha256_ok'] for r in rows)}/{len(rows)}")
    dec = [r for r in rows if r['verify']['decoded']]
    bad = [(r['file'], r['config'], r['verify']['decode_error']) for r in dec if not r['verify']['equal_to_reference']]
    R.check('all_records_decode_exact', not bad and len(dec) == len(rows), {'decoded': len(dec), 'records': len(rows), 'bad': bad[:20]})
    fb = [(r['file'], r['config']) for r in rows if r['ceil'] is not None and not r['ceil']['applied']]
    R.check('no_ceil_fallback', not fb, fb[:20], hard=False)
    # reference rows (exact sizes)
    if mode == 'ref' or full_train:
        mism, n = [], 0
        for r in rows:
            k = f"{r['idx']}:{r['config']}"
            if k in ref['rows']:
                n += 1
                q = ref['rows'][k]
                if (q['total'], q['used']) != (r['sizes']['total'], r['sizes']['used']):
                    mism.append({'key': k, 'ref': [q['total'], q['used']], 'here': [r['sizes']['total'], r['sizes']['used']]})
        R.check('record_sizes_equal_reference', n > 0 and not mism, {'compared': n, 'mismatch': len(mism), 'first': mism[:20]})
        if len(common) == 40:
            md = {c: {'here': tab[c]['mean_total_bytes'], 'ref': ref['summary_g0'][c]['mean_total_bytes']}
                  for c in cfgs if c in ref['summary_g0']}
            R.check('mean_totals_equal_reference', all(abs(v['here'] - v['ref']) < 1e-6 for v in md.values()), md)
    else:
        R.check('record_sizes_equal_reference', None, 'skipped: models trained on a subset')
    R.summary['reference'] = {'source': ref['source'], 'summary_g0': ref['summary_g0']}
    main_c = [c for c in ('f', 'fg0', 'fg0de', 'fg0dt', 'fg0cde') if c in tab and 'delta_total_vs_wnr2' in tab[c]]
    short = f"{len(common)} experts, {len(dec)} records decoded; d%raw vs WNR2: " + ', '.join(
        f"{c} {tab[c]['delta_total_vs_wnr2']['mean_pct_of_raw']:+.3f}" for c in main_c)
    R.result(short)


if __name__ == '__main__':
    main()
