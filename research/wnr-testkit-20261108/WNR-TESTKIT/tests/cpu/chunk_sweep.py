"""chunk_sweep: real WNR3 records (f + g0 + d + e, and f + g0 + d + t) of the held-out experts at several stream sizes.

Weight:scale chunk pairs (symbols per stream): 1024:256, 2048:512 (WNR2 / WNR3 v1 default), 4096:1024, 8192:2048 for
fg0de (ceil form, anchor 120, coded length table, 23-bit hash in the start state) and 2048:512, 4096:1024 for fg0dt
(raw tail seeding instead of the hash). Each payload also gives its sibling without the coded length table
(fg0e / fg0t, u16 lengths) for free. Models do not depend on the chunk: one fg0 model set (ceil form, anchor 120,
with the length-predictor table etab) trained on the 80 even-indexed sample experts, or the shipped ceil-ref model.
Coder: lib/wnr_common.py with team 6's set_chunks() (module-level chunk switch; decoder checks the header chunks).
Every fg0de / fg0dt record is decoded and compared byte for byte with the source (--decode-all: also fg0e / fg0t).
Statistics: sizes, length-table bytes, stream-length stats, deltas vs the same config at 2048:512 with
10,000-resample expert-bootstrap 95% CIs.
Reference: 2048:512 must equal team 4 ceil-ref heldout_g0 fgde / fgdt exactly (ref/heldout_ref.json); other chunks
are compared with team 6's PARTIAL run (ref/chunk_ref.json, 11 experts).
GPU-side cost of the chunk sizes is not measured here (see the GPU part of the kit).

Run (PowerShell, from the kit root):
  python -I tests\\cpu\\chunk_sweep.py --quick      # reference model, 2 test experts
  python -I tests\\cpu\\chunk_sweep.py              # train on 80, all 40 test experts
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'chunk_sweep', 'order': 23, 'needs': ['data'], 'quick_minutes': 1, 'full_minutes': 20,
            'description': 'fg0de / fg0dt real records at 1024..8192-symbol streams (CPU)'}
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import numpy as np  # noqa: E402
import kitcommon as K  # noqa: E402

NAME = 'chunk_sweep'
PLAN = [(1024, 256, ['fg0e']), (2048, 512, ['fg0e', 'fg0t']), (4096, 1024, ['fg0e', 'fg0t']), (8192, 2048, ['fg0e'])]
BASE = (2048, 512)
OLD = {'fg0e': 'fge', 'fg0de': 'fgde', 'fg0t': 'fgt', 'fg0dt': 'fgdt'}     # team 6 config names


def main():
    p = K.base_parser(NAME, __file__, __doc__.split('\n')[0])
    p.add_argument('--models', choices=('train', 'ref'), default=None,
                   help='train = on the 80 even experts (default full); ref = shipped ceil-ref fg model (default quick)')
    p.add_argument('--decode-all', action='store_true', help='also decode the fg0e / fg0t siblings')
    p.add_argument('--plan', default=None, help="override, e.g. '1024:256:fg0e;8192:2048:fg0e,fg0t'")
    a = K.finish_args(p.parse_args(), NAME)
    with K.Run(NAME, a) as R:
        run(R, a)
    sys.exit(R.exit_code())


def run(R, a):
    sample = K.find_sample(a, R)
    man = K.load_manifest(sample)
    plan = PLAN
    if a.plan:
        plan = []
        for it in a.plan.split(';'):
            cw, cs, cf = it.split(':')
            plan.append((int(cw), int(cs), cf.split(',')))
    test = K.test_indices()
    if a.quick:
        test = test[:2]
    if a.limit_test:
        test = test[:a.limit_test]
    mode = a.models or ('ref' if a.quick else 'train')
    ref4 = K.load_ref('ceilref_models_even80.npz')
    full_train = False
    if mode == 'train':
        tr = K.train_indices()
        if a.limit_train:
            tr = K.spread(tr, a.limit_train)
        full_train = len(tr) == 80
        cnt = K.ensure_counts(R, [('S', sample, i, man['experts'][i]) for i in tr], a.cache, a.jobs)
        cl = [cnt[k] for k in sorted(cnt)]
        R.check('train_sources_sha256', all(c['sha_ok'] for c in cl))
        m = K.train_model(cl, 'c', 'c120', etab=True)
        cmp = K.compare_model(m, ref4, 'fg_a0_K0_c0')
        R.check('model_equal_ceilref_fg', all(v is True for v in cmp.values()) if full_train else None,
                cmp if full_train else f'skipped: trained on {len(tr)} experts')
    else:
        m = K.model_from_ref(ref4, 'fg_a0_K0_c0')
    models = {'fg0': K.model_arrays(m)}
    R.summary['models'] = {'mode': mode, 'model_id_and_blob_bytes': K.model_id_of(models['fg0'])}
    jobs = []
    for i in test:
        for cw, cs, cfs in plan:
            for c in cfs:
                jobs.append({'data_dir': sample, 'idx': i, 'entry': man['experts'][i], 'chunk': (cw, cs),
                             'items': [{'config': c, 'model': 'fg0', 'decode': bool(a.decode_all), 'sib_decode': True}]})
    R.log(f'{len(test)} experts, plan {plan}: {len(jobs)} payload jobs on {a.jobs} workers')
    rows = []
    for job, res in K.run_jobs(K.encode_job, jobs, a.jobs, R, 'payload', K.init_models, (models,)):
        for r in res['rows']:
            rows.append(r)
            R.log(f"{r['file']} {r['chunk'][0]}:{r['chunk'][1]} {r['config']:6s} used {r['sizes']['used']} total {r['sizes']['total']} "
                  f"lt {r['sizes']['length_table']} dec {r['verify']['decoded']} ok {r['verify']['equal_to_reference']}")
    rows.sort(key=lambda r: (r['idx'], r['chunk'][0], r['config']))
    K.write_jsonl(os.path.join(a.out, 'records.jsonl'), rows)
    summarize(R, rows, mode, full_train)


def summarize(R, rows, mode, full_train):
    keys = sorted({(r['chunk'][0], r['chunk'][1], r['config']) for r in rows})
    by = {k: {r['idx']: r for r in rows if (r['chunk'][0], r['chunk'][1], r['config']) == k} for k in keys}
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    rng = np.random.default_rng(K.BOOT_SEED)
    tab = {}
    for k in keys:
        cw, cs, c = k
        rr = [by[k][i] for i in common]
        t = K.config_table(rr)
        t['weight_streams'] = rr[0]['n_weight_streams']
        t['scale_streams'] = rr[0]['n_scale_streams']
        t['scale_stream_bytes_mean_max'] = [float(np.mean([r['slen']['mean'] for r in rr])), int(max(r['slen']['max'] for r in rr))]
        if rr[0]['lt_parts'].get('lt_weight_part_bytes') is not None:
            t['length_table_parts_mean'] = {p: float(np.mean([r['lt_parts'][p] for r in rr])) for p in rr[0]['lt_parts']}
            t['len_resid_std_scale_weight'] = [float(np.mean([r['len_resid_std'][0] for r in rr])),
                                               float(np.mean([r['len_resid_std'][1] for r in rr]))]
        b = (BASE[0], BASE[1], c)
        if b in by and b != k:
            t['delta_used_vs_2048'] = K.describe_delta([by[k][i]['sizes']['used'] - by[b][i]['sizes']['used'] for i in common], rng)
            t['delta_total_vs_2048'] = K.describe_delta([by[k][i]['sizes']['total'] - by[b][i]['sizes']['total'] for i in common], rng)
        tab[f'{c}@{cw}:{cs}'] = t
    # tail vs hash at equal chunk
    pw = {}
    for cw, cs in sorted({(k[0], k[1]) for k in keys}):
        x, y = (cw, cs, 'fg0dt'), (cw, cs, 'fg0de')
        if x in by and y in by:
            pw[f'fg0dt vs fg0de @{cw}:{cs}'] = K.describe_delta(
                [by[x][i]['sizes']['used'] - by[y][i]['sizes']['used'] for i in common], rng)
    R.summary['experts'] = common
    R.summary['configs'] = tab
    R.summary['pairwise_used'] = pw
    for nm, t in tab.items():
        d = t.get('delta_total_vs_2048')
        R.log(f"{nm:18s} total {t['mean_total_bytes']:12.1f} ratio {t['ratio_of_raw']:.5f} lt {t['mean_parts']['length_table']:8.1f} "
              f"{t['decode_check']}" + (f" vs 2048 {d['mean_pct_of_raw']:+.4f}% CI {d['CI95_pct_of_raw'][0]:+.4f}..{d['CI95_pct_of_raw'][1]:+.4f}" if d else ''))
    R.check('sources_sha256', all(r['source_sha256_ok'] for r in rows))
    dec = [r for r in rows if r['verify']['decoded']]
    need = [r for r in rows if r['config'] in ('fg0de', 'fg0dt')]
    bad = [(r['file'], r['chunk'], r['config'], r['verify']['decode_error']) for r in dec if not r['verify']['equal_to_reference']]
    R.check('records_decode_byte_exact', not bad and all(r['verify']['decoded'] for r in need),
            {'decoded': len(dec), 'bad': bad[:20]})
    if mode == 'ref' or full_train:
        h = K.load_ref('heldout_ref.json')['rows']
        mism, n = [], 0
        for r in rows:
            if tuple(r['chunk']) == BASE:
                q = h.get(f"{r['idx']}:{r['config']}")
                if q:
                    n += 1
                    if (q['total'], q['used']) != (r['sizes']['total'], r['sizes']['used']):
                        mism.append({'key': f"{r['idx']}:{r['config']}", 'ref': [q['total'], q['used']],
                                     'here': [r['sizes']['total'], r['sizes']['used']]})
        R.check('2048_equals_ceilref_heldout', (not mism) if n else None, {'compared': n, 'mismatch': mism[:20]})
        c6 = K.load_ref('chunk_ref.json')['rows']
        mism6, n6 = [], 0
        for r in rows:
            q = c6.get(f"{r['idx']}:{r['chunk'][0]}:{r['chunk'][1]}:{OLD.get(r['config'], r['config'])}")
            if q:
                n6 += 1
                if (q['total'], q['used']) != (r['sizes']['total'], r['sizes']['used']):
                    mism6.append({'key': f"{r['idx']}:{r['chunk']}:{r['config']}", 'ref': [q['total'], q['used']],
                                  'here': [r['sizes']['total'], r['sizes']['used']]})
        R.check('equals_team6_partial_rows', (not mism6) if n6 else None, {'compared': n6, 'mismatch': mism6[:20]})
    parts = [f"{nm} {t['delta_total_vs_2048']['mean_pct_of_raw']:+.3f}%" for nm, t in tab.items()
             if 'delta_total_vs_2048' in t and nm.startswith('fg0d')]
    R.result(f'{len(common)} experts; total vs same config @2048:512: ' + ', '.join(parts))


if __name__ == '__main__':
    main()
