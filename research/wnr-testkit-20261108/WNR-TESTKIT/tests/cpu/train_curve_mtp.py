"""train_curve_mtp: (A) training-set learning curve and (B) storage options for the MTP experts (CPU, numpy).

Port of team 5 mtp-training (curve.py, mtp.py, spot.py) onto the kit's count cache (lib/kitcommon.py).
Experts: the 160-expert sample + the 76 extra experts (70 routed + 6 MTP); every file is checked against its manifest
sha256. Test set = the 40 held-out sample experts of team 3 / team 4; pool = the other 120 sample experts + the 70
extra routed experts = 190 (leakage checked by (layer, expert), file name and sha256).
(A) Learning curve, code-length estimates (exact 14-bit quantizer, real-flush model 3.5 B/stream, u16 lengths,
    header, 4096 alignment): families wnr2 (plain, lower median), wnr2g0 (plain, anchor 120), f (ceil form, median of
    the coded scales), fg0 (ceil form, anchor 120); N = 5, 10, 20, 40, 80, 120, 190 training experts, random and
    layer-stratified draws (team 5 seeds), the 80-even split, in-sample controls, power-law fits and a bootstrap of
    the fit's prediction at 7680 experts.
(B) MTP experts, estimates: WNR2 routed model; WNR2-g0 routed; (i) ceil routed set anchor 120 (g0); (ii) ceil routed
    set trained with per-matrix densest-pair anchors, MTP coded with its own densest pair; (ii-b) g0 tables + MTP pair
    anchor; MTP-own sets leave-one-out (LOO, 5 MTP train) and leave-one-MTP-layer-out (LOLO) with anchor rules
    (densest pair / median / 120 / constant mode of the densest pairs); (iv-K2) per-matrix choice between the routed
    g0 set and an MTP-own set; side check: does any routed test matrix pick the MTP set?
(C) Real records (byte-exact decode checked) for a subset: 5 routed test experts (WNR2 / fg0 / fg0de with the
    190-pool models, fg0 / WNR2 with a 10-expert draw) and the 6 MTP experts (WNR2 routed, fg0 routed, (ii) routed
    pair, MTP-own LOO pair / 120 / mode); K = 2 is composed from the real per-matrix stream bytes of two real records
    (the reference coder implements K only together with the A side symbols, so no single K = 2 record is written).
Reference: team 5 out/curve.json, mtp.json, spot_*.jsonl (ref/curve_mtp_ref.json); comparisons are soft (reported).

Run (PowerShell, from the kit root; needs the extra experts):
  python -I tests\\cpu\\train_curve_mtp.py --quick
  python -I tests\\cpu\\train_curve_mtp.py
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'train_curve_mtp', 'order': 24, 'needs': ['data', 'extra'], 'quick_minutes': 3, 'full_minutes': 15,
            'description': 'training-set learning curve and MTP storage options (CPU; needs the extra experts)'}
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import numpy as np  # noqa: E402
import kitcommon as K  # noqa: E402
import wnr_common as W  # noqa: E402

NAME = 'train_curve_mtp'
FAMS = {'wnr2': ('o', 'med'), 'wnr2g0': ('o', 'c120'), 'f': ('c', 'med'), 'fg0': ('c', 'c120')}
HDR = {'wnr2': 32, 'wnr2g0': 32, 'f': 48, 'fg0': 48}
NS_FULL = [5, 10, 20, 40, 80, 120, 190]
REPS_FULL = {5: 40, 10: 40, 20: 40, 40: 30, 80: 30, 120: 30, 190: 1}
NWS, NSS = 17280, 2160
LT = 2 * (NWS + NSS)
FLUSH = 3.5


class Model:
    def __init__(self, C, S):
        self.wf = W.quantize(C)
        self.sf = W.quantize(S)
        self.lw = -np.log2(self.wf / W.P)
        self.ls = -np.log2(self.sf / W.P)

    def bits(self, C, S):
        return (C * self.lw).sum((1, 2)), (S * self.ls).sum(1)


def rec_est(wb, sb, header):
    used = header + LT + float(np.sum(sb)) / 8 + float(np.sum(wb)) / 8 + FLUSH * (NWS + NSS)
    return used, -(-used // 4096) * 4096


class Ctx:
    """cached per-expert context counts for (form, rule)."""

    def __init__(self, ex):
        self.ex = ex
        self.cache = {}

    def get(self, k, form, rule):
        key = (k, form, str(rule))
        if key not in self.cache:
            self.cache[key] = K.expert_ctx(self.ex[k], form, rule)
        return self.cache[key]

    def train(self, keys, form, rule_of):
        C = np.zeros((3, W.NCTX, 16), np.int64)
        S = np.zeros((3, 256), np.int64)
        for k in keys:
            c, s, _ = self.get(k, form, rule_of(k) if callable(rule_of) else rule_of)
            C += c
            S += s
        return Model(C, S)


def mode_low(vals):
    v, c = np.unique(np.asarray(vals), return_counts=True)
    return int(v[np.argmax(c)])


def main():
    p = K.base_parser(NAME, __file__, __doc__.split('\n')[0])
    p.add_argument('--no-real', action='store_true', help='skip the real-record part (C)')
    p.add_argument('--real-limit', type=int, default=0, help='(debug/smoke) only the first N real-record jobs (MTP first)')
    a = K.finish_args(p.parse_args(), NAME)
    with K.Run(NAME, a) as R:
        run(R, a)
    sys.exit(R.exit_code())


def run(R, a):
    sample, extra = K.find_sample(a, R), K.find_extra(a, R)
    sman, xman = K.load_manifest(sample), K.load_manifest(extra)
    test_idx = K.test_indices()
    sel_s = list(range(len(sman['experts'])))
    sel_x = list(range(len(xman['experts'])))
    if a.quick:
        tq = K.spread(test_idx, 8)
        other = [i for i in sel_s if i not in test_idx]
        xr = [i for i in sel_x if xman['experts'][i].get('kind', 'layers') == 'layers']
        xm = [i for i in sel_x if xman['experts'][i].get('kind') == 'mtp']
        sel_s = sorted(tq + K.spread(other, 16))
        sel_x = sorted(K.spread(xr, 16) + xm)
        test_idx = tq
    items = [('S', sample, i, sman['experts'][i]) for i in sel_s] + [('X', extra, i, xman['experts'][i]) for i in sel_x]
    ex = K.ensure_counts(R, items, a.cache, a.jobs)
    keys = sorted(ex)                                   # team 5 order: 'S_L..' then 'X_L..' / 'X_M..'
    ks = {ex[k]['index']: k for k in keys if ex[k]['set'] == 'S'}
    TEST = [ks[i] for i in test_idx]
    ROUTED = [k for k in keys if ex[k]['kind'] == 'layers']
    POOL = [k for k in ROUTED if k not in TEST]
    MTP = sorted(k for k in keys if ex[k]['kind'] == 'mtp')
    EVEN80 = [ks[i] for i in range(160) if i % 2 == 0 and i in ks]
    sha = {('S', e['file']): e['sha256'] for e in sman['experts']}
    sha.update({('X', e['file']): e['sha256'] for e in xman['experts']})
    tl = {(ex[k]['layer'], ex[k]['expert']) for k in TEST}
    pl = [(ex[k]['layer'], ex[k]['expert']) for k in POOL]
    tsha = {sha[(ex[k]['set'], ex[k]['file'])] for k in TEST}
    psha = [sha[(ex[k]['set'], ex[k]['file'])] for k in POOL]
    checks = {'n_test': len(TEST), 'n_pool': len(POOL), 'n_mtp': len(MTP),
              'pool_layer_expert_overlap_with_test': len(tl & set(pl)), 'pool_duplicate_layer_expert': len(pl) - len(set(pl)),
              'pool_sha_overlap_with_test': len(tsha & set(psha)), 'pool_duplicate_sha': len(psha) - len(set(psha)),
              'pool_sample': sum(ex[k]['set'] == 'S' for k in POOL), 'pool_extra_routed': sum(ex[k]['set'] == 'X' for k in POOL),
              'roundtrip_fail': sum(m['roundtrip_ok'] is False for e in ex.values() for m in e['m']),
              'ceil_not_applicable': sum(not e['ceil_ok'] for e in ex.values()),
              'pool_per_layer': np.bincount([ex[k]['layer'] for k in POOL], minlength=40).tolist()}
    R.summary['data_checks'] = checks
    R.check('sources_sha256', all(e['sha_ok'] for e in ex.values()), f"{sum(e['sha_ok'] for e in ex.values())}/{len(ex)}")
    R.check('no_test_leakage', checks['pool_layer_expert_overlap_with_test'] == 0 and checks['pool_sha_overlap_with_test'] == 0
            and checks['pool_duplicate_sha'] == 0, checks)
    R.check('ceil_roundtrip_all_matrices', checks['roundtrip_fail'] == 0 and checks['ceil_not_applicable'] == 0)
    if not a.quick:
        R.check('sizes_full', len(TEST) == 40 and len(POOL) == 190 and len(MTP) == 6, checks)
    ref = K.load_ref('curve_mtp_ref.json')
    cx = Ctx(ex)
    curve = part_curve(R, a, ex, cx, TEST, POOL, EVEN80, ref)
    mtp = part_mtp(R, a, ex, cx, TEST, ROUTED, POOL, MTP, ref)
    real = None if a.no_real else part_real(R, a, ex, cx, sample, extra, sman, xman, TEST, ROUTED, POOL, MTP, curve, ref)
    # short result
    s = []
    if curve.get('d190') is not None:
        s.append(f"fg0-wnr2 @N={curve['nmax']} {curve['d190']:+.3f}%raw")
    if mtp.get('best'):
        s.append(f"MTP best est {mtp['best'][0]} {mtp['best'][1]:.4f}")
    if real and real.get('mtp_mean_ratio'):
        s.append('MTP real ' + ', '.join(f'{k} {v:.4f}' for k, v in real['mtp_mean_ratio'].items()))
    R.result('; '.join(s))


# ------------------------------------------------------------------------------------------------ (A)
def part_curve(R, a, ex, cx, TEST, POOL, EVEN80, ref):
    t0 = time.time()
    keys = TEST + POOL
    kix = {k: i for i, k in enumerate(keys)}
    CC, SS = {}, {}
    for fam, (form, rule) in FAMS.items():
        CC[fam] = np.zeros((len(keys), 3, W.NCTX, 16), np.int64)
        SS[fam] = np.zeros((len(keys), 3, 256), np.int64)
        for k in keys:
            c, s, _ = cx.get(k, form, rule)
            CC[fam][kix[k]] = c
            SS[fam][kix[k]] = s
    ti = np.array([kix[k] for k in TEST])
    R.log(f'curve: arrays for {len(keys)} experts x {len(FAMS)} families ({time.time() - t0:.0f}s)')

    def evaluate(train_keys, fam):
        idx = np.array([kix[k] for k in train_keys])
        m = Model(CC[fam][idx].sum(0), SS[fam][idx].sum(0))
        wb = np.einsum('emcs,mcs->e', CC[fam][ti].astype(np.float64), m.lw)
        sb = np.einsum('ems,ms->e', SS[fam][ti].astype(np.float64), m.ls)
        used = HDR[fam] + LT + sb / 8 + wb / 8 + FLUSH * (NWS + NSS)
        return {'used': used, 'total': np.ceil(used / 4096) * 4096, 'wbits': wb, 'sbits': sb}

    def summ(r):
        n = len(TEST)
        return {'used_mean': float(r['used'].mean()), 'total_mean': float(r['total'].mean()),
                'bpw': float(r['wbits'].sum() / (n * K.W_PER_REC) + 8 * FLUSH * NWS / K.W_PER_REC),
                'bps': float(r['sbits'].sum() / (n * K.S_PER_REC) + 8 * FLUSH * NSS / K.S_PER_REC)}

    res = {'test': TEST, 'pool_size': len(POOL), 'families': {f: list(v) for f, v in FAMS.items()}}
    if len(EVEN80) == 80 and len(TEST) == 40:
        crv = {}
        for fam in FAMS:
            crv[fam] = summ(evaluate(EVEN80, fam))
        rc = ref['curve']['cr_split_even80']
        crv['real_ceilref_used (heldout_summary_g0)'] = rc.get('real_cr_used')
        crv['est_minus_real_used'] = {f: crv[f]['used_mean'] - rc['real_cr_used'][f] for f in rc.get('real_cr_used', {}) if f in crv}
        res['cr_split_even80'] = crv
        R.log('curve: 80-even split est - real used', K.jdump(crv['est_minus_real_used']))
        R.check('curve_even80_equals_team5', all(abs(crv[f]['used_mean'] - rc[f]['used_mean']) < 0.5 for f in FAMS),
                {f: [crv[f]['used_mean'], rc[f]['used_mean']] for f in FAMS}, hard=False)
    if a.quick:
        NS = [n for n in (10, 20, 32) if n <= len(POOL)]
        REPS = {n: 3 for n in NS}
    else:
        NS, REPS = NS_FULL, REPS_FULL
    layers = sorted({ex[k]['layer'] for k in POOL})
    pool_by_layer = {L: [k for k in POOL if ex[k]['layer'] == L] for L in layers}
    draws = []
    for N in NS:
        for kind in ('random', 'strat'):
            if N == len(POOL) and kind == 'strat':
                continue
            if kind == 'strat' and N >= 40 and (len(layers) < 40 or min(len(v) for v in pool_by_layer.values()) < N // 40):
                continue
            if kind == 'strat' and N < 40 and N > len(layers):
                continue
            rng = np.random.default_rng(20261107 + 1000 * N + (0 if kind == 'random' else 1))
            for rep in range(REPS[N]):
                if kind == 'random':
                    sel = [POOL[i] for i in rng.choice(len(POOL), N, replace=False)]
                elif N >= 40:
                    sel = []
                    for L in layers:
                        pl_ = pool_by_layer[L]
                        sel += [pl_[i] for i in rng.choice(len(pl_), N // 40, replace=False)]
                else:
                    Ls = [layers[i] for i in rng.choice(len(layers), N, replace=False)]
                    sel = [pool_by_layer[L][rng.integers(len(pool_by_layer[L]))] for L in Ls]
                row = {'N': N, 'kind': kind, 'rep': rep, 'members': sel}
                for fam in FAMS:
                    row[fam] = summ(evaluate(sel, fam))
                draws.append(row)
        R.log(f'curve: N {N} done ({time.time() - t0:.0f}s)')
    res['draws'] = draws
    ins = {}
    for fam in FAMS:
        u = [evaluate(POOL + [k], fam)['used'][TEST.index(k)] for k in TEST]
        ins[fam] = {'pool_plus_self_used_mean': float(np.mean(u)),
                    'all_routed_used_mean': float(evaluate(POOL + TEST, fam)['used'].mean())}
    res['insample'] = ins
    agg = {}
    for kind in ('random', 'strat'):
        for fam in list(FAMS) + ['d_fg0_wnr2', 'd_fg0_wnr2g0', 'd_wnr2g0_wnr2', 'd_fg0_f']:
            pts = []
            for N in NS:
                rows = [d for d in draws if d['N'] == N and (d['kind'] == kind or N == len(POOL))]
                if not rows:
                    continue
                if fam.startswith('d_'):
                    x, y = fam[2:].split('_')
                    v = np.array([r[x]['used_mean'] - r[y]['used_mean'] for r in rows])
                else:
                    v = np.array([r[fam]['used_mean'] for r in rows])
                pts.append({'N': N, 'n_draws': len(v), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                            'min': float(v.min()), 'max': float(v.max()), 'mean_pct_of_raw': float(100 * v.mean() / K.RAW)})
            agg[f'{kind}:{fam}'] = pts
    res['agg'] = agg

    def fit(Ns, y, step=0.005):
        best = None
        for b in np.arange(0.05, 3.001, step):
            X = np.stack([np.ones_like(Ns), Ns ** -b], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(((X @ coef - y) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, b, coef)
        return best

    fits = {}
    for kind in ('random', 'strat'):
        for fam in FAMS:
            P_ = [q for q in agg[f'{kind}:{fam}'] if q['N'] >= 10]
            if len(P_) < 3:
                continue
            Ns = np.array([q['N'] for q in P_], float)
            y = np.array([q['mean'] for q in P_])
            rss, b, (c, aa) = fit(Ns, y)
            fits[f'{kind}:{fam}:N>=10'] = {'c_inf': float(c), 'a': float(aa), 'b': float(b), 'rms_resid': float(np.sqrt(rss / len(y))),
                                           'pred_7680': float(c + aa * 7680 ** -b), 'pred_190': float(c + aa * 190 ** -b)}
    res['fits'] = fits
    if not a.quick:
        rng = np.random.default_rng(7)
        boot = {f: [] for f in FAMS}
        boot['d_fg0_wnr2'], boot['d_fg0_wnr2g0'] = [], []
        Nf = [N for N in NS if N >= 10]
        byN = {N: [d for d in draws if d['N'] == N and (d['kind'] == 'random' or N == len(POOL))] for N in Nf}
        for it in range(300):
            pick = {N: [byN[N][i] for i in rng.integers(len(byN[N]), size=len(byN[N]))] for N in Nf}
            pred = {}
            for fam in FAMS:
                y = np.array([np.mean([r[fam]['used_mean'] for r in pick[N]]) for N in Nf])
                best = fit(np.array(Nf, float), y, 0.01)
                pred[fam] = float(best[2][0] + best[2][1] * 7680 ** -best[1])
                boot[fam].append(pred[fam])
            boot['d_fg0_wnr2'].append(pred['fg0'] - pred['wnr2'])
            boot['d_fg0_wnr2g0'].append(pred['fg0'] - pred['wnr2g0'])
        res['boot_pred_7680'] = {k: {'p2.5': float(np.percentile(v, 2.5)), 'p50': float(np.percentile(v, 50)),
                                     'p97.5': float(np.percentile(v, 97.5))} for k, v in boot.items()}
    # reference comparison (team 5): first draws + aggregate means
    if not a.quick:
        fd = {(d['N'], d['kind'], d['rep']): d for d in ref['curve']['first_draws']}
        same_m, same_v, n = 0, 0, 0
        for d in draws:
            q = fd.get((d['N'], d['kind'], d['rep']))
            if q:
                n += 1
                same_m += q['members'] == d['members']
                same_v += all(abs(q[f] - d[f]['used_mean']) < 0.5 for f in FAMS)
        R.check('curve_draws_equal_team5', n > 0 and same_m == n and same_v == n,
                {'compared': n, 'same_members': same_m, 'same_values': same_v}, hard=False)
        ra = ref['curve']['agg']
        dif = {k: max(abs(x['mean'] - y['mean']) for x, y in zip(v, ra.get(k, [])) if x['N'] == y['N'])
               for k, v in agg.items() if k in ra and v}
        R.check('curve_agg_equal_team5', all(v < 0.5 for v in dif.values()), {'max_abs_diff_B': dif}, hard=False)
    for kind in ('random', 'strat'):
        for fam in ('fg0', 'wnr2', 'd_fg0_wnr2', 'd_fg0_wnr2g0'):
            pts = agg.get(f'{kind}:{fam}', [])
            if pts:
                R.log(f'curve {kind:6s} {fam:13s} ' + ' '.join(
                    f"{q['N']}:{(q['mean_pct_of_raw'] if fam.startswith('d_') else q['mean'] / K.RAW):.5f}" for q in pts))
    R.summary['curve'] = {k: v for k, v in res.items() if k != 'draws'}
    K.write_jsonl(os.path.join(a.out, 'curve_draws.jsonl'), draws)
    last = agg.get('random:d_fg0_wnr2', [])
    return {'d190': last[-1]['mean_pct_of_raw'] if last else None, 'nmax': last[-1]['N'] if last else None,
            'draw10': next((d['members'] for d in draws if d['N'] == 10 and d['kind'] == 'random'), None)}


# ------------------------------------------------------------------------------------------------ (B)
def part_mtp(R, a, ex, cx, TEST, ROUTED, POOL, MTP, ref):
    if not MTP:
        R.log('mtp: no MTP experts')
        return {}
    t0 = time.time()
    options = {}

    def put(name, k, wb, sb, anchors, header, choice=None):
        options.setdefault(name, {})[k] = {'wb': np.asarray(wb, float), 'sb': np.asarray(sb, float),
                                            'anchors': [int(x) for x in anchors], 'header': header, 'choice': choice}

    def cost(model, k, form, rule, header, name):
        c, s, an = cx.get(k, form, rule)
        wb, sb = model.bits(c, s)
        put(name, k, wb, sb, an, header)

    M_R_omed = cx.train(ROUTED, 'o', 'med')
    M_R_o120 = cx.train(ROUTED, 'o', 'c120')
    M_R_c120 = cx.train(ROUTED, 'c', 'c120')
    M_R_cpair = cx.train(ROUTED, 'c', 'pair')
    M_P_c120 = cx.train(POOL, 'c', 'c120')
    for k in MTP:
        cost(M_R_omed, k, 'o', 'med', 32, 'WNR2 routed model (median anchors)')
        cost(M_R_o120, k, 'o', 'c120', 32, 'WNR2-g0 routed model (anchor 120)')
        cost(M_R_c120, k, 'c', 'c120', 48, '(i) ceil, routed set, anchor 120 (g0)')
        cost(M_P_c120, k, 'c', 'c120', 48, '(i) ceil, routed set (190 pool), anchor 120')
        cost(M_R_cpair, k, 'c', 'pair', 48, '(ii) ceil, routed set, per-matrix densest-pair anchor')
        cost(M_R_c120, k, 'c', 'pair', 48, '(ii-b) ceil, routed set g0 tables, MTP densest-pair anchor')
    layer_of = {k: ex[k]['layer'] for k in MTP}
    splits = {'LOO': {k: [j for j in MTP if j != k] for k in MTP},
              'LOLO': {k: [j for j in MTP if layer_of[j] != layer_of[k]] for k in MTP}}
    sets = {}
    for sp, tr_of in splits.items():
        for k in MTP:
            tr = tr_of[k]
            if not tr:
                continue
            cost(cx.train(tr, 'o', 'med'), k, 'o', 'med', 32, f'WNR2 MTP-own model ({sp}, median anchors)')
            for rule, lab in (('pair', 'per-matrix densest pair'), ('med', 'per-matrix median'), ('c120', 'constant 120')):
                cost(cx.train(tr, 'c', rule), k, 'c', rule, 48, f'(iii) ceil, MTP-own set ({sp}), {lab}')
            a0 = tuple(mode_low([ex[t]['m'][m]['pair_c'] for t in tr]) for m in range(3))
            cost(cx.train(tr, 'c', a0), k, 'c', a0, 48, f'(iii) ceil, MTP-own set ({sp}), constant anchor0 = mode of densest pair')
            sets.setdefault(sp, {})[k] = {'train': tr, 'anchor0_const': list(a0)}
        for mname in (f'(iii) ceil, MTP-own set ({sp}), constant anchor0 = mode of densest pair',
                      f'(iii) ceil, MTP-own set ({sp}), per-matrix densest pair'):
            for k in MTP:
                if k not in options.get(mname, {}):
                    continue
                x = options['(i) ceil, routed set, anchor 120 (g0)'][k]
                y = options[mname][k]
                wb, sb, an, ch = np.zeros(3), np.zeros(3), [], []
                for m in range(3):
                    pick = y if (y['wb'][m] + y['sb'][m]) < (x['wb'][m] + x['sb'][m]) else x
                    wb[m], sb[m] = pick['wb'][m], pick['sb'][m]
                    an.append(pick['anchors'][m])
                    ch.append('mtp' if pick is y else 'routed')
                put('(iv-K2) routed g0 set + ' + mname.split(', ', 1)[1] + ', per-matrix choice', k, wb, sb, an, 48, ch)
    raw_al = K.RAW
    table = {}
    w2 = {k: rec_est(options['WNR2 routed model (median anchors)'][k]['wb'], options['WNR2 routed model (median anchors)'][k]['sb'], 32)[1] for k in MTP}
    for name, d in options.items():
        per = {}
        for k, o in d.items():
            used, tot = rec_est(o['wb'], o['sb'], o['header'])
            per[k] = {'used': used, 'total': tot, 'ratio': tot / K.RAW, 'stored': min(tot, raw_al), 'stored_raw': tot >= raw_al,
                      'bpw': float(o['wb'].sum() / K.W_PER_REC + 8 * FLUSH * NWS / K.W_PER_REC),
                      'anchors': o['anchors'], 'choice': o['choice']}
        tot = np.array([v['total'] for v in per.values()])
        st = np.array([v['stored'] for v in per.values()])
        table[name] = {'n': len(per), 'mean_ratio': float(tot.mean() / K.RAW), 'mean_stored_ratio': float(st.mean() / K.RAW),
                       'n_stored_raw': int(sum(v['stored_raw'] for v in per.values())),
                       'vs_wnr2_routed_with_raw_fallback_pp': float(100 * (st.mean() - np.mean([min(w2[k], raw_al) for k in per])) / K.RAW),
                       'mean_bpw': float(np.mean([v['bpw'] for v in per.values()])), 'per_expert': per}
    # side check: does any routed test matrix pick an MTP set over the 190-pool g0 set?
    side = {}
    if len(MTP) >= 2:
        Mm = cx.train(MTP, 'c', 'pair')
        picks, gain = 0, 0.0
        for k in TEST:
            ca, sa, _ = cx.get(k, 'c', 'c120')
            cb, sb_, _ = cx.get(k, 'c', 'pair')
            wa, xa = M_P_c120.bits(ca, sa)
            wb_, xb = Mm.bits(cb, sb_)
            for m in range(3):
                if wb_[m] + xb[m] < wa[m] + xa[m]:
                    picks += 1
                    gain += (wb_[m] + xb[m] - wa[m] - xa[m]) / 8
        side = {'routed_test_matrices_picking_mtp_set': picks, 'of': 3 * len(TEST), 'bytes_gained_total': gain}
    R.summary['mtp_estimates'] = {'options': table, 'sets': sets, 'routed_side_check': side,
                                  'note': 'code-length estimates, 3.5 B/stream flush, u16 lengths, 4096 alignment'}
    for name, t in table.items():
        R.log(f"mtp est {name:100s} mean {t['mean_ratio']:.4f} stored {t['mean_stored_ratio']:.4f} raw#{t['n_stored_raw']}")
    R.log('mtp side check', K.jdump(side), f'({time.time() - t0:.0f}s)')
    if not a.quick:
        ro = ref['mtp_options']
        dif = {n: [t['mean_ratio'], ro[n]['mean_ratio']] for n, t in table.items() if n in ro}
        R.check('mtp_estimates_equal_team5', bool(dif) and all(abs(x - y) < 1e-6 for x, y in dif.values()),
                {'compared': len(dif), 'max_abs_diff_ratio': max(abs(x - y) for x, y in dif.values()) if dif else None}, hard=False)
    best = min(table.items(), key=lambda kv: kv[1]['mean_stored_ratio'])
    return {'best': (best[0], best[1]['mean_stored_ratio'])}


# ------------------------------------------------------------------------------------------------ (C)
def part_real(R, a, ex, cx, sample, extra, sman, xman, TEST, ROUTED, POOL, MTP, curve, ref):
    t0 = time.time()
    models, jobs = {}, []

    def mk(name, keys, form, rule, wnr2=False, by_key=None):
        if name not in models:
            m = K.train_model([ex[k] for k in keys], form, rule, etab=not wnr2, anchors_by_key=by_key)
            models[name] = K.model_arrays(m, wnr2=wnr2)
        return name

    def entry(k):
        e = ex[k]
        if e['set'] == 'S':
            return sample, sman['experts'][e['index']]
        return extra, xman['experts'][e['index']]

    def add(k, config, model, anchor, label, sib=None):
        d, en = entry(k)
        it = {'config': config, 'model': model, 'anchor': anchor, 'label': label}
        if sib is False:
            it['build_sibling'] = False
        jobs.append({'data_dir': d, 'idx': k, 'entry': en, 'chunk': (2048, 512), 'items': [it]})

    picks = [TEST[i] for i in (0, 10, 20, 30, 39)] if len(TEST) == 40 else TEST[:1]
    d10 = curve.get('draw10')
    w2p = mk('w2_pool', POOL, 'o', 'med', wnr2=True)
    fgp = mk('fg_pool', POOL, 'c', 'c120')
    for k in picks:
        add(k, 'wnr2', w2p, None, 'wnr2 pool190')
        add(k, 'fg0', fgp, 'c120', 'fg0 pool190', sib=False)
        if not a.quick:
            add(k, 'fg0e', fgp, 'c120', 'fg0e pool190 (+fg0de)')
        if d10 and not a.quick:
            add(k, 'fg0', mk('fg_d10', d10, 'c', 'c120'), 'c120', 'fg0 draw10', sib=False)
            add(k, 'wnr2', mk('w2_d10', d10, 'o', 'med', wnr2=True), None, 'wnr2 draw10')
    mtp_keys = MTP if not a.quick else [k for k in MTP if k.endswith(('M00_E001', 'M01_E024'))] or MTP[:2]
    if MTP:
        w2r = mk('w2_routed', ROUTED, 'o', 'med', wnr2=True)
        fgr = mk('fg_routed', ROUTED, 'c', 'c120')
        fpr = mk('fgpair_routed', ROUTED, 'c', 'pair')
        for k in mtp_keys:
            tr = [j for j in MTP if j != k]
            a0 = tuple(mode_low([ex[t]['m'][m]['pair_c'] for t in tr]) for m in range(3))
            if not a.quick:
                add(k, 'wnr2', w2r, None, 'wnr2 routed')
            add(k, 'fg0', fgr, 'c120', 'fg0 routed (i)', sib=False)
            if not a.quick:
                add(k, 'fgx', fpr, 'pair', 'fg routed pair-trained, own pair (ii)', sib=False)
            add(k, 'fgx', mk(f'mtp_pair_{k}', tr, 'c', 'pair'), 'pair', 'fg MTP-own LOO, densest pair', sib=False)
            add(k, 'fg0', mk(f'mtp_c120_{k}', tr, 'c', 'c120'), 'c120', 'fg MTP-own LOO, 120', sib=False)
            if not a.quick:
                add(k, 'fgx', mk(f'mtp_mode_{k}', tr, 'c', a0), list(a0), 'fg MTP-own LOO, const mode of pair', sib=False)
    if a.real_limit:
        jobs = sorted(jobs, key=lambda j: 0 if j['idx'] in MTP else 1)[:a.real_limit]
        R.log(f'real records limited to {a.real_limit} payload jobs (debug/smoke)')
    R.log(f'real records: {len(jobs)} payload jobs, {len(models)} models, {a.jobs} workers')
    rows = []
    for job, res in K.run_jobs(K.encode_job, jobs, a.jobs, R, 'record', K.init_models, (models,)):
        for r in res['rows']:
            r['key'] = job['idx']
            # estimate for the same model and expert
            mod = models[r['model']]
            form = 'o' if mod['wnr2'] else 'c'
            C, S, _ = K.expert_ctx(ex[r['key']], form, list(r['anchors']))
            lw = -np.log2(mod['wf'].reshape(3, W.NCTX, 16) / W.P)
            ls = -np.log2(mod['scf'] / W.P)
            used_est, _ = rec_est((C * lw).sum((1, 2)), (S * ls).sum(1), r['sizes']['header'])
            r['est_used_flush3.5'] = used_est
            r['real_minus_est_used'] = r['sizes']['used'] - used_est
            rows.append(r)
            R.log(f"{r['key']} {r['label']}{'' if r['payload_parent'] is None else ' -> ' + r['config']}: total {r['sizes']['total']} "
                  f"ratio {r['sizes']['total'] / K.RAW:.5f} anchors {r['anchors']} exact {r['verify']['equal_to_reference']} "
                  f"real-est {r['real_minus_est_used']:+.1f}")
    rows.sort(key=lambda r: (r['key'], str(r['label']), r['config']))
    # K = 2 composed from real per-matrix stream bytes
    comp = {}
    for k in mtp_keys:
        base = [r for r in rows if r['key'] == k and r['label'] == 'fg0 routed (i)']
        for lab in ('fg MTP-own LOO, densest pair', 'fg MTP-own LOO, const mode of pair', 'fg MTP-own LOO, 120'):
            alt = [r for r in rows if r['key'] == k and r['label'] == lab]
            if base and alt:
                x, y = base[0]['per_matrix_bytes'], alt[0]['per_matrix_bytes']
                ch = ['mtp' if y[m] < x[m] else 'routed' for m in range(3)]
                used = 48 + LT + sum(min(x[m], y[m]) for m in range(3))
                comp.setdefault(f'K2 real-composed: routed g0 + {lab}', {})[k] = {
                    'used': used, 'total': W.align(used), 'ratio': W.align(used) / K.RAW, 'choice': ch}
    mean = {}
    for lab in sorted({r['label'] for r in rows if r['key'] in mtp_keys}):
        rr = [r for r in rows if r['key'] in mtp_keys and r['label'] == lab and r['payload_parent'] is None]
        if len(rr) == len(mtp_keys):
            mean[lab] = float(np.mean([min(r['sizes']['total'], K.RAW) for r in rr]) / K.RAW)
    for nm, d in comp.items():
        if len(d) == len(mtp_keys):
            mean[nm] = float(np.mean([min(v['total'], K.RAW) for v in d.values()]) / K.RAW)
    R.summary['real_records'] = {'mtp_mean_stored_ratio': mean, 'k2_composed': comp,
                                 'rows': [{q: r[q] for q in ('key', 'file', 'label', 'config', 'model', 'anchors', 'sizes',
                                                             'per_matrix_bytes', 'record_sha256', 'verify', 'timing',
                                                             'est_used_flush3.5', 'real_minus_est_used')} for r in rows]}
    K.write_jsonl(os.path.join(a.out, 'real_records.jsonl'), rows)
    bad = [(r['key'], r['label'], r['config'], r['verify']['decode_error']) for r in rows if not r['verify']['equal_to_reference']]
    R.check('real_records_decode_byte_exact', not bad and all(r['verify']['decoded'] for r in rows), {'records': len(rows), 'bad': bad})
    # soft: team 5 spot totals (same expert, same model label)
    lab5 = {'wnr2 pool190': ('wnr2', 'pool190'), 'fg0 pool190': ('fg', 'pool190'), 'fg0 draw10': ('fg', 'draw10'),
            'wnr2 draw10': ('wnr2', 'draw10'), 'wnr2 routed': ('wnr2', 'routed230'), 'fg0 routed (i)': ('fg', 'routed230'),
            'fg MTP-own LOO, densest pair': ('fg', 'mtpLOO_pair'), 'fg MTP-own LOO, 120': ('fg', 'mtpLOO_const120'),
            'fg MTP-own LOO, const mode of pair': ('fg', 'mtpLOO_const_mode')}
    sp = {(q['key'], q['config'], q['model']): q for q in ref['spot']}
    n, same, diff = 0, 0, []
    for r in rows:
        if a.quick and 'MTP-own' not in str(r['label']):
            continue                      # quick: routed / pool models are trained on a subset, not comparable
        if r['payload_parent'] is None and r['label'] in lab5:
            q = sp.get((r['key'], *lab5[r['label']]))
            if q:
                n += 1
                if q['total'] == r['sizes']['total']:
                    same += 1
                else:
                    diff.append({'key': r['key'], 'label': r['label'], 'ref': q['total'], 'here': r['sizes']['total']})
        if r['config'] == 'fg0de':
            q = sp.get((r['key'], 'fgde', 'pool190'))
            if q:
                n += 1
                same += q['total'] == r['sizes']['total']
                if q['total'] != r['sizes']['total']:
                    diff.append({'key': r['key'], 'label': 'fg0de pool190', 'ref': q['total'], 'here': r['sizes']['total']})
    R.check('real_records_equal_team5_spot', (same == n) if n else None, {'compared': n, 'equal': same, 'diff': diff[:20]}, hard=False)
    R.log(f'real records done ({time.time() - t0:.0f}s): MTP mean stored ratio ' + K.jdump(mean))
    return {'mtp_mean_ratio': {k: v for k, v in mean.items() if 'LOO' in k or 'routed' in k}}


if __name__ == '__main__':
    main()
