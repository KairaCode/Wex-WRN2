#!/usr/bin/env python
"""capacity_update.py -- team 4 / capacity-update: team 3's value model (capacity_value.py, verbatim copy in this
folder, sha256 D5B06E88...E19B) re-run with the REAL record sizes that team 4 / ceil-ref measured for the ceil-form
formats, plus full V4.1 container sizes vs the free space in the WNR2 reference.

What is new here (everything else is team 3's model, unchanged):
  * record ratios from ceil-ref's real rANS records (40 held-out experts, models from the 80 even experts), applied as
    a per-record byte DELTA to the production WNR2 mean record (16,914,680 B over all 7680 records);
  * the record-size uncertainty enters the Monte Carlo: per draw, a bootstrap resample of the 40 test experts (paired
    across formats) gives that draw's ratios -> simulate2(), a copy of team 3's simulate() that accepts a per-draw
    ratio array (checked bit-for-bit against the original for scalar ratios);
  * sensitivities: no W2 (single-record helper), ceil-form decode-time penalty (dk), format-specific fixed-slot
    overhead from the measured max/mean record, fixed (non-bootstrapped) ratios, two extra MC seeds;
  * elasticity (% tok/s per % fewer record bytes) per draw for every format;
  * container sizes = header page + model (4096-aligned) + 72-B entry table (4096-aligned) + 7680 padded records.
ASSUMPTIONS KEPT FROM TEAM 3: G2 (V4.1 MoE on the GPU pipeline) and W2 (per-layer batched WNR decode) are BUILT;
neither exists in KAIRA src today (WNR2-REFERENCE.md:123-135). The no-W2 rows are a sensitivity only.
"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # python -I drops the script dir; this folder holds only our own files
import numpy as np
import capacity_value as cvm                   # verbatim copy of team 3's model

GIB, PCIE, KS, QCAP, DRIVES = cvm.GIB, cvm.PCIE, cvm.KS, cvm.QCAP, cvm.DRIVES
GEO = cvm.V41
RAW = GEO.raw                                  # 18,800,640 B
N_REC = GEO.N                                  # 7680
CEILREF = os.environ.get('WNR_CEILREF_RESULTS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ceil-ref', 'results')  # kit

# ---------------------------------------------------------------- reference constants (read in WNR2-REFERENCE.md)
REF = {
    'WNR2_container_bytes': 129_905_360_896,   # ref:80  E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex
    'WNR2_records_start': 618_496,             # ref:101 table_offset 65536; records start at 618,496
    'WNR2_table_offset': 65_536,               # ref:101, ref:190
    'WNM2_bytes': 59_408,                      # ref:229
    'WEX1_container_bytes': 144_389_472_256,   # ref:81-82
    'entry_bytes': 72,                         # ref:179
    'WNR2_mean_record': 16_914_680,            # brief:16, ref s12 (average over all 7680 records, 4096-padded)
    'free_GB_1711': {'U': 122.6, 'E': 105.9, 'D': 61.4},   # ref:110-115, 139, 641 (Get-PSDrive, decimal GB, 17:11)
    'min_free_gib_W1': 5.0,                    # ref:61, ref:320 (--min-free-gib 5 used for the W1 build)
    'builder_upfront_factor': 0.90,            # ref:300 build_nr2_full.py: 0.90 x source size + margin
}


def align4096(n):
    return int(-(-int(n) // 4096) * 4096)


# ---------------------------------------------------------------- 1. ceil-ref real records
def load_ceilref():
    def rows(files):
        out = []
        for f in files:
            with open(os.path.join(CEILREF, f)) as fh:
                out += [json.loads(l) for l in fh if l.strip()]
        return out
    A = rows(['heldout_A.jsonl', 'heldout_B.jsonl'])
    G0 = rows(['heldout_g0.jsonl'])
    with open(os.path.join(CEILREF, 'heldout_summary.json')) as fh:
        S = json.load(fh)
    with open(os.path.join(CEILREF, 'heldout_summary_g0.json')) as fh:
        S0 = json.load(fh)

    def table(rs):
        cfgs = sorted({x['config'] for x in rs})
        idx = sorted({x['idx'] for x in rs})
        T = {c: np.zeros(len(idx)) for c in cfgs}
        U = {c: np.zeros(len(idx)) for c in cfgs}
        ok = {c: 0 for c in cfgs}
        pos = {i: k for k, i in enumerate(idx)}
        seen = set()
        for x in rs:
            key = (x['idx'], x['config'])
            assert key not in seen, key
            seen.add(key)
            T[x['config']][pos[x['idx']]] = x['sizes']['total']
            U[x['config']][pos[x['idx']]] = x['sizes']['used']
            v = x['verify']
            good = v['decode_error'] is None and v['equal_to_reference'] and x['source_sha256_ok']
            if 'c' not in x['config'].replace('wnr2', ''):
                good = good and v['byte_exact_vs_source']
            ok[x['config']] += int(good)
        layers = {x['idx']: x['layer'] for x in rs}
        return T, U, ok, idx, [layers[i] for i in idx]
    T, Uu, ok, idx, layers = table(A)
    T0, U0, ok0, idx0, _ = table(G0)
    assert idx == idx0 and len(idx) == 40
    checks = {}
    for c in T:
        checks[c] = {'mean_total': float(T[c].mean()), 'summary_mean_total': S['configs'][c]['mean_total_bytes'],
                     'records_ok': ok[c],
                     'g0_mean_total': float(T0[c].mean()), 'g0_summary_mean_total': S0['configs'][c]['mean_total_bytes'],
                     'g0_records_ok': ok0[c]}
        assert abs(checks[c]['mean_total'] - checks[c]['summary_mean_total']) < 0.05, c
        assert abs(checks[c]['g0_mean_total'] - checks[c]['g0_summary_mean_total']) < 0.05, c
        assert ok[c] == 40 and ok0[c] == 40, c
    model_bytes = {c: S['configs'][c]['model_blob_bytes'] for c in T}
    hdr = {c: S['configs'][c]['mean_parts']['header'] for c in T if 'mean_parts' in S['configs'][c]}
    return {'T': T, 'U': Uu, 'T0': T0, 'U0': U0, 'idx': idx, 'layers': layers, 'checks': checks,
            'model_bytes': model_bytes, 'record_header_bytes': hdr, 'summary': S, 'summary_g0': S0}


# formats: key -> (source, config, description)
FORMATS = [
    ('WEX1', 'fixed', None, 'raw WEX1 record (ratio 1)'),
    ('WNR2', 'fixed', None, 'production WNR2, 16,914,680 B mean over all 7680 records (measured)'),
    ('F', 'A', 'f', 'WNR2 + ceil-form only (f)'),
    ('WNR3', 'A', 'fgde', 'WNR3 byte-exact = f+g+d+e (g = densest-pair anchor as defined by team 3)'),
    ('WNR3t', 'A', 'fgdt', 'WNR3 byte-exact with tail seeding instead of the hash = f+g+d+t'),
    ('WNR3z', 'A', 'fgcde', 'WNR3z value-exact = f+g+c+d+e'),
    ('WNR3_g0', 'G0', 'fgde', 'as WNR3 but constant srel anchor 120 (g0)'),
    ('WNR3t_g0', 'G0', 'fgdt', 'as WNR3t but g0'),
    ('WNR3z_g0', 'G0', 'fgcde', 'as WNR3z but g0'),
    ('WNR3zt_est', 'EST', None, 'ESTIMATE (not encoded): fgcd + per-expert tail-seeding increment (fgdt - fgd)'),
    ('T3_WNR3', 'T3', 'WNR3', "team 3's entropy-based WNR3 estimate (a+b+d+e), for comparison"),
    ('T3_WNR3z', 'T3', 'WNR3z', "team 3's entropy-based WNR3z estimate (a+b+c+d+e), for comparison"),
]
FKEYS = [f[0] for f in FORMATS]
CEIL_FORMATS = ['F', 'WNR3', 'WNR3t', 'WNR3z', 'WNR3_g0', 'WNR3t_g0', 'WNR3z_g0', 'WNR3zt_est']


def per_expert_totals(cr):
    """Per-expert padded record bytes for every format that is backed by ceil-ref records (40 experts)."""
    T, T0 = cr['T'], cr['T0']
    pe = {'WNR2_ref': T['wnr2']}
    for k, src, c, _ in FORMATS:
        if src == 'A':
            pe[k] = T[c]
        elif src == 'G0':
            pe[k] = T0[c]
        elif src == 'EST':
            # c + tail seeding was not encoded; add the measured per-expert tail increment (fgdt - fgd), then re-pad
            used = cr['U']['fgcd'] + (cr['U']['fgdt'] - cr['U']['fgd'])
            pe[k] = np.array([align4096(u) for u in used], float)
    return pe


def ratios_from(pe, sel=None):
    """Ratio = (production WNR2 mean + paired mean delta over the selected experts) / raw."""
    R3 = cvm.format_ratios()
    out = {}
    base = pe['WNR2_ref'] if sel is None else pe['WNR2_ref'][sel]
    for k, src, c, _ in FORMATS:
        if k == 'WEX1':
            out[k] = 1.0
        elif k == 'WNR2':
            out[k] = REF['WNR2_mean_record'] / RAW
        elif src == 'T3':
            out[k] = R3[c]
        else:
            a = pe[k] if sel is None else pe[k][sel]
            d = (a - base).mean(axis=-1)
            out[k] = (REF['WNR2_mean_record'] + d) / RAW
    return out


def boot_ratios(pe, D, rng):
    idx = rng.integers(0, 40, size=(D, 40))
    out = {}
    for k, src, c, _ in FORMATS:
        if src in ('A', 'G0', 'EST'):
            d = (pe[k][idx] - pe['WNR2_ref'][idx]).mean(-1)
            out[k] = (REF['WNR2_mean_record'] + d) / RAW
    return out, idx


# ---------------------------------------------------------------- 2. simulate2 = team 3's simulate with per-draw ratio
def capacities2(P, geo, r1, raw_fmt, cv, x, slot_ov=None):
    CV0 = P['CV0']
    ov = 1.0 if raw_fmt else (P['slot_ov'] if slot_ov is None else slot_ov)
    R_dec = P['R_tot'] * (1 - P['crc_share'])
    R_crc = P['R_tot'] * P['crc_share']
    C_ram = R_dec * GIB / geo.raw + R_crc * GIB / (geo.raw * r1 * ov)
    if cv == 'none' or raw_fmt:
        C_dec = CV0.copy()
        C_V = CV0.copy()
    elif cv == 'incl':
        C_dec = np.full_like(CV0, float(x))
        C_V = (CV0 - x) / (r1 * ov)
    elif cv == 'excl':
        C_dec = x * CV0
        C_V = C_dec + np.maximum(0.0, (1 - x) * CV0 - KS) / (r1 * ov)
    else:
        raise ValueError(cv)
    return C_dec, C_V, C_V + C_ram


def simulate2(P, geo, r, cv='none', x=None, drive='U', w2=True, U=None, sD=0.2, cv_ahead=False, dk=0.0,
              slot_ov=None):
    """Line-by-line copy of capacity_value.simulate (team 3) except: r may be an array (one ratio per draw);
    dk = relative decode-kernel slowdown (applied to cvtp, cv1, csingle; dk*cvtp is added to cdec_x, i.e. assumed
    NOT hidden under PCIe); slot_ov overrides the drawn fixed-slot overhead."""
    Usel, Upf, Udrv, Z = U
    D = len(P['T0'])
    r1 = np.broadcast_to(np.asarray(r, float), (D,)).copy()
    raw_fmt = bool(np.all(r1 >= 1))
    if not raw_fmt:
        assert bool(np.all(r1 < 1))
    if dk:
        P = dict(P)
        P['cdec_x'] = P['cdec_x'] + dk * P['cvtp']
        P['cvtp'] = P['cvtp'] * (1 + dk)
        P['cv1'] = P['cv1'] * (1 + dk)
        P['csingle'] = P['csingle'] * (1 + dk)
    C_dec, C_V, C_tot = capacities2(P, geo, r1, raw_fmt, cv, x, slot_ov)
    md0, mv0, mt0 = cvm.miss_curve(C_dec, P, geo), cvm.miss_curve(C_V, P, geo), cvm.miss_curve(C_tot, P, geo)
    s = P['sigma'][:, None]
    a = np.exp(s * Z - 0.5 * s * s)
    md = np.minimum(1.0, a * md0[:, None])
    mv = np.minimum(md, a * mv0[:, None])
    mt = np.minimum(mv, a * mt0[:, None])
    t1, t2, t3 = [(1 - m)[:, None, :, None] for m in (md, mv, mt)]
    cv_m = (Usel >= t1) & (Usel < t2)
    ram_m = (Usel >= t2) & (Usel < t3)
    dsk_m = Usel >= t3
    unc = Upf >= P['q'][:, None, None, None]
    k_cv = cv_m.sum(-1)
    k_ram = ram_m.sum(-1)
    k_dsk = dsk_m.sum(-1)
    k_ram_u = (ram_m & unc).sum(-1)
    k_dsk_u = (dsk_m & unc).sum(-1)

    def p3(name):
        return P[name][:, None, None]
    rb1 = r1 * geo.raw
    rb2 = rb1[:, None]
    rb3 = rb1[:, None, None]
    r3 = r1[:, None, None]
    if raw_fmt:
        cx = np.full(D, geo.raw / PCIE)[:, None, None]
        deck = 0.0
    elif w2:
        cx = (rb1 / PCIE + P['cdec_x'])[:, None, None]
        deck = p3('cvtp')
    else:
        cx = p3('csingle')
        deck = p3('cvtp')
    bws = p3('bw_scale')

    def disk_t(k, drv):
        bw, qd1 = DRIVES[drv]
        lat = (qd1 - geo.raw / bw) + (geo.raw / bw) * r3
        return np.where(k > 0, np.maximum(lat, k * rb3 / (bw * bws)), 0.0)

    w = P['w_extra'][:, None]
    if drive == 'U+D':
        onD = Udrv < sD
        kD_u = (dsk_m & unc & onD).sum(-1)
        kU_u = k_dsk_u - kD_u
        tdisk = np.maximum(disk_t(kU_u, 'U'), disk_t(kD_u, 'D'))
        kD = (dsk_m & onD).sum(-1).sum(-1)
        kU = k_dsk.sum(-1) - kD
        busy_disk = np.maximum(kU * (1 + w) * rb2 / (DRIVES['U'][0] * P['bw_scale'][:, None]),
                               kD * (1 + w) * rb2 / (DRIVES['D'][0] * P['bw_scale'][:, None]))
    else:
        tdisk = disk_t(k_dsk_u, drive)
        busy_disk = k_dsk.sum(-1) * (1 + w) * rb2 / (DRIVES[drive][0] * P['bw_scale'][:, None])
    tdisk0 = np.where(k_dsk_u > 0, tdisk + cx, 0.0)
    tcopy = (k_ram_u + k_dsk_u) * cx
    k_cv_lat = (cv_m & unc).sum(-1) if cv_ahead else k_cv
    tcv = np.where(k_cv_lat > 0, p3('cv1') + (k_cv_lat - 1) * p3('cvtp'), 0.0)
    fix = p3('cfix') * (k_ram_u + k_dsk_u)
    cont = p3('kappa') * (k_cv * p3('cvtp') + (k_ram + k_dsk * (1 + P['w_extra'][:, None, None])) * deck)
    base = p3('T0') / geo.L + fix + cont
    reads = k_dsk.sum(-1) * (1 + w)
    busy_copy = (k_ram.sum(-1) + reads) * cx[:, :, 0]
    bw_main = DRIVES['U' if drive == 'U+D' else drive][0] * P['bw_scale']
    svc = rb1 / bw_main
    rho = np.zeros(len(svc))
    for _ in range(14):
        Wq = P['qmix'] * np.minimum(QCAP, rho / (2 * (1 - rho))) * svc
        tdisk = np.where(k_dsk_u > 0, tdisk0 + Wq[:, None, None], 0.0)
        wait = np.maximum(np.maximum(tdisk, tcopy), tcv)
        crit = (base + wait).sum(-1)
        tt = np.maximum(np.maximum(crit, busy_disk), busy_copy)
        rho = 0.5 * rho + 0.5 * np.minimum(0.95, busy_disk.mean(1) / tt.mean(1))
    return {
        't': tt.mean(1),
        'crit': crit.mean(1), 'busy_disk': busy_disk.mean(1), 'busy_copy': busy_copy.mean(1),
        'disk_bound': (busy_disk >= np.maximum(crit, busy_copy)).mean(1),
        'n_vdec': (geo.L * geo.topk - k_cv.sum(-1) - k_ram.sum(-1) - k_dsk.sum(-1)).mean(1),
        'n_cv': k_cv.sum(-1).mean(1), 'n_ram': k_ram.sum(-1).mean(1),
        'n_disk_demand': k_dsk.sum(-1).mean(1), 'reads': reads.mean(1),
        'GB_per_token': reads.mean(1) * rb1 / 1e9,
        'C_dec': C_dec, 'C_V': C_V, 'C_tot': C_tot,
        'wait_ms': wait.sum(-1).mean(1) * 1e3, 'rho': rho,
    }


def equivalence_check(Pm, Um):
    """simulate2 with a scalar ratio must equal team 3's simulate exactly."""
    out = {}
    sub = 60
    Ps = {k: v[:sub] for k, v in Pm.items()}
    Us = tuple(u[:sub] for u in Um)
    cases = [(1.0, 'none', None, 'U', True), (cvm.format_ratios()['WNR2'], 'none', None, 'U', True),
             (0.88, 'none', None, 'E', True), (0.88, 'none', None, 'U', False), (0.85, 'none', None, 'E', False),
             (0.88, 'incl', 16, 'E', True), (0.88, 'excl', 0.5, 'U', True), (1.0, 'none', None, 'U+D', True),
             (0.88, 'none', None, 'U+D', True)]
    worst = 0.0
    for r, cv, x, drv, w2 in cases:
        a = cvm.simulate(Ps, GEO, r, cv, x, drv, w2, Us)
        b = simulate2(Ps, GEO, r, cv, x, drv, w2, Us)
        dmax = max(float(np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))) for k in a)
        worst = max(worst, dmax)
        out[f'r={r:.4f} cv={cv}/{x} drive={drv} w2={w2}'] = dmax
    out['max_abs_diff_all_outputs'] = worst
    out['identical'] = worst == 0.0
    return out


# ---------------------------------------------------------------- 3. statistics helpers
def q3(a):
    return cvm.q3(a)


def gain(b, n):
    return (b / n - 1.0) * 100.0


def summ(v):
    v = np.asarray(v, float)
    return {'p5_p50_p95': q3(v), 'P(>=+3%)': float((v >= 3.0).mean()), 'P(>0)': float((v > 0).mean()),
            'mean': float(v.mean())}


def run_mc(Pm, Um, rb, label, fkeys, modes=('W2',), dk=0.0, slot_fmt=None, fixed_r=None):
    """One MC block: t per (drive, format, mode). rb = per-draw ratio arrays for bootstrapped formats."""
    Rfix = fixed_r
    res = {}
    for drive in ('U', 'E'):
        for f in fkeys:
            r = rb[f] if (rb is not None and f in rb) else Rfix[f]
            for m in modes:
                if f == 'WEX1' and m != 'W2':
                    continue
                dk_f = dk if f in CEIL_FORMATS else 0.0
                so = None if slot_fmt is None else slot_fmt.get(f)
                res[(drive, f, m)] = simulate2(Pm, GEO, r, 'none', None, drive, m == 'W2', Um, dk=dk_f, slot_ov=so)
    return res


def comparisons(res, rb, Rfix, drive, mode='W2', fkeys=None):
    out = {}
    t = lambda f, m=mode: res[(drive, f, 'W2' if f == 'WEX1' else m)]['t']
    r = lambda f: (rb[f] if (rb is not None and f in rb) else np.full(len(t('WEX1')), Rfix[f]))
    for f in fkeys:
        if f == 'WEX1':
            continue
        g1 = gain(t('WEX1'), t(f))
        e1 = {'gain_vs_WEX1_%': summ(g1)}
        b1 = (1.0 / r(f) - 1) * 100
        e1['bytes_vs_WEX1_%(fewer)'] = q3(b1)
        e1['elasticity_vs_WEX1'] = q3(g1 / b1)
        if f != 'WNR2':
            g2 = gain(t('WNR2'), t(f))
            b2 = (r('WNR2') / r(f) - 1) * 100
            e1['gain_vs_WNR2_%'] = summ(g2)
            e1['bytes_vs_WNR2_%(fewer)'] = q3(b2)
            e1['elasticity_vs_WNR2'] = q3(g2 / b2)
        tw = t(f)
        e1['tok_s'] = q3(1.0 / tw)
        e1['ms_token'] = q3(tw * 1e3)
        e1['GB_token'] = q3(res[(drive, f, 'W2' if f == 'WEX1' else mode)]['GB_per_token'])
        e1['reads_token'] = q3(res[(drive, f, 'W2' if f == 'WEX1' else mode)]['reads'])
        out[f] = e1
    out['WEX1'] = {'tok_s': q3(1.0 / t('WEX1')), 'ms_token': q3(t('WEX1') * 1e3),
                   'disk_busy_share': q3(res[(drive, 'WEX1', 'W2')]['busy_disk'] / t('WEX1')),
                   'GB_token': q3(res[(drive, 'WEX1', 'W2')]['GB_per_token']),
                   'reads_token': q3(res[(drive, 'WEX1', 'W2')]['reads'])}
    pairs = [('WNR3t', 'WNR3'), ('WNR3z', 'WNR3'), ('WNR3z', 'WNR3t'), ('WNR3_g0', 'WNR3'), ('WNR3', 'F'),
             ('WNR3', 'T3_WNR3'), ('WNR3z', 'T3_WNR3z')]
    for a, b in pairs:
        if a in fkeys and b in fkeys:
            out[f'{a} vs {b}'] = summ(gain(t(b), t(a)))
    return out


# ---------------------------------------------------------------- 4. containers
def containers(cr, pe, Rmean):
    out = {}
    tbl = align4096(REF['entry_bytes'] * N_REC)
    # self-checks against the reference
    wex1 = 4096 + tbl + N_REC * RAW
    wnr2 = 4096 + align4096(REF['WNM2_bytes']) + tbl + N_REC * REF['WNR2_mean_record']
    out['checks'] = {
        'entry_table_bytes': tbl,
        'WEX1 formula == ref': [wex1, REF['WEX1_container_bytes'], wex1 == REF['WEX1_container_bytes']],
        'WNR2 formula == ref': [wnr2, REF['WNR2_container_bytes'], wnr2 == REF['WNR2_container_bytes']],
        'WNR2 records start == ref': [4096 + align4096(REF['WNM2_bytes']) + tbl, REF['WNR2_records_start']],
        'WNR2 mean record from ref container': (REF['WNR2_container_bytes'] - REF['WNR2_records_start']) / N_REC,
    }
    rng = np.random.default_rng(20261107)
    bidx = rng.integers(0, 40, size=(10000, 40))
    model_for = {'F': 'f', 'WNR3': 'fgde', 'WNR3t': 'fgdt', 'WNR3z': 'fgcde', 'WNR3_g0': 'fgde', 'WNR3t_g0': 'fgdt',
                 'WNR3z_g0': 'fgcde', 'WNR3zt_est': 'fgcde'}
    free = REF['free_GB_1711']
    for k, src, c, desc in FORMATS:
        if src == 'T3':
            mean_rec = Rmean[k] * RAW
            ci = None
            mb = 59_708
        elif k == 'WEX1':
            mean_rec, ci, mb = RAW, None, 0
        elif k == 'WNR2':
            mean_rec, ci, mb = REF['WNR2_mean_record'], None, REF['WNM2_bytes']
        else:
            d = pe[k] - pe['WNR2_ref']
            mean_rec = REF['WNR2_mean_record'] + d.mean()
            bd = d[bidx].mean(-1)
            ci = [REF['WNR2_mean_record'] + float(np.percentile(bd, 2.5)),
                  REF['WNR2_mean_record'] + float(np.percentile(bd, 97.5))]
            mb = cr['model_bytes'][model_for[k]]
        toff = 4096 + (align4096(mb) if mb else 0)
        overhead = toff + tbl
        size = overhead + N_REC * mean_rec
        e = {'description': desc, 'mean_record_bytes': float(mean_rec), 'ratio': float(mean_rec / RAW),
             'model_bytes': mb, 'table_offset': toff, 'records_start': overhead,
             'container_bytes': float(size), 'container_GB': size / 1e9, 'container_GiB': size / GIB}
        if ci is not None:
            e['mean_record_CI95'] = ci
            e['container_GB_CI95'] = [(overhead + N_REC * ci[0]) / 1e9, (overhead + N_REC * ci[1]) / 1e9]
        if k in CEIL_FORMATS:
            # WNR3-DRAFT.md:52 uses a 64-B record header; ceil-ref's records carry 48 B -> +16 B used per record,
            # i.e. +16 B padded on average (used mod 4096 ~ uniform)  [inferred]
            e['container_GB_with_64B_header'] = (size + 16 * N_REC) / 1e9
            # WNR3-DRAFT.md:166-168: compacted layout-0 K=1 WNM3 with LENPRED = 11,596 B -> table_offset 16,384
            e['container_GB_with_draft_layout0_model'] = (size - toff + 16_384) / 1e9
        fr = {}
        for drv, gb in free.items():
            fr[drv] = {'free_GB_1711': gb, 'margin_GB (free - size)': gb - size / 1e9,
                       'margin_GB with 5 GiB reserve': gb - size / 1e9 - REF['min_free_gib_W1'] * GIB / 1e9,
                       'fits': bool(gb * 1e9 >= size)}
        e['vs_free_space'] = fr
        out[k] = e
    up = REF['builder_upfront_factor'] * REF['WEX1_container_bytes']
    out['builder_upfront_check'] = {
        'rule (ref:300)': '0.90 x source size + margin (build_nr2_full.py)',
        'source WEX1 bytes': REF['WEX1_container_bytes'], '0.90 x source GB': up / 1e9,
        'U: free GB at 17:11': free['U'], 'passes on U: (before margin)': bool(free['U'] * 1e9 >= up),
        'note': 'applies to the WNR2 builder; a WNR3 builder with the same rule would refuse U: for every format '
                '[inferred]; copying a finished container needs only its size'}
    out['note_drive_letters'] = ('U:, E:, D: and their free space are the KAIRA machine (ref section 2, M0 probe '
                                 '2026-11-07). This PC is a different machine: its U: has ~607.9 GB and E: ~1102.9 GB '
                                 'free (Get-PSDrive here), so nothing about free space was measured on the KAIRA drives.')
    return out


# ---------------------------------------------------------------- 5. main
def run():
    t0 = time.time()
    res = {'assumptions': {
        'G2': 'BUILT (V4.1 MoE on the GPU pipeline: VRAM mirrors, prefetch, copy worker, RAM record cache) -- '
              'team 3 assumption; not in KAIRA src (ref:123-126)',
        'W2': 'BUILT (per-layer batched WNR decode) -- team 3 assumption; NOT in src (ref:127-135). '
              'No-W2 rows = sensitivity',
        'decode_cost': 'ceil-form formats use WNR2 decode costs (dk = 0) in the primary runs; ceil-gpu measured '
                       'CF5/CF0 median +0.3..+1.7%, max +3.4% (RTX 3070, deferred patch) -> sensitivity dk = 0.034, '
                       'and 0.157 = naive per-block byte store (CF/CF0 m3 median)',
        'record_sizes': 'ceil-ref real rANS records, 40 odd-indexed held-out experts (one per layer), models from the '
                        '80 even experts; applied as a paired byte delta to production WNR2 16,914,680 B; per MC draw a '
                        'bootstrap resample of the 40 experts',
        'value_exact': 'WNR3z (canonical zero) is value-exact, not byte-exact: needs the logprob check (brief:13, 26)',
        'other': 'all other parameters, ranges, seeds (MC rng 7, D=400, T=40; center rng 20261107, 3000 tokens) are '
                 "team 3's, unchanged",
    }}
    cr = load_ceilref()
    pe = per_expert_totals(cr)
    Rmean = ratios_from(pe)
    res['ceilref_checks'] = cr['checks']
    res['ceilref_record_header_bytes'] = cr['record_header_bytes']
    res['ratios_mean'] = Rmean
    res['formats'] = {k: {'source': s, 'ceilref_config': c, 'description': d} for k, s, c, d in FORMATS}
    res['per_expert_delta_vs_wnr2_bytes'] = {
        k: {'mean': float((pe[k] - pe['WNR2_ref']).mean()), 'min': float((pe[k] - pe['WNR2_ref']).min()),
            'max': float((pe[k] - pe['WNR2_ref']).max()),
            'pct_of_raw_mean': float((pe[k] - pe['WNR2_ref']).mean() / RAW * 100)}
        for k in pe if k != 'WNR2_ref'}
    mx = {k: float(pe[k].max() / pe[k].mean()) for k in pe}
    res['max_over_mean_record_40experts'] = mx

    # MC draws: identical to team 3 (rng 7)
    D, T = 400, 40
    rng = np.random.default_rng(7)
    Pm = cvm.sample_params(D, rng)
    Um = cvm.make_uniforms(rng, D, T, GEO.L, GEO.topk)
    res['equivalence_simulate2_vs_team3'] = equivalence_check(Pm, Um)

    rngb = np.random.default_rng(20261108)
    rb, _ = boot_ratios(pe, D, rngb)
    res['ratios_boot_p5_p50_p95'] = {k: q3(v) for k, v in rb.items()}

    main_keys = FKEYS
    mc = run_mc(Pm, Um, rb, 'main', main_keys, modes=('W2', 'single'), fixed_r=Rmean)
    comp = {d: comparisons(mc, rb, Rmean, d, 'W2', main_keys) for d in ('U', 'E')}
    comp_single = {d: comparisons(mc, rb, Rmean, d, 'single', main_keys) for d in ('U', 'E')}
    res['mc'] = {'setup': {'draws': D, 'tokens_per_draw': T, 'param_rng': 7, 'boot_rng': 20261108,
                           'note': 'paired: all formats share the parameter draws, the token uniforms and the expert '
                                   'bootstrap'},
                 'W2': comp, 'noW2_single_helper': comp_single}

    # reproduction of team 3's numbers (same seeds; WEX1/WNR2/T3_* are fixed ratios)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'team3-capacity-value', 'capacity_value_results.json')) as fh:
        t3 = json.load(fh)
    rep = {}
    for d in ('U', 'E'):
        rep[d] = {
            'WNR2 vs WEX1 mine': comp[d]['WNR2']['gain_vs_WEX1_%']['p5_p50_p95'],
            'WNR2 vs WEX1 team3': t3['mc'][d]['WNR2 vs WEX1']['p5_p50_p95'],
            'WNR3z(T3) vs WNR2 mine': comp[d]['T3_WNR3z']['gain_vs_WNR2_%']['p5_p50_p95'],
            'WNR3z vs WNR2 team3': t3['mc'][d]['WNR3z vs WNR2']['p5_p50_p95'],
            'WNR3(T3) vs WNR2 mine': comp[d]['T3_WNR3']['gain_vs_WNR2_%']['p5_p50_p95'],
            'WNR3 vs WNR2 team3': t3['mc'][d]['WNR3 vs WNR2']['p5_p50_p95'],
            'elasticity WNR2->WNR3z(T3) mine': comp[d]['T3_WNR3z']['elasticity_vs_WNR2'],
            'elasticity team3': t3['mc'][d]['elasticity tok/s % per bytes % (WNR2->WNR3z)']['p5_p50_p95'],
        }
        rep[d]['identical'] = all(np.allclose(rep[d][a], rep[d][b], rtol=0, atol=1e-9) for a, b in
                                  (('WNR2 vs WEX1 mine', 'WNR2 vs WEX1 team3'),
                                   ('WNR3z(T3) vs WNR2 mine', 'WNR3z vs WNR2 team3'),
                                   ('WNR3(T3) vs WNR2 mine', 'WNR3 vs WNR2 team3'),
                                   ('elasticity WNR2->WNR3z(T3) mine', 'elasticity team3')))
    res['reproduction_team3'] = rep

    # sensitivities on the same draws
    sens = {}
    key4 = ['WEX1', 'WNR2', 'WNR3', 'WNR3t', 'WNR3z']
    # (i) fixed (mean) ratios, no record bootstrap
    m_fix = run_mc(Pm, Um, None, 'fixed', key4, fixed_r=Rmean)
    sens['fixed_mean_ratios'] = {d: comparisons(m_fix, None, Rmean, d, 'W2', key4) for d in ('U', 'E')}
    # (ii) ceil-form decode penalty
    for dk in (0.034, 0.157):
        m_dk = run_mc(Pm, Um, rb, f'dk{dk}', key4, dk=dk, fixed_r=Rmean)
        sens[f'ceil_decode_dk={dk}'] = {d: comparisons(m_dk, rb, Rmean, d, 'W2', key4) for d in ('U', 'E')}
    # (iii) format-specific fixed-slot overhead = measured max/mean over the 40 test experts (fixed, not drawn)
    slot_fmt = {'WNR2': mx['WNR2_ref'], 'WNR3': mx['WNR3'], 'WNR3t': mx['WNR3t'], 'WNR3z': mx['WNR3z']}
    m_so = run_mc(Pm, Um, rb, 'slot', key4, slot_fmt=slot_fmt, fixed_r=Rmean)
    sens['slot_ov_measured_max_over_mean'] = {'slot_ov': slot_fmt,
                                              **{d: comparisons(m_so, rb, Rmean, d, 'W2', key4) for d in ('U', 'E')}}
    slot_g0 = {'WNR2': mx['WNR2_ref'], 'WNR3': mx['WNR3_g0'], 'WNR3t': mx['WNR3t_g0'], 'WNR3z': mx['WNR3z_g0']}
    rb_g0 = {'WNR3': rb['WNR3_g0'], 'WNR3t': rb['WNR3t_g0'], 'WNR3z': rb['WNR3z_g0']}
    Rg0 = dict(Rmean, WNR3=Rmean['WNR3_g0'], WNR3t=Rmean['WNR3t_g0'], WNR3z=Rmean['WNR3z_g0'])
    m_so0 = run_mc(Pm, Um, rb_g0, 'slot_g0', key4, slot_fmt=slot_g0, fixed_r=Rg0)
    sens['g0_with_slot_ov_measured'] = {'slot_ov': slot_g0,
                                        **{d: comparisons(m_so0, rb_g0, Rg0, d, 'W2', key4) for d in ('U', 'E')}}
    # (iv) two extra seeds (new parameter draws, uniforms and bootstrap), pooled with the main run -> 1200 draws
    pooled = {d: {} for d in ('U', 'E')}
    chunks = [(mc, rb, Pm)]
    for seed in (11, 12):
        rg = np.random.default_rng(seed)
        P2 = cvm.sample_params(D, rg)
        U2 = cvm.make_uniforms(rg, D, T, GEO.L, GEO.topk)
        rb2, _ = boot_ratios(pe, D, np.random.default_rng(seed + 1000))
        m2 = run_mc(P2, U2, rb2, f'seed{seed}', key4, fixed_r=Rmean)
        chunks.append((m2, rb2, P2))
        del U2
    for d in ('U', 'E'):
        tt = {f: np.concatenate([c[0][(d, f, 'W2')]['t'] for c in chunks]) for f in key4}
        rr = {f: np.concatenate([c[1][f] for c in chunks]) for f in ('WNR3', 'WNR3t', 'WNR3z')}
        rr['WNR2'] = np.full(len(tt['WEX1']), Rmean['WNR2'])
        for f in key4[1:]:
            e = {'gain_vs_WEX1_%': summ(gain(tt['WEX1'], tt[f]))}
            if f != 'WNR2':
                g2 = gain(tt['WNR2'], tt[f])
                e['gain_vs_WNR2_%'] = summ(g2)
                e['elasticity_vs_WNR2'] = q3(g2 / ((rr['WNR2'] / rr[f] - 1) * 100))
            e['tok_s'] = q3(1.0 / tt[f])
            pooled[d][f] = e
        pooled[d]['WEX1'] = {'tok_s': q3(1.0 / tt['WEX1'])}
        # per-seed medians (convergence)
        pooled[d]['per_seed_median_gain_vs_WNR2'] = {
            f: [float(np.median(gain(c[0][(d, 'WNR2', 'W2')]['t'], c[0][(d, f, 'W2')]['t']))) for c in chunks]
            for f in ('WNR3', 'WNR3t', 'WNR3z')}
        pooled[d]['per_seed_median_gain_WNR2_vs_WEX1'] = [
            float(np.median(gain(c[0][(d, 'WEX1', 'W2')]['t'], c[0][(d, 'WNR2', 'W2')]['t']))) for c in chunks]
    sens['pooled_3_seeds_1200_draws'] = pooled
    res['sensitivities'] = sens

    # rank correlations of parameters with the WNR3z-vs-WNR2 and WNR3-vs-WNR2 gains (main run)
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    dom = {}
    for d in ('U', 'E'):
        dom[d] = {}
        for f in ('WNR3', 'WNR3z'):
            g = gain(mc[(d, 'WNR2', 'W2')]['t'], mc[(d, f, 'W2')]['t'])
            cor = {n: round(spearman(Pm[n], g), 3) for n in cvm.PNAMES}
            cor['record_ratio(bootstrap)'] = round(spearman(rb[f], -g), 3)
            dom[d][f'{f} vs WNR2'] = dict(sorted(cor.items(), key=lambda kv: -abs(kv[1]))[:7])
    res['rank_correlations'] = dom

    # decision aid: gain vs WEX1 disk-busy share (U and E draws pooled), main run
    share = np.concatenate([mc[(d, 'WEX1', 'W2')]['busy_disk'] / mc[(d, 'WEX1', 'W2')]['t'] for d in ('U', 'E')])
    gg = {f: np.concatenate([gain(mc[(d, 'WNR2', 'W2')]['t'], mc[(d, f, 'W2')]['t']) for d in ('U', 'E')])
          for f in ('WNR3', 'WNR3t', 'WNR3z')}
    gw = {f: np.concatenate([gain(mc[(d, 'WEX1', 'W2')]['t'], mc[(d, f, 'W2')]['t']) for d in ('U', 'E')])
          for f in ('WNR2', 'WNR3', 'WNR3t', 'WNR3z')}
    aid = {}
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (share >= lo) & (share < hi)
        if sel.sum() < 5:
            continue
        aid[f'{lo:.1f}-{min(hi, 1.0):.1f}'] = {
            'n': int(sel.sum()),
            **{f'{f} vs WEX1 p50': float(np.median(gw[f][sel])) for f in gw},
            **{f'P({f} vs WEX1 >=3%)': float((gw[f][sel] >= 3).mean()) for f in gw},
            **{f'{f} vs WNR2 p50': float(np.median(gg[f][sel])) for f in gg},
            **{f'P({f} vs WNR2 >=3%)': float((gg[f][sel] >= 3).mean()) for f in gg}}
    res['decision_aid_by_WEX1_disk_busy_share'] = aid

    # center scenario (team 3: all params at center, 3000 tokens, rng 20261107)
    rngc = np.random.default_rng(20261107)
    Uc = cvm.make_uniforms(rngc, 1, 3000, GEO.L, GEO.topk)
    Pc = cvm.center_params()
    cen = {}
    for d in ('U', 'E'):
        base_w = simulate2(Pc, GEO, 1.0, 'none', None, d, True, Uc)['t'][0]
        base_2 = simulate2(Pc, GEO, Rmean['WNR2'], 'none', None, d, True, Uc)['t'][0]
        for f in FKEYS:
            q = simulate2(Pc, GEO, Rmean[f], 'none', None, d, True, Uc)
            cen[f'{d}|{f}|W2'] = {'tok_s': float(1 / q['t'][0]), 'ms_token': float(q['t'][0] * 1e3),
                                  'vs_WEX1_%': float(gain(base_w, q['t'][0])),
                                  'vs_WNR2_%': float(gain(base_2, q['t'][0])),
                                  'reads': float(q['reads'][0]), 'GB_token': float(q['GB_per_token'][0]),
                                  'C_tot': float(q['C_tot'][0]), 'disk_bound': float(q['disk_bound'][0])}
    res['center'] = cen

    # simple linear cross-check: team 3's median elasticity x new byte deltas
    lin = {}
    e3 = {'U': t3['mc']['U']['elasticity tok/s % per bytes % (WNR2->WNR3z)']['p5_p50_p95'][1],
          'E': t3['mc']['E']['elasticity tok/s % per bytes % (WNR2->WNR3z)']['p5_p50_p95'][1]}
    for d in ('U', 'E'):
        lin[d] = {f: {'bytes_vs_WNR2_%': (Rmean['WNR2'] / Rmean[f] - 1) * 100,
                      'linear_pred_gain_%': e3[d] * (Rmean['WNR2'] / Rmean[f] - 1) * 100,
                      'simulated_p50_%': comp[d][f]['gain_vs_WNR2_%']['p5_p50_p95'][1]}
                  for f in ('WNR3', 'WNR3t', 'WNR3z')}
        lin[d]['team3_elasticity_p50'] = e3[d]
    res['linear_elasticity_crosscheck'] = lin

    res['containers'] = containers(cr, pe, Rmean)
    res['runtime_s'] = time.time() - t0
    return res


def fmt(res):
    L = []
    L.append('ASSUMPTIONS: ' + json.dumps(res['assumptions'], indent=1))
    L.append('\nsimulate2 == team 3 simulate (scalar r): ' + str(res['equivalence_simulate2_vs_team3']['identical'])
             + f"  max abs diff {res['equivalence_simulate2_vs_team3']['max_abs_diff_all_outputs']}")
    L.append('reproduction of team 3 MC numbers: ' + ', '.join(f"{d} {res['reproduction_team3'][d]['identical']}"
                                                              for d in ('U', 'E')))
    L.append('\n== record ratios (mean; production WNR2 + paired ceil-ref delta) and bootstrap p5/p50/p95 ==')
    for k, v in res['ratios_mean'].items():
        b = res['ratios_boot_p5_p50_p95'].get(k)
        L.append(f'  {k:12s} {v:.5f}' + (f'   boot {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}' if b else ''))
    L.append('  max/mean record over the 40 test experts: ' +
             ', '.join(f'{k} {v:.4f}' for k, v in res['max_over_mean_record_40experts'].items()))
    for mode in ('W2', 'noW2_single_helper'):
        L.append(f'\n== Monte Carlo ({mode}), % tok/s: p5 / p50 / p95 | P(>=+3%) | P(>0) ==')
        for d in ('U', 'E'):
            c = res['mc'][mode][d]
            w = c['WEX1']
            L.append(f'-- drive {d}: WEX1 tok/s {w["tok_s"][0]:.2f} {w["tok_s"][1]:.2f} {w["tok_s"][2]:.2f}'
                     + (f'  disk-busy share {w["disk_busy_share"][1]:.2f}' if 'disk_busy_share' in w else ''))
            for f in FKEYS[1:]:
                e = c[f]
                a = e['gain_vs_WEX1_%']
                s = (f'   {f:11s} tok/s {e["tok_s"][1]:5.3f} | vs WEX1 {a["p5_p50_p95"][0]:6.2f} {a["p5_p50_p95"][1]:6.2f} '
                     f'{a["p5_p50_p95"][2]:6.2f} P3 {a["P(>=+3%)"]:.3f}')
                if 'gain_vs_WNR2_%' in e:
                    b = e['gain_vs_WNR2_%']
                    s += (f' | vs WNR2 {b["p5_p50_p95"][0]:5.2f} {b["p5_p50_p95"][1]:5.2f} {b["p5_p50_p95"][2]:5.2f} '
                          f'P3 {b["P(>=+3%)"]:.3f} P0 {b["P(>0)"]:.3f} | elast {e["elasticity_vs_WNR2"][1]:.3f} '
                          f'({e["elasticity_vs_WNR2"][0]:.3f}-{e["elasticity_vs_WNR2"][2]:.3f})')
                s += f' | elastWEX1 {e["elasticity_vs_WEX1"][1]:.3f}'
                L.append(s)
            for k in c:
                if ' vs ' in k:
                    a = c[k]
                    L.append(f'   {k:22s} {a["p5_p50_p95"][0]:6.2f} {a["p5_p50_p95"][1]:6.2f} {a["p5_p50_p95"][2]:6.2f} '
                             f'P3 {a["P(>=+3%)"]:.3f} P0 {a["P(>0)"]:.3f}')
    L.append('\n== sensitivities (W2): median [p5, p95] gain vs WNR2 and P(>=3%) ==')
    for name, sv in res['sensitivities'].items():
        if name == 'pooled_3_seeds_1200_draws':
            continue
        for d in ('U', 'E'):
            parts = []
            for f in ('WNR3', 'WNR3t', 'WNR3z'):
                b = sv[d][f]['gain_vs_WNR2_%']
                a = sv[d][f]['gain_vs_WEX1_%']
                parts.append(f'{f} {b["p5_p50_p95"][1]:.2f} [{b["p5_p50_p95"][0]:.2f},{b["p5_p50_p95"][2]:.2f}] '
                             f'P3 {b["P(>=+3%)"]:.2f} (vsWEX1 {a["p5_p50_p95"][1]:.2f} P3 {a["P(>=+3%)"]:.2f})')
            w2 = sv[d]['WNR2']['gain_vs_WEX1_%']
            L.append(f'  {name:32s} {d}: WNR2vsWEX1 {w2["p5_p50_p95"][1]:.2f} | ' + ' | '.join(parts))
    pl = res['sensitivities']['pooled_3_seeds_1200_draws']
    for d in ('U', 'E'):
        parts = []
        for f in ('WNR2', 'WNR3', 'WNR3t', 'WNR3z'):
            a = pl[d][f]['gain_vs_WEX1_%']
            s = f'{f} vsWEX1 {a["p5_p50_p95"][1]:.2f} [{a["p5_p50_p95"][0]:.2f},{a["p5_p50_p95"][2]:.2f}] P3 {a["P(>=+3%)"]:.3f}'
            if 'gain_vs_WNR2_%' in pl[d][f]:
                b = pl[d][f]['gain_vs_WNR2_%']
                s += f' vsWNR2 {b["p5_p50_p95"][1]:.2f} [{b["p5_p50_p95"][0]:.2f},{b["p5_p50_p95"][2]:.2f}] P3 {b["P(>=+3%)"]:.3f}'
            parts.append(s)
        L.append(f'  pooled 1200 draws {d}: ' + ' | '.join(parts))
        L.append(f'     per-seed medians vs WNR2: {pl[d]["per_seed_median_gain_vs_WNR2"]}; WNR2 vs WEX1: '
                 f'{pl[d]["per_seed_median_gain_WNR2_vs_WEX1"]}')
    L.append('\n== rank correlations ==')
    for d, v in res['rank_correlations'].items():
        for k, cor in v.items():
            L.append(f'  {d} {k}: ' + ', '.join(f'{n} {c:+.2f}' for n, c in cor.items()))
    L.append('\n== decision aid (gain vs WEX1 disk-busy share; U+E pooled) ==')
    for k, v in res['decision_aid_by_WEX1_disk_busy_share'].items():
        L.append(f'  share {k} n={v["n"]:3d}: ' + ', '.join(f'{kk} {vv:.2f}' for kk, vv in v.items() if kk != 'n'))
    L.append('\n== center scenario (W2) ==')
    for k, v in res['center'].items():
        L.append(f'  {k:18s} tok/s {v["tok_s"]:.3f} vs WEX1 {v["vs_WEX1_%"]:+6.2f}% vs WNR2 {v["vs_WNR2_%"]:+5.2f}% '
                 f'reads {v["reads"]:.1f} GB/t {v["GB_token"]:.3f} C_tot {v["C_tot"]:.0f}')
    L.append('\n== linear elasticity cross-check ==\n' + json.dumps(res['linear_elasticity_crosscheck'], indent=1))
    L.append('\n== containers ==')
    C = res['containers']
    L.append(json.dumps(C['checks'], indent=1))
    for k in FKEYS:
        e = C[k]
        ci = e.get('container_GB_CI95')
        fr = e['vs_free_space']
        L.append(f'  {k:11s} rec {e["mean_record_bytes"]:14,.1f} ratio {e["ratio"]:.5f} container {e["container_bytes"]:19,.0f} B '
                 f'= {e["container_GB"]:8.3f} GB = {e["container_GiB"]:7.2f} GiB'
                 + (f' CI95 {ci[0]:.3f}-{ci[1]:.3f} GB' if ci else '')
                 + f' | U: margin {fr["U"]["margin_GB (free - size)"]:+7.2f} GB (5 GiB reserve {fr["U"]["margin_GB with 5 GiB reserve"]:+7.2f})'
                 + f' | E: {fr["E"]["margin_GB (free - size)"]:+7.2f} | D: {fr["D"]["margin_GB (free - size)"]:+7.2f}')
    L.append(json.dumps(C['builder_upfront_check'], indent=1))
    L.append(C['note_drive_letters'])
    L.append(f'\nruntime {res["runtime_s"]:.1f} s')
    return '\n'.join(L)


if __name__ == '__main__':
    res = run()

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        return str(o)
    with open(os.path.join(HERE, 'capacity_update_results.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=conv)
    rep = fmt(res)
    with open(os.path.join(HERE, 'capacity_update_output.txt'), 'w') as fh:
        fh.write(rep)
    print(rep)
