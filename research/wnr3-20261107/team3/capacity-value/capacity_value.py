#!/usr/bin/env python
"""capacity_value.py -- what each WNR format change is worth in tok/s for DeepSeek V4.1 REAP-192
on the user's 32 GB-RAM + RTX 5090 machine, AFTER G2 (V4.1 MoE on the GPU pipeline).

Inputs are only the WNR2 reference's measurements (sections 2, 3, 10, 11, 12, 14 of
D:\\Desktop\\WNR2-REFERENCE.md) plus the measured format deltas of research\\measured-v41.md.
Every parameter carries its source; ranges are swept (Monte Carlo + one-at-a-time tornado).

Model (seconds unless noted)
  popularity   LRU stack-distance miss curve  m(C) = mA * (C/N / cA)^-beta,  N = 7680 records,
               anchored on M0 (V4.1, 360 records cached -> hit 34.6-37.1 %), slope beta swept.
               Per-layer heterogeneity: layer l uses a_l * m(C), a_l lognormal(sigma), mean 1.
  tiers        exclusive stack: decoded VRAM mirrors -> [compressed VRAM] -> RAM (decoded slots +
               compressed-record cache) -> disk.  Format ratio r shrinks the compressed tiers' slots
               (fixed slot = largest record -> factor slot_ov) and the bytes of every disk read.
  per layer    6 selections ~ categorical over the tiers; a fraction q of VRAM misses was prefetched
               in time (latency hidden, bytes still read).  Layer time =
               T0/L + max(disk wait, copy-engine wait, in-VRAM decode) + attach overhead + GPU contention.
  per token    max(sum of layer times, disk busy time, copy-engine busy time).
"""
import json, math, os, sys, time
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
GIB = float(1 << 30)
MIB = float(1 << 20)
PCIE = 48.6 * GIB          # h2d_bw 2026-11-05, pinned, with sync -> 0.36 ms for an 18.8 MB raw V4.1 record
KS = 16                    # decoded scratch records kept for a compressed VRAM tier (6 experts x 2 layers + slack)
QCAP = 2.0                 # max queued reads ahead of a demand read (M/D/1 term capped; closed system)

DRIVES = {                 # M0 probe 2026-11-07: GB/s random 18.8 MB QD8, QD1 p50 latency for 18.8 MB
    'U': (6.58e9, 3.2e-3),
    'E': (2.29e9, 8.3e-3),
    'D': (1.61e9, 12.2e-3),
}


class Geo:
    def __init__(self, name, N, L, topk, raw, cA, mA_override=None):
        self.name, self.N, self.L, self.topk, self.raw = name, N, L, topk, raw
        self.cA, self.mA_override = cA, mA_override


V41 = Geo('V4.1 REAP-192', 7680, 40, 6, 18_800_640, cA=360 / 7680)
# V4 geometry for the history check: anchor = Oct LRU replay, 70 % VRAM hit at 1530 of 11,008
V4 = Geo('V4-0731', 11008, 43, 6, 13_369_344, cA=1530 / 11008, mA_override=0.30)


def format_ratios():
    """Record bytes / raw bytes for each candidate (measured deltas, held-out)."""
    r2 = 16_914_680 / 18_800_640                        # WNR2 production, measured (0.8997)
    wshare = 0.8899164257825458 / 0.8999418779545942    # weight payload share of a WNR2 record (analysis.json)
    LEN = 0.00148                                       # (d) length table from scales: -0.148 % of RAW record
    INT = 1.00007                                       # (e) per-stream hash in x0: +0.007 % bytes

    def wb(pct):                                        # -pct % of weight bits
        return r2 * (1.0 - wshare * pct / 100.0)
    return {
        'WEX1': 1.0,
        'WNR2': r2,
        'WNR2+a': wb(0.213),                  # (a) block-max side bit + half contexts
        'WNR2+ab': wb(0.35),                  # (a)+(b) K=4 models per matrix
        'WNR2+d': r2 - LEN,                   # (d) alone
        'WNR2+e': r2 * INT,                   # (e) alone
        'WNR3': (wb(0.35) - LEN) * INT,       # byte-exact WNR3 = a+b+d+e  (~0.895)
        'WNR2z': wb(3.08),                    # (c) alone on the base context (in-sample -3.08 %)
        'WNR3z': (wb(3.38) - LEN) * INT,      # a+b+c+d+e (~0.868)
    }


# name, low, center, high, source
PARAMS = [
    ('CV0', 1030, 1070, 1110, 'ref s3 G2: ~1,030-1,110 decoded mirrors of 17.93 MiB expected'),
    ('beta', 0.42, 0.60, 1.00, 'slope of miss curve: 0.42-0.47 Zipf-IRM fit to M0; 0.55 Oct replay (70% VRAM, 45.8 reads @57 slots/layer); 0.71 M0<->V4 LRU two-point; ~1.0 local marginal of Oct VRAM sweep 1530->2000'),
    ('mA', 0.629, 0.641, 0.654, 'M0 R0c: hit 34.6-37.1 % with 360 records (9 slots/layer) cached'),
    ('sigma', 0.0, 0.25, 0.5, 'ASSUMED per-layer heterogeneity of miss rate (lognormal sd)'),
    ('q', 0.2, 0.5, 0.8, 'ASSUMED fraction of VRAM misses prefetched in time; V4 live 9-15 demand + ~71 prefetch reads/token'),
    ('w_extra', 0.0, 0.2, 0.5, 'ASSUMED wasted prefetch reads per demand disk miss'),
    ('T0', 0.060, 0.100, 0.180, 'non-expert-I/O time per token after G2 [ASSUMED]; V4 64GB A/B: attn 34.6 + hc 14.4 + GPU 13.9 ms + rest'),
    ('cfix', 0.05e-3, 0.15e-3, 0.30e-3, 'per VRAM-miss attach/publish overhead; V4 A1 attach 35.6 ms/token incl. upload'),
    ('cdec_x', 0.0, 0.04e-3, 0.08e-3, 'WNR batch (W2) time beyond the PCIe floor: V4 b8 0.281 vs 0.22; V4.1 b8 0.402 vs 0.324 (shared GPU)'),
    ('csingle', 0.72e-3, 0.76e-3, 1.00e-3, 'WNR single-record helper per record without W2: V4.1 single 0.718-0.759 (shared GPU), V4 helper ~1.0 ms'),
    ('kappa', 0.0, 0.15, 0.30, 'GPU contention per ms of decode kernel: bench_prio WNR2 decode inflates fg chain ~29 %'),
    ('cv1', 0.20e-3, 0.22e-3, 0.30e-3, 'in-VRAM single-record decode latency: rechunk 2048:512 0.1998 ms, task brief ~0.2 ms'),
    ('cvtp', 0.175e-3, 0.19e-3, 0.22e-3, 'in-VRAM decode throughput cost/record: 17.93 MiB / ~100 GiB/s (8 in flight) = 0.175 ms'),
    ('slot_ov', 1.00, 1.02, 1.03, 'fixed slot = largest record (crc code): per-layer mean 3.747-3.852 bit/w -> max ~+1.7 % + within-layer spread; 1.00 = byte-capped LRU'),
    ('R_tot', 6.0, 10.3, 14.0, 'RAM for expert records, GiB: M0 6.30 GiB slots + W3 4 GiB cache = 10.3; 32 GB machine'),
    ('crc_share', 0.39, 0.39, 1.0, 'share of that RAM held as compressed-record cache (W3: 4 of 10.3 GiB); 1.0 = all compressed'),
    ('bw_scale', 0.90, 1.00, 1.05, 'drive throughput uncertainty (thermal guard hit U: at 75 C in Oct, fragmentation)'),
    ('qmix', 0.0, 0.5, 1.0, 'ASSUMED weight of M/D/1 queueing of demand reads behind prefetch reads (0 = demand has priority)'),
]
PNAMES = [p[0] for p in PARAMS]


def center_params(n=1, **over):
    P = {name: np.full(n, c, float) for name, lo, c, hi, _ in PARAMS}
    for k, v in over.items():
        P[k] = np.full(n, float(v))
    return P


def sample_params(n, rng):
    P = {}
    for name, lo, c, hi, _ in PARAMS:
        P[name] = rng.uniform(lo, hi, n)
    return P


def make_uniforms(rng, D, T, L, k):
    return (rng.random((D, T, L, k)), rng.random((D, T, L, k)), rng.random((D, T, L, k)),
            rng.standard_normal((D, L)))


def miss_curve(C, P, geo):
    mA = P['mA'] if geo.mA_override is None else geo.mA_override
    c = np.maximum(np.asarray(C, float), 1e-9) / geo.N
    return np.minimum(1.0, mA * (c / geo.cA) ** (-P['beta']))


def capacities(P, geo, r, cv, x):
    CV0 = P['CV0']
    ov = P['slot_ov'] if r < 1 else 1.0
    R_dec = P['R_tot'] * (1 - P['crc_share'])
    R_crc = P['R_tot'] * P['crc_share']
    C_ram = R_dec * GIB / geo.raw + R_crc * GIB / (geo.raw * r * ov)
    if cv == 'none' or r >= 1:
        C_dec = CV0.copy()
        C_V = CV0.copy()
    elif cv == 'incl':      # every VRAM expert held compressed; a decoded LRU of x records on top
        C_dec = np.full_like(CV0, float(x))
        C_V = (CV0 - x) / (r * ov)
    elif cv == 'excl':      # optimistic exclusive split: fraction x decoded, the rest compressed
        C_dec = x * CV0
        C_V = C_dec + np.maximum(0.0, (1 - x) * CV0 - KS) / (r * ov)
    else:
        raise ValueError(cv)
    return C_dec, C_V, C_V + C_ram


def simulate(P, geo, r, cv='none', x=None, drive='U', w2=True, U=None, sD=0.2, cv_ahead=False):
    Usel, Upf, Udrv, Z = U
    C_dec, C_V, C_tot = capacities(P, geo, r, cv, x)
    md0, mv0, mt0 = miss_curve(C_dec, P, geo), miss_curve(C_V, P, geo), miss_curve(C_tot, P, geo)
    s = P['sigma'][:, None]
    a = np.exp(s * Z - 0.5 * s * s)                       # (D, L)
    md = np.minimum(1.0, a * md0[:, None])
    mv = np.minimum(md, a * mv0[:, None])
    mt = np.minimum(mv, a * mt0[:, None])
    t1, t2, t3 = [(1 - m)[:, None, :, None] for m in (md, mv, mt)]
    cv_m = (Usel >= t1) & (Usel < t2)
    ram_m = (Usel >= t2) & (Usel < t3)
    dsk_m = Usel >= t3
    unc = Upf >= P['q'][:, None, None, None]               # not prefetched in time
    k_cv = cv_m.sum(-1)
    k_ram = ram_m.sum(-1)
    k_dsk = dsk_m.sum(-1)
    k_ram_u = (ram_m & unc).sum(-1)
    k_dsk_u = (dsk_m & unc).sum(-1)

    def p3(name):
        return P[name][:, None, None]
    rb = r * geo.raw
    if r >= 1:
        cx = np.full(len(P['T0']), geo.raw / PCIE)[:, None, None]
        deck = 0.0
    elif w2:
        cx = (rb / PCIE + P['cdec_x'])[:, None, None]
        deck = p3('cvtp')
    else:
        cx = p3('csingle')
        deck = p3('cvtp')
    bws = p3('bw_scale')

    def disk_t(k, drv):
        bw, qd1 = DRIVES[drv]
        # QD1 latency = fixed part (qd1 - raw/bw: U 0.34, E 0.09, D 0.52 ms) + transfer part scaled by r
        lat = (qd1 - geo.raw / bw) + (geo.raw / bw) * r
        return np.where(k > 0, np.maximum(lat, k * rb / (bw * bws)), 0.0)

    w = P['w_extra'][:, None]
    if drive == 'U+D':
        onD = Udrv < sD
        kD_u = (dsk_m & unc & onD).sum(-1)
        kU_u = k_dsk_u - kD_u
        tdisk = np.maximum(disk_t(kU_u, 'U'), disk_t(kD_u, 'D'))
        kD = (dsk_m & onD).sum(-1).sum(-1)
        kU = k_dsk.sum(-1) - kD
        busy_disk = np.maximum(kU * (1 + w) * rb / (DRIVES['U'][0] * P['bw_scale'][:, None]),
                               kD * (1 + w) * rb / (DRIVES['D'][0] * P['bw_scale'][:, None]))
    else:
        tdisk = disk_t(k_dsk_u, drive)
        busy_disk = k_dsk.sum(-1) * (1 + w) * rb / (DRIVES[drive][0] * P['bw_scale'][:, None])
    tdisk0 = np.where(k_dsk_u > 0, tdisk + cx, 0.0)
    tcopy = (k_ram_u + k_dsk_u) * cx
    # cv_ahead: predicted compressed-VRAM hits are decoded during the previous layer (same coverage q
    # as prefetch): their latency is hidden, their GPU time still counts as contention
    k_cv_lat = (cv_m & unc).sum(-1) if cv_ahead else k_cv
    tcv = np.where(k_cv_lat > 0, p3('cv1') + (k_cv_lat - 1) * p3('cvtp'), 0.0)
    fix = p3('cfix') * (k_ram_u + k_dsk_u)
    cont = p3('kappa') * (k_cv * p3('cvtp') + (k_ram + k_dsk * (1 + P['w_extra'][:, None, None])) * deck)
    base = p3('T0') / geo.L + fix + cont
    reads = k_dsk.sum(-1) * (1 + w)
    busy_copy = (k_ram.sum(-1) + reads) * cx[:, :, 0]
    # demand reads queue behind prefetch / wasted reads: M/D/1 wait rho/(2(1-rho)) x service time of one
    # record, weighted by qmix (0 = demand reads have absolute priority), solved as a damped fixed point
    bw_main = DRIVES['U' if drive == 'U+D' else drive][0] * P['bw_scale']
    svc = rb / bw_main
    rho = np.zeros(len(svc))
    for _ in range(14):
        # capped at QCAP records ahead: the engine keeps only ~1.7 prefetch reads/layer in flight
        # (V4 live ~71 prefetch reads/token over 43 layers), so the queue is bounded (closed system)
        Wq = P['qmix'] * np.minimum(QCAP, rho / (2 * (1 - rho))) * svc
        tdisk = np.where(k_dsk_u > 0, tdisk0 + Wq[:, None, None], 0.0)
        wait = np.maximum(np.maximum(tdisk, tcopy), tcv)
        crit = (base + wait).sum(-1)                       # (D, T)
        tt = np.maximum(np.maximum(crit, busy_disk), busy_copy)
        rho = 0.5 * rho + 0.5 * np.minimum(0.95, busy_disk.mean(1) / tt.mean(1))
    return {
        't': tt.mean(1),
        'crit': crit.mean(1), 'busy_disk': busy_disk.mean(1), 'busy_copy': busy_copy.mean(1),
        'disk_bound': (busy_disk >= np.maximum(crit, busy_copy)).mean(1),
        'n_vdec': (geo.L * geo.topk - k_cv.sum(-1) - k_ram.sum(-1) - k_dsk.sum(-1)).mean(1),
        'n_cv': k_cv.sum(-1).mean(1), 'n_ram': k_ram.sum(-1).mean(1),
        'n_disk_demand': k_dsk.sum(-1).mean(1), 'reads': reads.mean(1),
        'GB_per_token': reads.mean(1) * rb / 1e9,
        'C_dec': C_dec, 'C_V': C_V, 'C_tot': C_tot,
        'wait_ms': wait.sum(-1).mean(1) * 1e3, 'rho': rho,
    }


CV_GRID = [('incl', 16), ('incl', 48), ('incl', 96), ('excl', 0.25), ('excl', 0.5), ('excl', 0.75)]


def best_cv(P, geo, r, drive, w2, U, cv_ahead=False):
    """Oracle-tuned compressed VRAM tier: per draw, the best of the CV_GRID designs (and never worse
    than 'none' is NOT enforced -- we report the CV tier's own best)."""
    res = [simulate(P, geo, r, cv, x, drive, w2, U, cv_ahead=cv_ahead) for cv, x in CV_GRID]
    T = np.stack([q['t'] for q in res])
    i = T.argmin(0)
    out = {}
    for k in res[0]:
        arr = np.stack([np.broadcast_to(q[k], T.shape[1:]) for q in res])
        out[k] = arr[i, np.arange(T.shape[1])]
    out['design'] = [CV_GRID[j] for j in i]
    return out


def all_compressed(P, geo, r, drive, w2, U):
    return simulate(P, geo, r, 'incl', KS, drive, w2, U)


def pct(base_t, new_t):
    return (base_t / new_t - 1.0) * 100.0


def q3(a):
    a = np.asarray(a, float)
    return [float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95))]


def run():
    t_start = time.time()
    R = format_ratios()
    results = {'ratios': R, 'params': [{'name': n, 'low': lo, 'center': c, 'high': hi, 'source': s}
                                       for n, lo, c, hi, s in PARAMS]}
    geo = V41
    S = geo.L * geo.topk

    # ---------------- 1. center scenario, long run -----------------
    rng = np.random.default_rng(20261107)
    Uc = make_uniforms(rng, 1, 3000, geo.L, geo.topk)
    Pc = center_params()
    center = {}
    for drive in ('U', 'E'):
        for f, r in R.items():
            for w2 in ((True,) if r >= 1 else (True, False)):
                res = simulate(Pc, geo, r, 'none', None, drive, w2, Uc)
                center[f'{drive}|{f}|none|{"W2" if w2 else "single"}'] = res
            if r < 1:
                center[f'{drive}|{f}|cvbest|W2'] = best_cv(Pc, geo, r, drive, True, Uc)
                center[f'{drive}|{f}|cvall|W2'] = all_compressed(Pc, geo, r, drive, True, Uc)
    center['U+D|WEX1|none|W2'] = simulate(Pc, geo, 1.0, 'none', None, 'U+D', True, Uc)

    def row(k):
        q = center[k]
        return {kk: (float(np.asarray(v).ravel()[0]) if not isinstance(v, list) else str(v[0]))
                for kk, v in q.items()}
    results['center'] = {k: row(k) for k in center}

    # ---------------- 2. Monte Carlo over the parameter ranges (paired draws) -----------------
    D, T = 400, 40
    rng = np.random.default_rng(7)
    Pm = sample_params(D, rng)
    Um = make_uniforms(rng, D, T, geo.L, geo.topk)
    mc = {}
    for drive in ('U', 'E'):
        for f, r in R.items():
            mc[f'{drive}|{f}|none|W2'] = simulate(Pm, geo, r, 'none', None, drive, True, Um)
            if r < 1 and f in ('WNR2', 'WNR3', 'WNR3z'):
                mc[f'{drive}|{f}|none|single'] = simulate(Pm, geo, r, 'none', None, drive, False, Um)
                mc[f'{drive}|{f}|cvbest|W2'] = best_cv(Pm, geo, r, drive, True, Um)
                mc[f'{drive}|{f}|cvall|W2'] = all_compressed(Pm, geo, r, drive, True, Um)
                if f != 'WNR3':
                    mc[f'{drive}|{f}|cvahead|W2'] = best_cv(Pm, geo, r, drive, True, Um, cv_ahead=True)
    mc['U+D|WEX1|none|W2'] = simulate(Pm, geo, 1.0, 'none', None, 'U+D', True, Um)

    comps = {}
    for drive in ('U', 'E'):
        g = lambda f, cv='none', m='W2': mc[f'{drive}|{f}|{cv}|{m}']['t']
        pr = lambda b, n: (b, n)           # keep the pair, compute % and ms below
        c = {
            'WNR2 vs WEX1': pr(g('WEX1'), g('WNR2')),
            'WNR2(no W2, single-record helper) vs WEX1': pr(g('WEX1'), g('WNR2', 'none', 'single')),
            'WNR3 vs WNR2': pr(g('WNR2'), g('WNR3')),
            'WNR3 vs WEX1': pr(g('WEX1'), g('WNR3')),
            'WNR3z vs WNR2': pr(g('WNR2'), g('WNR3z')),
            'WNR3z vs WNR3': pr(g('WNR3'), g('WNR3z')),
            'WNR3z vs WEX1': pr(g('WEX1'), g('WNR3z')),
            '(a) vs WNR2': pr(g('WNR2'), g('WNR2+a')),
            '(a)+(b) vs WNR2': pr(g('WNR2'), g('WNR2+ab')),
            '(d) vs WNR2': pr(g('WNR2'), g('WNR2+d')),
            '(e) vs WNR2': pr(g('WNR2'), g('WNR2+e')),
            '(c) alone vs WNR2': pr(g('WNR2'), g('WNR2z')),
            'CV best-design (WNR2) vs WNR2': pr(g('WNR2'), g('WNR2', 'cvbest')),
            'CV all-compressed (WNR2) vs WNR2': pr(g('WNR2'), g('WNR2', 'cvall')),
            'CV best-design (WNR3z) vs WNR3z': pr(g('WNR3z'), g('WNR3z', 'cvbest')),
            'CV best-design (WNR3z) vs WEX1': pr(g('WEX1'), g('WNR3z', 'cvbest')),
            'CV + decode-ahead (WNR2) vs WNR2': pr(g('WNR2'), g('WNR2', 'cvahead')),
            'CV + decode-ahead (WNR3z) vs WNR3z': pr(g('WNR3z'), g('WNR3z', 'cvahead')),
            'W2 batch vs single-record (WNR2)': pr(g('WNR2', 'none', 'single'), g('WNR2')),
        }
        comps[drive] = {}
        for k, (b, n) in c.items():
            v = (b / n - 1.0) * 100.0
            comps[drive][k] = {'p5_p50_p95': q3(v), 'P(>=+3%)': float((v >= 3.0).mean()),
                               'P(>0)': float((v > 0).mean()), 'dms_p5_p50_p95': q3((n - b) * 1e3)}
        abs_ = {}
        for k in mc:
            if k.startswith(drive + '|'):
                abs_[k] = {'tok_s': q3(1.0 / mc[k]['t']), 'ms_token': q3(mc[k]['t'] * 1e3),
                           'reads': q3(mc[k]['reads']), 'GB_token': q3(mc[k]['GB_per_token']),
                           'disk_bound_frac': q3(mc[k]['disk_bound']), 'n_cv': q3(mc[k]['n_cv'])}
        comps[drive + '_abs'] = abs_
    comps['cross-drive: WNR2 on E: vs WEX1 on U:'] = {
        'p5_p50_p95': q3(pct(mc['U|WEX1|none|W2']['t'], mc['E|WNR2|none|W2']['t']))}
    comps['cross-drive: WNR3z on E: vs WEX1 on U:'] = {
        'p5_p50_p95': q3(pct(mc['U|WEX1|none|W2']['t'], mc['E|WNR3z|none|W2']['t']))}
    comps['WEX1 U:+D: mirror (20% on D:) vs WEX1 U:'] = {
        'p5_p50_p95': q3(pct(mc['U|WEX1|none|W2']['t'], mc['U+D|WEX1|none|W2']['t']))}
    comps['WNR2 on U: vs WNR2 on E: (drive alone)'] = {
        'p5_p50_p95': q3(pct(mc['E|WNR2|none|W2']['t'], mc['U|WNR2|none|W2']['t']))}

    # elasticity: % tok/s per % record bytes, from WNR2 -> WNR3z, per draw
    for drive in ('U', 'E'):
        dt = pct(mc[f'{drive}|WNR2|none|W2']['t'], mc[f'{drive}|WNR3z|none|W2']['t'])
        db = (R['WNR2'] / R['WNR3z'] - 1) * 100
        comps[drive]['elasticity tok/s % per bytes % (WNR2->WNR3z)'] = {'p5_p50_p95': q3(dt / db)}

    # rank correlation of each parameter with key deltas (what dominates)
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    dom = {}
    for drive in ('U', 'E'):
        g = lambda f, cv='none': mc[f'{drive}|{f}|{cv}|W2']['t']
        targets = {'WNR2 vs WEX1': pct(g('WEX1'), g('WNR2')),
                   'WNR3z vs WNR2': pct(g('WNR2'), g('WNR3z')),
                   'CVbest(WNR2) vs WNR2': pct(g('WNR2'), g('WNR2', 'cvbest')),
                   'WEX1 tok/s': 1.0 / g('WEX1')}
        dom[drive] = {tk: dict(sorted(((n, round(spearman(Pm[n], tv), 3)) for n in PNAMES),
                                      key=lambda kv: -abs(kv[1]))) for tk, tv in targets.items()}
    # decision aid: format gain as a function of WEX1's disk-busy share (busy_disk / token time),
    # which is measurable after G2 from wexprof NVMe bytes/token, the drive's GB/s and ms/token
    share_all, g2_all, gz_all, g3z_all = [], [], [], []
    for drive in ('U', 'E'):
        w = mc[f'{drive}|WEX1|none|W2']
        share_all.append(w['busy_disk'] / w['t'])
        g2_all.append(pct(w['t'], mc[f'{drive}|WNR2|none|W2']['t']))
        gz_all.append(pct(w['t'], mc[f'{drive}|WNR3z|none|W2']['t']))
        g3z_all.append(pct(mc[f'{drive}|WNR2|none|W2']['t'], mc[f'{drive}|WNR3z|none|W2']['t']))
    share_all, g2_all, gz_all, g3z_all = map(np.concatenate, (share_all, g2_all, gz_all, g3z_all))
    aid = {}
    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        sel = (share_all >= lo_) & (share_all < hi_)
        if sel.sum() < 5:
            continue
        aid[f'{lo_:.1f}-{min(hi_, 1.0):.1f}'] = {
            'n': int(sel.sum()), 'WNR2 vs WEX1 p5/p50/p95': q3(g2_all[sel]),
            'P(WNR2>=+3%)': float((g2_all[sel] >= 3).mean()),
            'WNR3z vs WEX1 p50': float(np.median(gz_all[sel])), 'P(WNR3z>=+3%)': float((gz_all[sel] >= 3).mean()),
            'WNR3z vs WNR2 p50': float(np.median(g3z_all[sel]))}
    comps['decision aid: gain vs WEX1 disk-busy share (U and E draws pooled)'] = aid
    for drive in ('U', 'E'):
        w = mc[f'{drive}|WEX1|none|W2']
        comps[drive + '_abs'][f'{drive}|WEX1|none|W2']['disk_busy_share'] = q3(w['busy_disk'] / w['t'])

    results['mc'] = comps
    results['mc_rank_correlations'] = dom
    results['mc_setup'] = {'draws': D, 'tokens_per_draw': T, 'sampling': 'independent uniform on [low, high]'}

    # ---------------- 3. tornado (one at a time, others at center) -----------------
    rng = np.random.default_rng(99)
    Ut = make_uniforms(rng, 1, 1500, geo.L, geo.topk)
    torn = {}
    for drive in ('U', 'E'):
        torn[drive] = {}
        for name, lo, c, hi, _ in PARAMS:
            ent = {}
            for side, val in (('low', lo), ('high', hi)):
                P1 = center_params(**{name: val})
                tW = simulate(P1, geo, 1.0, 'none', None, drive, True, Ut)['t'][0]
                t2 = simulate(P1, geo, R['WNR2'], 'none', None, drive, True, Ut)['t'][0]
                tz = simulate(P1, geo, R['WNR3z'], 'none', None, drive, True, Ut)['t'][0]
                t3 = simulate(P1, geo, R['WNR3'], 'none', None, drive, True, Ut)['t'][0]
                tcv = best_cv(P1, geo, R['WNR2'], drive, True, Ut)['t'][0]
                ent[side] = {'WEX1_tok_s': 1 / tW, 'WNR2_vs_WEX1': pct(tW, t2), 'WNR3_vs_WNR2': pct(t2, t3),
                             'WNR3z_vs_WNR2': pct(t2, tz), 'CVbest_vs_WNR2': pct(t2, tcv)}
            torn[drive][name] = ent
    results['tornado'] = torn

    # ---------------- 4. RAM-tier scenarios (center params) -----------------
    ram = {}
    for drive in ('U', 'E'):
        for R_tot, share in ((10.3, 0.39), (10.3, 1.0), (6.0, 1.0), (14.0, 1.0), (14.0, 0.39), (6.0, 0.0)):
            P1 = center_params(R_tot=R_tot, crc_share=share)
            tW = simulate(P1, geo, 1.0, 'none', None, drive, True, Ut)
            t2 = simulate(P1, geo, R['WNR2'], 'none', None, drive, True, Ut)
            tz = simulate(P1, geo, R['WNR3z'], 'none', None, drive, True, Ut)
            ram[f'{drive} R={R_tot} GiB crc_share={share}'] = {
                'WEX1_tok_s': float(1 / tW['t'][0]), 'WNR2_vs_WEX1': float(pct(tW['t'], t2['t'])[0]),
                'WNR3z_vs_WNR2': float(pct(t2['t'], tz['t'])[0]),
                'WEX1_reads': float(tW['reads'][0]), 'WNR2_reads': float(t2['reads'][0]),
                'reads_cut_pct': float((1 - t2['reads'][0] / tW['reads'][0]) * 100),
                'bytes_cut_pct': float((1 - t2['GB_per_token'][0] / tW['GB_per_token'][0]) * 100)}
    results['ram_scenarios'] = ram

    # ---------------- 5. consistency checks against the reference -----------------
    chk = {}
    # (i) M0 CPU-only two-point fit: t = c + m * 18.8 MB / BW (serial), U: 1/1.12, D: 1/0.474 s
    tU, tD = 2 / (1.110 + 1.137), 2 / (0.468 + 0.480)
    k = (tD - tU) / (V41.raw * (1 / DRIVES['D'][0] - 1 / DRIVES['U'][0]))
    chk['M0 two-point fit (serial, no overlap)'] = {
        'misses_per_token': k, 'non_disk_s_per_token': tU - k * V41.raw / DRIVES['U'][0],
        'note': 'vs ~214 misses/token [inferred] in ref s12 and 240*(1-0.346..0.371)=151-157 from the M0 hit rate'}
    chk['ref disk floor 214 x 18.8 MB on D: (s)'] = 214 * V41.raw / DRIVES['D'][0]
    chk['measured M0 WEX1 on D: s/token'] = tD
    # (ii) miss curve at the operating point
    for b in (0.42, 0.6, 0.71, 1.0):
        P1 = center_params(beta=b)
        Cd, CVv, Ct = capacities(P1, V41, 1.0, 'none', None)
        chk[f'beta={b}: VRAM hit, total hit, disk demand reads/token'] = [
            float(1 - miss_curve(CVv, P1, V41)[0]), float(1 - miss_curve(Ct, P1, V41)[0]),
            float(S * miss_curve(Ct, P1, V41)[0])]
    # (iii) Zipf-IRM: exponent that gives 35.8 % static-top-9 hit over 192, and its local slope at 27/layer
    def zipf_hit(s, k, n=192):
        i = np.arange(1, n + 1, dtype=float) ** (-s)
        return i[:k].sum() / i.sum()
    lo_, hi_ = 0.1, 2.0
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        if zipf_hit(mid, 9) < 0.358: lo_ = mid
        else: hi_ = mid
    s_z = 0.5 * (lo_ + hi_)
    m27, m33 = 1 - zipf_hit(s_z, 27), 1 - zipf_hit(s_z, 33)
    chk['Zipf-IRM fit to M0'] = {'s': s_z, 'miss@27/layer': m27, 'miss@33/layer': m33,
                                 'local beta 27->33': -math.log(m33 / m27) / math.log(33 / 27)}
    # (iv) Oct replay VRAM sweep elasticity (V4): reads 45.8 @1530, 43.9 @1700, 41.0 @2000
    chk['Oct replay d ln reads / d ln C_VRAM (1530->2000)'] = math.log(41.0 / 45.8) / math.log(2000 / 1530)

    # (v) history replay: V4 equal-RAM A/B on the 64 GB machine (WEX1 5.62 vs WNR2+cache mode 2 5.44 = -3 %)
    rng = np.random.default_rng(5)
    Uh = make_uniforms(rng, 1, 1500, V4.L, V4.topk)
    rV4 = 11_661_338 / 13_369_344
    slot_ov_v4 = 11.76 * MIB / (rV4 * V4.raw)
    R_dec_v4 = 16 * 43 * V4.raw / GIB                    # 16 slots/layer
    R_crc_v4 = 12.86
    def Pv4(T0):
        return center_params(CV0=1530, R_tot=R_dec_v4 + R_crc_v4, crc_share=R_crc_v4 / (R_dec_v4 + R_crc_v4),
                             slot_ov=slot_ov_v4, T0=T0, csingle=1.0e-3)
    lo_, hi_ = 0.0, 0.3
    for _ in range(40):
        mid = 0.5 * (lo_ + hi_)
        t = simulate(Pv4(mid), V4, 1.0, 'none', None, 'U', True, Uh)['t'][0]
        if t < 1 / 5.62: lo_ = mid
        else: hi_ = mid
    T0v4 = 0.5 * (lo_ + hi_)
    hv = {'T0_calibrated_s': T0v4}
    for b in (0.42, 0.6, 1.0):
        for q in (0.2, 0.5, 0.8):
            P1 = Pv4(T0v4); P1['beta'][:] = b; P1['q'][:] = q
            tW = simulate(P1, V4, 1.0, 'none', None, 'U', True, Uh)
            ts = simulate(P1, V4, rV4, 'none', None, 'U', False, Uh)
            tb = simulate(P1, V4, rV4, 'none', None, 'U', True, Uh)
            hv[f'beta={b} q={q}'] = {'WEX1_tok_s': float(1 / tW['t'][0]),
                                     'WNR2 single (Oct helper ~1.0 ms) vs WEX1 %': float(pct(tW['t'], ts['t'])[0]),
                                     'WNR2 W2 vs WEX1 %': float(pct(tW['t'], tb['t'])[0]),
                                     'WEX1 reads/token': float(tW['reads'][0]),
                                     'disk_bound_frac': float(tW['disk_bound'][0])}
    chk['history: V4 equal-RAM A/B (measured -3 %, 64 GB machine)'] = hv
    results['checks'] = chk

    # PLAN 3.2 detectability from measured clean repeat runs (V4, 64 GB): pairs (5.573, 5.624), (5.446, 5.429), (5.401, 5.440)
    pairs = [(5.573, 5.624), (5.446, 5.429), (5.401, 5.440)]
    cvs = [abs(a - b) / math.sqrt(2) / ((a + b) / 2) for a, b in pairs]
    sigma_run = math.sqrt(sum(c * c for c in cvs) / len(cvs))
    se = sigma_run * math.sqrt(2 / 4)
    results['plan32'] = {'per_run_cv_pct': sigma_run * 100, 'abba2_se_diff_pct': se * 100,
                         'ci95_halfwidth_pct (t6=2.447)': 2.447 * se * 100,
                         'note': 'from 3 clean repeat pairs on V4/64 GB; a disk-bound 32 GB machine with NVMe thermals is likely noisier'}
    results['runtime_s'] = time.time() - t_start
    return results


def fmt_report(res):
    L = []
    R = res['ratios']
    L.append('record ratios: ' + ', '.join(f'{k} {v:.4f}' for k, v in R.items()))
    L.append('\n== center scenario (all params at center; 3000 tokens) ==')
    L.append(f'{"config":34s} {"tok/s":>6s} {"ms/tok":>7s} {"vdec":>6s} {"cv":>6s} {"ram":>6s} {"disk":>6s} {"reads":>6s} {"GB/t":>5s} {"bound":>5s} {"C_tot":>6s}')
    for k, q in res['center'].items():
        L.append(f'{k:34s} {1/q["t"]:6.3f} {q["t"]*1e3:7.1f} {q["n_vdec"]:6.1f} {q["n_cv"]:6.1f} {q["n_ram"]:6.1f} '
                 f'{q["n_disk_demand"]:6.1f} {q["reads"]:6.1f} {q["GB_per_token"]:5.2f} {q["disk_bound"]:5.2f} {q["C_tot"]:6.0f}'
                 + (f'  design={q["design"]}' if 'design' in q else ''))
    L.append('\n== Monte Carlo, paired deltas in % tok/s: p5 / p50 / p95, P(>=+3%), P(>0) ==')
    for drive in ('U', 'E'):
        L.append(f'-- drive {drive}:')
        for k, v in res['mc'][drive].items():
            a = v['p5_p50_p95']
            extra = f'  P>=3%: {v["P(>=+3%)"]:.2f}  P>0: {v["P(>0)"]:.2f}' if 'P(>=+3%)' in v else ''
            L.append(f'   {k:48s} {a[0]:7.2f} {a[1]:7.2f} {a[2]:7.2f}{extra}')
    for k in res['mc']:
        if k.startswith('cross') or k.startswith('WEX1 U') or k.startswith('WNR2 on U'):
            a = res['mc'][k]['p5_p50_p95']
            L.append(f'   {k:48s} {a[0]:7.2f} {a[1]:7.2f} {a[2]:7.2f}')
    L.append('\n== decision aid: format gain vs WEX1 disk-busy share after G2 ==')
    for k, v in res['mc']['decision aid: gain vs WEX1 disk-busy share (U and E draws pooled)'].items():
        a = v['WNR2 vs WEX1 p5/p50/p95']
        L.append(f'   share {k}: n={v["n"]:3d}  WNR2 vs WEX1 {a[0]:6.2f} {a[1]:6.2f} {a[2]:6.2f}  P>=3% {v["P(WNR2>=+3%)"]:.2f} | '
                 f'WNR3z vs WEX1 p50 {v["WNR3z vs WEX1 p50"]:6.2f} P>=3% {v["P(WNR3z>=+3%)"]:.2f} | WNR3z vs WNR2 p50 {v["WNR3z vs WNR2 p50"]:5.2f}')
    for drive in ('U', 'E'):
        a = res['mc'][drive + '_abs'][f'{drive}|WEX1|none|W2']['disk_busy_share']
        L.append(f'   WEX1 disk-busy share on {drive}: {a[0]:.2f} {a[1]:.2f} {a[2]:.2f}')
    L.append('\n== Monte Carlo absolute (p5 / p50 / p95) ==')
    for drive in ('U', 'E'):
        for k, v in res['mc'][drive + '_abs'].items():
            L.append(f'   {k:28s} tok/s {v["tok_s"][0]:5.2f} {v["tok_s"][1]:5.2f} {v["tok_s"][2]:5.2f} | reads {v["reads"][1]:5.1f} '
                     f'| GB/t {v["GB_token"][1]:4.2f} | disk-bound {v["disk_bound_frac"][1]:4.2f} | n_cv {v["n_cv"][1]:5.1f}')
    L.append('\n== rank correlations (Spearman) of parameters with key outputs ==')
    for drive, d in res['mc_rank_correlations'].items():
        for tk, cor in d.items():
            top = list(cor.items())[:5]
            L.append(f'   {drive} {tk:22s} ' + ', '.join(f'{n} {c:+.2f}' for n, c in top))
    L.append('\n== tornado (one-at-a-time; low -> high) ==')
    for drive, d in res['tornado'].items():
        L.append(f'-- drive {drive}: param: WEX1 tok/s | WNR2 vs WEX1 % | WNR3 vs WNR2 % | WNR3z vs WNR2 % | CVbest vs WNR2 %')
        for name, e in d.items():
            lo, hi = e['low'], e['high']
            L.append(f'   {name:10s} {lo["WEX1_tok_s"]:5.2f}->{hi["WEX1_tok_s"]:5.2f} | {lo["WNR2_vs_WEX1"]:6.2f}->{hi["WNR2_vs_WEX1"]:6.2f} | '
                     f'{lo["WNR3_vs_WNR2"]:5.2f}->{hi["WNR3_vs_WNR2"]:5.2f} | {lo["WNR3z_vs_WNR2"]:5.2f}->{hi["WNR3z_vs_WNR2"]:5.2f} | '
                     f'{lo["CVbest_vs_WNR2"]:6.2f}->{hi["CVbest_vs_WNR2"]:6.2f}')
    L.append('\n== RAM-tier scenarios (center params) ==')
    for k, v in res['ram_scenarios'].items():
        L.append(f'   {k:34s} WEX1 {v["WEX1_tok_s"]:5.2f} tok/s | WNR2 vs WEX1 {v["WNR2_vs_WEX1"]:+6.2f}% | WNR3z vs WNR2 {v["WNR3z_vs_WNR2"]:+5.2f}% | '
                 f'reads {v["WEX1_reads"]:5.1f}->{v["WNR2_reads"]:5.1f} (-{v["reads_cut_pct"]:.1f}%), bytes -{v["bytes_cut_pct"]:.1f}%')
    L.append('\n== checks ==')
    L.append(json.dumps(res['checks'], indent=1))
    L.append('\n== PLAN 3.2 detectability ==')
    L.append(json.dumps(res['plan32'], indent=1))
    L.append(f'\nruntime {res["runtime_s"]:.1f} s')
    return '\n'.join(L)


if __name__ == '__main__':
    res = run()
    with open(os.path.join(OUT, 'capacity_value_results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    rep = fmt_report(res)
    with open(os.path.join(OUT, 'capacity_value_output.txt'), 'w') as f:
        f.write(rep)
    print(rep)
