"""Test kit / audit: corruption cases on real ceil-form records, all integrity rules, paired trials.
(From team5 audit-ceil run_cases.py; kit changes: no globals / hard-coded paths, trial counts from a Params object,
shared trial-table builders for validate, M2 shift clamp no longer leaks from one rule to the next, sampled scale cases
for --quick, cases that need >= 2 experts are skipped with 1 expert.)

Paths:
  W  weight-stream jobs (cases W0, A..J, T, M2, N1..N3): a trial = 1-2 replaced weight streams (bytes, length, contexts,
     identity); the record outcome equals "every replaced stream passes" because nothing else in the record changes and a
     lane's result depends only on the bytes it consumed (ptr == offs + len) [I]; cross-checked against path R
     (validate.py, cases A and K).
  S  scale-stream jobs (cases S0, K, L, M3, N4, N5): stage 1 decodes the replaced scale streams; trials that pass get the
     record-level consequences: LENC weight lengths recomputed from the decoded coded scales (sum check), weight streams
     whose srel, length or offset changed are decoded (stage 2) from the clean payload.
  R  record patches through audit_rec.decode_r (cases M1, P1, P2, P3, O).
Rules per decoder kind: fg0d -> R0, fg0de -> HASH, fg0dh -> HASH32, fg0dt -> TAIL and TAILD (TAILD = TAIL streams +
record digest = XOR of H32 over all streams; for a trial it passes iff XOR of the new h of the replaced streams equals the
XOR of their clean h, all other streams being unchanged).
Acceptance with the spec decoder = stream rules pass and the spec ceil inverse accepts (cs == 0 in a halved block is the
only inverse rejection); 'cr' = ceil-ref's stricter inverse (also rejects decoded block max < 5).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import audit_lib as A
import audit_rec as R
from audit_lib import W, NWS, NSS, S_CHUNK, W_CHUNK, BPS

OTHER_M = {0: 2, 1: 2, 2: 0}       # team 2 case J: w1 -> w3 model, w2 -> w3 model; w3 -> w1 model

CASE_DOC = {
    'W0': 'clean weight streams (control: all must be accepted, identical output)',
    'A': 'team2 A: one random bit flip anywhere in a weight stream (state bytes included)',
    'B': 'team2 B: bit flip in the last byte of a weight stream',
    'C': 'team2 C: bit flip in the 4-byte initial state',
    'D': 'team2 D: random 2-byte overwrite',
    'E': 'team2 E: zero-fill from a random offset (>= 4) to the stream end (torn write)',
    'F': 'team2 F: length - 1 (truncated stream)',
    'G': 'team2 G: length + 1 (one extra random byte)',
    'H': 'team2 H: valid stream decoded at the next index (next stream srel and identity, own length)',
    'I': 'team2 I: valid stream at the next index with IDENTICAL contexts (own srel), identity of the next index',
    'J': 'team2 J: valid stream decoded with another matrix type model (w1->w3, w2->w3, w3->w1)',
    'T': 'tail magnitude change: valid re-encode with one magnitude changed in the last 5 symbols, integrity field of '
         'the ORIGINAL data (stale hash); for R0 and TAIL accepted by construction',
    'M2': 'length-table corruption, sum preserving: boundary between adjacent weight streams shifted by +-1..8 bytes',
    'N1': 'swapped adjacent weight streams (bytes and lengths consistent)',
    'N2': 'swapped weight streams j <-> random k',
    'N3': 'weight stream of another record (other expert, same index and kind)',
    'S0': 'clean scale streams (control)',
    'K': 'random single bit flip in a scale stream (5 passes over all scale streams)',
    'L': 'bit flip in the last byte of a scale stream (2 passes)',
    'M3': 'scale length boundary shift between adjacent scale streams (+-1..4 bytes)',
    'N4': 'swapped adjacent scale streams',
    'N5': 'scale stream of another record (same index)',
    'M1': 'record level: random bit flip in the LENC length section',
    'P1': 'record level: last 4096-byte page dropped',
    'P2': 'record level: torn write, zero from a random offset in the used region to the end',
    'P3': 'record level: last k = 1..4 payload bytes zeroed',
    'O': 'record level: whole valid record decoded in another expert slot (layer/expert binding)',
}
WCASES = ('W0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'T', 'M2', 'N1', 'N2', 'N3')
SCASES = ('S0', 'K', 'L', 'M3', 'N4', 'N5')
RCASES = ('M1', 'P1', 'P2', 'P3', 'O')
ALL_CASES = WCASES + SCASES + RCASES
NEED2 = ('N3', 'N5', 'O')          # need >= 2 experts
SEEDS = {c: 20261108 + 17 * i for i, c in enumerate(ALL_CASES)}


class Params:
    """Trial counts. per_w: streams per (record, matrix) for structured weight cases (team5 full: 3000 -> 24,000 per
    matrix with 8 experts); per_a: same for W0 / A (None = every stream, 46,080 per matrix with 8 experts); per_s: scale
    streams per (record, matrix, pass) (None = all 720); rper: record-level trials per expert per case."""

    def __init__(self, per_w=3000, per_a=None, per_s=None, rper=None):
        self.per_w, self.per_a, self.per_s = per_w, per_a, per_s
        self.rper = {'M1': 500, 'P1': 1, 'P2': 100, 'P3': 4, 'O': 2}
        if rper:
            self.rper.update(rper)

    def as_dict(self):
        return {'per_w': self.per_w, 'per_a': self.per_a, 'per_s': self.per_s, 'rper': dict(self.rper)}


# ------------------------------------------------------------------------------------------------ helpers
def gather(Es, r, m, j, what):
    """Stack per-position arrays: what in csym, srcn (weight streams), srel, cs, srcs (per stream blocks)."""
    r, m, j = np.asarray(r), np.asarray(m), np.asarray(j)
    first = getattr(Es[0], what)[0]
    if what in ('cs', 'srcs'):
        out = np.empty((r.size, BPS), first.dtype)
    else:
        out = np.empty((r.size,) + first.shape[1:], first.dtype)
    for ri in np.unique(r):
        for mi in range(3):
            sel = (r == ri) & (m == mi)
            if not sel.any():
                continue
            arr = getattr(Es[ri], what)[mi]
            if what in ('cs', 'srcs'):
                out[sel] = arr.reshape(NWS, BPS)[j[sel]]
            else:
                out[sel] = arr[j[sel]]
    return out


def base_table(rng, per, nrec, jmax=NWS, allstreams=False):
    rs, ms, js = [], [], []
    for r in range(nrec):
        for m in range(3):
            j = np.arange(jmax) if allstreams else np.sort(rng.choice(jmax, min(per, jmax), replace=False))
            rs.append(np.full(j.size, r))
            ms.append(np.full(j.size, m))
            js.append(j)
    r, m, j = np.concatenate(rs), np.concatenate(ms), np.concatenate(js)
    return {'r': r, 'm': m, 'j': j, 'u': rng.random(r.size), 'bit': rng.integers(0, 8, r.size),
            'v1': rng.integers(0, 256, r.size), 'v2': rng.integers(0, 256, r.size), 'u2': rng.random(r.size)}


def wtable(case, nrec, P):
    """Trial table of a weight case (deterministic from SEEDS[case], nrec and P)."""
    rng = np.random.default_rng(SEEDS[case])
    if case in ('W0', 'A'):
        B = base_table(rng, P.per_a, nrec, allstreams=P.per_a is None)
    else:
        jmax = NWS - 1 if case in ('H', 'I', 'M2', 'N1') else NWS
        B = base_table(rng, P.per_w, nrec, jmax=jmax)
    N = B['r'].size
    if case == 'M2':
        B['shift'] = rng.choice([-8, -4, -2, -1, 1, 2, 4, 8], N)
    if case == 'N2':
        k = rng.integers(0, NWS - 1, N)
        B['k'] = np.where(k >= B['j'], k + 1, k)
    return B


def stable(case, nrec, P):
    """Trial table of a scale case (deterministic from SEEDS[case], nrec and P)."""
    rng = np.random.default_rng(SEEDS[case])
    passes = {'S0': 1, 'K': 5, 'L': 2, 'M3': 1, 'N4': 1, 'N5': 1}[case]
    jmax = NSS - 1 if case in ('M3', 'N4') else NSS
    rs, ms, js = [], [], []
    for _ in range(passes):
        for r in range(nrec):
            for m in range(3):
                j = np.arange(jmax) if P.per_s is None else np.sort(rng.choice(jmax, min(P.per_s, jmax), replace=False))
                rs.append(np.full(j.size, r)); ms.append(np.full(j.size, m)); js.append(j)
    B = {'r': np.concatenate(rs), 'm': np.concatenate(ms), 'j': np.concatenate(js)}
    N = B['r'].size
    B['u'] = rng.random(N); B['bit'] = rng.integers(0, 8, N)
    B['shift'] = rng.choice([-4, -2, -1, 1, 2, 4], N)
    return B


def new_jobs():
    return {'trial': [], 'r': [], 'm': [], 'j': [], 'mm': [], 'sr_r': [], 'sr_m': [], 'sr_j': [], 'data': [], 'lens': [],
            'unchanged': []}


def add_job(J, t, r, m, j, data, ln, mm=None, sr=None, unchanged=False):
    J['trial'].append(t); J['r'].append(r); J['m'].append(m); J['j'].append(j)
    J['mm'].append(m if mm is None else mm)
    sr = (r, m, j) if sr is None else sr
    J['sr_r'].append(sr[0]); J['sr_m'].append(sr[1]); J['sr_j'].append(sr[2])
    J['data'].append(data); J['lens'].append(ln); J['unchanged'].append(unchanged)


# ------------------------------------------------------------------------------------------------ weight cases (path W)
def build_wjobs(case, Es, dk, B, shift=None):
    J = new_jobs()
    N = B['r'].size
    for t in range(N):
        r, m, j = int(B['r'][t]), int(B['m'][t]), int(B['j'][t])
        C = Es[r].recs[dk]
        b = C.wbytes(m, j)
        n = b.size
        if case == 'W0':
            add_job(J, t, r, m, j, b, n)
        elif case == 'A':
            d = b.copy(); p = int(B['u'][t] * n); d[p] ^= np.uint8(1 << int(B['bit'][t])); add_job(J, t, r, m, j, d, n)
        elif case == 'B':
            d = b.copy(); d[n - 1] ^= np.uint8(1 << int(B['bit'][t])); add_job(J, t, r, m, j, d, n)
        elif case == 'C':
            d = b.copy(); p = int(B['u'][t] * 4); d[p] ^= np.uint8(1 << int(B['bit'][t])); add_job(J, t, r, m, j, d, n)
        elif case == 'D':
            d = b.copy(); p = int(B['u'][t] * (n - 1)); d[p] = B['v1'][t]; d[p + 1] = B['v2'][t]
            add_job(J, t, r, m, j, d, n, unchanged=bool((d == b).all()))
        elif case == 'E':
            d = b.copy(); p = 4 + int(B['u'][t] * (n - 4)); d[p:] = 0
            add_job(J, t, r, m, j, d, n, unchanged=bool((d == b).all()))
        elif case == 'F':
            add_job(J, t, r, m, j, b[:n - 1].copy(), n - 1)
        elif case == 'G':
            d = np.concatenate([b, np.array([B['v1'][t]], np.uint8)]); add_job(J, t, r, m, j, d, n + 1)
        elif case == 'H':           # stream j decoded at position j+1 (srel, identity of j+1)
            add_job(J, t, r, m, j + 1, b, n)
        elif case == 'I':           # stream j at position j+1 with IDENTICAL contexts (own srel), identity j+1
            add_job(J, t, r, m, j + 1, b, n, sr=(r, m, j))
        elif case == 'J':           # stream decoded with another matrix type's model, own srel, identity (m', j)
            mo = OTHER_M[m]
            add_job(J, t, r, mo, j, b, n, mm=mo, sr=(r, m, j))
        elif case == 'M2':          # sum-preserving boundary shift between adjacent streams j, j+1
            b2 = C.wbytes(m, j + 1)
            d = int(shift[t])
            cat = np.concatenate([b, b2])
            add_job(J, t, r, m, j, cat[:n + d], n + d)
            add_job(J, t, r, m, j + 1, cat[n + d:], b2.size - d)
        elif case in ('N1', 'N2'):  # swap streams j and k (bytes + lengths consistent)
            k = j + 1 if case == 'N1' else int(B['k'][t])
            bk = C.wbytes(m, k)
            add_job(J, t, r, m, j, bk, bk.size)
            add_job(J, t, r, m, k, b, n)
        elif case == 'N3':          # same-index stream of another record (other expert, same kind)
            r2 = (r + 1) % len(Es)
            b2 = Es[r2].recs[dk].wbytes(m, j)
            add_job(J, t, r, m, j, b2, b2.size)
        else:
            raise ValueError(case)
    return J


def build_tail_jobs(Es, dk, B, tab):
    """Case T: valid re-encoded stream with one changed magnitude among the last 5 symbols (the TAIL positions),
    integrity field (hash in x0 / record digest) left as computed for the ORIGINAL data; TAIL: the tail is the data."""
    N = B['r'].size
    orig = gather(Es, B['r'], B['m'], B['j'], 'csym')                    # [N, 2048] coded
    mod = orig.copy()
    p = 2043 + (B['u'] * 5).astype(np.int64)
    old = mod[np.arange(N), p]
    oldmag = old & 7
    newmag = (oldmag + 1 + (B['u2'] * 7).astype(np.int64)) % 8         # uniform among the 7 other magnitudes
    mod[np.arange(N), p] = (old & 8) | newmag
    cs = gather(Es, B['r'], B['m'], B['j'], 'cs')                       # coded scales [N, 64]
    f = W.block_features(mod.ravel(), cs.ravel().astype(np.uint8), A.ANCHOR0)
    base = f['base'].reshape(N, W_CHUNK).astype(np.int64)
    idx = ((B['m'][:, None].astype(np.int64) * W.NCTX + base) << 4) | mod
    FT = tab.wf16[idx].T.copy()
    CT = tab.wc16[idx].T.copy()
    lay = np.array([Es[r].layer for r in B['r']])
    exp = np.array([Es[r].expert for r in B['r']])
    nt = 5 if dk == 'fg0dt' else 0
    if dk == 'fg0d':
        x0 = np.full(N, A.L, np.int64)
    elif dk == 'fg0de':
        seed = A.fnv_seeds(lay, exp, tab.model_id, 1, B['m'], B['j'])
        x0 = A.L + W.hash_symbols(seed, orig.T.copy()).astype(np.int64)
    elif dk == 'fg0dh':
        h = A.h32_streams(tab.model_id, lay, exp, A.gw(B['m'], B['j']), B['m'], 0x1C, A.pack_nib(orig))
        x0 = A.L + (h & A.U32(A.MASK23)).astype(np.int64)
    else:
        x0 = A.L + sum(mod[:, W_CHUNK - nt + q].astype(np.int64) << (4 * q) for q in range(nt))
    pay, lens = W.rans_encode_lockstep(FT[:W_CHUNK - nt], CT[:W_CHUNK - nt], x0)
    offs = np.zeros(N, np.int64)
    offs[1:] = np.cumsum(lens)[:-1]
    J = new_jobs()
    for t in range(N):
        add_job(J, t, int(B['r'][t]), int(B['m'][t]), int(B['j'][t]), pay[offs[t]:offs[t] + lens[t]], int(lens[t]))
    return J, {'pos': p, 'oldmag': oldmag, 'newmag': newmag}


def finish_trials(Es, dk, J, res, outs, ntrial, tab, mat_t):
    """Per-trial outcome per rule + damage of accepted trials. mat_t: matrix type of each trial (base table).
    Returns {rule: dict}."""
    trial = np.asarray(J['trial'])
    r, m, j = (np.asarray(J[k]) for k in ('r', 'm', 'j'))
    starts = np.flatnonzero(np.r_[True, trial[1:] != trial[:-1]])
    assert np.array_equal(trial[starts], np.arange(ntrial))
    exp_c = gather(Es, r, m, j, 'csym')
    wrong_job = (outs != exp_c).any(1)
    unchanged = np.asarray(J['unchanged'])
    out = {}
    for rule in A.RULES_OF_DEC[dk]:
        if rule == 'TAILD':
            okj = res['TAIL']
            hn = res['TAILD_h']
            hc = np.array([Es[a].hw_d[b, c] for a, b, c in zip(r, m, j)], A.U32)
            ok_t = np.logical_and.reduceat(okj, starts) & (np.bitwise_xor.reduceat(hn, starts) == np.bitwise_xor.reduceat(hc, starts))
        else:
            ok_t = np.logical_and.reduceat(res[rule], starts)
        wrong_t = np.logical_or.reduceat(wrong_job, starts)
        acc = np.flatnonzero(ok_t)
        dm = {k: np.zeros(ntrial, np.float64) for k in A.DAMAGE_KEYS}
        inv_spec = np.zeros(ntrial, bool)
        inv_cr = np.zeros(ntrial, bool)
        if acc.size:
            jsel = np.flatnonzero(np.isin(trial, acc))
            for a in range(0, jsel.size, 4096):
                q = jsel[a:a + 4096]
                d = A.damage(outs[q], gather(Es, r[q], m[q], j[q], 'cs'), exp_c[q], gather(Es, r[q], m[q], j[q], 'cs'),
                             gather(Es, r[q], m[q], j[q], 'srcn'), gather(Es, r[q], m[q], j[q], 'srcs'))
                tq = trial[q]
                for k in A.DAMAGE_KEYS:
                    v = d[k].astype(np.float64)
                    if k in ('err_units', 'err_abs', 'nan_scale'):
                        np.maximum.at(dm[k], tq, v)
                    elif k == 'sign_only':
                        continue
                    else:
                        np.add.at(dm[k], tq, v)
                np.logical_or.at(inv_spec, tq, d['inv_spec'])
                np.logical_or.at(inv_cr, tq, d['inv_cr'])
            dm['sign_only'] = (dm['nib_diff'] > 0) & (dm['scale_diff'] == 0)
            # sign-only per trial: every differing output nibble differs only by bit 3
            so = np.zeros(ntrial, bool)
            for a in range(0, jsel.size, 4096):
                q = jsel[a:a + 4096]
                sn = gather(Es, r[q], m[q], j[q], 'srcn')
                on, osc, _, _, _ = A.inverse(outs[q], gather(Es, r[q], m[q], j[q], 'cs'))
                bad = ((on != sn) & ((on ^ sn) != 8)).any(1) | (osc != gather(Es, r[q], m[q], j[q], 'srcs')).any(1)
                np.logical_or.at(so, trial[q], bad)
            dm['sign_only'] = dm['sign_only'] & ~so
        out[rule] = summarize_rule(ok_t, wrong_t, inv_spec, inv_cr, dm, np.asarray(mat_t),
                                   np.logical_and.reduceat(unchanged, starts))
    return out


def summarize_rule(ok_t, wrong_t, inv_spec, inv_cr, dm, mat_t, unchanged_t):
    """Counts per matrix type and overall; damage records for accepted-and-wrong trials (spec decoder)."""
    acc_spec = ok_t & ~inv_spec
    acc_cr = ok_t & ~inv_cr
    silent = acc_spec & wrong_t
    S = {}
    for key, sel in (('w1', mat_t == 0), ('w2', mat_t == 1), ('w3', mat_t == 2), ('all', np.ones(mat_t.size, bool))):
        n = int(sel.sum())
        k = int(silent[sel].sum())
        lo, hi = A.ci_fast(k, n)
        S[key] = {'trials': n, 'unchanged_trials': int(unchanged_t[sel].sum()), 'stream_rules_pass': int(ok_t[sel].sum()),
                  'rejected_by_spec_inverse_only': int((ok_t & inv_spec)[sel].sum()),
                  'accepted_spec': int(acc_spec[sel].sum()), 'accepted_ceilref': int(acc_cr[sel].sum()),
                  'accepted_identical_output': int((acc_spec & ~wrong_t)[sel].sum()),
                  'silent': k, 'silent_rate': k / n if n else None, 'silent_ci95': [lo, hi],
                  'silent_ceilref': int((acc_cr & wrong_t)[sel].sum()),
                  'silent_with_block_flip': int((silent & (dm['block_flips'] > 0))[sel].sum())}
    idx = np.flatnonzero(silent)
    S['silent_damage'] = {'trial': idx.tolist(), 'mat': mat_t[idx].tolist(),
                          **{k: dm[k][idx].tolist() for k in A.DAMAGE_KEYS}}
    return S


def run_wcase(case, Es, dks, tab, P, log=print):
    B = wtable(case, len(Es), P)
    N = B['r'].size
    res_all = {'case': case, 'seed': SEEDS[case], 'trials': int(N), 'by_kind': {}}
    for dk in dks:
        t0 = time.time()
        shift = None
        if case == 'M2':     # keep both lengths >= 4 (else the trial is not a valid table): clamp per kind, on a COPY
            shift = B['shift'].copy()
            for t in range(N):
                C = Es[B['r'][t]].recs[dk]
                g = B['m'][t] * NWS + B['j'][t]
                d = int(shift[t])
                if C.wlens[g] + d < 4 or C.wlens[g + 1] - d < 4:
                    shift[t] = 1 if d < 0 else -1
        extra = None
        if case == 'T':
            J, extra = build_tail_jobs(Es, dk, B, tab)
        else:
            J = build_wjobs(case, Es, dk, B, shift)
        r, m, j = (np.asarray(J[k]) for k in ('r', 'm', 'j'))
        lay = np.array([Es[a].layer for a in r])
        exp = np.array([Es[a].expert for a in r])
        srel = gather(Es, J['sr_r'], J['sr_m'], J['sr_j'], 'srel')
        ids = {'lay': lay, 'exp': exp, 'm': m, 'j': j}
        res, outs = A.run_lanes(dk, 'w', J['data'], np.asarray(J['lens']), srel, np.asarray(J['mm']), ids, tab)
        t1 = time.time()
        rr = finish_trials(Es, dk, J, res, outs, N, tab, B['m'])
        if extra is not None:
            for rule in rr:
                ti = np.asarray(rr[rule]['silent_damage']['trial'], np.int64)
                rr[rule]['silent_damage']['tail_pos'] = extra['pos'][ti].tolist()
                rr[rule]['silent_damage']['oldmag'] = extra['oldmag'][ti].tolist()
                rr[rule]['silent_damage']['newmag'] = extra['newmag'][ti].tolist()
        for rule, v in rr.items():
            v['seconds_decode'] = round(t1 - t0, 1)
            v['seconds_total'] = round(time.time() - t0, 1)
            res_all['by_kind'][rule] = v
            log(f"{case} {rule} " + short(v) + f" {time.time() - t0:.0f}s")
    return res_all


def short(v):
    a = v['all']
    return (f"trials {a['trials']} accepted {a['accepted_spec']} silent {a['silent']} "
            f"(w1 {v['w1']['silent']}, w2 {v['w2']['silent']}, w3 {v['w3']['silent']}) "
            f"rate {a['silent_rate']:.3e} CI95 [{a['silent_ci95'][0]:.2e}, {a['silent_ci95'][1]:.2e}] "
            f"block_flip {a['silent_with_block_flip']}")


# ------------------------------------------------------------------------------------------------ scale cases (path S)
def run_scase(case, Es, dks, tab, P, log=print):
    B = stable(case, len(Es), P)
    N = B['r'].size
    res_all = {'case': case, 'seed': SEEDS[case], 'trials': int(N), 'by_kind': {}}
    for dk in dks:
        t0 = time.time()
        J = {'trial': [], 'r': [], 'm': [], 'j': [], 'data': [], 'lens': []}

        def add(t, r, m, j, d, ln):
            J['trial'].append(t); J['r'].append(r); J['m'].append(m); J['j'].append(j); J['data'].append(d); J['lens'].append(ln)
        for t in range(N):
            r, m, j = int(B['r'][t]), int(B['m'][t]), int(B['j'][t])
            C = Es[r].recs[dk]
            b = C.sbytes(m, j)
            n = b.size
            if case == 'S0':
                add(t, r, m, j, b, n)
            elif case == 'K':
                d = b.copy(); d[int(B['u'][t] * n)] ^= np.uint8(1 << int(B['bit'][t])); add(t, r, m, j, d, n)
            elif case == 'L':
                d = b.copy(); d[n - 1] ^= np.uint8(1 << int(B['bit'][t])); add(t, r, m, j, d, n)
            elif case == 'M3':
                b2 = C.sbytes(m, j + 1)
                sh = int(B['shift'][t])
                if n + sh < 4 or b2.size - sh < 4:
                    sh = 1 if sh < 0 else -1
                cat = np.concatenate([b, b2])
                add(t, r, m, j, cat[:n + sh], n + sh)
                add(t, r, m, j + 1, cat[n + sh:], b2.size - sh)
            elif case == 'N4':
                b2 = C.sbytes(m, j + 1)
                add(t, r, m, j, b2, b2.size)
                add(t, r, m, j + 1, b, n)
            elif case == 'N5':
                b2 = Es[(r + 1) % len(Es)].recs[dk].sbytes(m, j)
                add(t, r, m, j, b2, b2.size)
        r, m, j = (np.asarray(J[k]) for k in ('r', 'm', 'j'))
        ids = {'lay': np.array([Es[a].layer for a in r]), 'exp': np.array([Es[a].expert for a in r]), 'm': m, 'j': j}
        res, outs = A.run_lanes(dk, 's', J['data'], np.asarray(J['lens']), None, m, ids, tab)
        trial = np.asarray(J['trial'])
        starts = np.flatnonzero(np.r_[True, trial[1:] != trial[:-1]])
        rule0 = A.RULES_OF_DEC[dk][0]
        ok1 = np.logical_and.reduceat(res[rule0], starts)
        t1 = time.time()
        # ---- stage 2 per stage-1-accepted trial
        st2 = stage2(Es, dk, B, J, res, outs, starts, ok1, tab)
        mat_t = B['m']
        for rule in A.RULES_OF_DEC[dk]:
            ok_t = st2['ok'][rule]
            v = summarize_rule(ok_t, st2['wrong'], st2['inv_spec'], st2['inv_cr'], st2['dm'], mat_t, np.zeros(N, bool))
            for key in ('w1', 'w2', 'w3', 'all'):
                sel = slice(None) if key == 'all' else (mat_t == ('w1', 'w2', 'w3').index(key))
                v[key]['scale_stream_rules_pass'] = int(ok1[sel].sum()) if rule != 'TAILD' else None
                v[key]['rejected_by_wlength_check'] = int(st2['rej_len'][sel].sum())
                v[key]['rejected_in_weight_streams'] = int((st2['rej_w'][rule])[sel].sum())
            v['seconds_stage1'] = round(t1 - t0, 1)
            v['seconds_total'] = round(time.time() - t0, 1)
            res_all['by_kind'][rule] = v
            log(f"{case} {rule} " + short(v) + f" {time.time() - t0:.0f}s")
    return res_all


def stage2(Es, dk, B, J, res, outs, starts, ok1, tab):
    N = B['r'].size
    rules = A.RULES_OF_DEC[dk]
    rule0 = rules[0]
    trial = np.asarray(J['trial'])
    jm, jj = np.asarray(J['m']), np.asarray(J['j'])
    ok = {rl: np.zeros(N, bool) for rl in rules}
    rej_w = {rl: np.zeros(N, bool) for rl in rules}
    rej_len = np.zeros(N, bool)
    wrong = np.zeros(N, bool)
    inv_spec = np.zeros(N, bool)
    inv_cr = np.zeros(N, bool)
    dm = {k: np.zeros(N, np.float64) for k in A.DAMAGE_KEYS}
    pend = []        # (trial, weight job list, affected stream list, cs2 per affected stream)
    W2 = {'trial': [], 'r': [], 'm': [], 'j': [], 'data': [], 'lens': [], 'srel': []}
    for t in np.flatnonzero(ok1):
        r, m = int(B['r'][t]), int(B['m'][t])
        E = Es[r]
        C = E.recs[dk]
        q = np.arange(starts[t], starts[t + 1] if t + 1 < N else trial.size)
        cs2 = E.cs[m].copy()
        for qi in q:
            js = jj[qi]
            cs2[js * S_CHUNK:(js + 1) * S_CHUNK] = outs[qi]
        chb = np.flatnonzero(cs2 != E.cs[m])
        if chb.size == 0:            # scales identical: weight streams unchanged
            ok[rule0][t] = True
            if 'TAILD' in rules:
                hn = np.bitwise_xor.reduce(res['TAILD_h'][q])
                hc = np.bitwise_xor.reduce(E.hs_d[m, jj[q]])
                ok['TAILD'][t] = hn == hc
            continue
        wst = np.unique(chb // BPS)                                   # weight streams with a changed block scale
        H = C.H
        d_new = np.clip(cs2.reshape(NWS, BPS)[wst] - H['meds'][m], -W.DBC, W.DBC) + W.DBC
        d_old = np.clip(E.cs[m].reshape(NWS, BPS)[wst] - H['meds'][m], -W.DBC, W.DBC) + W.DBC
        pn = W.pred_from_q(W.predict_wlen_q(d_new, tab.etab[m]), H['alphas'][m])
        po = W.pred_from_q(W.predict_wlen_q(d_old, tab.etab[m]), H['alphas'][m])
        g0 = m * NWS
        wl = C.wlens[g0 + wst] + (pn - po)
        if (wl < 4).any() or (wl > 65535).any() or wl.sum() != C.wlens[g0 + wst].sum():
            rej_len[t] = True
            continue
        # new lengths of matrix m; offsets shift between the first and last changed stream
        wl_m = C.wlens[g0:g0 + NWS].copy()
        wl_m[wst] = wl
        st_m = C.wstart[g0] + np.r_[0, np.cumsum(wl_m)[:-1]]
        srel2 = A.srel_of(cs2, H['meds'][m]).reshape(NWS, BPS)
        chg = (st_m != C.wstart[g0:g0 + NWS]) | (wl_m != C.wlens[g0:g0 + NWS]) | (srel2 != E.srel[m]).any(1)
        jobs = np.flatnonzero(chg)
        aff = np.union1d(wst, jobs)
        pend.append((t, r, m, jobs, aff, cs2))
        for k in jobs:
            W2['trial'].append(t); W2['r'].append(r); W2['m'].append(m); W2['j'].append(int(k))
            W2['data'].append(C.rec[st_m[k]:st_m[k] + wl_m[k]]); W2['lens'].append(int(wl_m[k])); W2['srel'].append(srel2[k])
    # ---- stage 2 batch decode
    if W2['trial']:
        r2, m2, j2 = (np.asarray(W2[k]) for k in ('r', 'm', 'j'))
        ids = {'lay': np.array([Es[a].layer for a in r2]), 'exp': np.array([Es[a].expert for a in r2]), 'm': m2, 'j': j2}
        res2, outs2 = A.run_lanes(dk, 'w', W2['data'], np.asarray(W2['lens']), np.stack(W2['srel']), m2, ids, tab)
        tr2 = np.asarray(W2['trial'])
    else:
        res2, outs2, tr2 = {}, None, np.zeros(0, np.int64)
        j2 = np.zeros(0, np.int64)
    pos2 = {}
    for i, t in enumerate(tr2.tolist()):
        pos2.setdefault(t, []).append(i)
    for (t, r, m, jobs, aff, cs2) in pend:
        E = Es[r]
        q = np.arange(starts[t], starts[t + 1] if t + 1 < N else trial.size)
        ii = pos2.get(t, [])
        okw = bool(res2[rule0][ii].all()) if ii else True
        ok[rule0][t] = okw
        rej_w[rule0][t] = not okw
        if 'TAILD' in rules:
            hn = np.bitwise_xor.reduce(res['TAILD_h'][q]) ^ (np.bitwise_xor.reduce(res2['TAILD_h'][ii]) if ii else A.U32(0))
            hc = np.bitwise_xor.reduce(E.hs_d[m, jj[q]]) ^ (np.bitwise_xor.reduce(E.hw_d[m, jobs]) if len(jobs) else A.U32(0))
            ok['TAILD'][t] = okw and hn == hc
            rej_w['TAILD'][t] = not okw
        if not okw:
            continue
        dec = E.csym[m][aff].copy()
        jpos = {int(k): n for n, k in enumerate(aff)}
        for i in ii:
            dec[jpos[int(j2[i])]] = outs2[i]
        csb = cs2.reshape(NWS, BPS)[aff]
        d = A.damage(dec, csb, E.csym[m][aff], E.cs[m].reshape(NWS, BPS)[aff], E.srcn[m][aff], E.srcs[m].reshape(NWS, BPS)[aff])
        ag = R.agg_damage(d)
        for k in A.DAMAGE_KEYS:
            dm[k][t] = float(ag[k])
        inv_spec[t] = bool(d['inv_spec'].any())
        inv_cr[t] = bool(d['inv_cr'].any())
        wrong[t] = ag['coded_sym_diff'] > 0 or ag['coded_scale_diff'] > 0
    # trials accepted at stage 1 with unchanged scales: wrong = False (identical output)
    return {'ok': ok, 'rej_w': rej_w, 'rej_len': rej_len, 'wrong': wrong, 'inv_spec': inv_spec, 'inv_cr': inv_cr, 'dm': dm}


# ------------------------------------------------------------------------------------------------ record-level cases (path R)
def run_rcase(case, Es, kinds, tab, P, log=print):
    rng = np.random.default_rng(SEEDS[case])
    res_all = {'case': case, 'seed': SEEDS[case], 'by_kind': {}}
    nper = P.rper[case]
    U = rng.random((len(Es), nper))
    BIT = rng.integers(0, 8, (len(Es), nper))
    for kind in kinds:
        rule = R.RULE_OF_KIND[kind]
        t0 = time.time()
        rows = []
        for ri, E in enumerate(Es):
            C = E.recs[kind]
            H = C.H
            for t in range(nper):
                bind = None
                if case == 'M1':          # random bit in the LENC length section
                    rec = C.rec.copy()
                    p = H['hlen'] + int(U[ri, t] * H['ltb'])
                    rec[p] ^= np.uint8(1 << int(BIT[ri, t]))
                    info = {'pos_rel': p - H['hlen']}
                elif case == 'P1':        # last 4096-B page dropped
                    rec = C.rec[:-4096].copy()
                    info = {}
                elif case == 'P2':        # torn write: zero from a random offset in the used region to the end
                    rec = C.rec.copy()
                    p = H['hlen'] + int(U[ri, t] * (H['used_end'] - H['hlen']))
                    rec[p:] = 0
                    info = {'pos': p, 'used_end': H['used_end']}
                elif case == 'P3':        # zero the last k = 1..4 payload bytes (last weight stream's last bytes)
                    rec = C.rec.copy()
                    k = t % 4 + 1
                    rec[H['used_end'] - k:H['used_end']] = 0
                    info = {'k': k, 'unchanged': bool((rec == C.rec).all())}
                elif case == 'O':         # whole valid record decoded at another expert's slot
                    r2 = (ri + (1 if t % 2 == 0 else -1)) % len(Es)
                    rec = C.rec
                    bind = (Es[r2].layer, Es[r2].expert)
                    info = {'slot': Es[r2].file}
                else:
                    raise ValueError(case)
                d = R.decode_r(E, kind, rec, tab, bind=bind)
                row = {'rec': E.file, 'status': d['status'], 'stage': d['stage'], 'msg': d['msg'],
                       'ceilref_reject': d['ceilref_reject'], 'n_dec_s': d['n_dec_s'], 'n_dec_w': d['n_dec_w'], **info}
                if d['status'] == 'accepted':
                    if case == 'O':
                        row['damage'] = whole_record_damage(E, Es[r2])
                    else:
                        row['damage'] = d['damage']
                rows.append(row)
        n = len(rows)
        acc = sum(1 for x in rows if x['status'] == 'accepted')
        silent = sum(1 for x in rows if x['status'] == 'accepted' and x['damage']['coded_sym_diff'] + x['damage']['coded_scale_diff']
                     + x['damage'].get('nib_diff', 0) > 0)
        bflip = sum(1 for x in rows if x['status'] == 'accepted' and x['damage']['block_flips'] > 0)
        stages = {}
        for x in rows:
            stages[str(x['stage'])] = stages.get(str(x['stage']), 0) + 1
        lo, hi = A.ci_fast(silent, n)
        res_all['by_kind'][rule] = {'trials': n, 'accepted_spec': acc, 'silent': silent, 'silent_rate': silent / n if n else None,
                                    'silent_ci95': [lo, hi], 'silent_with_block_flip': bflip, 'stages': stages, 'rows': rows,
                                    'seconds': round(time.time() - t0, 1)}
        log(f"{case} {rule} trials {n} accepted {acc} silent {silent} CI95 [{lo:.2e}, {hi:.2e}] stages {stages} "
            f"{time.time() - t0:.0f}s")
    return res_all


def whole_record_damage(E, Y):
    """Damage when record E is accepted in Y's slot: output = E's source, reference = Y's source."""
    tot = None
    for m in range(3):
        for a in range(0, NWS, 2048):
            sl = slice(a, min(NWS, a + 2048))
            d = A.damage(E.csym[m][sl], E.cs[m].reshape(NWS, BPS)[sl], Y.csym[m][sl], Y.cs[m].reshape(NWS, BPS)[sl],
                         Y.srcn[m][sl], Y.srcs[m].reshape(NWS, BPS)[sl])
            ag = R.agg_damage(d)
            if tot is None:
                tot = ag
            else:
                for k in ag:
                    if k in ('err_units', 'err_abs'):
                        tot[k] = max(tot[k], ag[k])
                    elif k == 'nan_scale':
                        tot[k] = tot[k] or ag[k]
                    elif k == 'sign_only':
                        tot[k] = False
                    else:
                        tot[k] += ag[k]
    tot['sign_only'] = False
    return tot


def run_case(case, Es, dks, tab, P, log=print):
    """Run one case for decoder kinds dks (path R uses the record kinds of dks, plus fg0dtD for fg0dt)."""
    if case in NEED2 and len(Es) < 2:
        return {'case': case, 'skipped': 'needs >= 2 experts', 'by_kind': {}}
    if case in WCASES:
        return run_wcase(case, Es, dks, tab, P, log)
    if case in SCASES:
        return run_scase(case, Es, dks, tab, P, log)
    if case in RCASES:
        rk = [k for k in A.KINDS if k in dks or (k == 'fg0dtD' and 'fg0dt' in dks)]
        return run_rcase(case, Es, rk, tab, P, log)
    raise ValueError(case)
