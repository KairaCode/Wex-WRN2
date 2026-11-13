"""Test kit / audit (from team5 audit-ceil summarize.py): silent-acceptance rates with exact 95% CIs per
rule x case x matrix type, and damage histograms of the silently accepted corruptions (after the spec ceil inverse)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import audit_lib as A

RULES = ('R0', 'HASH', 'HASH32', 'TAIL', 'TAILD')
BINS = {
    'coded_sym_diff': [(0, 0), (1, 1), (2, 5), (6, 31), (32, 2047), (2048, 1e18)],
    'nib_diff': [(0, 0), (1, 1), (2, 3), (4, 31), (32, 32), (33, 64), (65, 2048), (2049, 1e18)],
    'val_diff': [(0, 0), (1, 1), (2, 3), (4, 31), (32, 2048), (2049, 1e18)],
    'scale_diff': [(0, 0), (1, 1), (2, 8), (9, 1e18)],
    'block_flips': [(0, 0), (1, 1), (2, 1e18)],
}
EBINS = [0, 0.5, 1, 2, 4, 8, 12, np.inf]


def hist(vals, bins):
    v = np.asarray(vals, np.float64)
    return {f'{int(a)}' if a == b else (f'{int(a)}-{int(b)}' if b < 1e17 else f'>={int(a)}'): int(((v >= a) & (v <= b)).sum())
            for a, b in bins}


def ehist(vals):
    v = np.asarray(vals, np.float64)
    out = {'0': int((v == 0).sum())}
    for a, b in zip(EBINS[:-1], EBINS[1:]):
        out[f'({a},{b}]'] = int(((v > a) & (v <= b)).sum())
    return out


def damage_hist(sd, sel=None):
    if not sd['trial']:
        return None
    idx = np.arange(len(sd['trial'])) if sel is None else sel
    if len(idx) == 0:
        return None
    h = {k: hist(np.asarray(sd[k])[idx], b) for k, b in BINS.items()}
    h['err_units'] = ehist(np.asarray(sd['err_units'])[idx])
    h['sign_only'] = int(np.asarray(sd['sign_only'], bool)[idx].sum())
    h['nan_scale'] = int(np.asarray(sd['nan_scale'], bool)[idx].sum())
    h['n'] = int(len(idx))
    h['max_err_units'] = float(np.max(np.asarray(sd['err_units'])[idx]))
    h['max_err_abs'] = float(np.max(np.asarray(sd['err_abs'])[idx]))
    h['weights_differing_total'] = int(np.asarray(sd['val_diff'])[idx].sum())
    h['scales_differing_total'] = int(np.asarray(sd['scale_diff'])[idx].sum())
    h['blocks_flipped_total'] = int(np.asarray(sd['block_flips'])[idx].sum())
    h['trials_with_block_flip'] = int((np.asarray(sd['block_flips'])[idx] > 0).sum())
    h['trials_value_damage_eq_1_weight'] = int((np.asarray(sd['val_diff'])[idx] == 1).sum())
    h['trials_nibble_damage_ge_32'] = int((np.asarray(sd['nib_diff'])[idx] >= 32).sum())
    return h


def summarize(M):
    """M: {case: run_case output}. Returns (summary dict, text lines)."""
    S = {}
    lines = []
    for case in M:
        J = M[case]
        if J.get('skipped'):
            S[case] = {'skipped': J['skipped']}
            lines.append(f"{case:3s} skipped: {J['skipped']}")
            continue
        C = {}
        for rule in RULES:
            if rule not in J['by_kind']:
                continue
            v = J['by_kind'][rule]
            if 'rows' in v:                      # record-level case
                rows = v['rows']
                acc = [x for x in rows if x['status'] == 'accepted']
                C[rule] = {'all': {k: v[k] for k in ('trials', 'accepted_spec', 'silent', 'silent_rate', 'silent_ci95',
                                                     'silent_with_block_flip')},
                           'stages': v['stages']}
                dm = [x['damage'] for x in acc if x['damage']['coded_sym_diff'] + x['damage']['coded_scale_diff'] + x['damage']['nib_diff'] > 0]
                if dm:
                    sd = {k: [d[k] for d in dm] for k in A.DAMAGE_KEYS}
                    sd['trial'] = list(range(len(dm)))
                    C[rule]['damage_hist'] = damage_hist(sd)
                rate = v['silent_rate'] if v['silent_rate'] is not None else float('nan')
                lines.append(f"{case:3s} {rule:6s} all  {v['trials']:7d} silent {v['silent']:6d} rate {rate:.3e} "
                             f"CI [{v['silent_ci95'][0]:.2e},{v['silent_ci95'][1]:.2e}] stages {v['stages']}")
                continue
            C[rule] = {}
            for mk in ('w1', 'w2', 'w3', 'all'):
                C[rule][mk] = dict(v[mk])
                x = v[mk]
                rate = x['silent_rate'] if x['silent_rate'] is not None else float('nan')
                lines.append(f"{case:3s} {rule:6s} {mk:4s} {x['trials']:7d} pass {x['stream_rules_pass']:6d} acc {x['accepted_spec']:6d} "
                             f"silent {x['silent']:6d} rate {rate:.3e} CI [{x['silent_ci95'][0]:.2e},{x['silent_ci95'][1]:.2e}] "
                             f"blockflip {x['silent_with_block_flip']}"
                             + (f" lenrej {x.get('rejected_by_wlength_check')}" if 'rejected_by_wlength_check' in x else ''))
            sd = v['silent_damage']
            C[rule]['damage_hist'] = damage_hist(sd)
            mats = np.asarray(sd['mat'])
            C[rule]['damage_hist_by_mat'] = {mk: damage_hist(sd, np.flatnonzero(mats == i)) for i, mk in enumerate(('w1', 'w2', 'w3'))}
            if case == 'T' and sd['trial']:
                tp = np.asarray(sd['tail_pos'])
                bf = np.asarray(sd['block_flips']) > 0
                C[rule]['T_block_flip_by_tail_pos'] = {int(p): [int((tp == p).sum()), int(((tp == p) & bf).sum())]
                                                       for p in np.unique(tp)}
                om, nm = np.asarray(sd['oldmag']), np.asarray(sd['newmag'])
                C[rule]['T_block_flip_by_mag_change'] = {f'{a}->{b}': int(((om == a) & (nm == b) & bf).sum())
                                                         for a in range(8) for b in range(8) if ((om == a) & (nm == b)).any()}
        S[case] = C
    return S, lines
