"""Test kit / audit (from team5 audit-ceil validate.py): cross-check the stream-level engines against record-level decoding.

For cases A (weight-stream bit flips) and K (scale-stream bit flips): regenerate the paired trial tables (cases.wtable /
cases.stable, same seeds and Params), take up to n_each silent and n_each non-silent trials per rule from the case
results, apply the same bit flip to the FULL record bytes and decode with audit_rec.decode_r (record-level, unchanged
streams skipped). Agreement required on silent / not silent and on the damage numbers. Then, for kinds fg0d / fg0de /
fg0dt, up to n_full + n_full of those patched records are decoded with the ORIGINAL ceil-ref decode_record (byte copy;
strict inverse) and with decode_r(use_clean=False) (full decoder).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import audit_lib as A
import audit_rec as R
import cases as CS
from audit_lib import W, NWS, NSS

CMP = ('coded_sym_diff', 'nib_diff', 'val_diff', 'scale_diff', 'block_flips')


def validate(Es, tab, P, results, n_each=25, n_full=3, log=print):
    """results: {case: run_case output} with cases 'A' and/or 'K'. Returns dict with per (case, rule) agreement."""
    out = {'n_each': n_each, 'n_full': n_full, 'cases': {}, 'all_agree': True, 'full_consistent': True}
    for case in ('A', 'K'):
        if case not in results or 'by_kind' not in results[case] or not results[case]['by_kind']:
            continue
        B = CS.wtable('A', len(Es), P) if case == 'A' else CS.stable('K', len(Es), P)
        N = B['r'].size
        J = results[case]
        pick_rng = np.random.default_rng(7)
        out['cases'][case] = {}
        for rule, kind in A.RULE_KIND.items():
            if rule not in J['by_kind'] or kind not in Es[0].recs:
                continue
            sd = J['by_kind'][rule]['silent_damage']
            sil = np.asarray(sd['trial'], np.int64)
            dmg = {k: np.asarray(sd[k]) for k in CMP}
            pos_of = {int(t): n for n, t in enumerate(sil)}
            ns = pick_rng.choice(sil, min(n_each, sil.size), replace=False) if sil.size else np.zeros(0, np.int64)
            non = np.setdiff1d(np.arange(N), sil)
            nn = pick_rng.choice(non, min(n_each, non.size), replace=False)
            rows, agree, full_rows = [], 0, []
            t0 = time.time()
            for t in np.concatenate([ns, nn]):
                r, m, j = int(B['r'][t]), int(B['m'][t]), int(B['j'][t])
                E = Es[r]
                C = E.recs[kind]
                rec = C.rec.copy()
                if case == 'A':
                    n = int(C.wlens[m * NWS + j])
                    p = int(C.wstart[m * NWS + j]) + int(B['u'][t] * n)
                else:
                    n = int(C.slens[m * NSS + j])
                    p = int(C.sstart[m * NSS + j]) + int(B['u'][t] * n)
                rec[p] ^= np.uint8(1 << int(B['bit'][t]))
                d = R.decode_r(E, kind, rec, tab)
                silent_r = d['status'] == 'accepted' and (d['damage']['coded_sym_diff'] + d['damage']['coded_scale_diff']) > 0
                silent_w = int(t) in pos_of
                ok = silent_r == silent_w
                if ok and silent_w:
                    for k in CMP:
                        ok &= float(d['damage'][k]) == float(dmg[k][pos_of[int(t)]])
                agree += ok
                row = {'trial': int(t), 'rec': E.file, 'm': m, 'j': j, 'abs_pos': p, 'path_WS_silent': silent_w,
                       'path_R_status': d['status'], 'path_R_stage': d['stage'], 'path_R_silent': silent_r, 'agree': bool(ok),
                       'n_dec_s': d['n_dec_s'], 'n_dec_w': d['n_dec_w']}
                if silent_w:
                    row['damage_WS'] = {k: float(dmg[k][pos_of[int(t)]]) for k in CMP}
                    row['damage_R'] = {k: float(d['damage'][k]) for k in CMP} if d['status'] == 'accepted' else None
                rows.append(row)
                nfull = sum(1 for x in full_rows if x['silent'] == silent_w)
                if kind in ('fg0d', 'fg0de', 'fg0dt') and nfull < n_full:
                    fr = {'trial': int(t), 'silent': silent_w}
                    d2 = R.decode_r(E, kind, rec, tab, use_clean=False, build_slab=True)
                    fr['full_mine_status'] = d2['status']
                    fr['full_mine_slab_equals_source'] = bool(d2['status'] == 'accepted' and (d2['slab'] == E.raw).all())
                    try:
                        slab, _ = W.decode_record(rec.tobytes(), tab, E.layer, E.expert, E.shapes)
                        fr['ceilref_status'] = 'accepted'
                        fr['ceilref_slab_equals_source'] = bool((slab == E.raw).all())
                    except W.DecodeError as ex:
                        fr['ceilref_status'] = 'rejected: ' + str(ex)
                    fr['mine_incremental'] = d['status']
                    fr['mine_ceilref_reject'] = d['ceilref_reject']
                    fr['consistent'] = bool(fr['full_mine_status'] == d['status'] and
                                            (fr['ceilref_status'] == 'accepted') == (d['status'] == 'accepted' and not d['ceilref_reject']))
                    full_rows.append(fr)
            ok_all = agree == len(rows)
            full_ok = all(x['consistent'] for x in full_rows)
            out['all_agree'] &= bool(ok_all)
            out['full_consistent'] &= bool(full_ok)
            out['cases'][case][rule] = {'checked': len(rows), 'agree': int(agree), 'silent_checked': int(ns.size),
                                        'full_decoder_checked': len(full_rows), 'full_decoder_consistent': bool(full_ok),
                                        'rows': rows, 'full_decoder_rows': full_rows, 'seconds': round(time.time() - t0, 1)}
            log(f"validate {case} {rule}: checked {len(rows)} agree {agree} (silent {ns.size}); "
                f"full decoders {len(full_rows)} consistent {full_ok} {time.time() - t0:.0f}s")
    return out
