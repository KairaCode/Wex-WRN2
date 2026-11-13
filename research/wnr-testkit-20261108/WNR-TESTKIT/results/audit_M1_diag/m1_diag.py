"""Re-run the kit audit's case M1 only, unchanged, except: an accepted record with NO affected stream (decode_r
returns damage=None) gets an all-zero damage dict instead of crashing run_rcase, and each such trial is logged
(record, kind, flipped byte offset in the LENC table, table size, whether the decoded lengths equal the clean ones).
Usage: python -I m1_diag.py <kit root> <out dir>   (runs run_audit.py --cases M1 --jobs 1 in-process)"""
import json, os, runpy, sys

kit, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
here = os.path.join(kit, 'tests', 'audit')
sys.path.insert(0, here)
import numpy as np  # noqa: E402
import audit_lib as A  # noqa: E402
import audit_rec as R  # noqa: E402

ZERO = {'coded_sym_diff': 0, 'coded_scale_diff': 0, 'nib_diff': 0, 'val_diff': 0, 'scale_diff': 0, 'block_flips': 0,
        'err_units': 0.0, 'err_abs': 0.0, 'sign_only': False, 'nan_scale': False}
hits = []
orig = R.decode_r


def patched(E, kind, rec, tab, bind=None, use_clean=True, build_slab=False):
    d = orig(E, kind, rec, tab, bind=bind, use_clean=use_clean, build_slab=build_slab)
    if d['status'] == 'accepted' and d.get('damage') is None:
        C = E.recs[kind]
        H = A.parse_record(rec, tab, E.shapes)
        Hc = A.parse_record(C.rec, tab, E.shapes)
        pos = np.flatnonzero(rec != C.rec)
        same = bool((H['slens'] == Hc['slens']).all() and all((a == b).all() for a, b in zip(H['wres'], Hc['wres'])))
        hits.append({'rec': E.file, 'kind': kind, 'byte_rel': [int(p - Hc['hlen']) for p in pos], 'ltb': int(Hc['ltb']),
                     'n_aff_streams': d.get('n_aff_streams'), 'lengths_identical': same})
        d['damage'] = dict(ZERO)
    return d


R.decode_r = patched
os.makedirs(out, exist_ok=True)
sys.argv = ['run_audit.py', '--root', kit, '--cases', 'M1', '--jobs', '1', '--out', out]
rc = 0
try:
    runpy.run_path(os.path.join(here, 'run_audit.py'), run_name='__main__')
except SystemExit as ex:
    rc = ex.code or 0
json.dump({'none_damage_trials': hits}, open(os.path.join(out, 'm1_none_damage_trials.json'), 'w'), indent=1)
print(f'M1DIAG none-damage trials: {len(hits)}; all lengths identical: {all(h["lengths_identical"] for h in hits)}; '
      f'all n_aff_streams 0: {all(h["n_aff_streams"] == 0 for h in hits)}; '
      f'byte_rel/ltb: {[(h["kind"], h["byte_rel"], h["ltb"]) for h in hits]}')
sys.exit(rc)
