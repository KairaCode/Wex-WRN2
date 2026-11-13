"""Encode real V4.1 experts in the signplane-z layouts, decode, verify value-exact, write one JSON line per (expert, config).

Usage: python -I run_z.py SAMPLE_DIR ZMODEL_NPZ CEILREF_MODELS_NPZ OUT_JSONL EXPERT_INDICES CONFIGS [SAVE_DIR]
  CONFIGS: comma list of
    z1de  f + g0 + Z1 (compacted tail sign plane) + d + e
    z2de  f + g0 + Z2 (tail plane + last <=7 signs in x0 above the hash) + d + e
    z1e   Z1 with the u16 length table (payload shared with z1de)
    ref_fgcde / ref_fgde  team 4's configs re-run with the COPIED ceil-ref coder (reproduction check)
  SAVE_DIR: if given, the records are written there as <file>.<config>.rec (for the C decoder).
Anchor: constant 120 (g0) for every config, as team 4's heldout_g0 set.
"""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
import signplane as Z

REF_CFG = {'ref_fgcde': W.Config('fgcde', c=True, d=True, e=True, f=True, g=True),
           'ref_fgde': W.Config('fgde', d=True, e=True, f=True, g=True)}
REF_KEY = {'ref_fgcde': 'fg_a0_K0_c1', 'ref_fgde': 'fg_a0_K0_c0'}


def main():
    sample_dir, zmodel, refmodels, out_jsonl, idxs, cfgs = sys.argv[1:7]
    save_dir = sys.argv[7] if len(sys.argv) > 7 else None
    Z.use_g0(120)
    idxs = [int(x) for x in idxs.split(',')]
    cfgs = cfgs.split(',')
    man = W.load_manifest(sample_dir)
    tz = Z.load_tables(zmodel)
    R = np.load(refmodels)
    reft = {}
    fo = open(out_jsonl, 'a')
    for i in idxs:
        e = man['experts'][i]
        raw, mats = W.load_expert(sample_dir, e)
        off, sha_ok = 0, True
        for dsc in e['tensors']:
            if 'sha256' in dsc:
                sha_ok &= hashlib.sha256(raw[off:off + dsc['bytes']].tobytes()).hexdigest() == dsc['sha256']
            off += dsc['bytes']
        shapes = [(mt['R'], mt['C']) for mt in mats]
        nsb = sum(mt['sc'].size for mt in mats)
        craw = Z.canon_bytes(raw, nsb)
        n_canon_bytes = int((craw != raw).sum())
        n_neg0 = int(sum((mt['sym'] == 8).sum() for mt in mats))
        for c in cfgs:
            tm = {}
            zinfo = None
            if c.startswith('ref_'):
                cfg = REF_CFG[c]
                k = REF_KEY[c]
                if k not in reft:
                    reft[k] = W.Tables(R[k + '_wf'], R[k + '_scf'], None, R[k + '_etab'], K=0, nctx=W.NCTX)
                tab = reft[k]
                rec, info, st, bi = W.encode_record(mats, tab, cfg, e['layer'], e['expert'], tm)
                try:
                    slab, _ = W.decode_record(rec, tab, e['layer'], e['expert'], shapes, tm)
                    derr = None
                except W.DecodeError as ex:
                    slab, derr = np.zeros(0, np.uint8), str(ex)
                ref = craw if cfg.c else raw
            else:
                z2 = c.startswith('z2')
                d = c.endswith('de')
                rec, info = Z.encode_record_z(mats, tz, e['layer'], e['expert'], d=d, e=True, z2=z2, timing=tm)
                zinfo = info['z']
                try:
                    slab, dinfo = Z.decode_record_z(rec, tz, e['layer'], e['expert'], shapes, tm)
                    derr = None
                except W.DecodeError as ex:
                    slab, derr = np.zeros(0, np.uint8), str(ex)
                ref = craw
            if save_dir:
                with open(os.path.join(save_dir, e['file'].replace('.bin', f'.{c}.rec')), 'wb') as fr:
                    fr.write(rec)
            row = {'idx': i, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'config': c,
                   'rec_flags': info['rec_flags'], 'anchors': info['meds'], 'source_sha256_ok': bool(sha_ok),
                   'ceil_applied': None if info.get('ceil') is None else info['ceil']['applied'],
                   'sizes': {'header': info['header'], 'length_table': info['length_table'],
                             'scale_payload': info['scale_payload'], 'weight_payload': info['weight_payload'],
                             'used': info['used'], 'padding': info['padding'], 'total': info['total'],
                             'record_bytes_check': len(rec)},
                   'est': {'weight_bits': info['est_weight_bits'], 'scale_bits': info['est_scale_bits']},
                   'z': zinfo,
                   'len_resid_std': [info.get('len_resid_std_s'), info.get('len_resid_std_w')],
                   'verify': {'decode_error': derr,
                              'equal_to_canonical_source': bool(slab.size == ref.size and (slab == ref).all()),
                              'bytes_differing_from_source': int((slab != raw).sum()) if slab.size == raw.size else -1,
                              'expected_differing_bytes (bytes holding a -0 nibble)': n_canon_bytes,
                              'neg0_nibbles_in_source': n_neg0},
                   'timing': {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in tm.items()},
                   'record_sha256': hashlib.sha256(rec).hexdigest()}
            fo.write(json.dumps(row) + '\n')
            fo.flush()
            print(e['file'], c, row['sizes']['total'], row['sizes']['used'], row['verify']['equal_to_canonical_source'],
                  derr, f"enc {tm.get('enc_total_s', -1):.2f}s dec {tm.get('dec_total_s', -1):.2f}s", flush=True)
            del rec, slab
    fo.close()


if __name__ == '__main__':
    main()
