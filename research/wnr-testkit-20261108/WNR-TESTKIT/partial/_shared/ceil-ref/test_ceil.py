"""Unit / fallback tests for the ceil-form extension (team4 / ceil-ref). Works on in-memory copies only.

Usage: python -I test_ceil.py SAMPLE_DIR MODELS_NPZ OUT_JSON EXPERT_INDEX
1. ceil_inverse(ceil_forward(x)) == x on synthetic blocks (every max-6/7 pattern class incl. sign bits, -0).
2. Fallback path on a real expert, modified in memory:
   a) one w1 block set to all +0 (block max 0 < 6)            -> F must be cleared, record decodes byte-exact
   b) one even w2 block's scale set to 254 (transformed 255)  -> F cleared, byte-exact
   c) one even w3 block's scale set to 253 (transformed 254)  -> F kept, byte-exact
   d) unmodified                                              -> F kept, byte-exact
   for configs fg (u16 lengths), fgdt (tail seeding + D) and fgaK2ce (A 3-valued side symbol, K=2, C, E).
3. Decoder rejects an F record whose payload was re-encoded from a plain block with max < 5 (ceil_inverse check).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
from run_ceil import CONFIGS, canon_bytes


def slab_of(mats):
    parts = [mt['sc'] for mt in mats]
    for mt in mats:
        s = mt['sym']
        parts.append((s[0::2] | (s[1::2] << 4)).astype(np.uint8))
    return np.concatenate(parts)


def main():
    sample_dir, models_npz, out_json, ei = sys.argv[1:5]
    res = {}
    rng = np.random.default_rng(5)
    # ---- 1. synthetic round trip
    nb = 200000
    mag = rng.integers(0, 8, (nb, 32)).astype(np.uint8)
    kind = rng.integers(0, 3, nb)
    ev = kind == 0                                   # even blocks: magnitudes from {0,2,4,5,6,7} with a 7
    mag[ev] = np.array([0, 2, 4, 5, 6, 7], np.uint8)[rng.integers(0, 6, (int(ev.sum()), 32))]
    mag[ev, rng.integers(0, 32, int(ev.sum()))] = 7
    m6 = kind == 1                                   # max exactly 6
    mag[m6] = np.minimum(mag[m6], 6)
    mag[m6, rng.integers(0, 32, int(m6.sum()))] = 6
    m7 = kind == 2
    mag[m7, rng.integers(0, 32, int(m7.sum()))] = 7
    sign = (rng.integers(0, 2, (nb, 32)) * 8).astype(np.uint8)
    sym = (mag | sign).ravel()
    sc = rng.integers(0, 254, nb).astype(np.uint8)
    ts, tsc, info = W.ceil_forward(sym, sc)
    bs, bsc = W.ceil_inverse(ts, tsc)
    tb = ts.reshape(-1, 32) & 7
    res['synthetic'] = {'blocks': nb, 'even': info['even'], 'roundtrip_equal': bool((bs == sym).all() and (bsc == sc).all()),
                        'transformed_max_values': sorted(set(np.unique(tb.max(1)).tolist())),
                        'canon_then_ceil_roundtrip': bool((W.ceil_inverse(*W.ceil_forward(W.canon(sym), sc)[:2])[0] == W.canon(sym)).all())}
    # ---- 2. fallback on real data
    man = W.load_manifest(sample_dir)
    e = man['experts'][int(ei)]
    raw0, mats0 = W.load_expert(sample_dir, e)
    assert (slab_of(mats0) == raw0).all()
    Z = np.load(models_npz)
    shapes = [(mt['R'], mt['C']) for mt in mats0]

    def tabs(cfg):
        k = cfg.model_key()
        sf = Z[k + '_sf'] if (k + '_sf') in Z.files else None
        return W.Tables(Z[k + '_wf'], Z[k + '_scf'], sf, Z[k + '_etab'], K=cfg.K, nctx=W.NACTX if cfg.a else W.NCTX)

    def even_block(m):
        blk = mats0[m]['sym'].reshape(-1, 32) & 7
        return int(np.flatnonzero((blk.max(1) == 7) & ~((blk == 1) | (blk == 3)).any(1))[3])

    cases = {}
    for case in ('a_block_max0', 'b_even_scale254', 'c_even_scale253', 'd_unmodified'):
        mats = [{**mt, 'sym': mt['sym'].copy(), 'sc': mt['sc'].copy()} for mt in mats0]
        if case == 'a_block_max0':
            mats[0]['sym'][32 * 7:32 * 8] = 0
        elif case == 'b_even_scale254':
            mats[1]['sc'][even_block(1)] = 254
        elif case == 'c_even_scale253':
            mats[2]['sc'][even_block(2)] = 253
        raw = slab_of(mats)
        craw = canon_bytes(raw, sum(mt['sc'].size for mt in mats))
        out = {}
        for cn in ('fg', 'fgdt', 'fgaK2ce'):
            cfg = CONFIGS[cn]
            tab = tabs(cfg)
            if cfg.d:                             # D variants are built from the payload of their non-D sibling
                base_cfg = CONFIGS[cn.replace('d', '', 1)]
                _, _, st, bi = W.encode_record(mats, tab, base_cfg, e['layer'], e['expert'])
                rec, info = W.build_record(st, tab, cfg, bi)
            else:
                rec, info, st, bi = W.encode_record(mats, tab, cfg, e['layer'], e['expert'])
            slab, dinfo = W.decode_record(rec, tab, e['layer'], e['expert'], shapes)
            ref = craw if cfg.c else raw
            out[cn] = {'F_in_record': bool(info['rec_flags'] & W.FLAG_F), 'fallback': not info['ceil']['applied'],
                       'below6': info['ceil']['below6'], 'overflow': info['ceil']['overflow'],
                       'decoded_equal_to_reference': bool((slab == ref).all()), 'mode': 'value-exact' if cfg.c else 'byte-exact',
                       'total_bytes': info['total']}
        cases[case] = out
        print(case, json.dumps(out), flush=True)
    res['fallback'] = cases
    # ---- 3. F record whose decoded block max < 5 must be rejected: encode plain data containing a max-0 block
    cfg = CONFIGS['fg']
    tab = tabs(cfg)
    mats = [{**mt, 'sym': mt['sym'].copy(), 'sc': mt['sc'].copy()} for mt in mats0]
    mats[0]['sym'][0:32] = 0
    rec, info, st, bi = W.encode_record(mats, tab, cfg, e['layer'], e['expert'])
    assert not (info['rec_flags'] & W.FLAG_F)
    b = bytearray(rec)
    b[27] |= W.FLAG_F                             # forge the F flag on a plain record
    try:
        W.decode_record(bytes(b), tab, e['layer'], e['expert'], shapes)
        r3 = 'ACCEPTED'
    except W.DecodeError as ex:
        r3 = 'rejected: ' + str(ex)
    res['forged_F_on_plain_record_with_max0_block'] = r3
    print(json.dumps(res, indent=1))
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
