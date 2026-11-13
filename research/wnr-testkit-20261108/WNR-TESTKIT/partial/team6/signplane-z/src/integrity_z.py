"""Single-bit corruption trials on real Z records (one expert): every weight stream gets exactly one random bit flip in
one region (its rANS bytes incl. the 4-B state, or its sign plane), all streams in one lockstep decode (each stream is
decoded independently; a stream that reads outside its own bytes fails its exact length rule anyway).
Per stream: rejected / accepted with correct output / accepted with WRONG output (silent corruption).
Records: Z1 with hash (E), Z1 without hash (x0 = L, final rule x == L), Z2 with hash.
Usage: python -I integrity_z.py SAMPLE_DIR ZMODEL_NPZ OUT_JSON EXPERT_IDX ROUNDS
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
import signplane as Z


def main():
    sample_dir, zmodel, out_json, idx, rounds = sys.argv[1:6]
    idx, rounds = int(idx), int(rounds)
    Z.use_g0(120)
    man = W.load_manifest(sample_dir)
    e = man['experts'][idx]
    raw, mats = W.load_expert(sample_dir, e)
    tz = Z.load_tables(zmodel)
    res = {'expert': e['file'], 'rounds': rounds, 'cases': {}}
    rng = np.random.default_rng(20261108)
    for name, z2, hashed in (('Z1_hash', False, True), ('Z1_nohash', False, False), ('Z2_hash', True, True)):
        rec, info = Z.encode_record_z(mats, tz, e['layer'], e['expert'], d=True, e=hashed, z2=z2, keep=True)
        k = info['_keep']
        symT, nws, feats = k['symT'], k['nws'], k['feats']
        WS = symT.shape[1]
        wl = info['wlens'].astype(np.int64)
        rl = k['rlens'].astype(np.int64)
        offs = np.zeros(WS, np.int64); offs[1:] = np.cumsum(wl)[:-1]
        p0 = info['header'] + info['length_table'] + info['scale_payload']
        pay = np.frombuffer(rec, np.uint8)[p0:p0 + int(wl.sum())].copy()
        srelT = np.concatenate([feats[m]['srel'].reshape(nws[m], 64) for m in range(3)]).T.astype(np.int64)
        wmat = np.repeat(np.arange(3), nws)
        widx = np.concatenate([np.arange(n) for n in nws])
        hw = W.hash_seeds(e['layer'], e['expert'], tz.model_id, 1, wmat, widx, np.zeros(WS, np.int64)) if hashed else None
        pad = np.zeros(2 * (2048 + 64) + 16, np.uint8)
        front = np.zeros(Z.FRONT, np.uint8)
        # sanity: clean decode
        t0 = time.time()
        out, ok, _ = Z.decode_weights_z(np.concatenate([front, pay, pad]), offs, wl, srelT, wmat * W.NCTX, tz, hw, z2)
        clean = {'ok_streams': int(ok.sum()), 'exact_streams': int((out == symT).all(0).sum()), 'seconds': time.time() - t0}
        case = {'clean_decode': clean, 'streams': WS, 'rec_used': info['used'], 'regions': {}}
        print(name, 'clean', clean, flush=True)
        for region in ('rans', 'plane'):
            agg = {'trials': 0, 'rejected': 0, 'accepted_correct': 0, 'accepted_wrong': 0, 'wrong_weights_hist': {},
                   'accepted_wrong_only_sign_changes': 0}
            for r in range(rounds):
                if region == 'rans':
                    lo, n = offs, rl
                else:
                    lo, n = offs + rl, wl - rl
                byte = lo + (rng.random(WS) * n).astype(np.int64)
                bit = rng.integers(0, 8, WS).astype(np.uint8)
                cp_ = pay.copy()
                cp_[byte] ^= (np.uint8(1) << bit)
                out, ok, _ = Z.decode_weights_z(np.concatenate([front, cp_, pad]), offs, wl, srelT, wmat * W.NCTX, tz, hw, z2)
                diff = (out != symT)
                nd = diff.sum(0)
                wrong = nd > 0
                agg['trials'] += WS
                agg['rejected'] += int((~ok).sum())
                agg['accepted_correct'] += int((ok & ~wrong).sum())
                aw = ok & wrong
                agg['accepted_wrong'] += int(aw.sum())
                if aw.any():
                    for v in nd[aw]:
                        agg['wrong_weights_hist'][str(int(v))] = agg['wrong_weights_hist'].get(str(int(v)), 0) + 1
                    magdiff = ((out & 7) != (symT & 7)).sum(0)
                    agg['accepted_wrong_only_sign_changes'] += int((aw & (magdiff == 0)).sum())
                print(name, region, r, {k2: v2 for k2, v2 in agg.items() if k2 != 'wrong_weights_hist'}, flush=True)
            agg['accept_rate'] = (agg['accepted_correct'] + agg['accepted_wrong']) / agg['trials']
            agg['silent_rate'] = agg['accepted_wrong'] / agg['trials']
            case['regions'][region] = agg
        res['cases'][name] = case
        json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
