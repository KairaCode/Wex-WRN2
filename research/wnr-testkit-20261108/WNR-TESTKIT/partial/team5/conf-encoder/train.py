"""team5 / conf-encoder: deterministic reference trainer for the v1 WNM3 blob (S:213-218).
Usage: python -I train.py SAMPLE_DIR OUT_BLOB OUT_REPORT_JSON [CEILREF_MODELS_NPZ]
Training experts: the 80 even manifest indices (CR protocol). Set 0 = form 1 (ceil) trained on T(m) of every training
matrix that passes the gate; set 1 = form 0 (plain) trained on the same training matrices untransformed.
anchor0[t] = mode over the set's training matrices of type t of the densest-pair lower value of the coded scales
(ties of the mode -> smallest value: not specified by S:351). Counts use the decode contexts with the set's own anchor0.
etab = round(256 * mean coded bits of a block of (set, type, dbin)), coded bits = sum of -log2(f/16384) under the set's
quantised table; dbins without training blocks take the nearest filled dbin (lower on ties) - copied from
ceil-ref train_ceil.py:76-80 because S:217 does not define empty bins.
Optional cross-check: set 0 against ceil-ref's fg_a0_K0_c0 model (trained with per-matrix densest-pair anchors, which
are 120 on all 240 training matrices).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W


def main():
    sample_dir, out_blob, out_rep = sys.argv[1:4]
    cr_npz = sys.argv[4] if len(sys.argv) > 4 else None
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = [(i, e) for i, e in enumerate(man['experts']) if i % 2 == 0]
    assert len(ex) == 80
    C = np.zeros((2, 3, 256 * 81 * 16), np.int64)          # [form, type, scale*81 + pmx, sym]
    SH = np.zeros((2, 3, 256), np.int64)
    pairs = {0: [[], [], []], 1: [[], [], []]}
    gate_fail = []
    stats = []
    t0 = time.time()
    cache = os.path.join(os.path.dirname(os.path.abspath(out_rep)), 'train_counts.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        C, SH = z['C'], z['SH']
        pairs = {f: [list(z[f'pairs{f}'][t]) for t in range(3)] for f in (0, 1)}
        stats, gate_fail = [], json.loads(str(z['gate_fail']))
        ex_loop = []
    else:
        ex_loop = ex
    for n, (i, e) in enumerate(ex_loop):
        raw, mats = W.load_expert(os.path.join(sample_dir, e['file']), e)
        row = {'idx': i, 'file': e['file']}
        for m, mt in enumerate(mats):
            ok, tsym, tsc, st = W.gate_and_T(mt['sym'], mt['sc'])
            row[f'm{m}'] = st
            forms = [(0, mt['sym'], mt['sc'])]
            if ok:
                forms.append((1, tsym, tsc))
            else:
                gate_fail.append((e['file'], m))
            for f, sym, sc in forms:
                blk = sym.reshape(-1, 32)
                mag = (blk & 7).astype(np.int64)
                cm = np.maximum.accumulate(mag, axis=1)
                pmx = np.empty_like(mag)
                pmx[:, 0] = 80
                pmx[:, 1:] = mag[:, :-1] * 9 + cm[:, :-1]
                key = ((sc.astype(np.int64)[:, None] * 81 + pmx) << 4) | blk
                C[f, m] += np.bincount(key.ravel(), minlength=256 * 81 * 16)
                SH[f, m] += np.bincount(sc, minlength=256)[:256]
                pairs[f][m].append(W.densest_pair(sc))
        stats.append(row)
        print(n, e['file'], f'{time.time() - t0:.1f}s', flush=True)
    if ex_loop:
        np.savez_compressed(cache, C=C, SH=SH, pairs0=np.array(pairs[0]), pairs1=np.array(pairs[1]),
                            gate_fail=json.dumps(gate_fail))
    C = C.reshape(2, 3, 256, 81, 16)
    sets, etabs, rep_sets = [], [], []
    for form in (1, 0):                                      # set 0 = ceil, set 1 = plain (S:215)
        a0 = []
        for t in range(3):
            v = np.bincount(np.array(pairs[form][t]), minlength=256)
            a0.append(int(v.argmax()))
        wf = np.zeros((3, W.ROWS, 16), np.int64)
        scf = np.zeros((3, 256), np.int64)
        et = np.zeros((3, W.NBINS), np.int64)
        et_float = np.zeros((3, W.NBINS))
        svals = np.arange(256)
        for t in range(3):
            srel = np.clip(svals - a0[t], -3, 3) + 3
            full = np.zeros((7, 81, 16), np.int64)
            np.add.at(full, srel, C[form, t])
            full = full.reshape(W.FULLCTX, 16)
            reach = W.INDEX != 0xFFFF
            assert full[~reach].sum() == 0
            rows = np.zeros((W.ROWS, 16), np.int64)
            rows[W.INDEX[reach]] = full[reach]
            wf[t] = W.quantize(rows)
            scf[t] = W.quantize(SH[form, t])
            cost_full = np.zeros((W.FULLCTX, 16))
            cost_full[reach] = -np.log2(wf[t][W.INDEX[reach]] / W.P)
            cost3 = cost_full.reshape(7, 81, 16)
            bits_s = (C[form, t].astype(np.float64) * cost3[srel]).sum((1, 2))      # per scale value
            blocks_s = C[form, t][:, 80, :].sum(1)
            dbin = np.clip(svals - a0[t], -W.DBC, W.DBC) + W.DBC
            bits_d = np.bincount(dbin, bits_s, W.NBINS)
            blk_d = np.bincount(dbin, blocks_s, W.NBINS)
            good = blk_d > 0
            e_d = np.where(good, bits_d / np.maximum(blk_d, 1), 0)
            gi = np.flatnonzero(good)
            near = gi[np.abs(np.arange(W.NBINS)[:, None] - gi[None, :]).argmin(1)]
            et_float[t] = e_d[near]
            et[t] = np.round(e_d[near] * 256).astype(np.int64)
        sets.append({'form': form, 'role': 0, 'anchor0': a0, 'wf': wf, 'scf': scf})
        etabs.append(et)
        rep_sets.append({'form': form, 'anchor0': a0,
                         'densest_pair_hist': {['w1', 'w2', 'w3'][t]:
                                               {str(k): int(v) for k, v in enumerate(np.bincount(pairs[form][t], minlength=256)) if v}
                                               for t in range(3)},
                         'training_matrices': [len(pairs[form][t]) for t in range(3)],
                         'etab_bits_per_block': np.round(et_float, 3).tolist()})
    blob = W.build_wnm3(sets, np.stack(etabs))
    md = W.parse_wnm3(blob)
    open(out_blob, 'wb').write(blob)
    rep = {'blob_bytes': len(blob), 'model_id': md['model_id'], 'model_id_hex': f"0x{md['model_id']:08X}",
           'blob_sha256': __import__('hashlib').sha256(blob).hexdigest(), 'gate_failures': gate_fail,
           'sets': rep_sets, 'seconds': time.time() - t0, 'training_experts': [i for i, _ in ex]}
    if cr_npz:
        Z = np.load(cr_npz)
        cwf = Z['fg_a0_K0_c0_wf'].reshape(3, W.FULLCTX, 16)
        reach = W.INDEX != 0xFFFF
        ck = {}
        for t in range(3):
            mine = sets[0]['wf'][t]
            theirs = np.zeros_like(mine)
            theirs[W.INDEX[reach]] = cwf[t][reach]
            ck[f'w{t+1}_weight_freqs_equal'] = bool((mine == theirs).all())
            ck[f'w{t+1}_scale_freqs_equal'] = bool((sets[0]['scf'][t] == Z['fg_a0_K0_c0_scf'][t]).all())
            ck[f'w{t+1}_etab_equal'] = bool((etabs[0][t] == Z['fg_a0_K0_c0_etab'][t]).all())
            ck[f'w{t+1}_etab_maxabs_diff'] = int(np.abs(etabs[0][t] - Z['fg_a0_K0_c0_etab'][t]).max())
        rep['check_set0_vs_ceilref_fg_a0_K0_c0'] = ck
        print(ck)
    json.dump(rep, open(out_rep, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'sets'}, indent=1))
    print([s['anchor0'] for s in sets])


if __name__ == '__main__':
    main()
