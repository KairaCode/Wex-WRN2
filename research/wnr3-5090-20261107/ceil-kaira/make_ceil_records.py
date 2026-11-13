"""Build ceil-form WNM2 models (F: lower median on ceil scales; FG: densest-pair rule g on ceil scales)
and encode the 40 held-out test records + 12 largest tail records with KAIRA's C encoder (g0codec.dll).
Verifies:
1) CPU decode of ceil records + ceil_inverse == 100% byte-identical to original sc + wt.
2) Writes records_CF.bin and records_CFG.bin (with original raw slabs!) so test_decode_batch.exe +
   kaira_wex_cuda_ceil.dll can verify 100% GPU byte-exact reconstruction and benchmark single/batch latency.
"""
import ctypes as C, hashlib, json, os, random, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G0_DIR = os.path.join(HERE, '..', 'g0')
sys.path.insert(0, G0_DIR)
import t2_g0 as T2

SC_BINS = r'<TMP>\data\g0bins'
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)


def ceil_forward_packed(wt_packed, sc):
    """Apply ceil-form transform T to packed FP4 weights (uint8) and UE8M0 scales (uint8)."""
    sym = np.empty(wt_packed.size * 2, np.uint8)
    sym[0::2] = wt_packed & 15
    sym[1::2] = wt_packed >> 4
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    bmax = mag.max(1)
    has13 = ((mag == 1) | (mag == 3)).any(1)
    even = (bmax == 7) & ~has13
    assert (bmax >= 6).all() and (sc[even] < 254).all()
    tblk = np.where(even[:, None], (blk & 8) | HALF[mag], blk).astype(np.uint8).ravel()
    twt = (tblk[0::2] | (tblk[1::2] << 4)).astype(np.uint8)
    tsc = (sc.astype(np.int16) + even).astype(np.uint8)
    return twt, tsc


def ceil_inverse_packed(twt_packed, tsc):
    """Apply ceil-form inverse T^-1 to packed FP4 weights and UE8M0 scales."""
    sym = np.empty(twt_packed.size * 2, np.uint8)
    sym[0::2] = twt_packed & 15
    sym[1::2] = twt_packed >> 4
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    odd = mag.max(1) == 5
    ob = np.where(odd[:, None], (blk & 8) | DOUBLE[mag], blk).astype(np.uint8).ravel()
    owt = (ob[0::2] | (ob[1::2] << 4)).astype(np.uint8)
    osc = (tsc.astype(np.int16) - odd).astype(np.uint8)
    return owt, osc


def train_ceil(sh, idx, anchor):
    wc = np.zeros(3 * 567 * 16, np.uint64)
    scn = np.zeros(3 * 256, np.uint64)
    T2.lib.g0_set_anchor(anchor)
    for L, e in idx:
        sc, wt, _ = sh.expert(L, e)
        tsc, twt = [], []
        for m in range(3):
            tw, ts = ceil_forward_packed(wt[m], sc[m])
            tsc.append(ts); twt.append(tw)
        a, b = T2.layout(tsc, twt)
        T2.lib.g0_count(T2.ptr(a), T2.ptr(b), T2.ptrs(tsc), T2.ptrs(twt), T2.ptr(wc), T2.ptr(scn))
    blob = np.zeros(59408, np.uint8)
    T2.lib.g0_model_build(T2.ptr(wc), T2.ptr(scn), T2.ptr(blob))
    T2.lib.g0_set_anchor(-1)
    return blob


def cpu_decode_ceil_check(blob, rec, sc_orig, wt_orig):
    a, b = T2.layout(sc_orig, wt_orig)
    so = [np.zeros_like(s) for s in sc_orig]
    wo = [np.zeros_like(w) for w in wt_orig]
    med = np.zeros(3, np.uint8)
    mid = C.c_uint32(0)
    rc = T2.lib.g0_decode(T2.ptr(blob), T2.ptr(rec), rec.size, T2.ptr(a), T2.ptr(b), T2.ptrs(so), T2.ptrs(wo), T2.ptr(med), C.byref(mid))
    if rc != 0:
        return False, rc
    for m in range(3):
        ow, os_ = ceil_inverse_packed(wo[m], so[m])
        if not (np.array_equal(ow, wt_orig[m]) and np.array_equal(os_, sc_orig[m])):
            return False, -10 - m
    return True, 0


def main():
    t0 = time.time()
    sh = T2.Shards()
    pk = T2.picks()
    tr = [pk[i] for i in range(0, 160, 2)]
    te_idx = [4 * L + (1 if L % 2 == 0 else 3) for L in range(40)]
    te = [pk[i] for i in te_idx]
    ents, _ = T2.nr2_entries()
    t1 = json.load(open(os.path.join(HERE, '..', 't1_record_sizes.json')))['rows']
    big = [(r['i'] // 192, r['i'] % 192) for r in sorted(t1, key=lambda r: -r['total'])[:12]]
    big = [x for x in big if x not in tr and x not in te]

    pathCF = os.path.join(HERE, 'model_CF_median.wnm2')
    pathCFG = os.path.join(HERE, 'model_CFG_rule_g.wnm2')
    if os.path.exists(pathCF) and os.path.exists(pathCFG):
        blobCF = np.fromfile(pathCF, np.uint8)
        blobCFG = np.fromfile(pathCFG, np.uint8)
    else:
        print("Training CF (median) and CFG (rule g) models on 80 train experts...", flush=True)
        blobCF = train_ceil(sh, tr, -1)
        blobCFG = train_ceil(sh, tr, -2)
        blobCF.tofile(pathCF)
        blobCFG.tofile(pathCFG)

    idCF, idCFG = T2.fnv(blobCF), T2.fnv(blobCFG)
    print(f"Models CF={idCF:#x} CFG={idCFG:#x} ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(SC_BINS, exist_ok=True)
    binCF_path = os.path.join(SC_BINS, 'records_CF.bin')
    binCFG_path = os.path.join(SC_BINS, 'records_CFG.bin')
    binCFG_all52_path = os.path.join(SC_BINS, 'records_CFG_all52.bin')

    fCF = open(binCF_path, 'wb')
    fCF.write(struct.pack('<I', blobCF.size)); fCF.write(blobCF.tobytes()); fCF.write(struct.pack('<I', len(te)))

    fCFG = open(binCFG_path, 'wb')
    fCFG.write(struct.pack('<I', blobCFG.size)); fCFG.write(blobCFG.tobytes()); fCFG.write(struct.pack('<I', len(te)))

    fCFG52 = open(binCFG_all52_path, 'wb')
    fCFG52.write(struct.pack('<I', blobCFG.size)); fCFG52.write(blobCFG.tobytes()); fCFG52.write(struct.pack('<I', len(te) + len(big)))

    rows = []
    for set_name, exps in (('test', te), ('big', big)):
        for i, (L, e) in enumerate(exps):
            sc, wt, _ = sh.expert(L, e)
            tsc, twt = [], []
            for m in range(3):
                tw, ts = ceil_forward_packed(wt[m], sc[m])
                tsc.append(ts); twt.append(tw)

            recCF, uCF, tCF, _ = T2.encode(blobCF, -1, tsc, twt)
            recCFG, uCFG, tCFG, _ = T2.encode(blobCFG, -2, tsc, twt)

            okCF, rcCF = cpu_decode_ceil_check(blobCF, recCF, sc, wt)
            okCFG, rcCFG = cpu_decode_ceil_check(blobCFG, recCFG, sc, wt)
            assert okCF and okCFG, (L, e, okCF, rcCF, okCFG, rcCFG)

            raw_orig = T2.slab(ents[L * 192 + e], sc, wt)
            entCF = list(ents[L * 192 + e]); entCF[0] = 0; entCF[1] = recCF.size
            entCFG = list(ents[L * 192 + e]); entCFG[0] = 0; entCFG[1] = recCFG.size

            if set_name == 'test':
                fCF.write(T2.ENTRY.pack(*entCF)); fCF.write(struct.pack('<I', raw_orig.size))
                fCF.write(raw_orig.tobytes()); fCF.write(recCF.tobytes())
                fCFG.write(T2.ENTRY.pack(*entCFG)); fCFG.write(struct.pack('<I', raw_orig.size))
                fCFG.write(raw_orig.tobytes()); fCFG.write(recCFG.tobytes())

            fCFG52.write(T2.ENTRY.pack(*entCFG)); fCFG52.write(struct.pack('<I', raw_orig.size))
            fCFG52.write(raw_orig.tobytes()); fCFG52.write(recCFG.tobytes())

            rows.append({
                'set': set_name, 'layer': L, 'expert': e,
                'CF_used': uCF, 'CF_total': tCF,
                'CFG_used': uCFG, 'CFG_total': tCFG,
                'prod_total': ents[L * 192 + e][1],
                'cpu_decode_ceil_exact': True
            })
            if (i + 1) % 10 == 0 or set_name == 'big':
                print(f"[{set_name}] {i+1}/{len(exps)} L{L:02d}_E{e:03d}: CF={tCF} CFG={tCFG} prod={ents[L*192+e][1]} ({time.time()-t0:.1f}s)", flush=True)

    fCF.close(); fCFG.close(); fCFG52.close()

    T = [r for r in rows if r['set'] == 'test']
    B = [r for r in rows if r['set'] == 'big']
    summ = {
        'model_CF_id': idCF,
        'model_CFG_id': idCFG,
        'n_test': len(T),
        'n_big': len(B),
        'all_cpu_decode_ceil_exact': True,
        'test_CF_mean_total': float(np.mean([r['CF_total'] for r in T])),
        'test_CFG_mean_total': float(np.mean([r['CFG_total'] for r in T])),
        'test_CF_ratio': float(np.mean([r['CF_total'] for r in T]) / T2.RAW),
        'test_CFG_ratio': float(np.mean([r['CFG_total'] for r in T]) / T2.RAW),
        'big_CF_mean_total': float(np.mean([r['CF_total'] for r in B])),
        'big_CFG_mean_total': float(np.mean([r['CFG_total'] for r in B])),
        'big_CF_max_total': int(max(r['CF_total'] for r in B)),
        'big_CFG_max_total': int(max(r['CFG_total'] for r in B)),
        'big_prod_max_total': int(max(r['prod_total'] for r in B)),
        'elapsed_s': round(time.time() - t0, 2)
    }
    json.dump({'summary': summ, 'rows': rows}, open(os.path.join(HERE, 'ceil_kaira_records.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
