"""Task 2 supplement:
1) Write records_G.bin (40 held-out + 12 big tail records = 52 records) for GPU decode verification
   with the UNCHANGED engine decoder (kaira_wex_cuda.dll + test_decode_batch.exe).
2) OPEN-K #11 / T13: Verify byte-for-byte identity between KAIRA's C encoder (wex_nr2_encode in g0codec.dll)
   and Team 4's Python encoder (wnr_common.py) on the 40 held-out test records.
"""
import ctypes as C, hashlib, json, os, random, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'<HOME>\Wex-WRN2\research\wnr3-20261107\team4\ceil-ref')
import wnr_common as WC

import t2_g0 as T2

SC_BINS = r'<TMP>\data\g0bins'


def tables_from_wnm2(blob_bytes):
    """Parse a 59,408 B WNM2 blob into wnr_common.Tables (exact same frequencies)."""
    magic, ver, pb, nctx, nm, z0, z1, z2 = struct.unpack_from('<8I', blob_bytes, 0)
    assert magic == WC.WNM2_MAGIC and ver == 1 and pb == 14 and nctx == 567 and nm == 3
    off = 32
    wcdf = np.frombuffer(blob_bytes, '<u2', 3 * 567 * 17, off).reshape(3, 567, 17).astype(np.int64)
    off += 3 * 567 * 17 * 2
    scdf = np.frombuffer(blob_bytes, '<u2', 3 * 257, off).reshape(3, 257).astype(np.int64)
    wf = np.diff(wcdf, axis=-1).reshape(3 * 567, 16)
    scf = np.diff(scdf, axis=-1)
    tab = WC.Tables(wf, scf)
    assert tab.blob == bytes(blob_bytes), "Serialized WNM2 blob mismatch"
    return tab


def mats_from_sc_wt(sc, wt, info):
    """Build wnr_common mats list from Shards.expert(L, e) output."""
    mats = []
    for m in range(3):
        ss, sd, ws, wd = info[m]
        R, Cb = ws
        C = 2 * Cb
        w2 = wt[m].reshape(R, Cb)
        sym = np.empty((R, C), np.uint8)
        sym[:, 0::2] = w2 & 15
        sym[:, 1::2] = w2 >> 4
        mats.append({'sym': sym.ravel(), 'sc': np.ascontiguousarray(sc[m]), 'R': R, 'C': C})
    return mats


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

    blobA = np.fromfile(os.path.join(HERE, 'model_A_median.wnm2'), dtype=np.uint8)
    blobG = np.fromfile(os.path.join(HERE, 'model_G_rule_g.wnm2'), dtype=np.uint8)
    tabA = tables_from_wnm2(blobA)
    tabG = tables_from_wnm2(blobG)
    cfg_wnr2 = WC.Config('wnr2', wnr2=True)

    os.makedirs(SC_BINS, exist_ok=True)
    binG_test_path = os.path.join(SC_BINS, 'records_G.bin')
    binG_big_path = os.path.join(SC_BINS, 'records_G_big.bin')

    fG_test = open(binG_test_path, 'wb')
    fG_test.write(struct.pack('<I', blobG.size)); fG_test.write(blobG.tobytes()); fG_test.write(struct.pack('<I', len(te)))

    fG_big = open(binG_big_path, 'wb')
    fG_big.write(struct.pack('<I', blobG.size)); fG_big.write(blobG.tobytes()); fG_big.write(struct.pack('<I', len(big)))

    fG_all_path = os.path.join(SC_BINS, 'records_G_all52.bin')
    fG_all = open(fG_all_path, 'wb')
    fG_all.write(struct.pack('<I', blobG.size)); fG_all.write(blobG.tobytes()); fG_all.write(struct.pack('<I', len(te) + len(big)))

    byte_equal_A = 0
    for i, (L, e) in enumerate(te):
        sc, wt, info = sh.expert(L, e)
        recA_c, uA_c, tA_c, _ = T2.encode(blobA, -1, sc, wt)
        recG_c, uG_c, tG_c, _ = T2.encode(blobG, -2, sc, wt)

        mats = mats_from_sc_wt(sc, wt, info)
        recA_py, infoA_py, _, _ = WC.encode_record(mats, tabA, cfg_wnr2, L, e)

        eqA = (recA_c.tobytes() == recA_py)
        byte_equal_A += int(eqA)
        assert eqA, f"Mismatch on test expert ({L}, {e})"

        raw = T2.slab(ents[L * 192 + e], sc, wt)
        ent = list(ents[L * 192 + e]); ent[0] = 0; ent[1] = recG_c.size
        for fh in (fG_test, fG_all):
            fh.write(T2.ENTRY.pack(*ent)); fh.write(struct.pack('<I', raw.size))
            fh.write(raw.tobytes()); fh.write(recG_c.tobytes())

        if (i + 1) % 10 == 0:
            print(f"test {i+1}/40: C vs Python WNR2 byte-identical={byte_equal_A}/{i+1} ({time.time()-t0:.1f}s)", flush=True)

    fG_test.close()

    for i, (L, e) in enumerate(big):
        sc, wt, info = sh.expert(L, e)
        recG_c, uG_c, tG_c, _ = T2.encode(blobG, -2, sc, wt)
        raw = T2.slab(ents[L * 192 + e], sc, wt)
        ent = list(ents[L * 192 + e]); ent[0] = 0; ent[1] = recG_c.size
        for fh in (fG_big, fG_all):
            fh.write(T2.ENTRY.pack(*ent)); fh.write(struct.pack('<I', raw.size))
            fh.write(raw.tobytes()); fh.write(recG_c.tobytes())

    fG_big.close()
    fG_all.close()

    res = {
        'model_A_blob_byte_identical_c_vs_py': True,
        'test_records_compared': len(te),
        'test_records_byte_identical_c_vs_py': byte_equal_A,
        'records_G_bin': binG_test_path,
        'records_G_big_bin': binG_big_path,
        'elapsed_s': round(time.time() - t0, 2)
    }
    json.dump(res, open(os.path.join(HERE, 't2_conformance_and_g_bins.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
