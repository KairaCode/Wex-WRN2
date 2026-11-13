"""Task 2 extension: Evaluate Rule g (densest adjacent pair, anchor=-2) vs Model A (lower median)
and Model B (fixed 120) on DeepSeek V4.1 REAP-192.

Demonstrates whether Rule g eliminates the +13.2% tail expansion seen on L39 E120 under fixed 120,
while preserving or improving compression on the heldout test set.
"""
import ctypes as C, hashlib, json, os, random, struct, sys, time
import numpy as np

MODEL_DIR = r'U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
NR2 = r'E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = 18800640
WCHUNK, SCHUNK = 2048, 512
ANCHOR_B = 120
ANCHOR_G = -2  # Rule g: densest adjacent pair in wex_neural_rans_g0.h
ENTRY = struct.Struct('<QIII6H6H3I3II')

lib = C.CDLL(os.path.join(HERE, 'g0codec.dll'))
P = C.c_void_p
lib.g0_set_anchor.argtypes = [C.c_int]
lib.g0_count.argtypes = [P] * 6
lib.g0_model_build.argtypes = [P] * 3
lib.g0_encode.argtypes = [P, P, P, P, P, C.c_uint32, C.c_uint32, P, C.c_size_t, P]
lib.g0_decode.argtypes = [P, P, C.c_size_t, P, P, P, P, P, P]


def ptr(a):
    return a.ctypes.data


def ptrs(arrs):
    return (C.c_void_p * 3)(*[ptr(a) for a in arrs])


def picks():
    rng = random.Random(20261107)
    return [(L, e) for L in range(40) for e in sorted(rng.sample(range(192), 4))]


class Shards:
    def __init__(self):
        self.wm = json.load(open(os.path.join(MODEL_DIR, 'model.safetensors.index.json')))['weight_map']
        self.hdr = {}

    def tensor(self, name):
        f = self.wm[name]
        if f not in self.hdr:
            with open(os.path.join(MODEL_DIR, f), 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                self.hdr[f] = (8 + n, json.loads(fh.read(n)))
        base, h = self.hdr[f]
        t = h[name]
        a, b = t['data_offsets']
        with open(os.path.join(MODEL_DIR, f), 'rb') as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        assert len(buf) == b - a
        return np.frombuffer(buf, np.uint8).copy(), t['shape'], t['dtype']

    def expert(self, L, e):
        sc, wt, info = [], [], []
        for m in (1, 2, 3):
            s, ss, sd = self.tensor(f'layers.{L}.ffn.experts.{e}.w{m}.scale')
            w, ws, wd = self.tensor(f'layers.{L}.ffn.experts.{e}.w{m}.weight')
            assert s.size * 16 == w.size, (L, e, m, ss, ws)
            sc.append(s); wt.append(w); info.append((ss, sd, ws, wd))
        return sc, wt, info


def layout(sc, wt):
    return (np.array([s.size for s in sc], np.uint32), np.array([2 * w.size for w in wt], np.uint32))


def lower_median(s):
    h = np.bincount(s, minlength=256).cumsum()
    return int(np.argmax(h * 2 >= s.size))


def rule_g_anchor(s):
    h = np.bincount(s, minlength=256)
    pairs = h[:-1] + h[1:]
    return int(np.argmax(pairs))


def train(sh, idx, anchor):
    wc = np.zeros(3 * 567 * 16, np.uint64); scn = np.zeros(3 * 256, np.uint64)
    lib.g0_set_anchor(anchor)
    for L, e in idx:
        sc, wt, _ = sh.expert(L, e)
        a, b = layout(sc, wt)
        lib.g0_count(ptr(a), ptr(b), ptrs(sc), ptrs(wt), ptr(wc), ptr(scn))
    blob = np.zeros(59408, np.uint8)
    lib.g0_model_build(ptr(wc), ptr(scn), ptr(blob))
    lib.g0_set_anchor(-1)
    return blob, wc, scn


def fnv(b):
    h = 2166136261
    for x in bytes(b):
        h = ((h ^ x) * 16777619) & 0xffffffff
    return h or 1


def encode(blob, anchor, sc, wt):
    a, b = layout(sc, wt)
    out = np.zeros(2 * RAW, np.uint8); n = C.c_size_t(0)
    lib.g0_set_anchor(anchor)
    t = time.perf_counter()
    rc = lib.g0_encode(ptr(blob), ptr(a), ptr(b), ptrs(sc), ptrs(wt), WCHUNK, SCHUNK, ptr(out), out.size, C.byref(n))
    dt = time.perf_counter() - t
    lib.g0_set_anchor(-1)
    assert rc == 0, rc
    used = n.value
    total = (used + 4095) // 4096 * 4096
    return out[:total].copy(), used, total, dt


def decode_check(blob, rec, sc, wt):
    """UNCHANGED src decoder; returns (ok, median bytes, model id, seconds)."""
    a, b = layout(sc, wt)
    so = [np.zeros_like(s) for s in sc]; wo = [np.zeros_like(w) for w in wt]
    med = np.zeros(3, np.uint8); mid = C.c_uint32(0)
    t = time.perf_counter()
    rc = lib.g0_decode(ptr(blob), ptr(rec), rec.size, ptr(a), ptr(b), ptrs(so), ptrs(wo), ptr(med), C.byref(mid))
    dt = time.perf_counter() - t
    ok = rc == 0 and all(np.array_equal(x, y) for x, y in zip(so + wo, sc + wt))
    return ok, rc, med.tolist(), mid.value, dt


def nr2_entries():
    with open(NR2, 'rb') as f:
        hdr = struct.unpack('<IIIIQQII', f.read(40))
        f.seek(hdr[5])
        tab = f.read(hdr[2] * hdr[3] * ENTRY.size)
    return [list(ENTRY.unpack_from(tab, i * ENTRY.size)) for i in range(hdr[2] * hdr[3])], NR2


def prod_median(f, entry):
    f.seek(entry[0])
    return list(f.read(32)[24:27])


def boot(d, n=10000, seed=1):
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    sh = Shards()
    pk = picks()
    tr = [pk[i] for i in range(0, 160, 2)]
    te_idx = [4 * L + (1 if L % 2 == 0 else 3) for L in range(40)]
    te = [pk[i] for i in te_idx]
    assert not set(tr) & set(te)
    ents, _ = nr2_entries()
    
    # 12 largest production records
    t1 = json.load(open(os.path.join(HERE, '..', 't1_record_sizes.json')))['rows']
    big = [(r['i'] // 192, r['i'] % 192) for r in sorted(t1, key=lambda r: -r['total'])[:12]]
    big = [x for x in big if x not in tr and x not in te]

    # Load existing blobs A and B if present, or train
    pathA = os.path.join(out_dir, 'model_A_median.wnm2')
    pathB = os.path.join(out_dir, 'model_B_g0_120.wnm2')
    pathG = os.path.join(out_dir, 'model_G_rule_g.wnm2')

    if os.path.exists(pathA):
        blobA = np.fromfile(pathA, dtype=np.uint8)
    else:
        blobA, _, _ = train(sh, tr, -1)
        blobA.tofile(pathA)

    if os.path.exists(pathB):
        blobB = np.fromfile(pathB, dtype=np.uint8)
    else:
        blobB, _, _ = train(sh, tr, ANCHOR_B)
        blobB.tofile(pathB)

    # Train Model G (Rule g)
    if os.path.exists(pathG):
        blobG = np.fromfile(pathG, dtype=np.uint8)
    else:
        print("Training Model G (Rule g: densest adjacent pair)...", flush=True)
        blobG, _, _ = train(sh, tr, ANCHOR_G)
        blobG.tofile(pathG)

    idA, idB, idG = fnv(blobA), fnv(blobB), fnv(blobG)
    print(f'models A {idA:#x}  B {idB:#x}  G {idG:#x}  prep {time.time() - t0:.1f}s', flush=True)

    rows = []
    fprod = open(NR2, 'rb')
    for set_name, exps in (('test', te), ('big', big)):
        for L, e in exps:
            sc, wt, _ = sh.expert(L, e)
            src_sha = hashlib.sha256(b''.join(x.tobytes() for x in sc + wt)).hexdigest()
            meds = [lower_median(s) for s in sc]
            g_anchors = [rule_g_anchor(s) for s in sc]

            recA, uA, tA, eA = encode(blobA, -1, sc, wt)
            recB, uB, tB, eB = encode(blobB, ANCHOR_B, sc, wt)
            recG, uG, tG, eG = encode(blobG, ANCHOR_G, sc, wt)

            okA, rcA, mA, midA, dA = decode_check(blobA, recA, sc, wt)
            okB, rcB, mB, midB, dB = decode_check(blobB, recB, sc, wt)
            okG, rcG, mG, midG, dG = decode_check(blobG, recG, sc, wt)

            assert okA and okB and okG, f"Decode mismatch on ({L}, {e}): okA={okA}, okB={okB}, okG={okG}"
            assert mA == meds and midA == idA
            assert mB == [ANCHOR_B] * 3 and midB == idB
            assert mG == g_anchors and midG == idG, f"Anchor mismatch G on ({L}, {e}): {mG} vs expected {g_anchors}"

            pm = prod_median(fprod, ents[L * 192 + e])
            row = dict(
                set=set_name, layer=L, expert=e, src_sha256=src_sha,
                medians=meds, g_anchors=g_anchors, prod_medians=pm,
                A_total=tA, B_total=tB, G_total=tG,
                A_used=uA, B_used=uB, G_used=uG,
                delta_G_vs_A_bytes=tG - tA,
                delta_G_vs_B_bytes=tG - tB,
                delta_G_vs_A_pct_raw=(tG - tA) / RAW * 100,
                delta_G_vs_B_pct_raw=(tG - tB) / RAW * 100,
                cpu_decode_ok=True,
                prod_total=ents[L * 192 + e][1]
            )
            rows.append(row)
            print(f"[{set_name:4s}] L{L:02d}_E{e:03d} | meds={meds} g_anc={g_anchors} | A={tA} B={tB} G={tG} | G-A: {tG-tA:+d} ({row['delta_G_vs_A_pct_raw']:+.2f}%) | G-B: {tG-tB:+d} ({row['delta_G_vs_B_pct_raw']:+.2f}%)", flush=True)

    fprod.close()

    T = [r for r in rows if r['set'] == 'test']
    dGA = np.array([r['G_total'] - r['A_total'] for r in T], float)
    dGB = np.array([r['G_total'] - r['B_total'] for r in T], float)
    dBA = np.array([r['B_total'] - r['A_total'] for r in T], float)

    summ = dict(
        model_A_id=idA, model_B_id=idB, model_G_id=idG,
        n_test=len(T),
        all_cpu_decode_exact=all(r['cpu_decode_ok'] for r in rows),
        A_mean_total=float(np.mean([r['A_total'] for r in T])),
        B_mean_total=float(np.mean([r['B_total'] for r in T])),
        G_mean_total=float(np.mean([r['G_total'] for r in T])),
        A_ratio=float(np.mean([r['A_total'] for r in T]) / RAW),
        B_ratio=float(np.mean([r['B_total'] for r in T]) / RAW),
        G_ratio=float(np.mean([r['G_total'] for r in T]) / RAW),
        delta_G_vs_A_bytes=float(dGA.mean()),
        delta_G_vs_A_pct_raw=float(dGA.mean() / RAW * 100),
        delta_G_vs_A_pct_raw_CI95=[x / RAW * 100 for x in boot(dGA)],
        delta_G_vs_B_bytes=float(dGB.mean()),
        delta_G_vs_B_pct_raw=float(dGB.mean() / RAW * 100),
        delta_G_vs_B_pct_raw_CI95=[x / RAW * 100 for x in boot(dGB)],
        delta_B_vs_A_pct_raw=float(dBA.mean() / RAW * 100),
        experts_G_le_A=int((dGA <= 0).sum()),
        experts_G_lt_B=int((dGB < 0).sum()),
        seconds=time.time() - t0
    )

    Bg = [r for r in rows if r['set'] == 'big']
    if Bg:
        db_GA = np.array([r['G_total'] - r['A_total'] for r in Bg], float)
        db_GB = np.array([r['G_total'] - r['B_total'] for r in Bg], float)
        db_BA = np.array([r['B_total'] - r['A_total'] for r in Bg], float)
        summ['big'] = dict(
            n=len(Bg),
            A_max=max(r['A_total'] for r in Bg),
            B_max=max(r['B_total'] for r in Bg),
            G_max=max(r['G_total'] for r in Bg),
            prod_max=max(r['prod_total'] for r in Bg),
            delta_G_vs_A_pct_raw_mean=float(db_GA.mean() / RAW * 100),
            delta_G_vs_B_pct_raw_mean=float(db_GB.mean() / RAW * 100),
            delta_B_vs_A_pct_raw_mean=float(db_BA.mean() / RAW * 100),
            rows=[(r['layer'], r['expert'], r['medians'], r['g_anchors'], r['A_total'], r['B_total'], r['G_total'], r['prod_total']) for r in Bg]
        )

    out_json = os.path.join(out_dir, 't2_rule_g.json')
    json.dump(dict(summary=summ, rows=rows), open(out_json, 'w'), indent=1)
    print("\n" + "="*80)
    print(f"Summary written to {out_json}")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
