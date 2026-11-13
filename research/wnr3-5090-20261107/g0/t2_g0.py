"""Task 2: WNR2-g0 encoder-only test on V4.1 REAP-192 (lab only; reads U:/E:, writes only OUT_DIR/BIN_DIR).

Same split as team 4 ceil-ref: sample = fetch_experts.py picks (seed 20261107, 4 experts per layer, 160),
train = the 80 even indices, test = 40 odd indices, one per layer (4L+1 for even L, 4L+3 for odd L).
Model A = normal WNR2 (srel anchor = lower median, as production); model B = g0 (anchor fixed at 120).
Both trained with the KAIRA trainer (wex_nr2_count/model_build) from the scratch copy wex_neural_rans_g0.h,
records encoded with the KAIRA reference encoder of that copy and decoded with the UNCHANGED src decoder.

    python -I t2_g0.py OUT_DIR [BIN_DIR]
BIN_DIR: writes records_A.bin / records_B.bin for test_decode_batch.exe (GPU decoder wex_cuda.cuh).
"""
import ctypes as C, hashlib, json, os, random, struct, sys, time
import numpy as np

MODEL_DIR = r'U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
NR2 = r'E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = 18800640
WCHUNK, SCHUNK = 2048, 512
ANCHOR = 120
TEAM4 = r'<HOME>\Wex-WRN2\research\wnr3-20261107\team4\ceil-ref\results\heldout_A.jsonl'
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


def train(sh, idx, anchor):
    wc = np.zeros(3 * 567 * 16, np.uint64); scn = np.zeros(3 * 256, np.uint64)
    lib.g0_set_anchor(anchor)
    for L, e in idx:
        sc, wt, _ = sh.expert(L, e)
        a, b = layout(sc, wt)
        lib.g0_count(ptr(a), ptr(b), ptrs(sc), ptrs(wt), ptr(wc), ptr(scn))
    blob = np.zeros(59408, np.uint8)
    lib.g0_model_build(ptr(wc), ptr(scn), ptr(blob))
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


def slab(entry, sc, wt):
    sb, wb = entry[2], entry[3]
    raw = np.zeros(sb + wb, np.uint8)
    so, wo = entry[16:19], entry[19:22]
    for m in range(3):
        assert entry[4 + 2 * m] * entry[5 + 2 * m] == sc[m].size and entry[10 + 2 * m] * entry[11 + 2 * m] == wt[m].size
        raw[so[m]:so[m] + sc[m].size] = sc[m]
        raw[wo[m]:wo[m] + wt[m].size] = wt[m]
    return raw


def boot(d, n=10000, seed=1):
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(out_dir, bin_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    sh = Shards()
    pk = picks()
    tr = [pk[i] for i in range(0, 160, 2)]
    te_idx = [4 * L + (1 if L % 2 == 0 else 3) for L in range(40)]
    te = [pk[i] for i in te_idx]
    assert not set(tr) & set(te)
    ents, _ = nr2_entries()
    # supplementary: the 12 largest production records (task 1), if not in the training split
    t1 = json.load(open(os.path.join(out_dir, '..', 't1_record_sizes.json')))['rows']
    big = [(r['i'] // 192, r['i'] % 192) for r in sorted(t1, key=lambda r: -r['total'])[:12]]
    big = [x for x in big if x not in tr and x not in te]

    blobA, wcA, scA = train(sh, tr, -1)
    blobB, wcB, scB = train(sh, tr, ANCHOR)
    idA, idB = fnv(blobA), fnv(blobB)
    for nm, bl in (('model_A_median.wnm2', blobA), ('model_B_g0_120.wnm2', blobB)):
        bl.tofile(os.path.join(out_dir, nm))
    print(f'models A {idA:#x} B {idB:#x}  train {time.time() - t0:.1f}s', flush=True)

    team4 = {}
    if os.path.exists(TEAM4):
        for l in open(TEAM4):
            r = json.loads(l)
            if r['config'] == 'wnr2':
                team4[(r['layer'], r['expert'])] = (r['sizes']['total'], r['model_id'], r['anchors'])

    bins = {}
    if bin_dir:
        os.makedirs(bin_dir, exist_ok=True)
        for k, bl in (('A', blobA), ('B', blobB)):
            f = open(os.path.join(bin_dir, f'records_{k}.bin'), 'wb')
            f.write(struct.pack('<I', bl.size)); f.write(bl.tobytes()); f.write(struct.pack('<I', len(te)))
            bins[k] = f
    rows = []
    fprod = open(NR2, 'rb')
    for set_name, exps in (('test', te), ('big', big)):
        for L, e in exps:
            sc, wt, info = sh.expert(L, e)
            src_sha = hashlib.sha256(b''.join(x.tobytes() for x in sc + wt)).hexdigest()
            meds = [lower_median(s) for s in sc]
            recA, uA, tA, eA = encode(blobA, -1, sc, wt)
            recB, uB, tB, eB = encode(blobB, ANCHOR, sc, wt)
            recM, uM, tM, _ = encode(blobB, ANCHOR + 1, sc, wt)     # anchor byte != training anchor (cost control)
            okA, rcA, mA, midA, dA = decode_check(blobA, recA, sc, wt)
            okB, rcB, mB, midB, dB = decode_check(blobB, recB, sc, wt)
            okM, rcM, mM, _, _ = decode_check(blobB, recM, sc, wt)
            # anchor assertions: record byte == the anchor the model was trained with
            assert mB == [ANCHOR] * 3 and midB == idB, (L, e, mB)
            assert mA == meds and midA == idA, (L, e, mA, meds)
            assert mM == [ANCHOR + 1] * 3
            pm = prod_median(fprod, ents[L * 192 + e])
            row = dict(set=set_name, layer=L, expert=e, src_sha256=src_sha, medians=meds, prod_medians=pm,
                       A_used=uA, A_total=tA, B_used=uB, B_total=tB, M_total=tM,
                       cpu_decode_A=okA, cpu_decode_B=okB, cpu_decode_M=okM, rc=[rcA, rcB, rcM],
                       enc_s=[round(eA, 3), round(eB, 3)], dec_s=[round(dA, 3), round(dB, 3)],
                       prod_total=ents[L * 192 + e][1])
            if (L, e) in team4:
                row['team4_A_total'], row['team4_model_id'], row['team4_anchors'] = team4[(L, e)]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('set', 'layer', 'expert', 'medians', 'A_total', 'B_total', 'M_total',
                                                  'cpu_decode_A', 'cpu_decode_B', 'cpu_decode_M')}), flush=True)
            if bins and set_name == 'test':
                raw = slab(ents[L * 192 + e], sc, wt)
                for k, rec in (('A', recA), ('B', recB)):
                    ent = list(ents[L * 192 + e]); ent[0] = 0; ent[1] = rec.size
                    bins[k].write(ENTRY.pack(*ent)); bins[k].write(struct.pack('<I', raw.size))
                    bins[k].write(raw.tobytes()); bins[k].write(rec.tobytes())
    for f in bins.values():
        f.close()

    T = [r for r in rows if r['set'] == 'test']
    dB = np.array([r['B_total'] - r['A_total'] for r in T], float)
    dU = np.array([r['B_used'] - r['A_used'] for r in T], float)
    dM = np.array([r['M_total'] - r['B_total'] for r in T], float)
    summ = dict(
        model_A_id=idA, model_B_id=idB, train=tr, test=te, big=big,
        n_test=len(T), all_cpu_decode_exact=all(r['cpu_decode_A'] and r['cpu_decode_B'] and r['cpu_decode_M'] for r in rows),
        A_mean_total=float(np.mean([r['A_total'] for r in T])), B_mean_total=float(np.mean([r['B_total'] for r in T])),
        A_ratio=float(np.mean([r['A_total'] for r in T]) / RAW), B_ratio=float(np.mean([r['B_total'] for r in T]) / RAW),
        delta_total_bytes=float(dB.mean()), delta_total_pct_raw=float(dB.mean() / RAW * 100),
        delta_total_pct_raw_CI95=[x / RAW * 100 for x in boot(dB)],
        delta_used_pct_raw=float(dU.mean() / RAW * 100), delta_used_pct_raw_CI95=[x / RAW * 100 for x in boot(dU)],
        median_expert_delta_pct_raw=float(np.median(dB) / RAW * 100),
        experts_B_smaller=int((dB < 0).sum()), experts_B_larger=int((dB > 0).sum()),
        mismatch_anchor121_vs_B_pct_raw=float(dM.mean() / RAW * 100),
        matrices_with_median_ne_120=sum(sum(m != ANCHOR for m in r['medians']) for r in T),
        team4_conformance=dict(
            experts_compared=sum(1 for r in T if 'team4_A_total' in r),
            A_total_equal=sum(1 for r in T if r.get('team4_A_total') == r['A_total']),
            model_id_equal=all(r.get('team4_model_id', idA) == idA for r in T)),
        medians_equal_production=sum(1 for r in rows if r['medians'] == r['prod_medians']),
        seconds=time.time() - t0)
    Bg = [r for r in rows if r['set'] == 'big']
    if Bg:
        db = np.array([r['B_total'] - r['A_total'] for r in Bg], float)
        summ['big'] = dict(n=len(Bg), A_max=max(r['A_total'] for r in Bg), B_max=max(r['B_total'] for r in Bg),
                           prod_max=max(r['prod_total'] for r in Bg),
                           delta_pct_raw_mean=float(db.mean() / RAW * 100),
                           rows=[(r['layer'], r['expert'], r['medians'], r['A_total'], r['B_total'], r['prod_total']) for r in Bg])
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(out_dir, 't2_g0.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == '__main__':
    main(*sys.argv[1:])
