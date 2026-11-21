# Independent re-derivation (numpy, own code; only wex_nr3_prepare from the production tool library) of the
# held-out ideal cost of the WNX MAG model: 7 srel contexts per matrix, 8 magnitudes + 1 raw bit per non-zero.
# Train counts on TRAIN, score TEST. Reports exact-probability and 12-bit-quantised (simple largest-remainder) costs.
import ctypes as C, os, struct, numpy as np
D = "<TMP>/wnr3-data"; H = os.path.dirname(os.path.abspath(__file__))
lib = C.CDLL(D + "/libwex_nr3_tools.so")
ENT = struct.Struct("<QIII6H6H3I3II")
def counts(path):
    b = os.path.basename(path)[2:-4]
    z = open(f"{H}/data/Z_{b}.bin", "rb").read(); eb = open(f"{H}/data/E_{b}.bin", "rb").read()
    v = list(ENT.unpack(eb)); v[0], v[1], v[-1] = 0, 0, 0; e0 = ENT.pack(*v)
    coded = C.create_string_buffer(len(z)); st = (C.c_uint32 * 16)()
    assert lib.wex_nr3_prepare(e0, z, 1, coded, st) >= 0
    a = np.frombuffer(coded.raw, np.uint8)
    so, wo = v[4:7], v[10:13]   # scale_offsets[3]? verify below
    return a, v, [st[9 + m] for m in range(3)]
# entry layout: Q offset, I total, I ?, I ?, 6H, 6H, 3I scale_offsets?, 3I weight_offsets?, I flags
def split(v):
    return v[16:19], v[19:22]
def cnt(path):
    a, v, anc = counts(path)
    sofs, wofs = split(v)
    out = np.zeros((3, 7, 16), np.int64); sc = np.zeros((3, 256), np.int64)
    bounds = sorted(list(sofs) + list(wofs)) + [len(a)]
    for m in range(3):
        s0 = sofs[m]; s1 = min(b for b in bounds if b > s0)
        w0 = wofs[m]; w1 = min(b for b in bounds if b > w0)
        s = a[s0:s1]; w = a[w0:w1]; nb = len(s)
        assert len(w) == nb * 16, (len(w), nb)
        srel = np.clip(s.astype(np.int32) - int(anc[m]), -3, 3) + 3
        nib = np.stack([w & 15, w >> 4], 1).reshape(nb, 32)
        idx = np.repeat(srel, 32) * 16 + nib.reshape(-1)
        out[m] = np.bincount(idx, minlength=112).reshape(7, 16)
        sc[m] = np.bincount(s, minlength=256)
    return out, sc
def quant(c, total):
    c = np.asarray(c, np.float64); n = len(c); spare = total - n
    if c.sum() == 0: c = np.ones(n)
    f = 1 + np.floor(c * spare / c.sum()); rem = c * spare / c.sum() - np.floor(c * spare / c.sum())
    for i in np.argsort(-rem, kind="stable")[: int(total - f.sum())]: f[i] += 1
    return f / total
TR = sum(cnt(p)[0] for p in open(H + "/train.txt").read().split())
te = [cnt(p) for p in open(H + "/test.txt").read().split()]
TE = sum(x[0] for x in te)
W = TE.sum(); n = len(te)
mag_tr = np.zeros((3, 7, 8)); mag_te = np.zeros((3, 7, 8))
for s in range(16): mag_tr[:, :, s & 7] += TR[:, :, s]; mag_te[:, :, s & 7] += TE[:, :, s]
signs = W - mag_te[:, :, 0].sum()
pr = np.where(mag_tr.sum(2, keepdims=True) > 0, mag_tr / np.maximum(mag_tr.sum(2, keepdims=True), 1), 1 / 8)
print('empty train contexts (m,srel):', [(m, r, int(mag_te[m, r].sum())) for m in range(3) for r in range(7) if mag_tr[m, r].sum() == 0])
exact = -(mag_te * np.log2(np.where(mag_te > 0, pr, 1))).sum() + signs
q12 = sum(-(mag_te[m, r] * np.log2(quant(mag_tr[m, r], 4096))).sum() for m in range(3) for r in range(7)) + signs
print(f"TEST: {n} records, {W/n:.0f} weights each; -0 nibbles present in coded TEST: {TE[:,:,8].sum()}")
print(f"MAG 7 srel ctx + raw sign: exact-prob {exact/W:.5f} bit/w, 12-bit quantised {q12/W:.5f} bit/w (lane: 3.57662 at 12 bit)")
# order-0 per matrix (no srel) for reference
o0 = sum(-(mag_te[m].sum(0) * np.log2(mag_tr[m].sum(0) / mag_tr[m].sum())).sum() for m in range(3)) + signs
print(f"MAG order-0 per matrix (no srel): {o0/W:.5f} bit/w")
