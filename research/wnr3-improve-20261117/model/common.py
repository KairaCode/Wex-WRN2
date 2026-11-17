"""Shared helpers: data access, quantisers, the C prototype (mlib.so) and the tool library."""
import ctypes, json, os, struct, hashlib, numpy as np
D = "/tmp/wnr3-data"; W = "/tmp/wnr3-research/model"
RAW = 18800640
NC = 567
PREP = json.load(open(W + "/prep.json"))
TABLE = open(D + "/recs/entries.bin", "rb").read()
PROD_BLOB = open(D + "/recs/model.bin", "rb").read()
TAGS = [r["tag"] for r in PREP]
BY = {r["tag"]: r for r in PREP}
EXTRA = json.load(open(W + "/prep_extra.json")) if os.path.exists(W + "/prep_extra.json") else []
BY.update({r["tag"]: r for r in EXTRA})   # the 37 anchor-shifted records (never used for training)

SPLIT = os.environ.get("SPLIT", "A")
SFX = "" if SPLIT == "A" else "_split" + SPLIT
def split():
    """A (the README's): TRAIN = experts (37l+11)%192 and (37l+101)%192, TEST = (53l+7)%192 + L39 E120.
    B (robustness check): TRAIN = (37l+11) and (53l+7), TEST = (37l+101) + L39 E120."""
    train, test = [], []
    for r in PREP:
        l, e = r["layer"], r["expert"]
        tr = ((37 * l + 11) % 192, (37 * l + 101) % 192) if SPLIT == "A" else ((37 * l + 11) % 192, (53 * l + 7) % 192)
        if e in tr and not (l == 39 and e == 120):
            train.append(r["tag"])
        else:
            test.append(r["tag"])
    return train, test

def entry(tag):
    l, e = BY[tag]["layer"], BY[tag]["expert"]
    return TABLE[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]

def coded(tag):
    return open(W + "/coded/%s.bin" % tag, "rb").read()

def counts(tag):
    z = np.load(W + "/counts/%s.npz" % tag)
    return z["wc"].astype(np.int64).reshape(3, NC, 16), z["sc"].astype(np.int64).reshape(3, 256)

def sum_counts(tags):
    wc = np.zeros((3, NC, 16), np.int64); sc = np.zeros((3, 256), np.int64)
    for t in tags:
        a, b = counts(t); wc += a; sc += b
    return wc, sc

# ---------------- quantisers: counts [..., n] -> frequencies [..., n] summing to 2^P ----------------
def q_rule(c, P):
    """wex_nr2_quantize generalised to 2^P: f = 1 + floor(c*spare/total), remainders largest first (ties: lower symbol)."""
    c = np.asarray(c, np.int64); shp = c.shape; c = c.reshape(-1, shp[-1]); n = shp[-1]; M = 1 << P
    out = np.zeros_like(c)
    for t in range(c.shape[0]):
        tot = int(c[t].sum()); spare = M - n
        if tot:
            sc = c[t] * spare; f = 1 + sc // tot; rem = sc % tot
        else:
            f = np.ones(n, np.int64) + spare // n; rem = np.full(n, spare % n)
        extra = M - int(f.sum())
        order = sorted(range(n), key=lambda s: (-int(rem[s]), s))
        for s in order[:extra]: f[s] += 1
        out[t] = f
    return out.reshape(shp)

def q_opt(c, P, minf=None):
    """Frequencies minimising sum c_s * log2(M / f_s) subject to sum f = M and f_s >= minf_s (default 1).
    Separable concave objective: greedy start + pairwise exchange until no improving move = global optimum."""
    c = np.asarray(c, np.float64); shp = c.shape; c = c.reshape(-1, shp[-1]); n = shp[-1]; M = 1 << P
    if minf is None: minf = np.ones(n, np.int64)
    minf = np.asarray(minf, np.int64)
    out = np.zeros(c.shape, np.int64)
    for t in range(c.shape[0]):
        ct = c[t]; tot = ct.sum()
        if tot == 0:
            out[t] = q_rule(np.zeros(n, np.int64) + (minf > 0), P) if minf.min() == 0 else q_rule(np.zeros(n, np.int64), P)
            out[t][minf == 0] = 0; out[t][np.argmax(out[t])] += M - out[t].sum()
            continue
        f = np.maximum(minf, np.floor(ct * (M - minf.sum()) / tot).astype(np.int64))
        f = np.maximum(f, minf)
        with np.errstate(divide="ignore", invalid="ignore"):
            while f.sum() < M:
                g = np.where(f > 0, ct * np.log1p(1.0 / np.maximum(f, 1)), np.where(ct > 0, np.inf, 0.0))
                g[(minf == 0) & (ct == 0)] = -1
                k = int(f.sum()); need = M - k
                # add `need` units greedily (batch: top gains, one unit each, repeat)
                idx = np.argsort(-g)[:need]; f[idx] += 1
            while f.sum() > M:
                loss = np.where(f > minf, ct * np.log(f / np.maximum(f - 1, 1e-300)), np.inf)
                loss[(f > minf) & (f == 1)] = np.where(ct[(f > minf) & (f == 1)] > 0, np.inf, 0)
                f[int(np.argmin(loss))] -= 1
            for _ in range(100000):
                g = ct * np.log1p(1.0 / np.maximum(f, 1)); g[f == 0] = np.where(ct[f == 0] > 0, np.inf, -1)
                g[(minf == 0) & (ct == 0)] = -1
                loss = np.where(f > minf, ct * np.log(f / np.maximum(f - 1, 1)), np.inf)
                loss[(f == 1) & (f > minf)] = np.where(ct[(f == 1) & (f > minf)] > 0, np.inf, 0)
                a = int(np.argmax(g)); b = int(np.argmin(loss))
                if a == b or g[a] <= loss[b] + 1e-12: break
                f[a] += 1; f[b] -= 1
        out[t] = f
    return out.reshape(shp)

def cdf_of(f):
    f = np.asarray(f, np.int64)
    z = np.zeros(f.shape[:-1] + (f.shape[-1] + 1,), np.int64); z[..., 1:] = np.cumsum(f, -1)
    return z

def ideal_bits(c, f, P):
    c = np.asarray(c, np.float64); f = np.asarray(f, np.float64)
    if np.any((c > 0) & (f == 0)): return np.inf
    with np.errstate(divide="ignore"):
        return float(np.sum(np.where(c > 0, c * (P - np.log2(np.maximum(f, 1))), 0.0)))

def blob_freqs(blob):
    """WNM2 blob -> weight freqs [3,567,16], scale freqs [3,256] (P = 14)."""
    a = np.frombuffer(blob, np.uint16, offset=32, count=3 * NC * 17).reshape(3, NC, 17).astype(np.int64)
    b = np.frombuffer(blob, np.uint16, offset=32 + 3 * NC * 34, count=3 * 257).reshape(3, 257).astype(np.int64)
    return np.diff(a, axis=-1), np.diff(b, axis=-1)

# ---------------- C prototype ----------------
LIB = ctypes.CDLL(W + "/mlib.so")
TOOLS = ctypes.CDLL(D + "/libwex_nr3_tools.so")
class Mdl(ctypes.Structure):
    _fields_ = [("P", ctypes.c_uint32), ("K", ctypes.c_uint32), ("KS", ctypes.c_uint32), ("smode", ctypes.c_uint32),
                ("wcdf", ctypes.c_void_p), ("scdf", ctypes.c_void_p), ("ncls", ctypes.c_uint32), ("thr", ctypes.c_void_p),
                ("ns", ctypes.c_uint32), ("smap", ctypes.c_void_p), ("sval", ctypes.c_void_p)]
LIB.m_lloyd.restype = ctypes.c_double
LIB.m_lloyd2.restype = ctypes.c_double
LIB.m_lloyd2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
LIB.m_lloyd.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

class Model:
    """wf: weight freqs [K,3,567,16]; sf: scale freqs [KS,3,256]; P bits."""
    def __init__(self, wf, sf, P, smode=0, model_id=0x27e8a1e0, thr=(), K=None, salpha=None):
        wf = np.asarray(wf); sf = np.asarray(sf)
        if wf.ndim == 3: wf = wf[None]
        if sf.ndim == 2: sf = sf[None]
        assert np.all(wf.sum(-1) == 1 << P) and np.all(sf.sum(-1) == 1 << P)
        self.wf, self.sf, self.P, self.smode, self.model_id = wf, sf, P, smode, model_id
        self.wc = np.ascontiguousarray(cdf_of(wf).astype(np.uint32))
        sc = cdf_of(sf); full = np.zeros(sf.shape[:-1] + (257,), np.uint32); full[..., :sc.shape[-1]] = sc; self.sc = np.ascontiguousarray(full)
        self.smap = np.full(256, 255, np.uint8); self.sval = np.zeros(256, np.uint8); ns = sf.shape[-1]
        if smode == 2:
            assert len(salpha) == ns - 1
            for i, v in enumerate(salpha): self.smap[v & 255] = i; self.sval[i] = v & 255
        self.thr = np.ascontiguousarray(np.asarray(thr, np.int32).reshape(-1)) if len(thr) else np.zeros(1, np.int32)
        ncls = len(thr) + 1; K = K or wf.shape[0] // ncls; assert K * ncls == wf.shape[0]
        self.K, self.ncls = K, ncls
        self.m = Mdl(P, K, sf.shape[0], smode, self.wc.ctypes.data, self.sc.ctypes.data, ncls, self.thr.ctypes.data,
                     ns, self.smap.ctypes.data, self.sval.ctypes.data)
    def table_bytes(self):
        """container bytes for the tables as u16 frequencies (one fewer per table is implied)"""
        return 2 * (self.wf.size - self.wf.shape[0] * 3 * NC + self.sf.size - self.sf.shape[0] * 3)
    def encode(self, tag, wchunk=2048, schunk=512):
        r = BY[tag]; cap = 2 * RAW; out = ctypes.create_string_buffer(cap); used = ctypes.c_size_t()
        sel = (ctypes.c_uint8 * 40000)()
        rc = LIB.m_encode(ctypes.byref(self.m), entry(tag), coded(tag), wchunk, schunk, r["flags"],
                          (r["layer"] << 16) | r["expert"], self.model_id, out, ctypes.c_size_t(cap), ctypes.byref(used), sel)
        assert rc == 0
        return out.raw[:used.value], bytes(sel)
    def decode_ok(self, tag, rec):
        out = ctypes.create_string_buffer(RAW)
        rc = LIB.m_decode(ctypes.byref(self.m), entry(tag), rec, ctypes.c_size_t(len(rec)), self.model_id, out)
        return rc == 0 and hashlib.sha256(out.raw).hexdigest() == BY[tag]["zsha"]

def stream_counts():
    return 3 * 720, 3 * 5760
