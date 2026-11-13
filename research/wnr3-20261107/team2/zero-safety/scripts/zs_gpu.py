"""Run colibri's CUDA FP4 expert kernels (backend_cuda_dsv4.cu, via NVRTC + driver API) on the local GPU,
original vs canonicalised (-0 -> +0) MXFP4 records, and compare output bits.
Usage: <venv python> -I zs_gpu.py <kernels.cu> <nvrtc_bin_dir> <sample_dir> <out.json> <files_csv>
"""
import ctypes as C, json, os, sys, time
import numpy as np

src_path, nvrtc_dir, sample_dir, out_path, files_csv = sys.argv[1:6]
os.add_dll_directory(nvrtc_dir)
_builtins = C.CDLL(os.path.join(nvrtc_dir, "nvrtc-builtins64_129.dll"))   # preload: nvrtc LoadLibrary()s it by name
nvrtc = C.CDLL(os.path.join(nvrtc_dir, "nvrtc64_120_0.dll"))
cu = C.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))

def ck(r, what):
    if r != 0:
        s = C.c_char_p(); cu.cuGetErrorString(r, C.byref(s)); raise RuntimeError(f"{what}: {r} {s.value}")
def nk(r, what):
    if r != 0:
        nvrtc.nvrtcGetErrorString.restype = C.c_char_p; raise RuntimeError(f"{what}: {nvrtc.nvrtcGetErrorString(r)}")

names = ["mv_fp4_grouped<1>", "mv_fp4_grouped<2>", "mv_fp4_grouped<4>", "mv_fp4_grouped<8>",
         "mv_fp4_expert_major<4>", "mv<4>", "mv_rows<4,4>", "mv_rows<4,16>", "mm_batch<4>"]
src = open(src_path, "rb").read()
prog = C.c_void_p()
nk(nvrtc.nvrtcCreateProgram(C.byref(prog), C.c_char_p(src), b"zs.cu", 0, None, None), "create")
for n in names: nk(nvrtc.nvrtcAddNameExpression(prog, n.encode()), "nameexpr")
cu.cuInit(0); dev = C.c_int(); ck(cu.cuDeviceGet(C.byref(dev), 0), "dev")
maj, mnr = C.c_int(), C.c_int()
cu.cuDeviceGetAttribute(C.byref(maj), 75, dev); cu.cuDeviceGetAttribute(C.byref(mnr), 76, dev)
devname = C.create_string_buffer(256); cu.cuDeviceGetName(devname, 256, dev)
arch = f"--gpu-architecture=compute_{maj.value}{mnr.value}".encode()
opts = (C.c_char_p * 4)(arch, b"-std=c++17", b"--ftz=false", b"--fmad=true")   # nvcc defaults + Makefile -ftz=false
r = nvrtc.nvrtcCompileProgram(prog, 4, opts)
ls = C.c_size_t(); nvrtc.nvrtcGetProgramLogSize(prog, C.byref(ls)); log = C.create_string_buffer(ls.value); nvrtc.nvrtcGetProgramLog(prog, log)
if r != 0: print(log.value.decode(errors="replace")); nk(r, "compile")
ps = C.c_size_t(); nvrtc.nvrtcGetPTXSize(prog, C.byref(ps)); ptx = C.create_string_buffer(ps.value); nvrtc.nvrtcGetPTX(prog, ptx)
lowered = {}
for n in names:
    p = C.c_char_p(); nk(nvrtc.nvrtcGetLoweredName(prog, n.encode(), C.byref(p)), "lowered"); lowered[n] = p.value
lowered["v4f_rows"] = b"v4f_rows"
ctx = C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "ctx"); ck(cu.cuCtxSetCurrent(ctx), "setctx")
mod = C.c_void_p(); ck(cu.cuModuleLoadData(C.byref(mod), ptx), "module")
KF = {}
for n, l in lowered.items():
    f = C.c_void_p(); ck(cu.cuModuleGetFunction(C.byref(f), mod, l), "fn " + n); KF[n] = f
g = C.c_uint64(); gb = C.c_size_t(); ck(cu.cuModuleGetGlobal_v2(C.byref(g), C.byref(gb), mod, b"e8_table"), "e8_table")
v = np.arange(256)
e8 = np.ldexp(np.float32(1), np.where(v == 0, -127, v - 127)).astype(np.float32)   # dsv4_cuda_init (:1605)
ck(cu.cuMemcpyHtoD_v2(g, C.c_void_p(e8.ctypes.data), C.c_size_t(1024)), "e8 upload")

def dalloc(n):
    p = C.c_uint64(); ck(cu.cuMemAlloc_v2(C.byref(p), C.c_size_t(max(int(n), 4))), "alloc"); return p.value
def h2d(dp, a): ck(cu.cuMemcpyHtoD_v2(C.c_uint64(dp), C.c_void_p(a.ctypes.data), C.c_size_t(a.nbytes)), "h2d")
def d2h(a, dp): ck(cu.cuMemcpyDtoH_v2(C.c_void_p(a.ctypes.data), C.c_uint64(dp), C.c_size_t(a.nbytes)), "d2h")
def launch(name, grid, block, args, shmem=0):
    holders = []; ptrs = (C.c_void_p * len(args))()
    for i, (t, val) in enumerate(args):
        h = t(val); holders.append(h); ptrs[i] = C.addressof(h)
    ck(cu.cuLaunchKernel(KF[name], C.c_uint(grid[0]), C.c_uint(grid[1]), C.c_uint(1), C.c_uint(block), C.c_uint(1), C.c_uint(1),
                         C.c_uint(shmem), None, ptrs, None), "launch " + name)
    ck(cu.cuCtxSynchronize(), "sync " + name)
U64, I32, I64 = C.c_uint64, C.c_int, C.c_longlong

SC, WB = 368640, 5898240
rng = np.random.default_rng(20261107)
def canon(a):
    lo = a & 15; hi = a >> 4
    return (np.where(lo == 8, 0, lo) | (np.where(hi == 8, 0, hi) << 4)).astype(np.uint8)
AN = ["gauss", "gauss_25pct+0_25pct-0", "all-0", "all+0", "one_nonzero_rest-0", "denormals_10pct", "inf_at_100", "nan_at_200", "huge_1e30"]
def act(I, k):
    x = rng.standard_normal(I).astype(np.float32)
    if k == 1: r = rng.integers(0, 4, I); x[r == 0] = 0.0; x[r == 1] = np.float32(-0.0)
    elif k == 2: x[:] = np.float32(-0.0)
    elif k == 3: x[:] = 0.0
    elif k == 4: x[:] = np.float32(-0.0); x[7] = 1.0
    elif k == 5: m = rng.integers(0, 10, I) == 0; x[m] = np.where(rng.integers(0, 2, m.sum()) == 1, np.float32(1e-41), np.float32(-1e-41))
    elif k == 6: x[100] = np.inf
    elif k == 7: x[200] = np.nan
    elif k == 8: x *= np.float32(1e30)
    return x
def batch4(x):
    t = x.copy(); t[1::2] = np.float32(-0.0)
    return np.ascontiguousarray(np.stack([x, -x, t, x * np.float32(1e-30)]).astype(np.float32))

KN = ["mv_fp4_grouped<1>", "mv_fp4_grouped<2>", "mv_fp4_grouped<4>", "mv_fp4_grouped<8>", "mv_fp4_expert_major<4>_T4",
      "mv<4>", "mv_rows<4,4>", "mv_rows<4,16>", "mm_batch<4>_T4", "v4f_dot_fp4(warp/row)"]
def new_stats(): return {k: dict(compared=0, mismatch_value=0, mismatch_signzero_only=0, mismatch_nan_payload=0, zero_outputs=0, neg_zero_outputs=0, nan_outputs=0, rerun_nondeterministic=0) for k in KN}
def compare(st, a, b, a2):
    ua, ub = a.view(np.uint32), b.view(np.uint32); d = ua != ub
    both0 = ((ua | ub) & 0x7fffffff) == 0; bothnan = np.isnan(a) & np.isnan(b)
    st["compared"] += a.size; st["mismatch_signzero_only"] += int((d & both0).sum())
    st["mismatch_nan_payload"] += int((d & bothnan & ~both0).sum()); st["mismatch_value"] += int((d & ~both0 & ~bothnan).sum())
    st["zero_outputs"] += int(((ua & 0x7fffffff) == 0).sum()); st["neg_zero_outputs"] += int((ua == 0x80000000).sum())
    st["nan_outputs"] += int(np.isnan(a).sum()); st["rerun_nondeterministic"] += int((a2.view(np.uint32) != ua).sum())

dbuf = {}
def run_all(w, s, I, O, x1, x4):
    """returns dict kernel -> output (np.float32) for one weight version"""
    dw, ds = dalloc(w.nbytes), dalloc(s.nbytes); h2d(dw, w); h2d(ds, s)
    dx1, dx4 = dalloc(x1.nbytes), dalloc(x4.nbytes); h2d(dx1, x1); h2d(dx4, x4)
    dy = dalloc(4 * O * 4); out = {}
    desc = np.array([dw, ds, dx1, dy], dtype=np.uint64); dd = dalloc(32); h2d(dd, desc)
    for R, k in ((1, 0), (2, 1), (4, 2), (8, 3)):
        launch(f"mv_fp4_grouped<{R}>", ((O + R - 1) // R, 1), 256, [(U64, dd), (I32, 1), (I32, O), (I32, I)])
        y = np.empty(O, np.float32); d2h(y, dy); out[KN[k]] = y
    ex = np.array([0], np.int32); off = np.array([0, 4], np.int32); rt = np.arange(4, dtype=np.int32)
    dex, doff, drt = dalloc(4), dalloc(8), dalloc(16); h2d(dex, ex); h2d(doff, off); h2d(drt, rt)
    launch("mv_fp4_expert_major<4>", ((O + 7) // 8, 1), 256,
           [(U64, dw), (U64, ds), (U64, w.nbytes), (U64, s.nbytes), (U64, dex), (U64, doff), (U64, drt), (U64, dx4),
            (I64, I), (I32, 1), (U64, dy), (U64, dy), (I32, O), (I64, O), (I32, O), (I32, I)])
    y = np.empty(4 * O, np.float32); d2h(y, dy); out[KN[4]] = y
    launch("mv<4>", (O, 1), 256, [(U64, dw), (U64, ds), (U64, dx1), (U64, dy), (I32, O), (I32, I), (I32, 1)])
    y = np.empty(O, np.float32); d2h(y, dy); out[KN[5]] = y
    for R, k in ((4, 6), (16, 7)):
        launch(f"mv_rows<4,{R}>", ((O + R - 1) // R, 1), 256, [(U64, dw), (U64, ds), (U64, dx1), (U64, dy), (I32, O), (I32, I), (I32, 1)])
        y = np.empty(O, np.float32); d2h(y, dy); out[KN[k]] = y
    launch("mm_batch<4>", (O, 1), 256, [(U64, dw), (U64, ds), (U64, dx4), (U64, dy), (I32, O), (I32, I), (I32, 4)])
    y = np.empty(4 * O, np.float32); d2h(y, dy); out[KN[8]] = y
    launch("v4f_rows", ((O + 7) // 8, 1), 256, [(U64, dw), (U64, ds), (U64, dx1), (U64, dy), (I32, I), (I32, O)])
    y = np.empty(O, np.float32); d2h(y, dy); out[KN[9]] = y
    for p in (dw, ds, dx1, dx4, dy, dd, dex, doff, drt): cu.cuMemFree_v2(C.c_uint64(p))
    return out

res = {"real_matrices": new_stats(), "synthetic_zero_rows": new_stats(), "positive_control": new_stats()}
t0 = time.time(); meta = {"device": devname.value.decode(), "arch": arch.decode(), "nvrtc": "12.9 (nvidia-cuda-nvrtc-cu12 12.9.86)", "files": []}
dumped = [False]
def test(group, wA, wB, s, I, O, acts):
    for a in acts:
        x1 = np.ascontiguousarray(act(I, a)); x4 = batch4(x1)
        oa = run_all(wA, s, I, O, x1, x4); ob = run_all(wB, s, I, O, x1, x4); oa2 = run_all(wA, s, I, O, x1, x4)
        if not dumped[0] and group == "real_matrices":   # first real case (w1 of the first file, gauss): for the CPU cross-check
            np.savez(out_path + ".xcheck.npz", x1=x1, x4=x4, I=I, O=O, **{k.replace("<", "_").replace(">", "_").replace(",", "_").replace("(", "_").replace(")", "_").replace("/", "_"): v for k, v in oa.items()})
            dumped[0] = True
        for k in KN: compare(res[group][k], oa[k], ob[k], oa2[k])
for fi, fn in enumerate([f for f in files_csv.split(",") if f]):
    rec = np.fromfile(os.path.join(sample_dir, fn), dtype=np.uint8); meta["files"].append(fn)
    mats = [(rec[3*SC:3*SC+WB].copy(), rec[0:SC].copy(), 5120, 2304), (rec[3*SC+WB:3*SC+2*WB].copy(), rec[SC:2*SC].copy(), 2304, 5120),
            (rec[3*SC+2*WB:].copy(), rec[2*SC:3*SC].copy(), 5120, 2304)]
    for (w, s, I, O) in mats: test("real_matrices", w, canon(w), s, I, O, range(9))
    for (w, s, I, O) in mats[:2]:
        rb = I // 2; sa = w[:64*rb].copy(); ss = s[:64*(I//32)].copy()
        sa[:32*rb] = (np.where(rng.integers(0, 2, 32*rb) == 1, 8, 0) | (np.where(rng.integers(0, 2, 32*rb) == 1, 8, 0) << 4)).astype(np.uint8)
        test("synthetic_zero_rows", sa, canon(sa), ss, I, 64, range(9))
    if fi == 0:
        w, s, I, O = mats[0]; c = canon(w); c2 = c.copy(); j = int(np.nonzero((c2[:I//2] & 15) == 1)[0][0]); c2[j] = (c2[j] & 0xF0) | 9
        test("positive_control", c, c2, s, I, O, [0])
    print("gpu", fn, round(time.time() - t0, 1), "s", flush=True)
for gname in res: res[gname] = {k: v for k, v in res[gname].items() if v["compared"]}
out = {"meta": meta, "act_sets": AN, "batch4_rows": ["x", "-x", "x with odd entries -0", "x*1e-30"], "results": res, "seconds": round(time.time() - t0, 1)}
json.dump(out, open(out_path, "w"), indent=1); print(json.dumps(out, indent=1))
