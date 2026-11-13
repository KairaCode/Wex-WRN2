"""Tensor-core signed-zero micro-test on the local GPU (sm_86): mma.sync m16n8k16 f16 x f16 -> f32.
A = decoded e2m1 values as fp16 (exact), code 8 as -0 (0x8000) vs canonical +0 (0x0000); B random fp16;
C = +0 (or -0 / random). Compares D bits. Also all-zero A (random zero signs) to see the sign of an exact-zero D.
Usage: <venv python> -I zs_mma.py <nvrtc_bin_dir> <out.json>"""
import ctypes as C, json, os, sys
import numpy as np
nvrtc_dir, out_path = sys.argv[1:3]
os.add_dll_directory(nvrtc_dir); C.CDLL(os.path.join(nvrtc_dir, "nvrtc-builtins64_129.dll"))
nvrtc = C.CDLL(os.path.join(nvrtc_dir, "nvrtc64_120_0.dll")); cu = C.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
def ck(r, w):
    if r: raise RuntimeError(f"{w}: {r}")
SRC = rb'''
extern "C" __global__ void mma_f16(const unsigned* A,const unsigned* B,const float* Cin,float* D){
  int t=blockIdx.x, lane=threadIdx.x;
  const unsigned* a=A+(size_t)t*128+lane*4; const unsigned* b=B+(size_t)t*64+lane*2; const float* c=Cin+(size_t)t*128+lane*4;
  float d0,d1,d2,d3;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
  float* d=D+(size_t)t*128+lane*4; d[0]=d0; d[1]=d1; d[2]=d2; d[3]=d3;
}
extern "C" __global__ void mma_bf16(const unsigned* A,const unsigned* B,const float* Cin,float* D){
  int t=blockIdx.x, lane=threadIdx.x;
  const unsigned* a=A+(size_t)t*128+lane*4; const unsigned* b=B+(size_t)t*64+lane*2; const float* c=Cin+(size_t)t*128+lane*4;
  float d0,d1,d2,d3;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
  float* d=D+(size_t)t*128+lane*4; d[0]=d0; d[1]=d1; d[2]=d2; d[3]=d3;
}'''
prog = C.c_void_p(); ck(nvrtc.nvrtcCreateProgram(C.byref(prog), C.c_char_p(SRC), b"mma.cu", 0, None, None), "create")
cu.cuInit(0); dev = C.c_int(); ck(cu.cuDeviceGet(C.byref(dev), 0), "dev")
opts = (C.c_char_p * 1)(b"--gpu-architecture=compute_86")
r = nvrtc.nvrtcCompileProgram(prog, 1, opts)
if r:
    ls = C.c_size_t(); nvrtc.nvrtcGetProgramLogSize(prog, C.byref(ls)); lg = C.create_string_buffer(ls.value); nvrtc.nvrtcGetProgramLog(prog, lg); print(lg.value.decode()); raise SystemExit(1)
ps = C.c_size_t(); nvrtc.nvrtcGetPTXSize(prog, C.byref(ps)); ptx = C.create_string_buffer(ps.value); nvrtc.nvrtcGetPTX(prog, ptx)
ctx = C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "ctx"); ck(cu.cuCtxSetCurrent(ctx), "set")
mod = C.c_void_p(); ck(cu.cuModuleLoadData(C.byref(mod), ptx), "mod")
KF = {}
for n in (b"mma_f16", b"mma_bf16"):
    f = C.c_void_p(); ck(cu.cuModuleGetFunction(C.byref(f), mod, n), "fn"); KF[n] = f
def dalloc(n): p = C.c_uint64(); ck(cu.cuMemAlloc_v2(C.byref(p), C.c_size_t(n)), "alloc"); return p.value
def h2d(dp, a): ck(cu.cuMemcpyHtoD_v2(C.c_uint64(dp), C.c_void_p(a.ctypes.data), C.c_size_t(a.nbytes)), "h2d")
def d2h(a, dp): ck(cu.cuMemcpyDtoH_v2(C.c_void_p(a.ctypes.data), C.c_uint64(dp), C.c_size_t(a.nbytes)), "d2h")
def mma(kind, A16, B16, Cf):
    T = Cf.shape[0]
    dA, dB, dC, dD = dalloc(A16.nbytes), dalloc(B16.nbytes), dalloc(Cf.nbytes), dalloc(Cf.nbytes)
    h2d(dA, A16); h2d(dB, B16); h2d(dC, Cf)
    hold = [C.c_uint64(dA), C.c_uint64(dB), C.c_uint64(dC), C.c_uint64(dD)]; ptrs = (C.c_void_p * 4)(*[C.addressof(h) for h in hold])
    ck(cu.cuLaunchKernel(KF[kind], C.c_uint(T), C.c_uint(1), C.c_uint(1), C.c_uint(32), C.c_uint(1), C.c_uint(1), C.c_uint(0), None, ptrs, None), "launch")
    ck(cu.cuCtxSynchronize(), "sync"); D = np.empty_like(Cf); d2h(D, dD)
    for p in (dA, dB, dC, dD): cu.cuMemFree_v2(C.c_uint64(p))
    return D
rng = np.random.default_rng(7)
T = 20000
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6])
def to16(vals, kind):
    if kind == b"mma_f16": return vals.astype(np.float16).view(np.uint16)
    f = vals.astype(np.float32).view(np.uint32); return ((f + 0x7fff + ((f >> 16) & 1)) >> 16).astype(np.uint16)   # RNE to bf16 (exact here)
res = {}
for kind in (b"mma_f16", b"mma_bf16"):
    codes = rng.integers(0, 16, (T, 256))                         # 16x16 A tile per trial, realistic: all 16 codes
    Aorig = E2M1[codes]; Acan = E2M1[np.where(codes == 8, 0, codes)]
    Bv = rng.standard_normal((T, 128)) * rng.choice([1e-3, 1, 30], (T, 1))
    a1 = to16(Aorig, kind).view(np.uint32).reshape(T, 128).copy(); a2 = to16(Acan, kind).view(np.uint32).reshape(T, 128).copy()
    b = to16(Bv, kind).view(np.uint32).reshape(T, 64).copy()
    assert (a1 != a2).any()
    out = {}
    for cname, Cf in (("C=+0", np.zeros((T, 128), np.float32)), ("C=-0", np.full((T, 128), -0.0, np.float32)),
                      ("C=random", rng.standard_normal((T, 128)).astype(np.float32))):
        D1 = mma(kind, a1, b, Cf); D2 = mma(kind, a2, b, Cf)
        out[f"random A ({T} tiles), {cname}: differing D"] = int((D1.view(np.uint32) != D2.view(np.uint32)).sum())
    # all-zero A with random zero signs vs all +0: sign of an exactly-zero D
    zs = np.where(rng.integers(0, 2, (T, 256)) == 1, -0.0, 0.0); zp = np.zeros((T, 256))
    az1 = to16(zs, kind).view(np.uint32).reshape(T, 128).copy(); az2 = to16(zp, kind).view(np.uint32).reshape(T, 128).copy()
    for cname, Cf in (("C=+0", np.zeros((T, 128), np.float32)), ("C=-0", np.full((T, 128), -0.0, np.float32))):
        D1 = mma(kind, az1, b, Cf); D2 = mma(kind, az2, b, Cf)
        u1, u2 = D1.view(np.uint32), D2.view(np.uint32)
        out[f"all-zero A (random signs vs +0), {cname}: differing D"] = int((u1 != u2).sum())
        out[f"all-zero A (random signs), {cname}: D==-0 count / D==+0 count"] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
        out[f"all-zero A (+0), {cname}: D==-0 count / D==+0 count"] = [int((u2 == 0x80000000).sum()), int((u2 == 0).sum())]
    # all-negative-zero products: A all -0, B all positive (every product is -0)
    am = to16(np.full((T, 256), -0.0), kind).view(np.uint32).reshape(T, 128).copy(); bp = to16(np.abs(Bv) + 0.1, kind).view(np.uint32).reshape(T, 64).copy()
    for cname, Cf in (("C=+0", np.zeros((T, 128), np.float32)), ("C=-0", np.full((T, 128), -0.0, np.float32))):
        D1 = mma(kind, am, bp, Cf); u1 = D1.view(np.uint32)
        out[f"A all -0, B all >0 (every product -0), {cname}: D==-0 / D==+0"] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
    res[kind.decode()] = out
dn = C.create_string_buffer(256); cu.cuDeviceGetName(dn, 256, dev)
res["device"] = dn.value.decode()
json.dump(res, open(out_path, "w"), indent=1); print(json.dumps(res, indent=1))
