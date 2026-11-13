# SPDX-License-Identifier: Apache-2.0
# Derived from colibri c/backend_cuda_dsv4.cu (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
# Modified by Ludovico Zanichelli, 2026.
"""gpu_probe.py - does FP4 -0 vs +0 change any GPU result bit on this RTX 3070 (sm_86)?

Compiles with NVRTC (pip wheel in the venv) and runs, on REAL V4.1 expert matrices with original
codes vs canonical (-0 -> +0) codes:
  K1 mv_fp4_grouped<4>  - copied verbatim from colibri backend_cuda_dsv4.cu:99,127-143 (f4() gives -0)
  K2 v4f_dot_fp4        - copied from backend_cuda_dsv4.cu:2250-2257,2279-2289 (fused6 decode MoE,
                          shared-memory LUT whose entry 8 is -0.f, line 2313)
  K3 FP16 tensor-core mma.sync.m16n8k16 (f32 accumulate) over the full K of the matrix, weights as
     E2M1 values in f16 (code 8 -> 0x8000), 8 activation columns; plus all-(-0)/all-(+0) tiles to
     characterise the tensor core's signed-zero behaviour.
Run: venv\\Scripts\\python.exe -I gpu_probe.py <sample_dir> <out_json> [n_experts]
"""
import json, os, sys, ctypes
import numpy as np
from cuda.bindings import driver as cu, nvrtc

SRC = r'''
__constant__ float e8_table[256];
__device__ float e8(unsigned char b){return e8_table[b];}
__device__ float f4(unsigned char q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
struct MvDesc { const unsigned char *w,*scale; const float *x; float *y; };
#define R 4
extern "C" __global__ void mv_fp4_grouped4(const MvDesc *desc,int count,int O,int I){
    int matrix=blockIdx.y,base=blockIdx.x*R;if(matrix>=count)return;MvDesc d=desc[matrix];float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){unsigned char p=d.w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;
            sum[r]+=d.x[i]*f4(q)*e8(d.scale[(long long)o*(I/32)+i/32]);}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)d.y[base+r]=s[r][0];}
}
__device__ __forceinline__ float v4f_tree(float p[8]){
    #pragma unroll
    for(int k=16;k;k>>=1){
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__shfl_down_sync(0xffffffffu,p[r],k));}
    p[0]=__fadd_rn(p[0],p[4]);p[1]=__fadd_rn(p[1],p[5]);p[2]=__fadd_rn(p[2],p[6]);p[3]=__fadd_rn(p[3],p[7]);
    p[0]=__fadd_rn(p[0],p[2]);p[1]=__fadd_rn(p[1],p[3]);
    return __fadd_rn(p[0],p[1]);}
__device__ __forceinline__ float v4f_e8f(unsigned b){ return b? __uint_as_float(b<<23) : __uint_as_float(0x00400000u); }
__device__ __forceinline__ float v4f_dot_fp4(const unsigned char* __restrict__ w,const unsigned char* __restrict__ srow,const float* xs,const float* lut,int I,int lane){
    float p[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int m=0;m<I;m+=256){
        int i=m+8*lane;
        unsigned u=__ldcs((const unsigned*)(w+(i>>1)));
        float sc=v4f_e8f(__ldg(srow+(i>>5)));
        const float4 *xp=(const float4*)(xs+i); float4 xa=xp[0],xb=xp[1];
        const float xv[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__fmul_rn(__fmul_rn(xv[r],lut[(u>>(4*r))&15]),sc));}
    return v4f_tree(p);}
extern "C" __global__ void fused_dot(const unsigned char*w,const unsigned char*s,const float*x,float*y,int O,int I){
    __shared__ __align__(16) float xs[5120]; __shared__ float lut[16];
    if(threadIdx.x<16){ const float t[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6}; lut[threadIdx.x]=t[threadIdx.x]; }
    for(int i=threadIdx.x;i<I;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    int lane=threadIdx.x&31, o=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(o<O){ float v=v4f_dot_fp4(w+(size_t)o*(I/2),s+(size_t)o*(I/32),xs,lut,I,lane); if(!lane) y[o]=v; }
}
/* one warp per 16-row block; A regs [blk][kt][lane][4], B regs [kt][lane][2], D [blk][lane][4] */
extern "C" __global__ void tc_gemm(const unsigned*A,const unsigned*B,float*D,int nblk,int nk,unsigned cinit){
    int blk=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5), lane=threadIdx.x&31; if(blk>=nblk)return;
    float c0=__uint_as_float(cinit),c1=c0,c2=c0,c3=c0;
    for(int k=0;k<nk;k++){
        const unsigned*a=A+(((size_t)blk*nk+k)*32+lane)*4; const unsigned*b=B+((size_t)k*32+lane)*2;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
    float*d=D+((size_t)blk*32+lane)*4; d[0]=c0;d[1]=c1;d[2]=c2;d[3]=c3;
}
'''


def chk(res):
    err = res[0]
    if isinstance(err, cu.CUresult):
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error {err}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error {err}")
    return res[1] if len(res) == 2 else res[1:]


def compile_cubin():
    prog = chk(nvrtc.nvrtcCreateProgram(SRC.encode(), b"probe.cu", 0, [], []))
    opts = [b"--gpu-architecture=sm_86", b"-std=c++17"]
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if r[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        n = chk(nvrtc.nvrtcGetProgramLogSize(prog)); log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log); raise RuntimeError(log.decode(errors="replace"))
    n = chk(nvrtc.nvrtcGetCUBINSize(prog)); cub = b" " * n
    chk((nvrtc.nvrtcGetCUBIN(prog, cub)[0], None))
    return cub


class Gpu:
    def __init__(self):
        chk((cu.cuInit(0)[0], None))
        self.dev = chk(cu.cuDeviceGet(0))
        name = chk(cu.cuDeviceGetName(64, self.dev))
        self.name = name.split(b"\0")[0].decode()
        self.ctx = chk(cu.cuDevicePrimaryCtxRetain(self.dev))
        chk((cu.cuCtxSetCurrent(self.ctx)[0], None))
        self.mod = chk(cu.cuModuleLoadData(compile_cubin()))
        self.allocs = []

    def fn(self, name):
        return chk(cu.cuModuleGetFunction(self.mod, name.encode()))

    def put(self, arr):
        arr = np.ascontiguousarray(arr)
        p = chk(cu.cuMemAlloc(max(arr.nbytes, 4)))
        chk((cu.cuMemcpyHtoD(p, arr, arr.nbytes)[0], None))
        self.allocs.append(p)
        return p

    def alloc(self, n):
        p = chk(cu.cuMemAlloc(n)); self.allocs.append(p); return p

    def get(self, p, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        chk((cu.cuMemcpyDtoH(out, p, out.nbytes)[0], None))
        return out

    def launch(self, f, grid, block, args):
        # args: list of (value, numpy dtype); device pointers as uint64
        holders = [np.array([int(v)], dtype=t) for v, t in args]
        ptrs = np.array([h.ctypes.data for h in holders], dtype=np.uint64)
        chk((cu.cuLaunchKernel(f, grid[0], grid[1], 1, block, 1, 1, 0, 0, ptrs.ctypes.data, 0)[0], None))
        chk((cu.cuCtxSynchronize()[0], None))

    def free(self):
        for p in self.allocs:
            cu.cuMemFree(p)
        self.allocs = []


def load(sample, ex):
    raw = np.fromfile(os.path.join(sample, ex["file"]), dtype=np.uint8)
    out, off = {}, 0
    for t in ex["tensors"]:
        out[t["name"]] = raw[off:off + t["bytes"]].reshape(t["shape"]); off += t["bytes"]
    return out


def canon_bytes(wb):
    lo = wb & 15; hi = wb >> 4
    lo = np.where(lo == 8, 0, lo); hi = np.where(hi == 8, 0, hi)
    return (lo | (hi << 4)).astype(np.uint8)


F16 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float16).view(np.uint16)


def tc_pack(codes, nblk, nk):
    """codes rows x cols (uint8 nibbles) -> A regs [blk][kt][lane][4] u32 using the PTX m16n8k16 A layout."""
    h = F16[codes[:nblk * 16, :nk * 16]].reshape(nblk, 16, nk, 16).transpose(0, 2, 1, 3)  # blk,kt,row,col
    lane = np.arange(32); g = lane >> 2; t = lane & 3
    rows = [g, g, g + 8, g + 8, g, g, g + 8, g + 8]
    cols = [2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9]
    halves = np.stack([h[:, :, rows[j], cols[j]] for j in range(8)], axis=-1).astype(np.uint32)  # blk,kt,32,8
    regs = halves[..., 0::2] | (halves[..., 1::2] << 16)
    return np.ascontiguousarray(regs.astype(np.uint32))


def tc_bpack(xh, nk):
    """xh: (nk*16) x 8 f16 bits -> B regs [kt][lane][2] (col layout: k = 2t+(0,1) [+8], n = g)."""
    lane = np.arange(32); g = lane >> 2; t = lane & 3
    xb = xh.reshape(nk, 16, 8)
    ks = [2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9]
    halves = np.stack([xb[:, ks[j], g] for j in range(4)], axis=-1).astype(np.uint32)  # kt,32,4
    regs = halves[..., 0::2] | (halves[..., 1::2] << 16)
    return np.ascontiguousarray(regs.astype(np.uint32))


def bits_diff(a, b):
    return int((np.ascontiguousarray(a).view(np.uint32) != np.ascontiguousarray(b).view(np.uint32)).sum())


def main():
    sample, out_json = sys.argv[1], sys.argv[2]
    nex = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    man = json.load(open(os.path.join(sample, "manifest.json")))
    g = Gpu()
    res = {"device": g.name, "runs": []}
    k_mv, k_fd, k_tc = g.fn("mv_fp4_grouped4"), g.fn("fused_dot"), g.fn("tc_gemm")
    e8 = (np.arange(256, dtype=np.uint32) << 23).view(np.float32)
    dptr, size = chk(cu.cuModuleGetGlobal(g.mod, b"e8_table"))
    chk((cu.cuMemcpyHtoD(dptr, e8, e8.nbytes)[0], None))
    rng = np.random.default_rng(7)
    for ex in man["experts"][::40][:nex]:
        t = load(sample, ex)
        for m in ("w1", "w2"):
            wb = t[m + ".weight"]; sc = t[m + ".scale"]
            O, I = wb.shape[0], wb.shape[1] * 2
            wc = canon_bytes(wb)
            for xname in ("gauss", "sparse90"):
                x = rng.standard_normal(I).astype(np.float32)
                if xname == "sparse90":
                    x[rng.random(I) < 0.9] = 0
                dw, dwc, ds, dx = g.put(wb), g.put(wc), g.put(sc), g.put(x)
                y0, y1 = g.alloc(O * 4), g.alloc(O * 4)
                desc = np.array([[int(dw), int(ds), int(dx), int(y0)], [int(dwc), int(ds), int(dx), int(y1)]], dtype=np.uint64)
                dd = g.put(desc)
                g.launch(k_mv, ((O + 3) // 4, 2), 256, [(dd, np.uint64), (2, np.int32), (O, np.int32), (I, np.int32)])
                a, b = g.get(y0, O, np.float32), g.get(y1, O, np.float32)
                r_mv = bits_diff(a, b); z_mv = int((a == 0).sum())
                # same data through the fused6 dot (8 warps per block)
                g.launch(k_fd, ((O + 7) // 8, 1), 256, [(dw, np.uint64), (ds, np.uint64), (dx, np.uint64), (y0, np.uint64), (O, np.int32), (I, np.int32)])
                g.launch(k_fd, ((O + 7) // 8, 1), 256, [(dwc, np.uint64), (ds, np.uint64), (dx, np.uint64), (y1, np.uint64), (O, np.int32), (I, np.int32)])
                c, d = g.get(y0, O, np.float32), g.get(y1, O, np.float32)
                r_fd = bits_diff(c, d)
                cross = bits_diff(a, c)  # mv vs fused (same operands/order by design)
                rec = {"expert": ex["file"], "matrix": m, "x": xname, "mv_fp4_grouped_bitdiff": r_mv,
                       "mv_exact_zero_outputs": z_mv, "fused6_dot_bitdiff": r_fd, "mv_vs_fused_bitdiff": cross,
                       "outputs": O}
                g.free()
                print(rec, flush=True)
                res["runs"].append(rec)
            # tensor-core FP16 GEMM over the whole matrix, 8 activation columns
            codes = np.empty((O, I), dtype=np.uint8); codes[:, 0::2] = wb & 15; codes[:, 1::2] = wb >> 4
            canon = codes.copy(); canon[canon == 8] = 0
            nblk, nk = min(O // 16, 32), I // 16
            xh = rng.standard_normal((nk * 16, 8)).astype(np.float16).view(np.uint16)
            B = tc_bpack(xh, nk)
            outs = []
            for cc in (codes, canon):
                A = tc_pack(cc, nblk, nk)
                dA, dB, dD = g.put(A), g.put(B), g.alloc(nblk * 32 * 4 * 4)
                g.launch(k_tc, ((nblk + 7) // 8, 1), 256, [(dA, np.uint64), (dB, np.uint64), (dD, np.uint64), (nblk, np.int32), (nk, np.int32), (0, np.uint32)])
                outs.append(g.get(dD, nblk * 128, np.float32)); g.free()
            # reference check of the layout: fp32 matmul of the same f16 values
            wv = F16[codes].view(np.float16).astype(np.float32); ref = wv[:nblk * 16, :nk * 16] @ xh.view(np.float16).astype(np.float32)
            lane = np.arange(32); gg = lane >> 2; tt = lane & 3
            D = outs[0].reshape(nblk, 32, 4)
            got = np.empty((nblk, 16, 8), np.float32)
            got[:, gg, 2 * tt] = D[:, :, 0]; got[:, gg, 2 * tt + 1] = D[:, :, 1]
            got[:, gg + 8, 2 * tt] = D[:, :, 2]; got[:, gg + 8, 2 * tt + 1] = D[:, :, 3]
            rel = float(np.max(np.abs(got.reshape(-1, 8) - ref)) / np.max(np.abs(ref)))
            rec = {"expert": ex["file"], "matrix": m, "tc_fp16_gemm_bitdiff": bits_diff(outs[0], outs[1]),
                   "tc_outputs": int(outs[0].size), "tc_layout_max_rel_err_vs_fp32": rel}
            print(rec, flush=True); res["runs"].append(rec)
    # signed-zero characterisation of the tensor core: all-zero A tiles
    nk = 4; nblk = 1
    char = {}
    for bsign in ("mixed", "positive"):
      xr = rng.standard_normal((nk * 16, 8))
      if bsign == "positive":
          xr = np.abs(xr) + 0.1
      xh = xr.astype(np.float16).view(np.uint16)
      B = tc_bpack(xh, nk)
      for label0, code, cinit in (("A_all_neg0_C_pos0", 8, 0), ("A_all_pos0_C_pos0", 0, 0),
                                 ("A_all_neg0_C_neg0", 8, 0x80000000), ("A_all_pos0_C_neg0", 0, 0x80000000)):
        label = f"B_{bsign}_{label0}"
        A = tc_pack(np.full((16, nk * 16), code, np.uint8), nblk, nk)
        dA, dB, dD = g.put(A), g.put(B), g.alloc(128 * 4)
        g.launch(k_tc, (1, 1), 32, [(dA, np.uint64), (dB, np.uint64), (dD, np.uint64), (nblk, np.int32), (nk, np.int32), (cinit, np.uint32)])
        D = g.get(dD, 128, np.float32); g.free()
        char[label] = {"all_zero": bool((D == 0).all()), "neg_zero_count": int(np.signbit(D).sum()), "n": 128}
    res["tc_signed_zero"] = char
    print(char, flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
