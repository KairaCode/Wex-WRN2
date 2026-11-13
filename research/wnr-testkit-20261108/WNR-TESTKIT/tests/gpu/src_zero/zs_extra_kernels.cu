// Extra kernels of the zero_bittest (compiled as a separate NVRTC module, independent of zs_gpu_kernels.cu).
//  tc_gemm  : team 2 zero-refuter/gpu_probe.py:69-78, unchanged. FP16 tensor-core mma.sync m16n8k16 (f32 accumulate)
//             over the full K of a matrix; one warp per 16-row block; A regs [blk][kt][lane][4], B regs [kt][lane][2].
//  mma_f16 / mma_bf16 : team 2 zero-safety/scripts/zs_mma.py:13-30, unchanged (single m16n8k16 tile per block).
//  negctl_seeded : NEGATIVE CONTROL written for the kit (not colibri code): a row dot product whose accumulator is
//             seeded with the FIRST product instead of +0 (the K12 'first-term' pattern of team 2's CPU tests). On rows
//             whose products are all zeros it returns the sign of those zeros, so it MUST show sign-only mismatches
//             between original and canonicalised records on the synthetic zero rows (activation set all -0).
typedef unsigned char u8;

extern "C" __global__ void tc_gemm(const unsigned* A, const unsigned* B, float* D, int nblk, int nk, unsigned cinit) {
    int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk >= nblk) return;
    float c0 = __uint_as_float(cinit), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < nk; k++) {
        const unsigned* a = A + (((size_t)blk * nk + k) * 32 + lane) * 4;
        const unsigned* b = B + ((size_t)k * 32 + lane) * 2;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    float* d = D + ((size_t)blk * 32 + lane) * 4;
    d[0] = c0; d[1] = c1; d[2] = c2; d[3] = c3;
}

extern "C" __global__ void mma_f16(const unsigned* A, const unsigned* B, const float* Cin, float* D) {
    int t = blockIdx.x, lane = threadIdx.x;
    const unsigned* a = A + (size_t)t * 128 + lane * 4; const unsigned* b = B + (size_t)t * 64 + lane * 2;
    const float* c = Cin + (size_t)t * 128 + lane * 4;
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    float* d = D + (size_t)t * 128 + lane * 4; d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

extern "C" __global__ void mma_bf16(const unsigned* A, const unsigned* B, const float* Cin, float* D) {
    int t = blockIdx.x, lane = threadIdx.x;
    const unsigned* a = A + (size_t)t * 128 + lane * 4; const unsigned* b = B + (size_t)t * 64 + lane * 2;
    const float* c = Cin + (size_t)t * 128 + lane * 4;
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    float* d = D + (size_t)t * 128 + lane * 4; d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

__device__ __forceinline__ float nc_f4(u8 q) { const float a[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return q & 8 ? -a[q & 7] : a[q]; }
__device__ __forceinline__ float nc_e8(u8 b) { return b ? __uint_as_float((unsigned)b << 23) : __uint_as_float(0x00400000u); }

extern "C" __global__ void negctl_seeded(const u8* w, const u8* sc, const float* x, float* y, int O, int I) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    float s = 0.f;
    for (int i = 0; i < I; i++) {
        u8 p = w[((long long)o * I + i) >> 1], q = (i & 1) ? p >> 4 : p & 15;
        float v = __fmul_rn(x[i], __fmul_rn(nc_f4(q), nc_e8(sc[(long long)o * (I / 32) + i / 32])));
        s = (i == 0) ? v : __fadd_rn(s, v);
    }
    y[o] = s;
}
