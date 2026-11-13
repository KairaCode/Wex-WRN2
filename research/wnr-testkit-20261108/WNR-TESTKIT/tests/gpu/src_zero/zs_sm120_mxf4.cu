// EXPERIMENTAL (written for the kit, never run on hardware when the kit was built): block-scaled FP4 tensor-core MMA
// of sm_120 (RTX 50xx), the instruction family DeepGEMM / CUTLASS sm120 FP4 GEMMs use:
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
// Needs -arch=sm_120a (arch-specific features; NVRTC >= 12.8). One warp = one 16x8x64 tile, K looped over `nk` tiles.
// A fragment: 4 x u32 per lane (32 e2m1 nibbles), B fragment: 2 x u32 per lane (16 nibbles), one ue8m0 scale word
// per lane for A and for B (byte-id 0, thread-id 0 selectors). The exact mapping of nibbles to matrix positions
// does NOT matter for this test: original vs canonicalised (code 8 -> 0) operands are packed the same way, so any
// differing D bit is a real difference caused by the sign of a zero. The host side fills A from real weight bytes.
extern "C" __global__ void mxf4_gemm(const unsigned* A, const unsigned* B, const unsigned* SA, const unsigned* SB,
                                     float* D, int ntile, int nk, unsigned cinit) {
    int tile = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (tile >= ntile) return;
    float c0 = __uint_as_float(cinit), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < nk; k++) {
        const unsigned* a = A + (((size_t)tile * nk + k) * 32 + lane) * 4;
        const unsigned* b = B + ((size_t)k * 32 + lane) * 2;
        unsigned sa = SA[((size_t)tile * nk + k) * 32 + lane];
        unsigned sb = SB[(size_t)k * 32 + lane];
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "r"(sa), "h"((unsigned short)0), "h"((unsigned short)0), "r"(sb), "h"((unsigned short)0), "h"((unsigned short)0));
    }
    float* d = D + ((size_t)tile * 32 + lane) * 4;
    d[0] = c0; d[1] = c1; d[2] = c2; d[3] = c3;
}
