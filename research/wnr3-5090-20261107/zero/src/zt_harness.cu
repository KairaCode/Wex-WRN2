// Canonical-zero (WNR2z) GPU bit test harness for KAIRA's DeepSeek V4 CUDA backend.
// The backend source is included VERBATIM (worktree beautiful-volhard-bfc14a, git HEAD 88ddb07),
// so every kernel and every static dispatch helper called below is the shipped code, compiled with
// the same flags as the shipped DLL flavour (generic: sm_120 C++17; DeepGEMM: sm_120a C++20).
// Only additions: the C exports zt_* (host glue), one warp-per-row wrapper around the device function
// v4f_dot_fp4 for V4.1 shapes, and (DeepGEMM flavour) an mma.sync micro-test kernel that calls
// DeepGEMM's own sm120 MMA wrappers.
#include "backend_cuda_dsv4.cu"

#define ZT extern "C" __declspec(dllexport)

static int zt_dev = 0;
static Dev *zc() { return ctx(zt_dev); }
static int zt_check(const char *what) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "[zt] %s: %s\n", what, cudaGetErrorString(e)); return 0; }
    return 1;
}

ZT int zt_putenv(const char *kv) { return _putenv(kv) == 0; }
ZT int zt_flavour() {
#ifdef COLI_DSV4_DEEPGEMM
    return 1;
#else
    return 0;
#endif
}
ZT int zt_init(int device) { zt_dev = device; return dsv4_cuda_init(&zt_dev, 1); }

// ---- matrix slots (one FP4 matrix each: packed nibbles low-first + one E8M0 per 32) ----
struct ZtSlot { void *w; uint8_t *s; size_t wcap, scap; Dsv4CudaTensor t; };
static ZtSlot zs[4];
ZT int zt_set_matrix(int slot, const uint8_t *w, const uint8_t *s, int O, int I) {
    if (slot < 0 || slot > 3 || I % 32) return 0;
    ZtSlot &z = zs[slot];
    size_t wb = (size_t)O * I / 2, sb = (size_t)O * (I / 32);
    if (!buf(&z.w, &z.wcap, wb) || !buf((void **)&z.s, &z.scap, sb)) return 0;
    if (cudaMemcpy(z.w, w, wb, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(z.s, s, sb, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (!zt_check("matrix upload")) return 0;
    memset(&z.t, 0, sizeof(z.t));
    z.t.w = z.w; z.t.scale = z.s; z.t.O = O; z.t.I = I; z.t.fmt = 4; z.t.device = zt_dev;
    z.t.bytes = (long long)(wb + sb);
    return 1;
}

static float *zx, *zy; static size_t zxcap, zycap;
static MvDesc *zdesc; static size_t zdesccap;
static int *zem; static size_t zemcap;

// v4f_dot_fp4 (the fused decode MoE's FP4 lane) at arbitrary I: one warp per row, x and the
// 16-entry LUT staged in shared memory exactly like v4f_moe_kernel does (harness wrapper).
__global__ void zt_v4f_rows(const uint8_t *w, const uint8_t *s, const float *x, float *y, int O, int I) {
    extern __shared__ __align__(16) float zt_xs[];
    __shared__ float lut[16];
    if (threadIdx.x < 16) { const float t[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6}; lut[threadIdx.x] = t[threadIdx.x]; }
    for (int i = threadIdx.x; i < I; i += blockDim.x) zt_xs[i] = x[i];
    __syncthreads();
    int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (o >= O) return;
    float p = v4f_dot_fp4(w + (size_t)o * (I / 2), s + (size_t)o * (I / 32), zt_xs, lut, I, lane);
    if (!lane) y[o] = p;
}

enum {
    K_MV = 0, K_MVW8, K_ROWS4, K_ROWS8, K_ROWS16, K_ROWS32, K_GRP1, K_GRP2, K_GRP4, K_GRP8,
    K_MMB, K_EM2, K_EM3, K_EM4, K_EM6, K_V4F, K_TC, K_COUNT
};
ZT const char *zt_kernel_name(int k) {
    static const char *n[K_COUNT] = {"mv<4>", "mv_warp8<4>", "mv_rows<4,4>", "mv_rows<4,8>", "mv_rows<4,16>", "mv_rows<4,32>",
        "mv_fp4_grouped<1>", "mv_fp4_grouped<2>", "mv_fp4_grouped<4>", "mv_fp4_grouped<8>", "mm_batch<4>",
        "mv_fp4_expert_major<2>", "mv_fp4_expert_major<3>", "mv_fp4_expert_major<4>", "mv_fp4_expert_major<6>",
        "v4f_dot_fp4(warp/row wrapper)", "TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul)"};
    return k >= 0 && k < K_COUNT ? n[k] : nullptr;
}
ZT int zt_kernel_count() { return K_COUNT; }

// Runs kernel k on slot's matrix [O x I] for T activation rows x (host, T x I, tokens-major);
// writes T x O outputs (tokens-major) to y. The output buffer is pre-filled with 0xA5A5A5A5 so an
// unwritten output is visible. Returns T*O, -1 when the kernel is not compiled in / not applicable, 0 on error.
ZT long long zt_run(int k, int slot, const float *x, int T, float *y) {
    ZtSlot &z = zs[slot]; int O = z.t.O, I = z.t.I; Dev *c = zc();
    if (!c || T < 1 || T > 16) return 0;
    size_t xb = (size_t)T * I * 4, yb = (size_t)T * O * 4;
    if (!buf((void **)&zx, &zxcap, xb) || !buf((void **)&zy, &zycap, yb)) return 0;
    const uint8_t *w = (const uint8_t *)z.w, *s = z.s; cudaStream_t st = c->stream;   // everything on the backend's (non-blocking) stream
    if (cudaMemcpyAsync(zx, x, xb, cudaMemcpyHostToDevice, st) != cudaSuccess || cudaMemsetAsync(zy, 0xA5, yb, st) != cudaSuccess) return 0;
    switch (k) {
    case K_MV: for (int t = 0; t < T; t++) mv<4><<<O, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1); break;
    case K_MVW8:
        if (!((I & 127) == 0 && I <= 8192 && (O & 7) == 0)) return -1;   // run_mv<4>'s condition for this kernel
        for (int t = 0; t < T; t++) mv_warp8<4><<<(O + 7) / 8, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1);
        break;
    case K_ROWS4: for (int t = 0; t < T; t++) mv_rows<4, 4><<<(O + 3) / 4, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1); break;
    case K_ROWS8: for (int t = 0; t < T; t++) mv_rows<4, 8><<<(O + 7) / 8, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1); break;
    case K_ROWS16: for (int t = 0; t < T; t++) mv_rows<4, 16><<<(O + 15) / 16, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1); break;
    case K_ROWS32: for (int t = 0; t < T; t++) mv_rows<4, 32><<<(O + 31) / 32, 256, 0, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I, 1); break;
    case K_GRP1: case K_GRP2: case K_GRP4: case K_GRP8: {
        if (!buf((void **)&zdesc, &zdesccap, (size_t)T * sizeof(MvDesc))) return 0;
        MvDesc hd[16]; for (int t = 0; t < T; t++) hd[t] = {w, s, zx + (size_t)t * I, zy + (size_t)t * O};
        if (cudaMemcpyAsync(zdesc, hd, (size_t)T * sizeof(MvDesc), cudaMemcpyHostToDevice, st) != cudaSuccess) return 0;
        if (k == K_GRP1) mv_fp4_grouped<1><<<dim3(O, T), 256, 0, st>>>(zdesc, T, O, I);
        else if (k == K_GRP2) mv_fp4_grouped<2><<<dim3((O + 1) / 2, T), 256, 0, st>>>(zdesc, T, O, I);
        else if (k == K_GRP4) mv_fp4_grouped<4><<<dim3((O + 3) / 4, T), 256, 0, st>>>(zdesc, T, O, I);
        else mv_fp4_grouped<8><<<dim3((O + 7) / 8, T), 256, 0, st>>>(zdesc, T, O, I);
        break; }
    case K_MMB: mm_batch<4><<<dim3(O, (T + 15) / 16), 256, 0, st>>>(w, s, zx, zy, O, I, T); break;
    case K_EM2: case K_EM3: case K_EM4: case K_EM6: {
#ifdef COLI_DSV4_DEEPGEMM
        return -1;   // compiled out of the DeepGEMM build (#ifndef COLI_DSV4_DEEPGEMM)
#else
        if ((I / 32) % 8) return -1;
        // one bank expert (slot 0 of a 1-expert bank), every token a route of it: run_fp4_expert_major
        if (!buf((void **)&zem, &zemcap, (size_t)(1 + 2 + 16) * sizeof(int))) return 0;
        int h[19]; h[0] = 0; h[1] = 0; h[2] = T; for (int t = 0; t < T; t++) h[3 + t] = t;
        if (cudaMemcpyAsync(zem, h, sizeof(h), cudaMemcpyHostToDevice, st) != cudaSuccess) return 0;
        int tb = k == K_EM2 ? 2 : k == K_EM3 ? 3 : k == K_EM4 ? 4 : 6;
        run_fp4_expert_major(tb, w, s, (size_t)O * I / 2, (size_t)O * (I / 32), zem, zem + 1, zem + 3, 1, zx, I, 1, zy, zy, O, O, O, I, st);
        break;
#endif
    }
    case K_V4F:
        if (I % 16 || (size_t)I * 4 > 48 * 1024) return -1;
        for (int t = 0; t < T; t++) zt_v4f_rows<<<(O + 7) / 8, 256, (size_t)I * 4, st>>>(w, s, zx + (size_t)t * I, zy + (size_t)t * O, O, I);
        break;
    case K_TC:
#if COLI_DSV4_TC
        { TcPlan *p = &c->plans[O > I]; if (p->ok && (p->O != O || p->I != I)) return -1; }   // tc_plan caches one plan per (O>I) without re-checking the shape
        for (int t = 0; t < T; t++) if (!tc_prepare_x(c, zx + (size_t)t * I, I) || !tc_fp4_matvec(c, &z.t, zy + (size_t)t * O)) return 0;
        break;
#else
        return -1;
#endif
    default: return 0;
    }
    if (!zt_check(zt_kernel_name(k))) return 0;
    if (cudaMemcpy(y, zy, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    return (long long)T * O;
}

// ---- public API end-to-end: dsv4_cuda_expert_group (generic routed experts, V4.1 shapes) ----
static Dsv4CudaTensor *zg[3][16];
ZT int zt_api_upload(int which, int e, const uint8_t *w, const uint8_t *s, int O, int I) {   // which: 0 gate (w1), 1 up (w3), 2 down (w2)
    if (which < 0 || which > 2 || e < 0 || e > 15) return 0;
    if (zg[which][e] && (zg[which][e]->O != O || zg[which][e]->I != I)) { dsv4_cuda_tensor_free(zg[which][e]); zg[which][e] = nullptr; }
    if (!zg[which][e]) return dsv4_cuda_upload_fp4(&zg[which][e], w, s, O, I, zt_dev);
    return dsv4_cuda_tensor_refill_fp4(zg[which][e], w, s, O, I, 1);
}
ZT int zt_api_expert_group(int count, const float *weights, float limit, float *y, const float *x) {
    return dsv4_cuda_expert_group(zg[0], zg[1], zg[2], weights, count, limit, y, x) && zt_check("expert_group");
}
// dsv4_cuda_moe (decode MoE: fused v4f_moe_kernel when DSV4_CUDA_MOE_FUSED6 != 0 and the shapes are
// exactly 6 x 4096/2048, else mv_fp4_grouped<8>) with shared experts sh[0..2] (fmt 8).
static Dsv4CudaTensor *zsh[3];
ZT int zt_api_shared(int which, const uint8_t *w, const uint8_t *s, int O, int I) {
    if (zsh[which]) { dsv4_cuda_tensor_free(zsh[which]); zsh[which] = nullptr; }
    return dsv4_cuda_upload_fp8(&zsh[which], w, s, O, I, zt_dev);
}
ZT int zt_api_moe(int count, const float *weights, float limit, float *y, const float *x) {
    return dsv4_cuda_moe(zg[0], zg[1], zg[2], weights, count, zsh[0], zsh[1], zsh[2], limit, y, x) && zt_check("moe");
}

#ifdef COLI_DSV4_DEEPGEMM
// ---- DeepGEMM routed-expert paths (V4 shapes only: H=4096, I=2048, 256-slot bank) ----
static Dsv4CudaExpertSet *zset;
ZT int zt_dg_bank(void) {
    if (!zset) zset = dsv4_cuda_expert_bank_create(256, 4096, 2048, zt_dev, nullptr, nullptr, nullptr);
    return zset != nullptr;
}
ZT int zt_dg_upload(int e, const uint8_t *gw, const uint8_t *gs, const uint8_t *uw, const uint8_t *us, const uint8_t *dw, const uint8_t *ds) {
    Dsv4CudaTensor *g = nullptr, *u = nullptr, *d = nullptr;
    if (!zset || !dsv4_cuda_expert_bank_upload(zset, e, gw, gs, uw, us, dw, ds, &g, &u, &d)) return 0;
    free(g); free(u); free(d);   // expert_view() structs (calloc, own nothing)
    return zt_check("bank upload");
}
// kind 0: dg_moe<256> (masked, decode); 1: dg_moe_contiguous_fast<256> (decode default);
// 2: dg_moe_batch_contiguous with host ids (expert-grouped layout, prefill default);
// 3: dg_moe_batch_contiguous without host ids (legacy replicated layout).
// x: T x 4096; ids/weights: T x 6 (kinds 0/1: T must be 1). out: T x 4096 floats.
// mm1/mm2 (kinds 0/1 only, may be null): the 6 routes' FC1 rows (4096 bf16 = gate|up) and FC2 rows (4096 bf16).
ZT int zt_dg_moe(int kind, const float *x, int T, const int *ids, const float *weights, float limit, float *out, uint16_t *mm1, uint16_t *mm2) {
    Dev *c = zc(); if (!c || !zset || T < 1 || T > 128 || (kind < 2 && T != 1)) return 0;
    size_t xb = (size_t)T * 4096 * 4, ib = (size_t)T * 6 * sizeof(int), wb = (size_t)T * 6 * sizeof(float);
    if (!buf((void **)&c->dx, &c->xcap, xb) || !buf((void **)&zy, &zycap, xb) ||
        !buf((void **)&c->expert_ids, &c->expert_idcap, ib) || !buf((void **)&c->expert_weights, &c->expert_weightscap, wb)) return 0;
    cudaStream_t st = c->stream;
    if (cudaMemcpyAsync(c->dx, x, xb, cudaMemcpyHostToDevice, st) != cudaSuccess || cudaMemcpyAsync(c->expert_ids, ids, ib, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(c->expert_weights, weights, wb, cudaMemcpyHostToDevice, st) != cudaSuccess || cudaMemsetAsync(zy, 0xA5, xb, st) != cudaSuccess) return 0;
    int r = 0;
    if (kind == 0) r = dg_moe<256>(c, zset, c->expert_weights, limit, zy);
    else if (kind == 1) r = dg_moe_contiguous_fast<256>(c, zset, c->expert_weights, limit, zy);
    else if (kind == 2) r = dg_moe_batch_contiguous(c, zset, c->dx, c->expert_ids, c->expert_weights, T, limit, zy, ids);
    else if (kind == 3) r = dg_moe_batch_contiguous(c, zset, c->dx, c->expert_ids, c->expert_weights, T, limit, zy, nullptr);
    if (!r || !zt_check("dg_moe")) return 0;
    if (cudaMemcpy(out, zy, xb, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (kind < 2) {
        const int M = kind == 0 ? 16 : 64;   // masked: rows id*16; contiguous: rows k*64 (dg_layout6, A=64)
        for (int k = 0; k < 6; k++) {
            size_t row = kind == 0 ? (size_t)ids[k] * M : (size_t)k * M;
            if (mm1 && cudaMemcpy(mm1 + (size_t)k * 4096, (__nv_bfloat16 *)c->dgmm1 + row * 4096, 4096 * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
            if (mm2 && cudaMemcpy(mm2 + (size_t)k * 4096, (__nv_bfloat16 *)c->dgmm2 + row * 4096, 4096 * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
        }
    }
    return 1;
}

// ---- instruction micro-test: the block-scaled mma.sync forms DeepGEMM uses on sm_120a ----
// op 0: kind::mxf8f6f4 e4m3 (A) x e2m1 (B, 1-byte container value<<2), scale_vec::1X, ue8m0 (KAIRA's FP8xFP4 MoE GEMM)
// op 1: kind::mxf8f6f4 e4m3 x e4m3 (the MXFP8 class cuBLASLt's VEC32_UE8M0 path maps to; B 0x80 vs 0x00)
// per tile: A 32 lanes x 4 u32, B 32 x 2 u32, C/D 32 x 4 f32, sfa/sfb one byte per lane.
__global__ void zt_mma_kernel(int op, const uint32_t *A, const uint32_t *B, const uint8_t *SA, const uint8_t *SB, const float *Cin, float *D) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    size_t t = blockIdx.x; int lane = threadIdx.x;
    const uint32_t *a = A + (t * 32 + lane) * 4, *b = B + (t * 32 + lane) * 2;
    uint32_t af[4] = {a[0], a[1], a[2], a[3]}, bf[2] = {b[0], b[1]};
    float d[4]; const float *ci = Cin + (t * 32 + lane) * 4; d[0] = ci[0]; d[1] = ci[1]; d[2] = ci[2]; d[3] = ci[3];
    uint8_t sa = SA[t * 32 + lane], sb = SB[t * 32 + lane];
    if (op == 0) deep_gemm::mma::sm120::fp8_fp4_mixed_mma_block_scaled(d, af, bf, sa, sb);
    else deep_gemm::mma::sm120::fp8_mma_block_scaled(d, af, bf[0], bf[1], sa, sb);
    float *o = D + (t * 32 + lane) * 4; o[0] = d[0]; o[1] = d[1]; o[2] = d[2]; o[3] = d[3];
#endif
}
ZT int zt_mma(int op, int tiles, const uint32_t *A, const uint32_t *B, const uint8_t *SA, const uint8_t *SB, const float *C, float *D) {
    void *dA, *dB, *dSA, *dSB, *dC, *dD; size_t n = (size_t)tiles * 32;
    if (cudaMalloc(&dA, n * 16) || cudaMalloc(&dB, n * 8) || cudaMalloc(&dSA, n) || cudaMalloc(&dSB, n) || cudaMalloc(&dC, n * 16) || cudaMalloc(&dD, n * 16)) return 0;
    cudaMemcpy(dA, A, n * 16, cudaMemcpyHostToDevice); cudaMemcpy(dB, B, n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dSA, SA, n, cudaMemcpyHostToDevice); cudaMemcpy(dSB, SB, n, cudaMemcpyHostToDevice); cudaMemcpy(dC, C, n * 16, cudaMemcpyHostToDevice);
    zt_mma_kernel<<<tiles, 32>>>(op, (uint32_t *)dA, (uint32_t *)dB, (uint8_t *)dSA, (uint8_t *)dSB, (float *)dC, (float *)dD);
    int r = zt_check("mma") && cudaMemcpy(D, dD, n * 16, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC); cudaFree(dD);
    return r;
}
#endif
