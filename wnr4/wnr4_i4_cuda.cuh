/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-I4 CUDA decoder (kind 3, wnr4/wnr4_i4.h; bit-exact with wnr4_i4_decode_tile). One warp decodes one tile
 * with its 32 lanes in lockstep: renormalising lanes take the next words of the tile in lane order from a
 * per-warp 4-slot shared-memory ring buffer fed asynchronously by cp.async (or synchronous loads on < sm_80),
 * checking refill only once every 2/4/8 nibble steps so the hot inner decode loop is branch-free. Lanes 2..31
 * carry their tail in the state, lanes 0 and 1 end at the tile hash, every failure sets error bits of the
 * record (fail closed: its destination may hold partial data).
 * A block holds 1, 2, 4, 8 or 16 tiles of one matrix (chosen automatically from tile count and SM count to
 * minimise single-record latency without extra waves, or overridden via wnr4_cuda_warps). The matrix's nibble
 * rows sit in 48-byte-strided shared memory (8 bank groups for LDS.128, pairs byte-swapped for zero-unpack
 * level-1 4-way register search: 19,872 B), plus 2,048 B of swapped scale summaries for Phase 1; the tile's
 * lattice indices sit in shared memory (32 * WNR4_I4_CUDA_MAX_G bytes per warp, reused in Phase 2 as a
 * conflict-free 8 x 32 planar nibble staging buffer). The lattice is computed with __fmul_rn and __fdiv_rn: do
 * not build with --use_fast_math or -ftz=true. Destination: the record's decoded bytes, 4-byte aligned.
 * Host side: wnr4_i4_cuda_table_image() (model), wnr4_i4_cuda_plan() (jobs of a record, from its host copy),
 * wnr4_i4_cuda_jobs(), wnr4_i4_cuda_launch(). */
#ifndef WNR4_I4_CUDA_CUH
#define WNR4_I4_CUDA_CUH

#include "wnr4_cuda.cuh"
#include "wnr4_i4.h"

#define WNR4_I4_CUDA_MAX_G 64u
#define WNR4_I4_CUDA_STATES 46u
#define WNR4_I4_CUDA_NIB_U4 (WNR4_I4_NCLS * WNR4_I4_CUDA_STATES * 2u)          /* 828 */
#define WNR4_I4_CUDA_MAT_U4 (WNR4_I4_CUDA_NIB_U4 + WNR4_I4_SROWS * 34u)        /* + 2312 */

struct Wnr4I4CudaJob {           /* one tile */
    const unsigned char *tile;   /* device: its 32 states then words */
    const unsigned char *bases;  /* device: the record's base area */
    unsigned char *codes;        /* device: the matrix's codes (group 0) */
    uint32_t *scales;            /* device: the matrix's scales (group 0) */
    uint32_t nwords, rec, t, m, c0, n, owner, G, gc, nbc, bbase, anchor, pairs;
};

/* compact state of (pm, mx): 0 = element 0 (9, 9), else 1 + mx (mx + 1) / 2 + pm (pm <= mx) */
static WNR4_HD inline uint32_t wnr4_i4_state(uint32_t pm, uint32_t mx) { return pm == 9u ? 0u : 1u + mx * (mx + 1u) / 2u + pm; }

/* ---- device helpers ------------------------------------------------------------- */
static __device__ __forceinline__ uint4 wnr4_i4_swap_u4(uint4 v) {
    return make_uint4(__byte_perm(v.x, 0u, 0x1032u), __byte_perm(v.y, 0u, 0x1032u),
                      __byte_perm(v.z, 0u, 0x1032u), __byte_perm(v.w, 0u, 0x1032u));
}

static __device__ __forceinline__ void wnr4_i4_prefetch_l2(const void *p) {
#if __CUDA_ARCH__ >= 700
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    (void)p;
#endif
}

static __device__ __forceinline__ void wnr4_i4_cp_async_16(void *smem_ptr, const void *gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(saddr), "l"(gmem_ptr));
#else
    *(uint4 *)smem_ptr = __ldg((const uint4 *)gmem_ptr);
#endif
}
static __device__ __forceinline__ void wnr4_i4_cp_commit(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
static __device__ __forceinline__ void wnr4_i4_cp_wait0(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}
static __device__ __forceinline__ void wnr4_i4_cp_wait1(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}

/* Forced-order dual 128-bit SMEM load: prevents ptxas from sinking LDS.128 behind the renorm LDS.U16. */
static __device__ __forceinline__ void wnr4_i4_lds128_x2(const uint4 *ptr, uint4 &a, uint4 &b) {
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(ptr);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%8];\n\t"
                 "ld.shared.v4.u32 {%4,%5,%6,%7}, [%8+16];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
                   "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "r"(saddr) : "memory");
}

/* Predicated SMEM uint16 renorm: if xn < WNR4_L (65536), loads word from saddr and returns (xn << 16) | word. */
static __device__ __forceinline__ uint32_t wnr4_i4_lds_renorm(uint32_t xn, uint32_t saddr) {
    uint32_t out = xn;
    asm volatile("{\n\t"
                 ".reg .pred %%p;\n\t"
                 ".reg .u32 %%w;\n\t"
                 "setp.lt.u32 %%p, %1, 65536;\n\t"
                 "@%%p ld.shared.u16 %%w, [%2];\n\t"
                 "@%%p shl.b32 %0, %1, 16;\n\t"
                 "@%%p or.b32 %0, %0, %%w;\n\t"
                 "}"
                 : "+r"(out) : "r"(xn), "r"(saddr) : "memory");
    return out;
}

/* 2-level 4-way register search on byte-swapped 16-symbol CDF row: a.x = (c0<<16)|c1, ..., b.w = (c14<<16)|c15. */
static __device__ __forceinline__ void wnr4_i4_decode_4way_swap(uint4 a, uint4 b, uint32_t x, uint32_t &y, uint32_t &xn) {
    const uint32_t slot = x & (WNR4_TOTAL - 1u);
    const uint32_t q = x >> WNR4_PREC;
    const uint32_t sh = (slot << 16) | 0xffffu;
    const bool p1 = sh >= a.z, p2 = sh >= b.x, p3 = sh >= b.z;
    const uint32_t r0  = p2 ? (p3 ? b.z : b.x) : (p1 ? a.z : a.x);
    const uint32_t r1  = p2 ? (p3 ? b.w : b.y) : (p1 ? a.w : a.y);
    const uint32_t c4r = p2 ? (p3 ? (WNR4_TOTAL << 16) : b.z) : (p1 ? b.x : a.z);
    const uint32_t yhi = p2 ? (p3 ? 12u : 8u) : (p1 ? 4u : 0u);
    const uint32_t c0 = r0 >> 16, c1 = r0 & 0xffffu, c2 = r1 >> 16, c3 = r1 & 0xffffu, c4 = c4r >> 16;
    const bool q1 = slot >= c1, q2 = sh >= r1, q3 = slot >= c3;
    const uint32_t c  = q2 ? (q3 ? c3 : c2) : (q1 ? c1 : c0);
    const uint32_t hi = q2 ? (q3 ? c4 : c3) : (q1 ? c2 : c1);
    y = yhi + (q2 ? (q3 ? 3u : 2u) : (q1 ? 1u : 0u));
    xn = (hi - c) * q + (slot - c);
}

/* 4-way search on natural-order 16-symbol CDF row: a.x = (c1<<16)|c0, ..., with explicit end bound top. */
static __device__ __forceinline__ uint32_t wnr4_i4_find16_nat(uint4 a, uint4 b, uint32_t slot, uint32_t top,
                                                              uint32_t &c, uint32_t &f) {
    const uint32_t k4 = a.z & 0xffffu, k8 = b.x & 0xffffu, k12 = b.z & 0xffffu;
    const bool p1 = slot >= k4, p2 = slot >= k8, p3 = slot >= k12;
    const uint32_t r0  = p2 ? (p3 ? b.z : b.x) : (p1 ? a.z : a.x);
    const uint32_t r1  = p2 ? (p3 ? b.w : b.y) : (p1 ? a.w : a.y);
    const uint32_t c4  = p2 ? (p3 ? top : k12) : (p1 ? k8 : k4);
    const uint32_t yhi = p2 ? (p3 ? 12u : 8u) : (p1 ? 4u : 0u);
    const uint32_t c0 = r0 & 0xffffu, c1 = r0 >> 16, c2 = r1 & 0xffffu, c3 = r1 >> 16;
    const bool q1 = slot >= c1, q2 = slot >= c2, q3 = slot >= c3;
    c = q2 ? (q3 ? c3 : c2) : (q1 ? c1 : c0);
    const uint32_t hi = q2 ? (q3 ? c4 : c3) : (q1 ? c2 : c1);
    f = hi - c;
    return yhi + (q2 ? (q3 ? 3u : 2u) : (q1 ? 1u : 0u));
}

/* 256-symbol two-level scale lookup with swapped summary in SMEM and natural block in L1/L2. */
static __device__ __forceinline__ uint32_t wnr4_i4_find256_sm(const uint4 *sum_sm, const uint4 *blk_base,
                                                              uint32_t slot, uint32_t &c, uint32_t &f) {
    const uint4 a = sum_sm[0], b = sum_sm[1];
    const uint32_t sh = (slot << 16) | 0xffffu;
    const bool p1 = sh >= a.z, p2 = sh >= b.x, p3 = sh >= b.z;
    const uint32_t r0  = p2 ? (p3 ? b.z : b.x) : (p1 ? a.z : a.x);
    const uint32_t r1  = p2 ? (p3 ? b.w : b.y) : (p1 ? a.w : a.y);
    const uint32_t c4r = p2 ? (p3 ? (WNR4_TOTAL << 16) : b.z) : (p1 ? b.x : a.z);
    const uint32_t yhi = p2 ? (p3 ? 12u : 8u) : (p1 ? 4u : 0u);
    const uint32_t c1 = r0 & 0xffffu, c3 = r1 & 0xffffu, c4 = c4r >> 16;
    const bool q1 = slot >= c1, q2 = sh >= r1, q3 = slot >= c3;
    const uint32_t hi = q2 ? (q3 ? c4 : c3) : (q1 ? (r1 >> 16) : c1);
    const uint32_t kb = yhi + (q2 ? (q3 ? 3u : 2u) : (q1 ? 1u : 0u));
    const uint32_t jb = wnr4_i4_find16_nat(__ldg(blk_base + 2u * kb), __ldg(blk_base + 2u * kb + 1u), slot, hi, c, f);
    return 16u * kb + jb;
}

/* 256-symbol two-level scale lookup straight through __ldg (for escape-byte contexts 64..67). */
static __device__ __forceinline__ uint32_t wnr4_i4_find256(const uint4 *row, uint32_t slot, uint32_t &c, uint32_t &f) {
    uint32_t sS, fS;
    const uint32_t kb = wnr4_i4_find16_nat(__ldg(row), __ldg(row + 1), slot, WNR4_TOTAL, sS, fS);
    const uint32_t jb = wnr4_i4_find16_nat(__ldg(row + 2u + 2u * kb), __ldg(row + 3u + 2u * kb), slot, sS + fS, c, f);
    return 16u * kb + jb;
}

/* 4-slot async SMEM ring buffer for the tile's 16-bit word stream.
 * SLOT_W words per slot (64, 128 or 256); 4 slots total -> 2 slots ready + 1 in flight + 1 safe ahead.
 * Checking ensure(SLOT_W * 2 bytes) guarantees at least SLOT_W words are ready for SLOT_W/32 branch-free steps. */
template<uint32_t SLOT_W>
struct Wnr4I4Ring {
    static constexpr uint32_t RING_W  = SLOT_W * 4u;
    static constexpr uint32_t MASK_B  = (RING_W * 2u) - 1u;
    static constexpr uint32_t SLOT_B  = SLOT_W * 2u;
    static constexpr uint32_t SLOT_U4 = SLOT_B / 16u;
    const unsigned char *gbase;   /* 16-byte aligned global start (always >= rec_dev because tile header >= 128 B) */
    uint16_t *ring;               /* [RING_W] 16-byte aligned in SMEM */
    uint32_t ring_saddr;          /* shared address space byte offset of ring */
    uint32_t ptr_b;               /* current read offset in bytes (= word_index * 2) */
    uint32_t ready_end_b;         /* bytes guaranteed present in ring */
    uint32_t issue_end;           /* next word offset to issue via cp.async */
    uint32_t nwords_adj;          /* nwords + misalign */
    uint32_t misalign;            /* (words - gbase) / 2 in 0..7 */
};

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_i4_ring_issue(const Wnr4I4Ring<SLOT_W> &w, uint32_t word_off, uint32_t lane) {
    const uint32_t slot = (word_off / SLOT_W) & 3u;
    uint16_t *dst = w.ring + slot * SLOT_W;
    if (__builtin_expect(word_off + SLOT_W <= w.nwords_adj, 1)) {
        if constexpr (Wnr4I4Ring<SLOT_W>::SLOT_U4 < 32u) {
            if (lane < Wnr4I4Ring<SLOT_W>::SLOT_U4)
                wnr4_i4_cp_async_16((unsigned char *)dst + lane * 16u, w.gbase + word_off * 2u + lane * 16u);
        } else {
            wnr4_i4_cp_async_16((unsigned char *)dst + lane * 16u, w.gbase + word_off * 2u + lane * 16u);
        }
    } else {
        const uint32_t *g32 = (const uint32_t *)w.gbase;
        uint32_t *dst32 = (uint32_t *)dst;
        for (uint32_t k = lane; k < SLOT_W / 2u; k += 32u) {
            const uint32_t wi = word_off + 2u * k;
            uint32_t v = 0u;
            if (wi + 1u < w.nwords_adj) v = __ldg(g32 + (wi >> 1));
            else if (wi < w.nwords_adj) v = (uint32_t)__ldg((const uint16_t *)w.gbase + wi);
            dst32[k] = v;
        }
    }
    wnr4_i4_cp_commit();
}

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_i4_ring_init(Wnr4I4Ring<SLOT_W> &w, const uint16_t *words,
                                                         uint32_t nwords, uint16_t *ring, uint32_t lane) {
    const uintptr_t addr = (uintptr_t)words;
    const uintptr_t aligned = addr & ~(uintptr_t)15u;
    w.gbase = (const unsigned char *)aligned;
    w.misalign = (uint32_t)((addr - aligned) >> 1);
    w.nwords_adj = nwords + w.misalign;
    w.ring = ring;
    w.ring_saddr = (uint32_t)__cvta_generic_to_shared(ring);
    w.ptr_b = w.misalign * 2u;
    wnr4_i4_ring_issue(w, 0u, lane);
    wnr4_i4_ring_issue(w, SLOT_W, lane);
    wnr4_i4_ring_issue(w, 2u * SLOT_W, lane);
    wnr4_i4_cp_wait1();
    __syncwarp();
    w.ready_end_b = 4u * SLOT_W;
    w.issue_end = 3u * SLOT_W;
}

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_i4_ring_ensure(Wnr4I4Ring<SLOT_W> &w, uint32_t need_bytes, uint32_t lane) {
    if (__builtin_expect(w.ready_end_b - w.ptr_b < need_bytes, 0)) {
        wnr4_i4_cp_wait0();
        __syncwarp();
        w.ready_end_b += SLOT_W * 2u;
        wnr4_i4_ring_issue(w, w.issue_end, lane);
        w.issue_end += SLOT_W;
    }
}

/* Spread 4 tail nibbles (bits 0..15) into the high nibbles of the 4 bytes of planar word 7. */
static __device__ __forceinline__ uint32_t wnr4_i4_planar_tail_hi(uint32_t tail) {
    return ((tail & 0x000fu) << 4) | ((tail & 0x00f0u) << 8) | ((tail & 0x0f00u) << 12) | ((tail & 0xf000u) << 16);
}

/* One branch-free pipelined nibble step: decodes from (ra, rb), fires VOTE -> LDS.128 x2 -> POPC -> LDS.U16. */
template<uint32_t SLOT_W>
static __device__ __forceinline__ uint32_t wnr4_i4_fast_step(
    const uint4 *rows, uint32_t &x, uint32_t &mx, uint4 &ra, uint4 &rb,
    Wnr4I4Ring<SLOT_W> &wr, uint32_t lmask, bool load_next, bool is_first) {
    uint32_t y, xn;
    wnr4_i4_decode_4way_swap(ra, rb, x, y, xn);
    uint32_t st = 0u;
    if (load_next) {
        const uint32_t a = (uint32_t)abs((int)y - 8);
        mx = is_first ? a : (a > mx ? a : mx);
        st = 1u + ((mx * (mx + 1u)) >> 1) + a;
    }
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    if (load_next) wnr4_i4_lds128_x2(rows + st * 3u, ra, rb);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4I4Ring<SLOT_W>::MASK_B);
    x = wnr4_i4_lds_renorm(xn, raddr);
    wr.ptr_b += 2u * __popc(vote);
    return y;
}

/* SMEM budget per block:
 *   W = 1, 2, 4 : SLOT_W = 256 (2048 B ring/warp) + static kbuf (2048 B/warp) -> 26 / 30 / 38 KB static (<= 48 KB)
 *   W = 8       : SLOT_W = 128 (1024 B ring/warp) + static kbuf (2048 B/warp) -> 46.5 KB static (<= 48 KB, 2 blk/SM)
 *   W = 16      : SLOT_W =  64 ( 512 B ring/warp) + dynamic kbuf (32 KB)      -> 30.1 KB static + 32 KB dyn (<= 64 KB)
 * Launch bounds match actual SMEM occupancy so ptxas gets >= 128 regs/thread -> 0 register spills at every W. */
template<uint32_t W, bool Pairs>
static __global__ void __launch_bounds__(W * 32u, W == 8u ? 2u : 1u)
wnr4_i4_cuda_kernel(const Wnr4I4CudaJob *jobs, uint32_t njobs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    constexpr uint32_t NIB_ROWS = WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;   /* 414 */
    constexpr uint32_t SLOT_W   = (W <= 4u) ? 256u : ((W == 8u) ? 128u : 64u);
    constexpr bool DYN_KBUF     = (W == 16u);

    __shared__ uint4 sm[NIB_ROWS * 3u];                                 /* 19,872 B (48 B/row -> 8 bank groups) */
    __shared__ uint4 sm_ssum[64u * 2u];                                 /*  2,048 B (Phase 1 scale summaries) */
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];
    __shared__ alignas(16) unsigned char kbuf_static[DYN_KBUF ? 1u : (W * 32u * WNR4_I4_CUDA_MAX_G)];
    extern __shared__ unsigned char kbuf_dyn[];

    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t m = jobs[first].m;   /* the host groups jobs so a block has one matrix */
    const uint4 *mat_tab = tab + (size_t)m * WNR4_I4_CUDA_MAT_U4;
    const uint4 *srows = mat_tab + WNR4_I4_CUDA_NIB_U4;

    /* Cooperatively load the matrix's nibble rows (swapped into 48 B stride) and scale summaries, and L2-prefetch srows. */
    for (uint32_t i = threadIdx.x; i < WNR4_I4_CUDA_NIB_U4; i += blockDim.x) {
        const uint32_t row = i >> 1, half = i & 1u;
        sm[row * 3u + half] = wnr4_i4_swap_u4(__ldg(mat_tab + i));
    }
    for (uint32_t s = threadIdx.x; s < 64u * 2u; s += blockDim.x) {
        const uint32_t ctx = s >> 1, half = s & 1u;
        sm_ssum[s] = wnr4_i4_swap_u4(__ldg(srows + ctx * 34u + half));
    }
    for (uint32_t p = threadIdx.x; p < 272u; p += blockDim.x)
        wnr4_i4_prefetch_l2((const unsigned char *)srows + p * 128u);
    if constexpr (W > 1u) __syncthreads();
    else __syncwarp();

    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4I4CudaJob job = jobs[jid];
    if (!job.n) return;   /* padding job */
    if (job.m != m || job.n > 32u * WNR4_I4_CUDA_MAX_G) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }

    const uint16_t *words = (const uint16_t *)(job.tile + WNR4_TILE_STATE_BYTES);
    unsigned char *kk = (DYN_KBUF ? kbuf_dyn : kbuf_static) + warp * (32u * WNR4_I4_CUDA_MAX_G);
    uint32_t lmask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lmask));

    uint32_t x = (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane + 1u) << 16;
    uint32_t err = 0u;
    bool ok = x >= WNR4_L;

    Wnr4I4Ring<SLOT_W> wr;
    wnr4_i4_ring_init<SLOT_W>(wr, words, job.nwords, sm_ring + warp * (SLOT_W * 4u), lane);

    auto renorm_scalar = [&](bool need) {
        wnr4_i4_ring_ensure<SLOT_W>(wr, 64u, lane);
        const uint32_t vote = __ballot_sync(0xffffffffu, need);
        const uint32_t rank = __popc(vote & lmask);
        if (need) {
            const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4I4Ring<SLOT_W>::MASK_B);
            uint32_t w16;
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(w16) : "r"(raddr));
            x = (x << 16) | w16;
        }
        wr.ptr_b += 2u * __popc(vote);
    };

    uint32_t hash = wnr4_lane_seed(model_id, job.owner, job.t, lane, wnr4_word(WNR4_KIND_I4, job.pairs, job.G),
                                   job.m | job.anchor << 16);
    const uint32_t n = job.n, c0 = job.c0;
    const uint32_t gc = job.gc, nbc = job.nbc, bbase = job.bbase;
    const uint32_t ref_d = gc >= 32u ? gc : ((32u + gc - 1u) / gc) * gc;
    const uint32_t *bases32 = (const uint32_t *)job.bases;

    /* phase 1: scales, row-major across the lanes; incremental (cur_row, cur_col) avoids per-step division */
    uint32_t cur_row = (c0 + lane) / gc;
    uint32_t cur_col = (c0 + lane) - cur_row * gc;
    const uint32_t dr = 32u / gc, dc = 32u - dr * gc;
    for (uint32_t k = 0; 32u * k < n && __all_sync(0xffffffffu, ok); k++) {
        const uint32_t off_k = 32u * k + lane;
        const uint32_t i = c0 + off_k;
        const bool active = off_k < n;
        bool esc = false;
        uint32_t val = 0u;
        if (active) {
            const bool has = off_k >= ref_d;
            const uint32_t kref = wnr4_i4_kref(has ? kk[off_k - ref_d] : 0u, (int)has);
            const uint32_t sctx = wnr4_i4_sctx(kref, (int)has);
            uint32_t slot = x & (WNR4_TOTAL - 1u), c, f;
            uint32_t y = wnr4_i4_find256_sm(sm_ssum + sctx * 2u, srows + sctx * 34u + 2u, slot, c, f);
            x = f * (x >> WNR4_PREC) + slot - c;
            uint32_t kv = (y + kref) & 255u;
            if (kv > WNR4_I4_ESC) { ok = false; kv = WNR4_I4_ESC; }
            kk[off_k] = (unsigned char)kv;
            esc = kv == WNR4_I4_ESC;
            if (!esc) val = wnr4_i4_lat(kv, __ldg(bases32 + bbase + (cur_row >> 7) * nbc + (cur_col >> 1)));
        }
        cur_row += dr; cur_col += dc;
        if (cur_col >= gc) { cur_col -= gc; cur_row += 1u; }
        renorm_scalar(active && x < WNR4_L);
        if (__any_sync(0xffffffffu, esc)) {
            for (uint32_t b = 0; b < 4u; b++) {
                if (esc) {
                    uint32_t slot = x & (WNR4_TOTAL - 1u), c, f;
                    uint32_t y = wnr4_i4_find256(srows + (64u + b) * 34u, slot, c, f);
                    x = f * (x >> WNR4_PREC) + slot - c;
                    val |= y << (24u - 8u * b);
                }
                renorm_scalar(esc && x < WNR4_L);
            }
        }
        if (active) { job.scales[i] = val; hash = wnr4_h32_block(hash, val); }
        __syncwarp();   /* kk of this step is read by other lanes at later steps */
    }
    if (!__all_sync(0xffffffffu, ok)) {
        wnr4_i4_cp_wait0();
        if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT);
        return;
    }

    /* phase 2: groups, lane l owning groups c0 + l + 32 j, their 64 nibbles in element order */
    const bool match_pairs = (job.pairs == (uint32_t)Pairs);
    const uint32_t fast_rounds = (match_pairs && n >= 64u) ? (n - 32u) / 32u : 0u;
    uint32_t next_cls = (lane < n) ? wnr4_i4_cls(job.scales[c0 + lane], job.anchor) : 0u;

    for (uint32_t j = 0; j < fast_rounds; j++) {
        const uint32_t i = c0 + 32u * j + lane;
        const uint32_t cls = next_cls;
        next_cls = wnr4_i4_cls(job.scales[i + 32u], job.anchor);
        const uint4 *rows = sm + cls * (WNR4_I4_CUDA_STATES * 3u);
        uint4 ra = rows[0], rb = rows[1];
        uint32_t mx = 0u;
        uint32_t *grp32 = (uint32_t *)(job.codes + 32u * (size_t)i);

        if constexpr (!Pairs) {
            uint32_t *ptmp = (uint32_t *)kk;   /* [8][32] conflict-free planar staging buffer (1024 B/warp) */
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                ptmp[wq * 32u + lane] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                ptmp[(wq + 1u) * 32u + lane] = y4 | (y5 << 8) | (y6 << 16) | (y7 << 24);
            }
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                const uint32_t v0 = ptmp[wq * 32u + lane] | (y0 << 4) | (y1 << 12) | (y2 << 20) | (y3 << 28);
                hash = wnr4_h32_block(hash, v0);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, wq < 6u, false);
                const uint32_t v1 = ptmp[(wq + 1u) * 32u + lane] | (y4 << 4) | (y5 << 12) | (y6 << 20) | (y7 << 28);
                hash = wnr4_h32_block(hash, v1);
                grp32[wq] = v0;
                grp32[wq + 1u] = v1;
            }
        } else {
            /* pairs == 1: 8 consecutive nibbles pack directly into one uint32_t (zero ptmp traffic!) */
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true,    false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, wq < 7u, false);
                const uint32_t v = y0 | (y1 << 4) | (y2 << 8) | (y3 << 12) | (y4 << 16) | (y5 << 20) | (y6 << 24) | (y7 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
        }
    }

    /* Fast tail round when n is a non-zero multiple of 32 and packing matches Pairs */
    if (match_pairs && (n & 31u) == 0u && fast_rounds * 32u < n) {
        const uint32_t i = c0 + 32u * fast_rounds + lane;
        const uint4 *rows = sm + next_cls * (WNR4_I4_CUDA_STATES * 3u);
        uint4 ra = rows[0], rb = rows[1];
        uint32_t mx = 0u;
        uint32_t *grp32 = (uint32_t *)(job.codes + 32u * (size_t)i);
        uint32_t w7;
        if constexpr (!Pairs) {
            uint32_t *ptmp = (uint32_t *)kk;
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                ptmp[wq * 32u + lane] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
            }
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 7u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                const uint32_t v = ptmp[wq * 32u + lane] | (y0 << 4) | (y1 << 12) | (y2 << 20) | (y3 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
            w7 = ptmp[7u * 32u + lane];
        } else {
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 7u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
                const uint32_t v = y0 | (y1 << 4) | (y2 << 8) | (y3 << 12) | (y4 << 16) | (y5 << 20) | (y6 << 24) | (y7 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
            /* elements 56..59 (all 32 lanes!) */
            wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
            uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
            uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
            if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
            uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
            uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows, x, mx, ra, rb, wr, lmask, true, false);
            w7 = y0 | (y1 << 4) | (y2 << 8) | (y3 << 12);
        }
        /* elements 60..63: lanes 0 and 1 decode; lanes 2..31 carry tail = x - L */
        for (uint32_t e = 60u; e < 64u; e++) {
            const bool coded = (lane < 2u);
            if (coded) {
                uint32_t y, xn;
                wnr4_i4_decode_4way_swap(ra, rb, x, y, xn);
                x = xn;
                if constexpr (!Pairs) w7 |= y << (((e & 3u) * 8u) + 4u);
                else                  w7 |= y << (16u + (e & 3u) * 4u);
                if (e < 63u) {
                    const uint32_t a = (uint32_t)abs((int)y - 8);
                    mx = a > mx ? a : mx;
                    const uint32_t st = 1u + ((mx * (mx + 1u)) >> 1) + a;
                    ra = rows[st * 3u]; rb = rows[st * 3u + 1u];
                }
            }
            renorm_scalar(coded && x < WNR4_L);
        }
        if (lane >= 2u) {
            const uint32_t tail = x - WNR4_L;
            if (x < WNR4_L || tail > 0xffffu) err |= WNR4_ERR_STRUCT;
            if constexpr (!Pairs) w7 |= wnr4_i4_planar_tail_hi(tail);
            else                  w7 |= tail << 16;
            x = WNR4_L;
        }
        hash = wnr4_h32_block(hash, w7);
        grp32[7] = w7;
    } else {
        /* Generic partial-tile or mismatched-pairs fallback (handles any n in 1..2048 and both planar & pairs) */
        uint32_t pv[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        unsigned char *pgrp = nullptr;
        for (uint32_t j = fast_rounds; 32u * j < n; j++) {
            const uint32_t i = c0 + 32u * j + lane;
            const bool active = 32u * j + lane < n;
            const uint32_t cls = active ? wnr4_i4_cls(job.scales[i], job.anchor) : 0u;
            const uint4 *rows = sm + cls * (WNR4_I4_CUDA_STATES * 3u);
            uint32_t v[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, pm = 9u, mx = 9u;
            const bool tail = active && wnr4_i4_tail_skip(lane, j, 63u, n);
            const uint32_t eend = tail ? 60u : 64u;
            for (uint32_t e = 0; e < 64u; e++) {
                const bool coded = active && e < eend;
                if (coded) {
                    const uint32_t st = wnr4_i4_state(pm, mx);
                    uint32_t y, xn;
                    wnr4_i4_decode_4way_swap(rows[st * 3u], rows[st * 3u + 1u], x, y, xn);
                    x = xn;
                    v[e >> 3] |= y << ((e & 7u) * 4u);
                    const uint32_t a = wnr4_i4_mag(y);
                    pm = a; mx = wnr4_i4_mx(mx, a);
                }
                renorm_scalar(coded && x < WNR4_L);
            }
            if (active) {
                unsigned char *grp = job.codes + 32u * (size_t)i;
                if (tail) { for (uint32_t q = 0; q < 8u; q++) pv[q] = v[q]; pgrp = grp; }
                else {
                    uint32_t w[8];
                    wnr4_i4_pack(v, job.pairs, w);
                    for (uint32_t q = 0; q < 8u; q++) { hash = wnr4_h32_block(hash, w[q]); ((uint32_t *)grp)[q] = w[q]; }
                }
            }
        }
        if (pgrp) {
            const uint32_t tail = x - WNR4_L;
            uint32_t w[8];
            if (x < WNR4_L || tail > 0xffffu) err |= WNR4_ERR_STRUCT;
            pv[7] |= tail << 16;
            x = WNR4_L;
            wnr4_i4_pack(pv, job.pairs, w);
            for (uint32_t q = 0; q < 8u; q++) { hash = wnr4_h32_block(hash, w[q]); ((uint32_t *)pgrp)[q] = w[q]; }
        }
    }

    wnr4_i4_cp_wait0();
    const uint32_t ptr = (wr.ptr_b >> 1) - wr.misalign;
    const uint32_t lc = wnr4_lane_cells(n, lane);
    uint32_t H = wnr4_h32_final(hash, wnr4_i4_hlen(lc, lc));
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);
    if (lane == 0u && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1u && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (lane == 0u && ptr != job.nwords) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + job.rec, err);
}

/* ---- host side --------------------------------------------------------------- */
static thread_local uint32_t wnr4_i4_planned_pairs = 0u;

static inline size_t wnr4_i4_cuda_table_bytes(void) { return 3u * (size_t)WNR4_I4_CUDA_MAT_U4 * 16u; }
/* device image of a model, per matrix: the nibble rows cdf[0..15] in compact state order (class-major), then the
 * 68 scale rows as two-level rows (cdf[0], cdf[16], .., cdf[240], then cdf[0..255]) */
static inline void wnr4_i4_cuda_table_image(const Wnr4I4Model *md, unsigned char *img) {
    uint32_t m, c, pm, mx, q;
    for (m = 0; m < 3u; m++) {
        uint16_t *o = (uint16_t *)(img + (size_t)m * WNR4_I4_CUDA_MAT_U4 * 16u);
        for (c = 0; c < WNR4_I4_NCLS; c++) {
            memcpy(o + 16u * (c * WNR4_I4_CUDA_STATES), md->w + ((size_t)m * WNR4_I4_WROWS + wnr4_i4_wrow(c, 9u, 9u)) * 17u, 32u);
            for (mx = 0; mx < 9u; mx++)
                for (pm = 0; pm <= mx; pm++)
                    memcpy(o + 16u * (c * WNR4_I4_CUDA_STATES + wnr4_i4_state(pm, mx)),
                           md->w + ((size_t)m * WNR4_I4_WROWS + wnr4_i4_wrow(c, pm, mx)) * 17u, 32u);
        }
        o += 16u * WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;
        for (c = 0; c < WNR4_I4_SROWS; c++, o += 272) {
            const uint16_t *cdf = md->s + ((size_t)m * WNR4_I4_SROWS + c) * 257u;
            for (q = 0; q < 16u; q++) o[q] = cdf[16u * q];
            for (q = 0; q < 256u; q++) o[16u + q] = cdf[q];
        }
    }
}

/* Jobs of one record (host copy rec_host, device copy rec_dev of the same bytes, 4-byte aligned), decoded into
 * dst (device, g->raw bytes, 4-byte aligned); jobs appended per matrix to by_m[3]. Returns 0 or -1 (invalid record,
 * or G above WNR4_I4_CUDA_MAX_G: nothing appended). */
static inline int wnr4_i4_cuda_plan(const Wnr4I4Geo *g, const Wnr4I4Model *model, const unsigned char *rec_host,
                                    size_t total, uint32_t owner, const unsigned char *rec_dev, uint32_t index,
                                    unsigned char *dst, std::vector<Wnr4I4CudaJob> by_m[3]) {
    Wnr4I4Record r;
    std::vector<Wnr4I4CudaJob> mine;
    uint32_t t, m;
    if (wnr4_i4_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner || r.G > WNR4_I4_CUDA_MAX_G)
        return -1;
    const size_t data_at = (size_t)(r.data - rec_host), bases_at = (size_t)(r.bases - rec_host);
    for (t = 0; t < r.ntiles; t++) {   /* the offsets are read again here: check them again */
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        Wnr4I4CudaJob j;
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r.payload) return -1;
        wnr4_tile_cells(r.tile_base, g->cells, r.G, t, &m, &j.c0, &j.n);
        j.tile = rec_dev + data_at + o0; j.bases = rec_dev + bases_at;
        j.codes = dst + g->coff[m]; j.scales = (uint32_t *)(dst + g->soff[m]);
        j.nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u; j.rec = index; j.t = t; j.m = m;
        j.owner = r.owner; j.G = r.G; j.gc = g->gc[m]; j.nbc = g->nbc[m]; j.bbase = g->bbase[m];
        j.anchor = r.anchor[m]; j.pairs = g->pairs;
        mine.push_back(j);
    }
    for (auto &j : mine) by_m[j.m].push_back(j);
    return 0;
}

static inline uint32_t wnr4_i4_cuda_sms(void) {
    static int cached[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void)cudaGetLastError(); return 128u; }
    if (cached[dev] > 0) return (uint32_t)cached[dev];
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) {
        (void)cudaGetLastError();
        return 128u;
    }
    cached[dev] = sms;
    return (uint32_t)sms;
}

/* Concatenates the per-matrix lists, each padded to a multiple of the block size (padding jobs have n = 0).
 * Returns the block size: wnr4_cuda_warps, or chosen from tile count and SM count (1, 2, 4 or 8 warps). */
static inline uint32_t wnr4_i4_cuda_jobs(std::vector<Wnr4I4CudaJob> by_m[3], std::vector<Wnr4I4CudaJob> &jobs) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    uint32_t W = wnr4_cuda_warps;
    if (!W) {
        const size_t sms = (size_t)wnr4_i4_cuda_sms();
        W = tiles <= sms ? 1u : tiles <= 2u * sms ? 2u : tiles <= 4u * sms ? 4u : 8u;
    }
    wnr4_i4_planned_pairs = 0u;
    for (uint32_t m = 0; m < 3u; m++) if (!by_m[m].empty()) { wnr4_i4_planned_pairs = by_m[m][0].pairs; break; }
    jobs.clear();
    for (uint32_t m = 0; m < 3u; m++) {
        for (auto &j : by_m[m]) jobs.push_back(j);
        Wnr4I4CudaJob pad = {};
        pad.m = m;
        while (jobs.size() % W) jobs.push_back(pad);
    }
    return W;
}

template<bool Pairs>
static inline cudaError_t wnr4_i4_cuda_launch_typed(const uint4 *tab, uint32_t model_id, const Wnr4I4CudaJob *jobs_dev,
                                                    uint32_t njobs, uint32_t W, uint32_t blocks, uint32_t *errors_dev,
                                                    cudaStream_t stream) {
    if (W == 1u)      wnr4_i4_cuda_kernel< 1u, Pairs><<<blocks,  32u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 2u) wnr4_i4_cuda_kernel< 2u, Pairs><<<blocks,  64u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 4u) wnr4_i4_cuda_kernel< 4u, Pairs><<<blocks, 128u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 8u) wnr4_i4_cuda_kernel< 8u, Pairs><<<blocks, 256u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 16u) {
        constexpr size_t dsm = 16u * 32u * WNR4_I4_CUDA_MAX_G;   /* 32,768 B dynamic kbuf; static SMEM is 30,112 B */
        static bool configured[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || !configured[dev]) {
            cudaError_t e = cudaFuncSetAttribute(wnr4_i4_cuda_kernel<16u, Pairs>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm);
            if (e != cudaSuccess) return e;
            if (dev >= 0 && dev < 64) configured[dev] = true;
        }
        wnr4_i4_cuda_kernel<16u, Pairs><<<blocks, 512u, dsm, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    } else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_i4_cuda_launch(const unsigned char *tab_dev, uint32_t model_id, const Wnr4I4CudaJob *jobs_dev,
                                              uint32_t njobs, uint32_t W, uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t blocks = (njobs + W - 1u) / W;
    const uint4 *tab = (const uint4 *)tab_dev;
    if (!blocks) return cudaSuccess;
    if (wnr4_i4_planned_pairs) return wnr4_i4_cuda_launch_typed<true>(tab, model_id, jobs_dev, njobs, W, blocks, errors_dev, stream);
    return wnr4_i4_cuda_launch_typed<false>(tab, model_id, jobs_dev, njobs, W, blocks, errors_dev, stream);
}

#endif
