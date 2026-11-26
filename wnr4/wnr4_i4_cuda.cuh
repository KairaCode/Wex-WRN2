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
 * rows sit in 48-byte-strided shared memory (8 bank groups for LDS.128, pairs pre-swapped on the host for
 * zero-unpack level-1 4-way register search: 19,872 B), plus 2,048 B of pre-swapped scale summaries for Phase 1;
 * the tile's lattice indices sit in shared memory (32 * WNR4_I4_CUDA_MAX_G bytes per warp, reused in Phase 2 as
 * a conflict-free 8 x 32 planar nibble staging buffer). The lattice is computed with __fmul_rn and __fdiv_rn: do
 * not build with --use_fast_math or -ftz=true. Destination: the record's decoded bytes, 4-byte aligned.
 * Host side: wnr4_i4_cuda_table_image() (model), record-level direct launch (wnr4_i4_cuda_launch_recs() /
 * wnr4_i4_cuda_launch_recs_ptr() for uniform-G batches with zero per-tile job structs), and job-list API
 * (wnr4_i4_cuda_plan(), wnr4_i4_cuda_jobs(), wnr4_i4_cuda_launch() for mixed-G batches or custom callers). */
#ifndef WNR4_I4_CUDA_CUH
#define WNR4_I4_CUDA_CUH

#include "wnr4_cuda.cuh"
#include "wnr4_i4.h"

#define WNR4_I4_CUDA_MAX_G 64u
#define WNR4_I4_CUDA_STATES 46u
#define WNR4_I4_CUDA_NIB_U4 (WNR4_I4_NCLS * WNR4_I4_CUDA_STATES * 2u)          /* 828 */
#define WNR4_I4_CUDA_MAT_U4 (WNR4_I4_CUDA_NIB_U4 + WNR4_I4_SROWS * 34u)        /* + 2312 */
#define WNR4_I4_CUDA_INLINE_RECS 64u

struct Wnr4I4CudaJob {           /* one tile (job-list API) */
    const unsigned char *tile;   /* device: its 32 states then words */
    const unsigned char *bases;  /* device: the record's base area */
    unsigned char *codes;        /* device: the matrix's codes (group 0) */
    uint32_t *scales;            /* device: the matrix's scales (group 0) */
    uint32_t nwords, rec, t, m, c0, n, owner, G, gc, nbc, bbase, anchor, pairs;
};

struct Wnr4I4CudaRec {           /* one validated record (record-level direct launch) */
    const unsigned char *rec;    /* device: start of record (64 B header, offsets, bases, payload) */
    unsigned char *dst;          /* device: record's decoded bytes (4-byte aligned) */
    uint32_t owner, rec_idx, bases_at, data_at, payload;
    uint16_t anchor[3], _pad;
};

struct Wnr4I4CudaBatchGeo {      /* shared geometry of a uniform-G batch */
    uint32_t nmat, G, pairs, nrecs;
    uint32_t cells[3], gc[3], nbc[3], bbase[3];
    uint32_t coff[3], soff[3];
    uint32_t tile_base[4];
    uint32_t mat_blocks[4];
};

struct Wnr4I4InlineRecs {        /* <= 64 records passed directly via kernel constant-memory args */
    Wnr4I4CudaRec r[WNR4_I4_CUDA_INLINE_RECS];
};

/* compact state of (pm, mx): 0 = element 0 (9, 9), else 1 + mx (mx + 1) / 2 + pm (pm <= mx) */
static WNR4_HD inline uint32_t wnr4_i4_state(uint32_t pm, uint32_t mx) { return pm == 9u ? 0u : 1u + mx * (mx + 1u) / 2u + pm; }

/* ---- device helpers ------------------------------------------------------------- */
static __device__ __forceinline__ void wnr4_i4_prefetch_l2(const void *p) {
#if __CUDA_ARCH__ >= 700
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    (void)p;
#endif
}

#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void wnr4_i4_cp_async_16_sz(void *smem_ptr, const void *gmem_ptr, uint32_t src_bytes) {
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(saddr), "l"(gmem_ptr), "r"(src_bytes));
}
#endif
static __device__ __forceinline__ void wnr4_i4_cp_commit(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
static __device__ __forceinline__ void wnr4_i4_cp_wait0(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
static __device__ __forceinline__ void wnr4_i4_cp_wait1(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

/* Read-only dual 128-bit SMEM load (no memory clobber so ptxas schedules LDS.128 ahead of IMAD/VOTE/LDS.U16). */
static __device__ __forceinline__ void wnr4_i4_lds128_x2_ro(uint32_t saddr, uint4 &a, uint4 &b) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%8];\n\t"
                 "ld.shared.v4.u32 {%4,%5,%6,%7}, [%8+16];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
                   "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "r"(saddr));
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
                 : "+r"(out) : "r"(xn), "r"(saddr));
    return out;
}

/* 2-level 4-way register search on pre-swapped 16-symbol CDF row: a.x = (c0<<16)|c1, ..., b.w = (c14<<16)|c15.
 * Returns (y, c, hi, slot, q) so callers can issue the next state's LDS.128 before computing xn = (hi-c)*q+(slot-c). */
static __device__ __forceinline__ void wnr4_i4_decode_ychi(
    uint4 a, uint4 b, uint32_t x, uint32_t &y, uint32_t &c, uint32_t &hi, uint32_t &slot, uint32_t &q) {
    slot = x & (WNR4_TOTAL - 1u);
    q = x >> WNR4_PREC;
    const uint32_t sh = (slot << 16) | 0xffffu;
    const bool p1 = sh >= a.z, p2 = sh >= b.x, p3 = sh >= b.z;
    const uint32_t r0  = p2 ? (p3 ? b.z : b.x) : (p1 ? a.z : a.x);
    const uint32_t r1  = p2 ? (p3 ? b.w : b.y) : (p1 ? a.w : a.y);
    const uint32_t c4r = p2 ? (p3 ? (WNR4_TOTAL << 16) : b.z) : (p1 ? b.x : a.z);
    const uint32_t yhi = p2 ? (p3 ? 12u : 8u) : (p1 ? 4u : 0u);
    const uint32_t c0 = r0 >> 16, c1 = r0 & 0xffffu, c2 = r1 >> 16, c3 = r1 & 0xffffu, c4 = c4r >> 16;
    const bool q1 = slot >= c1, q2 = sh >= r1, q3 = slot >= c3;
    c  = q2 ? (q3 ? c3 : c2) : (q1 ? c1 : c0);
    hi = q2 ? (q3 ? c4 : c3) : (q1 ? c2 : c1);
    y  = yhi + (q2 ? (q3 ? 3u : 2u) : (q1 ? 1u : 0u));
}

static __device__ __forceinline__ void wnr4_i4_decode_4way_swap(uint4 a, uint4 b, uint32_t x, uint32_t &y, uint32_t &xn) {
    uint32_t c, hi, slot, q;
    wnr4_i4_decode_ychi(a, b, x, y, c, hi, slot, q);
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

/* 256-symbol two-level scale lookup with pre-swapped summary in SMEM and natural block in L1/L2. */
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
    uint32_t nbytes_adj;          /* (nwords + misalign) * 2 */
    uint32_t misalign;            /* (words - gbase) / 2 in 0..7 */
};

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_i4_ring_issue(const Wnr4I4Ring<SLOT_W> &w, uint32_t word_off, uint32_t lane) {
    const uint32_t byte_off = word_off * 2u;
    const uint32_t slot = (word_off / SLOT_W) & 3u;
    unsigned char *dst = (unsigned char *)(w.ring + slot * SLOT_W);
#if __CUDA_ARCH__ >= 800
    if (__builtin_expect(byte_off < w.nbytes_adj, 1)) {
        const uint32_t lane_off = byte_off + lane * 16u;
        if constexpr (Wnr4I4Ring<SLOT_W>::SLOT_U4 < 32u) {
            if (lane < Wnr4I4Ring<SLOT_W>::SLOT_U4) {
                const uint32_t rem = lane_off < w.nbytes_adj ? w.nbytes_adj - lane_off : 0u;
                wnr4_i4_cp_async_16_sz(dst + lane * 16u, rem ? (w.gbase + lane_off) : w.gbase, rem > 16u ? 16u : rem);
            }
        } else {
            const uint32_t rem = lane_off < w.nbytes_adj ? w.nbytes_adj - lane_off : 0u;
            wnr4_i4_cp_async_16_sz(dst + lane * 16u, rem ? (w.gbase + lane_off) : w.gbase, rem > 16u ? 16u : rem);
        }
    }
    wnr4_i4_cp_commit();
#else
    if (__builtin_expect(byte_off + Wnr4I4Ring<SLOT_W>::SLOT_B <= w.nbytes_adj, 1)) {
        if constexpr (Wnr4I4Ring<SLOT_W>::SLOT_U4 < 32u) {
            if (lane < Wnr4I4Ring<SLOT_W>::SLOT_U4)
                *((uint4 *)dst + lane) = __ldg((const uint4 *)(w.gbase + byte_off) + lane);
        } else {
            *((uint4 *)dst + lane) = __ldg((const uint4 *)(w.gbase + byte_off) + lane);
        }
    } else {
        const uint32_t nwords_adj = w.nbytes_adj >> 1;
        const uint32_t *g32 = (const uint32_t *)w.gbase;
        uint32_t *dst32 = (uint32_t *)dst;
        for (uint32_t k = lane; k < SLOT_W / 2u; k += 32u) {
            const uint32_t wi = word_off + 2u * k;
            uint32_t v = 0u;
            if (wi + 1u < nwords_adj) v = __ldg(g32 + (wi >> 1));
            else if (wi < nwords_adj) v = (uint32_t)__ldg((const uint16_t *)w.gbase + wi);
            dst32[k] = v;
        }
    }
#endif
}

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_i4_ring_init(Wnr4I4Ring<SLOT_W> &w, const uint16_t *words,
                                                         uint32_t nwords, uint16_t *ring, uint32_t lane) {
    const uintptr_t addr = (uintptr_t)words;
    const uintptr_t aligned = addr & ~(uintptr_t)15u;
    w.gbase = (const unsigned char *)aligned;
    w.misalign = (uint32_t)((addr - aligned) >> 1);
    w.nbytes_adj = (nwords + w.misalign) * 2u;
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

/* One branch-free pipelined nibble step: decodes y from (ra, rb), fires the next state's LDS.128 x2 ahead of
 * IMAD + VOTE + POPC + LDS.U16 so SMEM latency overlaps the rANS renorm critical path. */
template<uint32_t SLOT_W>
static __device__ __forceinline__ uint32_t wnr4_i4_fast_step(
    uint32_t rows_saddr, uint32_t &x, uint32_t &mx, uint4 &ra, uint4 &rb,
    Wnr4I4Ring<SLOT_W> &wr, uint32_t &ring_off_b, uint32_t lmask, bool load_next, bool is_first) {
    uint32_t y, c, hi, slot, q;
    wnr4_i4_decode_ychi(ra, rb, x, y, c, hi, slot, q);
    if (load_next) {
        const uint32_t a = (uint32_t)abs((int)y - 8);
        mx = is_first ? a : (a > mx ? a : mx);
        const uint32_t st = 1u + ((mx * (mx + 1u)) >> 1) + a;
        wnr4_i4_lds128_x2_ro(rows_saddr + st * 48u, ra, rb);
    }
    const uint32_t xn = (hi - c) * q + (slot - c);
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((ring_off_b + 2u * rank) & Wnr4I4Ring<SLOT_W>::MASK_B);
    const uint32_t adv = 2u * __popc(vote);
    ring_off_b = (ring_off_b + adv) & Wnr4I4Ring<SLOT_W>::MASK_B;
    wr.ptr_b += adv;
    x = wnr4_i4_lds_renorm(xn, raddr);
    return y;
}

/* Cooperatively load one matrix's pre-swapped nibble rows (48 B stride) and scale summaries, and L2-prefetch srows. */
template<uint32_t W>
static __device__ __forceinline__ void wnr4_i4_load_mat_smem(
    uint4 *sm, uint4 *sm_ssum, const uint4 *mat_tab, const uint4 *srows) {
    for (uint32_t i = threadIdx.x; i < WNR4_I4_CUDA_NIB_U4; i += blockDim.x)
        sm[(i >> 1) * 3u + (i & 1u)] = __ldg(mat_tab + i);
    for (uint32_t s = threadIdx.x; s < 64u * 2u; s += blockDim.x)
        sm_ssum[s] = __ldg(srows + (s >> 1) * 34u + (s & 1u));
    for (uint32_t p = threadIdx.x; p < 272u; p += blockDim.x)
        wnr4_i4_prefetch_l2((const unsigned char *)srows + p * 128u);
    if constexpr (W > 1u) __syncthreads();
    else __syncwarp();
}

/* Warp-level tile decoder shared by the job-list and record-level kernels. */
template<uint32_t W, bool Pairs>
static __device__ __forceinline__ void wnr4_i4_decode_tile_dev(
    uint4 *sm, uint4 *sm_ssum, uint16_t *sm_ring_warp, unsigned char *kk,
    const uint4 *srows, uint32_t model_id, uint32_t *errors,
    const unsigned char *tile, const unsigned char *bases, unsigned char *codes, uint32_t *scales,
    uint32_t nwords, uint32_t rec_idx, uint32_t t, uint32_t m, uint32_t c0, uint32_t n,
    uint32_t owner, uint32_t G, uint32_t gc, uint32_t nbc, uint32_t bbase, uint32_t anchor, uint32_t pairs) {
    constexpr uint32_t SLOT_W = (W <= 4u) ? 256u : ((W == 8u) ? 128u : 64u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint16_t *words = (const uint16_t *)(tile + WNR4_TILE_STATE_BYTES);
    uint32_t lmask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lmask));

    uint32_t x = (uint32_t)__ldg((const uint16_t *)tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)tile + 2u * lane + 1u) << 16;
    uint32_t err = 0u;
    bool ok = x >= WNR4_L;

    Wnr4I4Ring<SLOT_W> wr;
    wnr4_i4_ring_init<SLOT_W>(wr, words, nwords, sm_ring_warp, lane);

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

    uint32_t hash = wnr4_lane_seed(model_id, owner, t, lane, wnr4_word(WNR4_KIND_I4, pairs, G), m | anchor << 16);
    const uint32_t ref_d = gc >= 32u ? gc : ((32u + gc - 1u) / gc) * gc;
    const uint32_t *bases32 = (const uint32_t *)bases;

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
        if (active) { scales[i] = val; hash = wnr4_h32_block(hash, val); }
        __syncwarp();   /* kk of this step is read by other lanes at later steps */
    }
    if (!__all_sync(0xffffffffu, ok)) {
        wnr4_i4_cp_wait0();
        if (!lane) atomicOr(errors + rec_idx, WNR4_ERR_STRUCT);
        return;
    }

    /* phase 2: groups, lane l owning groups c0 + l + 32 j, their 64 nibbles in element order */
    const uint32_t sm_base_saddr = (uint32_t)__cvta_generic_to_shared(sm);
    const bool match_pairs = (pairs == (uint32_t)Pairs);
    const uint32_t fast_rounds = (match_pairs && n >= 64u) ? (n - 32u) / 32u : 0u;
    uint32_t next_cls = (lane < n) ? wnr4_i4_cls(scales[c0 + lane], anchor) : 0u;
    uint32_t ring_off_b = wr.ptr_b & Wnr4I4Ring<SLOT_W>::MASK_B;

    for (uint32_t j = 0; j < fast_rounds; j++) {
        const uint32_t i = c0 + 32u * j + lane;
        const uint32_t cls = next_cls;
        next_cls = wnr4_i4_cls(scales[i + 32u], anchor);
        const uint32_t rows_saddr = sm_base_saddr + cls * (WNR4_I4_CUDA_STATES * 48u);
        uint4 ra, rb;
        wnr4_i4_lds128_x2_ro(rows_saddr, ra, rb);
        uint32_t mx = 0u;
        uint32_t *grp32 = (uint32_t *)(codes + 32u * (size_t)i);

        if constexpr (!Pairs && W == 8u) {
            /* W == 8 (large batches): unroll fast rounds into register p[8] and emit 128-bit STG.128 when aligned. */
            uint32_t p[8];
#pragma unroll
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                p[wq] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                p[wq + 1u] = y4 | (y5 << 8) | (y6 << 16) | (y7 << 24);
            }
            uint32_t v[8];
#pragma unroll
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                v[wq] = p[wq] | (y0 << 4) | (y1 << 12) | (y2 << 20) | (y3 << 28);
                hash = wnr4_h32_block(hash, v[wq]);
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, wq < 6u, false);
                v[wq + 1u] = p[wq + 1u] | (y4 << 4) | (y5 << 12) | (y6 << 20) | (y7 << 28);
                hash = wnr4_h32_block(hash, v[wq + 1u]);
            }
            if ((((uintptr_t)codes) & 15u) == 0u) {
                ((uint4 *)grp32)[0] = make_uint4(v[0], v[1], v[2], v[3]);
                ((uint4 *)grp32)[1] = make_uint4(v[4], v[5], v[6], v[7]);
            } else {
#pragma unroll
                for (uint32_t q = 0u; q < 8u; q++) grp32[q] = v[q];
            }
        } else if constexpr (!Pairs) {
            uint32_t *ptmp = (uint32_t *)kk;   /* [8][32] conflict-free planar staging buffer (1024 B/warp) */
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                ptmp[wq * 32u + lane] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                ptmp[(wq + 1u) * 32u + lane] = y4 | (y5 << 8) | (y6 << 16) | (y7 << 24);
            }
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq += 2u) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                const uint32_t v0 = ptmp[wq * 32u + lane] | (y0 << 4) | (y1 << 12) | (y2 << 20) | (y3 << 28);
                hash = wnr4_h32_block(hash, v0);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, wq < 6u, false);
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
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true,    false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, wq < 7u, false);
                const uint32_t v = y0 | (y1 << 4) | (y2 << 8) | (y3 << 12) | (y4 << 16) | (y5 << 20) | (y6 << 24) | (y7 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
        }
    }

    /* Fast tail round when n is a non-zero multiple of 32 and packing matches Pairs (kept compact for I-cache) */
    if (match_pairs && (n & 31u) == 0u && fast_rounds * 32u < n) {
        const uint32_t i = c0 + 32u * fast_rounds + lane;
        const uint32_t rows_saddr = sm_base_saddr + next_cls * (WNR4_I4_CUDA_STATES * 48u);
        uint4 ra, rb;
        wnr4_i4_lds128_x2_ro(rows_saddr, ra, rb);
        uint32_t mx = 0u;
        uint32_t *grp32 = (uint32_t *)(codes + 32u * (size_t)i);
        uint32_t w7;
        if constexpr (!Pairs) {
            uint32_t *ptmp = (uint32_t *)kk;
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 8u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                ptmp[wq * 32u + lane] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
            }
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 7u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                const uint32_t v = ptmp[wq * 32u + lane] | (y0 << 4) | (y1 << 12) | (y2 << 20) | (y3 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
            w7 = ptmp[7u * 32u + lane];
        } else {
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 7u; wq++) {
                wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, wq == 0u);
                uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W <= 128u) wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t y4 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y5 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
                uint32_t y6 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                uint32_t y7 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
                const uint32_t v = y0 | (y1 << 4) | (y2 << 8) | (y3 << 12) | (y4 << 16) | (y5 << 20) | (y6 << 24) | (y7 << 28);
                hash = wnr4_h32_block(hash, v);
                grp32[wq] = v;
            }
            /* elements 56..59 (all 32 lanes!) */
            wnr4_i4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
            uint32_t y0 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
            uint32_t y1 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
            if constexpr (SLOT_W == 64u) wnr4_i4_ring_ensure<SLOT_W>(wr, 128u, lane);
            uint32_t y2 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
            uint32_t y3 = wnr4_i4_fast_step<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, ring_off_b, lmask, true, false);
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
                    wnr4_i4_lds128_x2_ro(rows_saddr + st * 48u, ra, rb);
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
            const uint32_t cls = active ? wnr4_i4_cls(scales[i], anchor) : 0u;
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
                unsigned char *grp = codes + 32u * (size_t)i;
                if (tail) { for (uint32_t q = 0; q < 8u; q++) pv[q] = v[q]; pgrp = grp; }
                else {
                    uint32_t w[8];
                    wnr4_i4_pack(v, pairs, w);
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
            wnr4_i4_pack(pv, pairs, w);
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
    if (lane == 0u && ptr != nwords) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + rec_idx, err);
}

/* SMEM budget per block:
 *   W = 1, 2, 4 : SLOT_W = 256 (2048 B ring/warp) + static kbuf (2048 B/warp) -> 26 / 30 / 38 KB static (<= 48 KB)
 *   W = 8       : SLOT_W = 128 (1024 B ring/warp) + static kbuf (2048 B/warp) -> 46.5 KB static (<= 48 KB, 2 blk/SM)
 *   W = 16      : SLOT_W =  64 ( 512 B ring/warp) + dynamic kbuf (32 KB)      -> 30.1 KB static + 32 KB dyn (<= 64 KB)
 * Launch bounds match actual SMEM occupancy so ptxas gets >= 128 regs/thread -> 0 register spills at every W. */
template<uint32_t W, bool Pairs>
static __global__ void __launch_bounds__(W * 32u, W == 8u ? 2u : 1u)
wnr4_i4_cuda_kernel(const Wnr4I4CudaJob *jobs, uint32_t njobs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    constexpr uint32_t NIB_ROWS = WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;
    constexpr uint32_t SLOT_W   = (W <= 4u) ? 256u : ((W == 8u) ? 128u : 64u);
    constexpr bool DYN_KBUF     = (W == 16u);

    __shared__ uint4 sm[NIB_ROWS * 3u];
    __shared__ uint4 sm_ssum[64u * 2u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];
    __shared__ alignas(16) unsigned char kbuf_static[DYN_KBUF ? 1u : (W * 32u * WNR4_I4_CUDA_MAX_G)];
    extern __shared__ unsigned char kbuf_dyn[];

    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t m = jobs[first].m;   /* the host groups jobs so a block has one matrix */
    const uint4 *mat_tab = tab + (size_t)m * WNR4_I4_CUDA_MAT_U4;
    const uint4 *srows = mat_tab + WNR4_I4_CUDA_NIB_U4;
    wnr4_i4_load_mat_smem<W>(sm, sm_ssum, mat_tab, srows);

    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4I4CudaJob job = jobs[jid];
    if (!job.n) return;   /* padding job */
    if (job.m != m || job.n > 32u * WNR4_I4_CUDA_MAX_G) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }

    unsigned char *kk = (DYN_KBUF ? kbuf_dyn : kbuf_static) + warp * (32u * WNR4_I4_CUDA_MAX_G);
    wnr4_i4_decode_tile_dev<W, Pairs>(sm, sm_ssum, sm_ring + warp * (SLOT_W * 4u), kk, srows, model_id, errors,
                                      job.tile, job.bases, job.codes, job.scales, job.nwords, job.rec, job.t, m,
                                      job.c0, job.n, job.owner, job.G, job.gc, job.nbc, job.bbase, job.anchor, job.pairs);
}

static __device__ __forceinline__ uint32_t wnr4_i4_sel3(uint32_t m, uint32_t a, uint32_t b, uint32_t c) {
    return m == 0u ? a : (m == 1u ? b : c);
}

/* Record-level direct launch (<= 64 records inline in constant-memory args): zero per-tile job structs or job H2D DMA. */
template<uint32_t W, bool Pairs>
static __global__ void __launch_bounds__(W * 32u, W == 8u ? 2u : 1u)
wnr4_i4_cuda_recs_kernel(Wnr4I4CudaBatchGeo geo, Wnr4I4InlineRecs irecs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    constexpr uint32_t NIB_ROWS = WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;
    constexpr uint32_t SLOT_W   = (W <= 4u) ? 256u : ((W == 8u) ? 128u : 64u);
    constexpr bool DYN_KBUF     = (W == 16u);

    __shared__ uint4 sm[NIB_ROWS * 3u];
    __shared__ uint4 sm_ssum[64u * 2u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];
    __shared__ alignas(16) unsigned char kbuf_static[DYN_KBUF ? 1u : (W * 32u * WNR4_I4_CUDA_MAX_G)];
    extern __shared__ unsigned char kbuf_dyn[];

    if (blockIdx.x >= geo.mat_blocks[3]) return;
    const uint32_t m = blockIdx.x < geo.mat_blocks[1] ? 0u : (blockIdx.x < geo.mat_blocks[2] ? 1u : 2u);
    const uint32_t blk_in_m = blockIdx.x - wnr4_i4_sel3(m, geo.mat_blocks[0], geo.mat_blocks[1], geo.mat_blocks[2]);
    const uint4 *mat_tab = tab + (size_t)m * WNR4_I4_CUDA_MAT_U4;
    const uint4 *srows = mat_tab + WNR4_I4_CUDA_NIB_U4;
    wnr4_i4_load_mat_smem<W>(sm, sm_ssum, mat_tab, srows);

    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t tb0 = wnr4_i4_sel3(m, geo.tile_base[0], geo.tile_base[1], geo.tile_base[2]);
    const uint32_t tb1 = wnr4_i4_sel3(m, geo.tile_base[1], geo.tile_base[2], geo.tile_base[3]);
    const uint32_t mt = tb1 - tb0;
    if (!mt) return;
    const uint32_t local_tile = blk_in_m * W + warp;
    const uint32_t ridx = (geo.nrecs == 1u) ? (local_tile >= mt ? 1u : 0u) : (local_tile / mt);
    if (ridx >= geo.nrecs) return;
    const uint32_t lt = (geo.nrecs == 1u) ? local_tile : (local_tile - ridx * mt);
    const Wnr4I4CudaRec &rec = irecs.r[ridx];
    const uint32_t t = tb0 + lt;
    const uint32_t *offs = (const uint32_t *)(rec.rec + WNR4_HEADER_BYTES);
    const uint32_t o0 = __ldg(offs + t), o1 = __ldg(offs + t + 1u);
    const uint32_t cells_m = wnr4_i4_sel3(m, geo.cells[0], geo.cells[1], geo.cells[2]);
    const uint32_t c0 = lt * (32u * geo.G);
    const uint32_t rem = cells_m - c0;
    const uint32_t n = rem < (32u * geo.G) ? rem : (32u * geo.G);
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 1u) || o1 > rec.payload || !n || n > 32u * WNR4_I4_CUDA_MAX_G) {
        if (!lane) atomicOr(errors + rec.rec_idx, WNR4_ERR_STRUCT);
        return;
    }
    const uint32_t coff_m   = wnr4_i4_sel3(m, geo.coff[0],  geo.coff[1],  geo.coff[2]);
    const uint32_t soff_m   = wnr4_i4_sel3(m, geo.soff[0],  geo.soff[1],  geo.soff[2]);
    const uint32_t gc_m     = wnr4_i4_sel3(m, geo.gc[0],    geo.gc[1],    geo.gc[2]);
    const uint32_t nbc_m    = wnr4_i4_sel3(m, geo.nbc[0],   geo.nbc[1],   geo.nbc[2]);
    const uint32_t bbase_m  = wnr4_i4_sel3(m, geo.bbase[0], geo.bbase[1], geo.bbase[2]);
    const uint32_t anchor_m = wnr4_i4_sel3(m, (uint32_t)rec.anchor[0], (uint32_t)rec.anchor[1], (uint32_t)rec.anchor[2]);
    unsigned char *kk = (DYN_KBUF ? kbuf_dyn : kbuf_static) + warp * (32u * WNR4_I4_CUDA_MAX_G);
    wnr4_i4_decode_tile_dev<W, Pairs>(sm, sm_ssum, sm_ring + warp * (SLOT_W * 4u), kk, srows, model_id, errors,
                                      rec.rec + rec.data_at + o0, rec.rec + rec.bases_at,
                                      rec.dst + coff_m, (uint32_t *)(rec.dst + soff_m),
                                      (o1 - o0 - WNR4_TILE_STATE_BYTES) >> 1, rec.rec_idx, t, m, c0, n,
                                      rec.owner, geo.G, gc_m, nbc_m, bbase_m, anchor_m, (uint32_t)Pairs);
}

/* Record-level direct launch for large batches (> 64 records): reads Wnr4I4CudaRec array from device memory. */
template<uint32_t W, bool Pairs>
static __global__ void __launch_bounds__(W * 32u, W == 8u ? 2u : 1u)
wnr4_i4_cuda_recs_ptr_kernel(Wnr4I4CudaBatchGeo geo, const Wnr4I4CudaRec *recs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    constexpr uint32_t NIB_ROWS = WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;
    constexpr uint32_t SLOT_W   = (W <= 4u) ? 256u : ((W == 8u) ? 128u : 64u);
    constexpr bool DYN_KBUF     = (W == 16u);

    __shared__ uint4 sm[NIB_ROWS * 3u];
    __shared__ uint4 sm_ssum[64u * 2u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];
    __shared__ alignas(16) unsigned char kbuf_static[DYN_KBUF ? 1u : (W * 32u * WNR4_I4_CUDA_MAX_G)];
    extern __shared__ unsigned char kbuf_dyn[];

    if (blockIdx.x >= geo.mat_blocks[3]) return;
    const uint32_t m = blockIdx.x < geo.mat_blocks[1] ? 0u : (blockIdx.x < geo.mat_blocks[2] ? 1u : 2u);
    const uint32_t blk_in_m = blockIdx.x - wnr4_i4_sel3(m, geo.mat_blocks[0], geo.mat_blocks[1], geo.mat_blocks[2]);
    const uint4 *mat_tab = tab + (size_t)m * WNR4_I4_CUDA_MAT_U4;
    const uint4 *srows = mat_tab + WNR4_I4_CUDA_NIB_U4;
    wnr4_i4_load_mat_smem<W>(sm, sm_ssum, mat_tab, srows);

    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t tb0 = wnr4_i4_sel3(m, geo.tile_base[0], geo.tile_base[1], geo.tile_base[2]);
    const uint32_t tb1 = wnr4_i4_sel3(m, geo.tile_base[1], geo.tile_base[2], geo.tile_base[3]);
    const uint32_t mt = tb1 - tb0;
    if (!mt) return;
    const uint32_t local_tile = blk_in_m * W + warp;
    const uint32_t ridx = local_tile / mt;
    if (ridx >= geo.nrecs) return;
    const uint32_t lt = local_tile - ridx * mt;
    const Wnr4I4CudaRec rec = recs[ridx];
    const uint32_t t = tb0 + lt;
    const uint32_t *offs = (const uint32_t *)(rec.rec + WNR4_HEADER_BYTES);
    const uint32_t o0 = __ldg(offs + t), o1 = __ldg(offs + t + 1u);
    const uint32_t cells_m = wnr4_i4_sel3(m, geo.cells[0], geo.cells[1], geo.cells[2]);
    const uint32_t c0 = lt * (32u * geo.G);
    const uint32_t rem = cells_m - c0;
    const uint32_t n = rem < (32u * geo.G) ? rem : (32u * geo.G);
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 1u) || o1 > rec.payload || !n || n > 32u * WNR4_I4_CUDA_MAX_G) {
        if (!lane) atomicOr(errors + rec.rec_idx, WNR4_ERR_STRUCT);
        return;
    }
    const uint32_t coff_m   = wnr4_i4_sel3(m, geo.coff[0],  geo.coff[1],  geo.coff[2]);
    const uint32_t soff_m   = wnr4_i4_sel3(m, geo.soff[0],  geo.soff[1],  geo.soff[2]);
    const uint32_t gc_m     = wnr4_i4_sel3(m, geo.gc[0],    geo.gc[1],    geo.gc[2]);
    const uint32_t nbc_m    = wnr4_i4_sel3(m, geo.nbc[0],   geo.nbc[1],   geo.nbc[2]);
    const uint32_t bbase_m  = wnr4_i4_sel3(m, geo.bbase[0], geo.bbase[1], geo.bbase[2]);
    const uint32_t anchor_m = wnr4_i4_sel3(m, (uint32_t)rec.anchor[0], (uint32_t)rec.anchor[1], (uint32_t)rec.anchor[2]);
    unsigned char *kk = (DYN_KBUF ? kbuf_dyn : kbuf_static) + warp * (32u * WNR4_I4_CUDA_MAX_G);
    wnr4_i4_decode_tile_dev<W, Pairs>(sm, sm_ssum, sm_ring + warp * (SLOT_W * 4u), kk, srows, model_id, errors,
                                      rec.rec + rec.data_at + o0, rec.rec + rec.bases_at,
                                      rec.dst + coff_m, (uint32_t *)(rec.dst + soff_m),
                                      (o1 - o0 - WNR4_TILE_STATE_BYTES) >> 1, rec.rec_idx, t, m, c0, n,
                                      rec.owner, geo.G, gc_m, nbc_m, bbase_m, anchor_m, (uint32_t)Pairs);
}

/* ---- host side --------------------------------------------------------------- */
static thread_local uint32_t wnr4_i4_planned_pairs = 0u;

static inline void wnr4_i4_swap_pairs_u16(uint16_t *dst, const uint16_t *src, size_t n16) {
    for (size_t i = 0; i < n16; i += 2u) { dst[i] = src[i + 1u]; dst[i + 1u] = src[i]; }
}

static inline size_t wnr4_i4_cuda_table_bytes(void) { return 3u * (size_t)WNR4_I4_CUDA_MAT_U4 * 16u; }
/* device image of a model, per matrix: nibble rows cdf[0..15] in compact state order (class-major, u16 pairs
 * pre-swapped for zero-unpack 4-way register search), then 68 scale rows as two-level rows (summary cdf[0,16..240]
 * pre-swapped for contexts 0..63 and natural for escape contexts 64..67, followed by natural cdf[0..255]). */
static inline void wnr4_i4_cuda_table_image(const Wnr4I4Model *md, unsigned char *img) {
    uint32_t m, c, pm, mx, q;
    for (m = 0; m < 3u; m++) {
        uint16_t *o = (uint16_t *)(img + (size_t)m * WNR4_I4_CUDA_MAT_U4 * 16u);
        for (c = 0; c < WNR4_I4_NCLS; c++) {
            wnr4_i4_swap_pairs_u16(o + 16u * (c * WNR4_I4_CUDA_STATES),
                                   md->w + ((size_t)m * WNR4_I4_WROWS + wnr4_i4_wrow(c, 9u, 9u)) * 17u, 16u);
            for (mx = 0; mx < 9u; mx++)
                for (pm = 0; pm <= mx; pm++)
                    wnr4_i4_swap_pairs_u16(o + 16u * (c * WNR4_I4_CUDA_STATES + wnr4_i4_state(pm, mx)),
                                           md->w + ((size_t)m * WNR4_I4_WROWS + wnr4_i4_wrow(c, pm, mx)) * 17u, 16u);
        }
        o += 16u * WNR4_I4_NCLS * WNR4_I4_CUDA_STATES;
        for (c = 0; c < WNR4_I4_SROWS; c++, o += 272) {
            const uint16_t *cdf = md->s + ((size_t)m * WNR4_I4_SROWS + c) * 257u;
            uint16_t sum[16];
            for (q = 0; q < 16u; q++) sum[q] = cdf[16u * q];
            if (c < 64u) wnr4_i4_swap_pairs_u16(o, sum, 16u);
            else memcpy(o, sum, 32u);
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
    uint32_t t, m;
    if (wnr4_i4_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner || r.G > WNR4_I4_CUDA_MAX_G)
        return -1;
    const size_t saved[3] = {by_m[0].size(), by_m[1].size(), by_m[2].size()};
    const size_t data_at = (size_t)(r.data - rec_host), bases_at = (size_t)(r.bases - rec_host);
    for (t = 0; t < r.ntiles; t++) {   /* the offsets are read again here: check them again */
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        Wnr4I4CudaJob j;
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r.payload) {
            for (m = 0; m < 3u; m++) by_m[m].resize(saved[m]);
            return -1;
        }
        wnr4_tile_cells(r.tile_base, g->cells, r.G, t, &m, &j.c0, &j.n);
        j.tile = rec_dev + data_at + o0; j.bases = rec_dev + bases_at;
        j.codes = dst + g->coff[m]; j.scales = (uint32_t *)(dst + g->soff[m]);
        j.nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u; j.rec = index; j.t = t; j.m = m;
        j.owner = r.owner; j.G = r.G; j.gc = g->gc[m]; j.nbc = g->nbc[m]; j.bbase = g->bbase[m];
        j.anchor = r.anchor[m]; j.pairs = g->pairs;
        by_m[m].push_back(j);
    }
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

static inline uint32_t wnr4_i4_cuda_choose_warps(size_t tiles) {
    uint32_t W = wnr4_cuda_warps;
    if (!W) {
        const size_t sms = (size_t)wnr4_i4_cuda_sms();
        W = tiles <= sms ? 1u : tiles <= 2u * sms ? 2u : tiles <= 4u * sms ? 4u : 8u;
    }
    return W;
}

/* Concatenates the per-matrix lists, each padded to a multiple of the block size (padding jobs have n = 0).
 * Returns the block size: wnr4_cuda_warps, or chosen from tile count and SM count (1, 2, 4 or 8 warps). */
static inline uint32_t wnr4_i4_cuda_jobs(std::vector<Wnr4I4CudaJob> by_m[3], std::vector<Wnr4I4CudaJob> &jobs) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    const uint32_t W = wnr4_i4_cuda_choose_warps(tiles);
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

template<typename KernelT>
static inline cudaError_t wnr4_i4_cuda_cfg_w16(KernelT fn, bool configured[64], size_t dsm) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || !configured[dev]) {
        cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm);
        if (e != cudaSuccess) return e;
        if (dev >= 0 && dev < 64) configured[dev] = true;
    }
    return cudaSuccess;
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
        constexpr size_t dsm = 16u * 32u * WNR4_I4_CUDA_MAX_G;
        static bool configured[64] = {};
        cudaError_t e = wnr4_i4_cuda_cfg_w16(wnr4_i4_cuda_kernel<16u, Pairs>, configured, dsm);
        if (e != cudaSuccess) return e;
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

template<bool Pairs>
static inline cudaError_t wnr4_i4_cuda_launch_recs_typed(const uint4 *tab, uint32_t model_id, const Wnr4I4CudaBatchGeo &geo,
                                                         const Wnr4I4InlineRecs &irecs, uint32_t W, uint32_t *errors_dev,
                                                         cudaStream_t stream) {
    const uint32_t blocks = geo.mat_blocks[3];
    if (!blocks) return cudaSuccess;
    if (W == 1u)      wnr4_i4_cuda_recs_kernel< 1u, Pairs><<<blocks,  32u, 0, stream>>>(geo, irecs, tab, model_id, errors_dev);
    else if (W == 2u) wnr4_i4_cuda_recs_kernel< 2u, Pairs><<<blocks,  64u, 0, stream>>>(geo, irecs, tab, model_id, errors_dev);
    else if (W == 4u) wnr4_i4_cuda_recs_kernel< 4u, Pairs><<<blocks, 128u, 0, stream>>>(geo, irecs, tab, model_id, errors_dev);
    else if (W == 8u) wnr4_i4_cuda_recs_kernel< 8u, Pairs><<<blocks, 256u, 0, stream>>>(geo, irecs, tab, model_id, errors_dev);
    else if (W == 16u) {
        constexpr size_t dsm = 16u * 32u * WNR4_I4_CUDA_MAX_G;
        static bool configured[64] = {};
        cudaError_t e = wnr4_i4_cuda_cfg_w16(wnr4_i4_cuda_recs_kernel<16u, Pairs>, configured, dsm);
        if (e != cudaSuccess) return e;
        wnr4_i4_cuda_recs_kernel<16u, Pairs><<<blocks, 512u, dsm, stream>>>(geo, irecs, tab, model_id, errors_dev);
    } else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_i4_cuda_launch_recs(const unsigned char *tab_dev, uint32_t model_id, const Wnr4I4CudaBatchGeo &geo,
                                                   const Wnr4I4InlineRecs &irecs, uint32_t W, uint32_t *errors_dev,
                                                   cudaStream_t stream) {
    const uint4 *tab = (const uint4 *)tab_dev;
    if (geo.pairs) return wnr4_i4_cuda_launch_recs_typed<true>(tab, model_id, geo, irecs, W, errors_dev, stream);
    return wnr4_i4_cuda_launch_recs_typed<false>(tab, model_id, geo, irecs, W, errors_dev, stream);
}

template<bool Pairs>
static inline cudaError_t wnr4_i4_cuda_launch_recs_ptr_typed(const uint4 *tab, uint32_t model_id, const Wnr4I4CudaBatchGeo &geo,
                                                             const Wnr4I4CudaRec *recs_dev, uint32_t W, uint32_t *errors_dev,
                                                             cudaStream_t stream) {
    const uint32_t blocks = geo.mat_blocks[3];
    if (!blocks) return cudaSuccess;
    if (W == 1u)      wnr4_i4_cuda_recs_ptr_kernel< 1u, Pairs><<<blocks,  32u, 0, stream>>>(geo, recs_dev, tab, model_id, errors_dev);
    else if (W == 2u) wnr4_i4_cuda_recs_ptr_kernel< 2u, Pairs><<<blocks,  64u, 0, stream>>>(geo, recs_dev, tab, model_id, errors_dev);
    else if (W == 4u) wnr4_i4_cuda_recs_ptr_kernel< 4u, Pairs><<<blocks, 128u, 0, stream>>>(geo, recs_dev, tab, model_id, errors_dev);
    else if (W == 8u) wnr4_i4_cuda_recs_ptr_kernel< 8u, Pairs><<<blocks, 256u, 0, stream>>>(geo, recs_dev, tab, model_id, errors_dev);
    else if (W == 16u) {
        constexpr size_t dsm = 16u * 32u * WNR4_I4_CUDA_MAX_G;
        static bool configured[64] = {};
        cudaError_t e = wnr4_i4_cuda_cfg_w16(wnr4_i4_cuda_recs_ptr_kernel<16u, Pairs>, configured, dsm);
        if (e != cudaSuccess) return e;
        wnr4_i4_cuda_recs_ptr_kernel<16u, Pairs><<<blocks, 512u, dsm, stream>>>(geo, recs_dev, tab, model_id, errors_dev);
    } else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_i4_cuda_launch_recs_ptr(const unsigned char *tab_dev, uint32_t model_id, const Wnr4I4CudaBatchGeo &geo,
                                                       const Wnr4I4CudaRec *recs_dev, uint32_t W, uint32_t *errors_dev,
                                                       cudaStream_t stream) {
    const uint4 *tab = (const uint4 *)tab_dev;
    if (geo.pairs) return wnr4_i4_cuda_launch_recs_ptr_typed<true>(tab, model_id, geo, recs_dev, W, errors_dev, stream);
    return wnr4_i4_cuda_launch_recs_ptr_typed<false>(tab, model_id, geo, recs_dev, W, errors_dev, stream);
}

#endif
