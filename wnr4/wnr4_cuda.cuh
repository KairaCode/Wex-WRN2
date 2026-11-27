/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CUDA decoder for MXFP4 (kind 0) and NVFP4 (kind 1) (docs/WNR4-SPEC.md; bit-exact with
 * wnr4_decode_tile). One warp decodes one tile with its 32 rANS lanes in lockstep: renormalising
 * lanes take the next 16-bit words of the tile in lane order from a per-warp 4-slot shared-memory
 * ring buffer fed asynchronously by cp.async (or synchronous loads on < sm_80), checking refill
 * only once every 2/4/8 nibble steps so the hot inner decode loop is branch-free. Lanes 2..31 carry
 * their tail in the state, lanes 0 and 1 end at the tile hash, every failure sets error bits of the
 * record (fail closed: its destinations may hold partial data).
 * A block holds 1, 2, 4, 8 or 16 tiles of one matrix (chosen automatically from tile count and SM
 * count to minimise single-record latency without extra waves, or overridden via wnr4_cuda_warps).
 * The matrix's bucket-0 weight rows sit in 48-byte-strided shared memory (8 bank groups for
 * LDS.128, u16 pairs pre-swapped on the host for zero-unpack 2-level 4-way register search:
 * 12,432 B MX / 23,088 B NV); when pb == 5, buckets 1..4 are stored bucket-major in global memory
 * and streamed via read-only 128-bit loads overlapped with the rANS renorm path. Scale tables use
 * pre-swapped two-level 256-symbol rows (staged in SMEM for MX; read via __ldg for NV with
 * modulo-free incremental column tracking). Every configuration fits <= 48 KiB static SMEM.
 * Host side: wnr4_cuda_table_bytes() / wnr4_cuda_table_image() / wnr4_cuda_tables_view() (model),
 * record-level direct launch (wnr4_cuda_launch_recs() / wnr4_cuda_launch_recs_ptr() for uniform-G
 * batches with zero per-tile job structs), and job-list API (wnr4_cuda_plan(), wnr4_cuda_jobs(),
 * wnr4_cuda_launch() for mixed-G batches or custom callers). */
#ifndef WNR4_CUDA_CUH
#define WNR4_CUDA_CUH

#include <cuda_runtime.h>
#include <atomic>
#include <cstring>
#include <vector>
#include "wnr4.h"

#define WNR4_CUDA_WARPS_MAX 16u
#define WNR4_CUDA_INLINE_RECS 32u

struct Wnr4CudaTables {      /* device copy of a model */
    uint32_t kind, id, rows, rows0, sctx, pb;   /* rows0 = NCLS * 37 (bucket 0); rows = rows0 * pb */
    const uint4 *wrows;      /* alias of wrows0 */
    const uint4 *wrows0;     /* [3][rows0][2]: bucket-0 cdf[0..15], pre-swapped u16 pairs */
    const uint4 *wrows14;    /* [3][4][rows0][2]: buckets 1..4 (pb == 5), bucket-major, pre-swapped; nullptr if pb == 1 */
    const uint16_t *scdf;    /* nullptr (replaced by pre-swapped srows) */
    const uint4 *srows;      /* [3][sctx][34]: pre-swapped 2 uint4 summary + 32 uint4 blocks */
};
struct Wnr4CudaRec {         /* per record of a batch (job-list API) */
    uint32_t model_id, owner, flags, G, anchors, kind;
    uint32_t scols[3];
    unsigned char *dst_s[3], *dst_w[3];
};
struct Wnr4CudaJob {         /* one tile (job-list API) */
    const unsigned char *tile;   /* device: its 32 states then words */
    uint32_t nwords, rec, t, m, c0, n;
};
struct Wnr4CudaDirectRec {   /* one validated record (record-level direct launch) */
    const unsigned char *rec;    /* device: start of record (64 B header, offsets, payload) */
    unsigned char *dst_s[3], *dst_w[3];   /* device: destinations (dst_w 16-byte aligned) */
    uint32_t owner, flags, anchors, rec_idx, data_at, payload;
};
struct Wnr4CudaBatchGeo {    /* shared geometry of a uniform-G batch */
    uint32_t kind, G, nrecs;
    uint32_t cells[3], scols[3];
    uint32_t tile_base[4];
    uint32_t mat_blocks[4];
};
struct Wnr4InlineRecs {      /* <= 32 records passed directly via kernel constant-memory args */
    Wnr4CudaDirectRec r[WNR4_CUDA_INLINE_RECS];
};

/* ---- shared device helpers (kept for wnr4_gg_cuda.cuh and legacy callers) ---- */
static __device__ __forceinline__ uint32_t wnr4_half_sign(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(r) : "r"(x));
    return r;
}
/* symbol, start, frequency of slot in a 16-symbol natural-order row (cdf[16] = 2^14) */
static __device__ __forceinline__ void wnr4_row_find(uint4 a, uint4 b, uint32_t slot, uint32_t &symbol,
                                                     uint32_t &start, uint32_t &frequency) {
    const uint32_t g = slot * 0x10001u | 0x80008000u;
    uint32_t m0 = wnr4_half_sign(g - a.x), m1 = wnr4_half_sign(g - a.y);
    uint32_t m2 = wnr4_half_sign(g - a.z), m3 = wnr4_half_sign(g - a.w);
    uint32_t m4 = wnr4_half_sign(g - b.x), m5 = wnr4_half_sign(g - b.y);
    uint32_t m6 = wnr4_half_sign(g - b.z), m7 = wnr4_half_sign(g - b.w);
    uint32_t bits = (m0 & 0x00010001u) | (m1 & 0x00020002u) | (m2 & 0x00040004u) | (m3 & 0x00080008u) |
                    (m4 & 0x00100010u) | (m5 & 0x00200020u) | (m6 & 0x00400040u) | (m7 & 0x00800080u);
    symbol = (uint32_t)__popc(bits) - 1u;
    uint32_t lo = __vmaxu2(__vmaxu2(__vmaxu2(a.x & m0, a.y & m1), __vmaxu2(a.z & m2, a.w & m3)),
                           __vmaxu2(__vmaxu2(b.x & m4, b.y & m5), __vmaxu2(b.z & m6, b.w & m7)));
    uint32_t hi = __vminu2(__vminu2(__vminu2(a.x | m0, a.y | m1), __vminu2(a.z | m2, a.w | m3)),
                           __vminu2(__vminu2(b.x | m4, b.y | m5), __vminu2(b.z | m6, b.w | m7)));
    start = max(lo & 0xffffu, lo >> 16);
    frequency = min(min(hi & 0xffffu, hi >> 16), WNR4_TOTAL) - start;
}
/* symbol of slot in a 16-symbol natural-order row: number of cdf[k] <= slot, minus 1 */
static __device__ __forceinline__ uint32_t wnr4_row_sym(uint4 a, uint4 b, uint32_t slot) {
    const uint32_t g = slot * 0x10001u | 0x80008000u;
    uint32_t bits = (wnr4_half_sign(g - a.x) & 0x00010001u) | (wnr4_half_sign(g - a.y) & 0x00020002u) |
                    (wnr4_half_sign(g - a.z) & 0x00040004u) | (wnr4_half_sign(g - a.w) & 0x00080008u) |
                    (wnr4_half_sign(g - b.x) & 0x00100010u) | (wnr4_half_sign(g - b.y) & 0x00200020u) |
                    (wnr4_half_sign(g - b.z) & 0x00400040u) | (wnr4_half_sign(g - b.w) & 0x00800080u);
    return (uint32_t)__popc(bits) - 1u;
}
/* compact (pm, mx) state: 0 = block start (8, 8), else 1 + mx (mx + 1) / 2 + pm (pm <= mx) */
static WNR4_HD inline uint32_t wnr4_state(uint32_t pm, uint32_t mx) { return pm == 8u ? 0u : 1u + mx * (mx + 1u) / 2u + pm; }

/* Register word window (used by wnr4_gg_cuda.cuh) */
struct Wnr4Win { uint32_t w0, w1, base; };
static __device__ __forceinline__ uint32_t wnr4_win_load(const uint16_t *words, uint32_t i, uint32_t nwords) {
    return i < nwords ? (uint32_t)__ldg(words + i) : 0u;
}
static __device__ __forceinline__ void wnr4_win_init(Wnr4Win &w, const uint16_t *words, uint32_t nwords, uint32_t lane) {
    w.base = 0; w.w0 = wnr4_win_load(words, lane, nwords); w.w1 = wnr4_win_load(words, 32u + lane, nwords);
}
static __device__ __forceinline__ bool wnr4_renorm_win(uint32_t &x, bool active, const uint16_t *words, uint32_t &ptr,
                                                       uint32_t nwords, uint32_t lane, Wnr4Win &w) {
    bool need = active && x < WNR4_L;
    uint32_t vote = __ballot_sync(0xffffffffu, need);
    if (!vote) return true;
    uint32_t total = __popc(vote);
    if (ptr + total > nwords) return false;
    uint32_t idx = ptr - w.base + __popc(vote & ((1u << lane) - 1u));   /* < 64 */
    uint32_t v0 = __shfl_sync(0xffffffffu, w.w0, idx & 31u), v1 = __shfl_sync(0xffffffffu, w.w1, idx & 31u);
    if (need) x = x << 16 | (idx < 32u ? v0 : v1);
    ptr += total;
    if (ptr - w.base >= 32u) { w.w0 = w.w1; w.base += 32u; w.w1 = wnr4_win_load(words, w.base + 32u + lane, nwords); }
    return true;
}

/* ---- v2 device primitives ---------------------------------------------------- */
static __device__ __forceinline__ void wnr4_prefetch_l2(const void *p) {
#if __CUDA_ARCH__ >= 700
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    (void)p;
#endif
}
#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void wnr4_cp_async_16_sz(uint32_t saddr, const void *gmem_ptr, uint32_t src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(saddr), "l"(gmem_ptr), "r"(src_bytes));
}
#endif
static __device__ __forceinline__ void wnr4_cp_commit(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
static __device__ __forceinline__ void wnr4_cp_wait0(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
static __device__ __forceinline__ void wnr4_cp_wait1(void) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

/* Read-only dual 128-bit loads (no memory clobber so ptxas hoists next-nibble loads ahead of renorm). */
static __device__ __forceinline__ void wnr4_lds128_x2_ro(uint32_t saddr, uint4 &a, uint4 &b) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%8];\n\t"
                 "ld.shared.v4.u32 {%4,%5,%6,%7}, [%8+16];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
                   "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "r"(saddr));
}
static __device__ __forceinline__ void wnr4_ldg128_x2_ro(const uint4 *ptr, uint4 &a, uint4 &b) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%8];\n\t"
                 "ld.global.nc.v4.u32 {%4,%5,%6,%7}, [%8+16];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
                   "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "l"(ptr));
}

/* Predicated SMEM uint16 renorm: if xn < WNR4_L (65536), loads word from saddr and returns (xn << 16) | word. */
static __device__ __forceinline__ uint32_t wnr4_lds_renorm(uint32_t xn, uint32_t saddr) {
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

/* 2-level 4-way register search on pre-swapped 16-symbol CDF row: a.x = (c0<<16)|c1, ..., b.w = (c14<<16)|c15,
 * with explicit upper bound top (WNR4_TOTAL for weight rows & level-1 scale summaries; hi_lvl1 for level-2 blocks). */
static __device__ __forceinline__ void wnr4_find16_swap_top(
    uint4 a, uint4 b, uint32_t slot, uint32_t top, uint32_t &y, uint32_t &c, uint32_t &hi) {
    const uint32_t sh = (slot << 16) | 0xffffu;
    const bool p1 = sh >= a.z, p2 = sh >= b.x, p3 = sh >= b.z;
    const uint32_t r0  = p2 ? (p3 ? b.z : b.x) : (p1 ? a.z : a.x);
    const uint32_t r1  = p2 ? (p3 ? b.w : b.y) : (p1 ? a.w : a.y);
    const uint32_t c4r = p2 ? (p3 ? (top << 16) : b.z) : (p1 ? b.x : a.z);
    const uint32_t yhi = p2 ? (p3 ? 12u : 8u) : (p1 ? 4u : 0u);
    const uint32_t c0 = r0 >> 16, c1 = r0 & 0xffffu, c2 = r1 >> 16, c3 = r1 & 0xffffu, c4 = c4r >> 16;
    const bool q1 = slot >= c1, q2 = sh >= r1, q3 = slot >= c3;
    c  = q2 ? (q3 ? c3 : c2) : (q1 ? c1 : c0);
    hi = q2 ? (q3 ? c4 : c3) : (q1 ? c2 : c1);
    y  = yhi + (q2 ? (q3 ? 3u : 2u) : (q1 ? 1u : 0u));
}

static __device__ __forceinline__ void wnr4_decode_ychi_top(
    uint4 a, uint4 b, uint32_t x, uint32_t top,
    uint32_t &y, uint32_t &c, uint32_t &hi, uint32_t &slot, uint32_t &q) {
    slot = x & (WNR4_TOTAL - 1u);
    q = x >> WNR4_PREC;
    wnr4_find16_swap_top(a, b, slot, top, y, c, hi);
}

/* 4-slot async SMEM ring buffer for the tile's 16-bit word stream (compact 6-register state on sm_80+).
 * Invariants: ptr_b starts at misalign*2 and advances by 2 per word consumed, so exact word consumption
 * ptr == nwords holds iff ptr_b == nbytes_adj; next slot to issue is always byte_off = ready_end_b + SLOT_B. */
template<uint32_t SLOT_W>
struct Wnr4Ring {
    static constexpr uint32_t RING_W  = SLOT_W * 4u;
    static constexpr uint32_t MASK_B  = (RING_W * 2u) - 1u;
    static constexpr uint32_t SLOT_B  = SLOT_W * 2u;
    static constexpr uint32_t SLOT_U4 = SLOT_B / 16u;
    const unsigned char *gbase;   /* 16-byte aligned global start (always >= tile start) */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
    uint16_t *ring;               /* [RING_W] 16-byte aligned in SMEM (only needed on < sm_80) */
#endif
    uint32_t ring_saddr;          /* shared address space byte offset of ring */
    uint32_t ptr_b;               /* current read offset in bytes */
    uint32_t ready_end_b;         /* bytes guaranteed present in ring */
    uint32_t nbytes_adj;          /* (nwords + misalign) * 2 */
};

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_ring_issue(const Wnr4Ring<SLOT_W> &w, uint32_t byte_off, uint32_t lane) {
    const uint32_t slot = (byte_off / Wnr4Ring<SLOT_W>::SLOT_B) & 3u;
#if __CUDA_ARCH__ >= 800
    const uint32_t dst_saddr = w.ring_saddr + slot * Wnr4Ring<SLOT_W>::SLOT_B;
    if (__builtin_expect(byte_off < w.nbytes_adj, 1)) {
        const uint32_t lane_off = byte_off + lane * 16u;
        if constexpr (Wnr4Ring<SLOT_W>::SLOT_U4 < 32u) {
            if (lane < Wnr4Ring<SLOT_W>::SLOT_U4) {
                const uint32_t rem = lane_off < w.nbytes_adj ? w.nbytes_adj - lane_off : 0u;
                wnr4_cp_async_16_sz(dst_saddr + lane * 16u, rem ? (w.gbase + lane_off) : w.gbase, rem > 16u ? 16u : rem);
            }
        } else {
            const uint32_t rem = lane_off < w.nbytes_adj ? w.nbytes_adj - lane_off : 0u;
            wnr4_cp_async_16_sz(dst_saddr + lane * 16u, rem ? (w.gbase + lane_off) : w.gbase, rem > 16u ? 16u : rem);
        }
    }
    wnr4_cp_commit();
#else
    unsigned char *dst = (unsigned char *)(w.ring + slot * SLOT_W);
    if (__builtin_expect(byte_off < w.nbytes_adj && w.nbytes_adj - byte_off >= Wnr4Ring<SLOT_W>::SLOT_B, 1)) {
        if constexpr (Wnr4Ring<SLOT_W>::SLOT_U4 < 32u) {
            if (lane < Wnr4Ring<SLOT_W>::SLOT_U4)
                *((uint4 *)dst + lane) = __ldg((const uint4 *)(w.gbase + byte_off) + lane);
        } else {
            *((uint4 *)dst + lane) = __ldg((const uint4 *)(w.gbase + byte_off) + lane);
        }
    } else {
        const uint32_t word_off = byte_off >> 1;
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
static __device__ __forceinline__ void wnr4_ring_init(Wnr4Ring<SLOT_W> &w, const uint16_t *words,
                                                      uint32_t nwords, uint16_t *ring, uint32_t lane) {
    const uintptr_t addr = (uintptr_t)words;
    const uintptr_t aligned = addr & ~(uintptr_t)15u;
    const uint32_t mis_b = (uint32_t)(addr - aligned);
    w.gbase = (const unsigned char *)aligned;
    w.nbytes_adj = nwords * 2u + mis_b;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
    w.ring = ring;
#endif
    w.ring_saddr = (uint32_t)__cvta_generic_to_shared(ring);
    w.ptr_b = mis_b;
    wnr4_ring_issue(w, 0u, lane);
    wnr4_ring_issue(w, Wnr4Ring<SLOT_W>::SLOT_B, lane);
    wnr4_ring_issue(w, 2u * Wnr4Ring<SLOT_W>::SLOT_B, lane);
    wnr4_cp_wait1();
    __syncwarp();
    w.ready_end_b = 2u * Wnr4Ring<SLOT_W>::SLOT_B;
}

template<uint32_t SLOT_W>
static __device__ __forceinline__ void wnr4_ring_ensure(Wnr4Ring<SLOT_W> &w, uint32_t need_bytes, uint32_t lane) {
    if (__builtin_expect(w.ready_end_b - w.ptr_b < need_bytes, 0)) {
        wnr4_cp_wait0();
        __syncwarp();
        w.ready_end_b += Wnr4Ring<SLOT_W>::SLOT_B;
        wnr4_ring_issue(w, w.ready_end_b, lane);
    }
}

/* Branch-free pipelined nibble steps: fire the next state's LDS.128 or LDG.128 ahead of IMAD + VOTE + POPC + LDS.U16. */
template<uint32_t SLOT_W>
static __device__ __forceinline__ uint32_t wnr4_fast_step_smem(
    uint32_t rows_saddr, uint32_t &x, uint32_t &mx, uint4 &ra, uint4 &rb,
    Wnr4Ring<SLOT_W> &wr, uint32_t lmask) {
    uint32_t y, c, hi, slot, q;
    wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
    const uint32_t a = y & 7u;
    mx = a > mx ? a : mx;
    const uint32_t st = 1u + ((mx * (mx + 1u)) >> 1) + a;
    wnr4_lds128_x2_ro(rows_saddr + st * 48u, ra, rb);
    const uint32_t xn = (hi - c) * q + (slot - c);
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
    wr.ptr_b += 2u * __popc(vote);
    x = wnr4_lds_renorm(xn, raddr);
    return y;
}

template<uint32_t SLOT_W, bool PB5>
static __device__ __forceinline__ uint32_t wnr4_fast_step_bkt(
    uint32_t rows_saddr, const uint4 *grows_cls_b, uint32_t &x, uint32_t &mx, uint4 &ra, uint4 &rb,
    Wnr4Ring<SLOT_W> &wr, uint32_t lmask) {
    uint32_t y, c, hi, slot, q;
    wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
    const uint32_t a = y & 7u;
    mx = a > mx ? a : mx;
    const uint32_t st = 1u + ((mx * (mx + 1u)) >> 1) + a;
    if constexpr (PB5) wnr4_ldg128_x2_ro(grows_cls_b + st * 2u, ra, rb);
    else               wnr4_lds128_x2_ro(rows_saddr + st * 48u, ra, rb);
    const uint32_t xn = (hi - c) * q + (slot - c);
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
    wr.ptr_b += 2u * __popc(vote);
    x = wnr4_lds_renorm(xn, raddr);
    return y;
}

template<uint32_t SLOT_W, bool UpdateMx>
static __device__ __forceinline__ uint32_t wnr4_fast_step_end(
    uint32_t &x, uint32_t &mx, uint4 ra, uint4 rb,
    Wnr4Ring<SLOT_W> &wr, uint32_t lmask) {
    uint32_t y, c, hi, slot, q;
    wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
    if constexpr (UpdateMx) {
        const uint32_t a = y & 7u;
        mx = a > mx ? a : mx;
    }
    const uint32_t xn = (hi - c) * q + (slot - c);
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
    wr.ptr_b += 2u * __popc(vote);
    x = wnr4_lds_renorm(xn, raddr);
    return y;
}

template<uint32_t SLOT_W>
static __device__ __forceinline__ uint32_t wnr4_fast_step_nv_mid(
    uint32_t next_rows_saddr, uint32_t &x, uint4 &ra, uint4 &rb,
    Wnr4Ring<SLOT_W> &wr, uint32_t lmask) {
    uint32_t y, c, hi, slot, q;
    wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
    wnr4_lds128_x2_ro(next_rows_saddr, ra, rb);
    const uint32_t xn = (hi - c) * q + (slot - c);
    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
    const uint32_t rank = __popc(vote & lmask);
    const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
    wr.ptr_b += 2u * __popc(vote);
    x = wnr4_lds_renorm(xn, raddr);
    return y;
}

/* Slot word count per warp: every (Kind, W) fits cleanly inside 48 KiB static SMEM.
 *   W <= 4 : SLOT_W = 256 (2048 B/warp -> 15.3..21.4 KB MX, 25.2..31.3 KB NV)
 *   W == 8 : SLOT_W = 128 (1024 B/warp -> 21.4 KB MX, 31.3 KB NV; 2+ blk/SM)
 *   W == 16: SLOT_W = 128 (1024 B/warp -> 29.6 KB MX [2 blk/SM], 39.5 KB NV [1 blk/SM + 88 KB L1]) */
template<uint32_t Kind, uint32_t W>
static WNR4_HD constexpr uint32_t wnr4_slot_w() {
    return (W <= 4u) ? 256u : 128u;
}

/* Cooperatively load matrix m's bucket-0 weight rows (48 B stride) and MX scale table into SMEM. */
template<uint32_t Kind, bool PB5, uint32_t W>
static __device__ __forceinline__ void wnr4_load_mat_smem(
    uint4 *sm, uint4 *sm_ssum, uint4 *sm_sblk, const Wnr4CudaTables &tab, uint32_t m) {
    constexpr uint32_t ROWS0 = WNR4_NCLS(Kind) * 37u;
    const uint4 *w0 = tab.wrows0 + (size_t)m * ROWS0 * 2u;
    for (uint32_t i = threadIdx.x; i < ROWS0 * 2u; i += blockDim.x)
        sm[(i >> 1) * 3u + (i & 1u)] = __ldg(w0 + i);
    if constexpr (!Kind) {
        const uint4 *sr = tab.srows + (size_t)m * 34u;
        if (threadIdx.x < 2u) sm_ssum[threadIdx.x] = __ldg(sr + threadIdx.x);
        if (threadIdx.x < 32u) sm_sblk[(threadIdx.x >> 1) * 3u + (threadIdx.x & 1u)] = __ldg(sr + 2u + threadIdx.x);
    }
    if constexpr (PB5) {
        const unsigned char *w14 = (const unsigned char *)(tab.wrows14 + (size_t)m * 4u * ROWS0 * 2u);
        constexpr uint32_t LINES = ROWS0;   /* 4 * ROWS0 * 32 B / 128 B = ROWS0 cache lines */
        for (uint32_t p = threadIdx.x; p < LINES; p += blockDim.x)
            wnr4_prefetch_l2(w14 + p * 128u);
    }
    if constexpr (W > 1u) __syncthreads();
    else __syncwarp();
}

static __device__ __forceinline__ uint32_t wnr4_sel3(uint32_t m, uint32_t a, uint32_t b, uint32_t c) {
    return m == 0u ? a : (m == 1u ? b : c);
}

static __device__ __forceinline__ bool wnr4_has_neg_zero4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    const uint32_t t0 = w0 ^ 0x88888888u, t1 = w1 ^ 0x88888888u, t2 = w2 ^ 0x88888888u, t3 = w3 ^ 0x88888888u;
    return ((((t0 - 0x11111111u) & ~t0) | ((t1 - 0x11111111u) & ~t1) |
             ((t2 - 0x11111111u) & ~t2) | ((t3 - 0x11111111u) & ~t3)) & 0x88888888u) != 0u;
}

/* Warp-level tile decoder shared by the job-list and record-level kernels. */
template<uint32_t Kind, bool PB5, uint32_t W>
static __device__ __forceinline__ void wnr4_decode_tile_dev(
    uint4 *sm, const uint4 *sm_ssum, const uint4 *sm_sblk, uint16_t *sm_ring_warp,
    const Wnr4CudaTables &tab, uint32_t *err_ptr,
    const unsigned char *tile, unsigned char *so, uint4 *wo,
    uint32_t nwords, uint32_t t, uint32_t m, uint32_t c0, uint32_t n,
    uint32_t owner, uint32_t flags, uint32_t G, uint32_t anchors, uint32_t cols) {
    constexpr uint32_t SPC = Kind ? 2u : 1u, B = Kind ? 16u : 32u;
    constexpr int R = Kind ? 6 : 3;
    constexpr uint32_t ROWS0 = WNR4_NCLS(Kind) * 37u;
    constexpr uint32_t SLOT_W = wnr4_slot_w<Kind, W>();
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t lmask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lmask));

    uint32_t x = (uint32_t)__ldg((const uint16_t *)tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)tile + 2u * lane + 1u) << 16;
    if (!__all_sync(0xffffffffu, x >= WNR4_L)) {
        if (!lane) atomicOr(err_ptr, WNR4_ERR_STRUCT);
        return;
    }

    const uint16_t *words = (const uint16_t *)(tile + WNR4_TILE_STATE_BYTES);
    Wnr4Ring<SLOT_W> wr;
    wnr4_ring_init<SLOT_W>(wr, words, nwords, sm_ring_warp, lane);

    auto renorm_scalar = [&](bool need) {
        wnr4_ring_ensure<SLOT_W>(wr, 64u, lane);
        const uint32_t vote = __ballot_sync(0xffffffffu, need);
        const uint32_t rank = __popc(vote & lmask);
        if (need) {
            const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
            uint32_t w16;
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(w16) : "r"(raddr));
            x = (x << 16) | w16;
        }
        wr.ptr_b += 2u * __popc(vote);
    };

    uint32_t err = 0u;
    uint32_t hash = wnr4_lane_seed(tab.id, owner, t, lane, wnr4_word(Kind, flags, G), anchors);
    const uint32_t anchor = __shfl_sync(0xffffffffu, (anchors >> (8u * m)) & 0xffu, 0);

    /* ---- phase 1: scales, row-major across the lanes ---- */
    {
        const uint32_t ns = n * SPC, S0 = c0 * SPC;
        if constexpr (!Kind) {
            const uint4 ssum_a = sm_ssum[0], ssum_b = sm_ssum[1];
            const uint32_t sblk_saddr = (uint32_t)__cvta_generic_to_shared(sm_sblk);
            const uint32_t full_s = ns >> 5;
            for (uint32_t k = 0u; k < full_s; k++) {
                wnr4_ring_ensure<SLOT_W>(wr, 64u, lane);
                uint32_t kb, sS, hiS, slot, q;
                wnr4_decode_ychi_top(ssum_a, ssum_b, x, WNR4_TOTAL, kb, sS, hiS, slot, q);
                uint4 ba, bb;
                wnr4_lds128_x2_ro(sblk_saddr + kb * 48u, ba, bb);
                uint32_t jb, c, hi;
                wnr4_find16_swap_top(ba, bb, slot, hiS, jb, c, hi);
                const uint32_t y = (kb << 4) | jb;
                const uint32_t xn = (hi - c) * q + (slot - c);
                const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
                const uint32_t rank = __popc(vote & lmask);
                const uint32_t raddr = wr.ring_saddr + ((wr.ptr_b + 2u * rank) & Wnr4Ring<SLOT_W>::MASK_B);
                wr.ptr_b += 2u * __popc(vote);
                x = wnr4_lds_renorm(xn, raddr);
                so[S0 + (k << 5) + lane] = (unsigned char)y;
                hash = wnr4_h32_block(hash, y);
            }
            if (ns & 31u) {
                const uint32_t off = (full_s << 5) + lane;
                const bool active = off < ns;
                if (active) {
                    uint32_t kb, sS, hiS, slot, q;
                    wnr4_decode_ychi_top(ssum_a, ssum_b, x, WNR4_TOTAL, kb, sS, hiS, slot, q);
                    uint4 ba, bb;
                    wnr4_lds128_x2_ro(sblk_saddr + kb * 48u, ba, bb);
                    uint32_t jb, c, hi;
                    wnr4_find16_swap_top(ba, bb, slot, hiS, jb, c, hi);
                    const uint32_t y = (kb << 4) | jb;
                    x = (hi - c) * q + (slot - c);
                    so[S0 + off] = (unsigned char)y;
                    hash = wnr4_h32_block(hash, y);
                }
                renorm_scalar(active && x < WNR4_L);
            }
        } else {
            const uint4 *srows = tab.srows + (size_t)m * tab.sctx * 34u;
            if (cols >= 33u) {
                uint32_t cur_col = (S0 + lane) % cols;
                for (uint32_t k = 0u; 32u * k < ns; k++) {
                    const uint32_t off = (k << 5) + lane;
                    const bool active = off < ns;
                    if (active) {
                        const uint32_t i = S0 + off;
                        uint32_t pred = 0u, sctx = 63u;
                        if (off >= cols) {
                            const uint32_t up = (uint32_t)so[i - cols];
                            uint32_t dul = 7u, dur = 7u;
                            if (cur_col > 0u && off > cols) {
                                int d = (int)so[i - cols - 1u] - (int)up;
                                dul = (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3);
                            }
                            if (cur_col + 1u < cols) {
                                int d = (int)so[i - cols + 1u] - (int)up;
                                dur = (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3);
                            }
                            pred = up;
                            sctx = (1u + (up >> 2)) * 64u + dul * 8u + dur;
                        }
                        const uint4 *row = srows + (size_t)sctx * 34u;
                        uint4 sa, sb, ba, bb;
                        wnr4_ldg128_x2_ro(row, sa, sb);
                        uint32_t kb, sS, hiS, slot, q;
                        wnr4_decode_ychi_top(sa, sb, x, WNR4_TOTAL, kb, sS, hiS, slot, q);
                        wnr4_ldg128_x2_ro(row + 2u + 2u * kb, ba, bb);
                        uint32_t jb, c, hi;
                        wnr4_find16_swap_top(ba, bb, slot, hiS, jb, c, hi);
                        x = (hi - c) * q + (slot - c);
                        const uint32_t y = (((kb << 4) | jb) + pred) & 255u;
                        so[i] = (unsigned char)y;
                        hash = wnr4_h32_block(hash, y);
                    }
                    cur_col += 32u;
                    if (cur_col >= cols) cur_col -= cols;
                    renorm_scalar(active && x < WNR4_L);
                    __syncwarp();
                }
            } else {
                const uint4 *row63 = srows + 63u * 34u;
                uint4 sa, sb;
                wnr4_ldg128_x2_ro(row63, sa, sb);
                for (uint32_t k = 0u; 32u * k < ns; k++) {
                    const uint32_t off = (k << 5) + lane;
                    const bool active = off < ns;
                    if (active) {
                        uint32_t kb, sS, hiS, slot, q;
                        wnr4_decode_ychi_top(sa, sb, x, WNR4_TOTAL, kb, sS, hiS, slot, q);
                        uint4 ba, bb;
                        wnr4_ldg128_x2_ro(row63 + 2u + 2u * kb, ba, bb);
                        uint32_t jb, c, hi;
                        wnr4_find16_swap_top(ba, bb, slot, hiS, jb, c, hi);
                        x = (hi - c) * q + (slot - c);
                        const uint32_t y = (kb << 4) | jb;
                        so[S0 + off] = (unsigned char)y;
                        hash = wnr4_h32_block(hash, y);
                    }
                    renorm_scalar(active && x < WNR4_L);
                }
            }
        }
    }
    __syncwarp();   /* phase 2 reads scales written by other lanes (NV) */

    /* ---- phase 2: cells, lane l owning cells c0 + l + 32 j, 32 nibbles each ---- */
    const uint32_t flag_bits = __shfl_sync(0xffffffffu, ((flags & WNR4_FLAG_CZ) ? 2u : 0u) | ((!Kind && ((flags >> m) & 1u)) ? 1u : 0u), 0);
    const uint32_t sm_base_saddr = (uint32_t)__cvta_generic_to_shared(sm);
    const uint4 *grows14_m = PB5 ? (tab.wrows14 + (size_t)m * 4u * ROWS0 * 2u) : nullptr;
    const uint32_t fast_rounds = (n >= 64u) ? (n - 32u) / 32u : 0u;
    uint32_t next_s = 0u;
    if (lane < n) {
        next_s = (uint32_t)so[(size_t)(c0 + lane) * SPC];
        if constexpr (Kind) next_s |= ((uint32_t)so[(size_t)(c0 + lane) * 2u + 1u]) << 8;
    }

    auto decode_8_smem = [&](uint32_t rows_saddr, uint32_t &mx, uint4 &ra, uint4 &rb) -> uint32_t {
        wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
        uint32_t v = wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask);
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 4;
        if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 8;
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 12;
        if constexpr (SLOT_W <= 128u) wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 16;
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 20;
        if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 24;
        v |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 28;
        return v;
    };

    for (uint32_t j = 0u; j < fast_rounds; j++) {
        const uint32_t cell = c0 + (j << 5) + lane;
        const uint32_t cur_s = next_s;
        next_s = (uint32_t)so[(size_t)(cell + 32u) * SPC];
        if constexpr (Kind) next_s |= ((uint32_t)so[(size_t)(cell + 32u) * 2u + 1u]) << 8;
        uint32_t w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u, mx = 0u;
        if constexpr (!Kind) {
            const uint32_t cls = wnr4_cls(cur_s, anchor, R);
            const uint32_t rows_saddr = sm_base_saddr + cls * (37u * 48u);
            uint4 ra, rb;
            wnr4_lds128_x2_ro(rows_saddr, ra, rb);
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 3u; wq++) {
                const uint32_t v = decode_8_smem(rows_saddr, mx, ra, rb);
                if (wq == 0u) w0 = v;
                else if (wq == 1u) w1 = v;
                else w2 = v;
            }
            const uint4 *grows_cls = PB5 ? (grows14_m + cls * (37u * 2u)) : nullptr;
            /* nibbles 24..31: 24..26 load SMEM, 27..30 load buckets 1..4 (or SMEM if !PB5), 31 ends block */
            wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
            w3  = wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask);
            w3 |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 4;
            if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
            w3 |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 8;
            w3 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls, x, mx, ra, rb, wr, lmask) << 12;
            if constexpr (SLOT_W <= 128u) wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
            w3 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 2u, x, mx, ra, rb, wr, lmask) << 16;
            w3 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 4u, x, mx, ra, rb, wr, lmask) << 20;
            if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
            w3 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 6u, x, mx, ra, rb, wr, lmask) << 24;
            w3 |= wnr4_fast_step_end<SLOT_W, true>(x, mx, ra, rb, wr, lmask) << 28;
        } else {
            uint4 ra, rb;
            const uint32_t cls0 = wnr4_cls(cur_s & 0xffu, anchor, R);
            wnr4_lds128_x2_ro(sm_base_saddr + cls0 * (37u * 48u), ra, rb);
#pragma unroll 1
            for (uint32_t blk = 0u; blk < 2u; blk++) {
                const uint32_t cls = wnr4_cls(blk == 0u ? (cur_s & 0xffu) : (cur_s >> 8), anchor, R);
                const uint32_t rows_saddr = sm_base_saddr + cls * (37u * 48u);
                mx = 0u;
                const uint32_t va = decode_8_smem(rows_saddr, mx, ra, rb);
                const uint4 *grows_cls = PB5 ? (grows14_m + cls * (37u * 2u)) : nullptr;
                wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                uint32_t vb = wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask);
                vb |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 4;
                if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
                vb |= wnr4_fast_step_smem<SLOT_W>(rows_saddr, x, mx, ra, rb, wr, lmask) << 8;
                vb |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls, x, mx, ra, rb, wr, lmask) << 12;
                if constexpr (SLOT_W <= 128u) wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
                vb |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 2u, x, mx, ra, rb, wr, lmask) << 16;
                vb |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 4u, x, mx, ra, rb, wr, lmask) << 20;
                if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
                vb |= wnr4_fast_step_bkt<SLOT_W, PB5>(rows_saddr, grows_cls + ROWS0 * 6u, x, mx, ra, rb, wr, lmask) << 24;
                if (blk == 0u) {
                    const uint32_t rs1 = sm_base_saddr + wnr4_cls(cur_s >> 8, anchor, R) * (37u * 48u);
                    vb |= wnr4_fast_step_nv_mid<SLOT_W>(rs1, x, ra, rb, wr, lmask) << 28;
                    w0 = va; w1 = vb;
                } else {
                    vb |= wnr4_fast_step_end<SLOT_W, false>(x, mx, ra, rb, wr, lmask) << 28;
                    w2 = va; w3 = vb;
                }
            }
        }
        hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w0), w1), w2), w3);
        if ((flag_bits & 2u) && wnr4_has_neg_zero4(w0, w1, w2, w3)) err |= WNR4_ERR_CZ;
        if constexpr (!Kind) {
            if ((flag_bits & 1u) && mx == 5u) {
                if (!cur_s) err |= WNR4_ERR_CEIL;
                else {
                    so[cell] = (unsigned char)(cur_s - 1u);
                    w0 = wnr4_ceil_dbl8(w0); w1 = wnr4_ceil_dbl8(w1);
                    w2 = wnr4_ceil_dbl8(w2); w3 = wnr4_ceil_dbl8(w3);
                }
            }
        }
        wo[cell] = make_uint4(w0, w1, w2, w3);
    }

    /* Fast tail round when n is a non-zero multiple of 32 */
    if ((n & 31u) == 0u && fast_rounds * 32u < n) {
        const uint32_t cell = c0 + (fast_rounds << 5) + lane;
        const uint32_t cur_s = next_s;
        uint32_t w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u, mx = 0u;
        uint32_t tail_rs = 0u;
        const uint4 *tail_gcls = nullptr;
        uint4 ra, rb;

        if constexpr (!Kind) {
            const uint32_t cls = wnr4_cls(cur_s, anchor, R);
            tail_rs = sm_base_saddr + cls * (37u * 48u);
            tail_gcls = PB5 ? (grows14_m + cls * (37u * 2u)) : nullptr;
            wnr4_lds128_x2_ro(tail_rs, ra, rb);
#pragma unroll 1
            for (uint32_t wq = 0u; wq < 3u; wq++) {
                const uint32_t v = decode_8_smem(tail_rs, mx, ra, rb);
                if (wq == 0u) w0 = v;
                else if (wq == 1u) w1 = v;
                else w2 = v;
            }
        } else {
            const uint32_t cls0 = wnr4_cls(cur_s & 0xffu, anchor, R);
            const uint32_t cls1 = wnr4_cls(cur_s >> 8, anchor, R);
            const uint32_t rs0 = sm_base_saddr + cls0 * (37u * 48u);
            tail_rs = sm_base_saddr + cls1 * (37u * 48u);
            const uint4 *gcls0 = PB5 ? (grows14_m + cls0 * (37u * 2u)) : nullptr;
            tail_gcls = PB5 ? (grows14_m + cls1 * (37u * 2u)) : nullptr;
            wnr4_lds128_x2_ro(rs0, ra, rb);
            w0 = decode_8_smem(rs0, mx, ra, rb);
            wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
            w1  = wnr4_fast_step_smem<SLOT_W>(rs0, x, mx, ra, rb, wr, lmask);
            w1 |= wnr4_fast_step_smem<SLOT_W>(rs0, x, mx, ra, rb, wr, lmask) << 4;
            if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
            w1 |= wnr4_fast_step_smem<SLOT_W>(rs0, x, mx, ra, rb, wr, lmask) << 8;
            w1 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rs0, gcls0, x, mx, ra, rb, wr, lmask) << 12;
            if constexpr (SLOT_W <= 128u) wnr4_ring_ensure<SLOT_W>(wr, SLOT_W * 2u, lane);
            w1 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rs0, gcls0 + ROWS0 * 2u, x, mx, ra, rb, wr, lmask) << 16;
            w1 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rs0, gcls0 + ROWS0 * 4u, x, mx, ra, rb, wr, lmask) << 20;
            if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
            w1 |= wnr4_fast_step_bkt<SLOT_W, PB5>(rs0, gcls0 + ROWS0 * 6u, x, mx, ra, rb, wr, lmask) << 24;
            w1 |= wnr4_fast_step_nv_mid<SLOT_W>(tail_rs, x, ra, rb, wr, lmask) << 28;
            mx = 0u;
            w2 = decode_8_smem(tail_rs, mx, ra, rb);
        }

        /* nibbles 24..27 across all 32 lanes (27 loads bucket 1 for lanes 0..1 and updates mx for MX) */
        wnr4_ring_ensure<SLOT_W>(wr, SLOT_W <= 128u ? SLOT_W * 2u : 256u, lane);
        w3  = wnr4_fast_step_smem<SLOT_W>(tail_rs, x, mx, ra, rb, wr, lmask);
        w3 |= wnr4_fast_step_smem<SLOT_W>(tail_rs, x, mx, ra, rb, wr, lmask) << 4;
        if constexpr (SLOT_W == 64u) wnr4_ring_ensure<SLOT_W>(wr, 128u, lane);
        w3 |= wnr4_fast_step_smem<SLOT_W>(tail_rs, x, mx, ra, rb, wr, lmask) << 8;
        w3 |= wnr4_fast_step_bkt<SLOT_W, PB5>(tail_rs, tail_gcls, x, mx, ra, rb, wr, lmask) << 12;

        /* nibbles 28..31: lanes 0..1 decode; lanes 2..31 carry tail = x - L */
#pragma unroll 1
        for (uint32_t k = 28u; k < 32u; k++) {
            const bool coded = (lane < 2u);
            if (coded) {
                uint32_t y, c, hi, slot, q;
                wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
                x = (hi - c) * q + (slot - c);
                w3 |= y << ((k - 24u) * 4u);
                const uint32_t a = y & 7u;
                mx = a > mx ? a : mx;
                if (k < 31u) {
                    const uint32_t st = 1u + ((mx * (mx + 1u)) >> 1) + a;
                    if constexpr (PB5) wnr4_ldg128_x2_ro(tail_gcls + ((k - 27u) * ROWS0 + st) * 2u, ra, rb);
                    else               wnr4_lds128_x2_ro(tail_rs + st * 48u, ra, rb);
                }
            }
            renorm_scalar(coded && x < WNR4_L);
        }
        if (lane >= 2u) {
            const uint32_t tail = x - WNR4_L;
            if (x < WNR4_L || tail > 0xffffu) err |= WNR4_ERR_STRUCT;
            w3 |= tail << 16;
            if constexpr (!Kind) {
#pragma unroll
                for (uint32_t q = 0u; q < 4u; q++) {
                    const uint32_t a = (tail >> (4u * q)) & 7u;
                    mx = a > mx ? a : mx;
                }
            }
            x = WNR4_L;
        }
        hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w0), w1), w2), w3);
        if ((flag_bits & 2u) && wnr4_has_neg_zero4(w0, w1, w2, w3)) err |= WNR4_ERR_CZ;
        if constexpr (!Kind) {
            if ((flag_bits & 1u) && mx == 5u) {
                if (!cur_s) err |= WNR4_ERR_CEIL;
                else {
                    so[cell] = (unsigned char)(cur_s - 1u);
                    w0 = wnr4_ceil_dbl8(w0); w1 = wnr4_ceil_dbl8(w1);
                    w2 = wnr4_ceil_dbl8(w2); w3 = wnr4_ceil_dbl8(w3);
                }
            }
        }
        wo[cell] = make_uint4(w0, w1, w2, w3);
    } else if (fast_rounds * 32u < n) {
        /* Generic partial-tile fallback (handles any n with n % 32 != 0) */
        uint32_t pw0 = 0u, pw1 = 0u, pw2 = 0u, pw3 = 0u, pcell = 0u;
        bool pending = false;
#pragma unroll 1
        for (uint32_t j = fast_rounds; 32u * j < n; j++) {
            const uint32_t cell = c0 + (j << 5) + lane;
            const bool active = (j << 5) + lane < n;
            const uint32_t s0 = active ? (uint32_t)so[(size_t)cell * SPC] : 0u;
            const uint32_t s1 = (Kind && active) ? (uint32_t)so[(size_t)cell * SPC + 1u] : 0u;
            const bool is_tail = active && wnr4_tail_skip(lane, j, 31u, n);
            const uint32_t kend = is_tail ? 28u : 32u;
            uint32_t w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u, cls = 0u, pm = 8u, mx = 8u;
#pragma unroll 1
            for (uint32_t k = 0u; k < 32u; k++) {
                if (!(k & (B - 1u))) {
                    cls = active ? wnr4_cls((Kind && k >= 16u) ? s1 : s0, anchor, R) : 0u;
                    pm = 8u; mx = 8u;
                }
                const bool coded = active && k < kend;
                if (coded) {
                    const uint32_t p = k & (B - 1u);
                    const uint32_t st = wnr4_state(pm, mx);
                    uint4 ra, rb;
                    if (PB5 && (p + 5u > B)) {
                        const uint32_t bi = p + 4u - B;
                        const uint4 *gp = grows14_m + ((bi * ROWS0 + cls * 37u + st) * 2u);
                        ra = __ldg(gp); rb = __ldg(gp + 1u);
                    } else {
                        const uint32_t ri = (cls * 37u + st) * 3u;
                        ra = sm[ri]; rb = sm[ri + 1u];
                    }
                    uint32_t y, c, hi, slot, q;
                    wnr4_decode_ychi_top(ra, rb, x, WNR4_TOTAL, y, c, hi, slot, q);
                    x = (hi - c) * q + (slot - c);
                    if (k < 8u)       w0 |= y << (k * 4u);
                    else if (k < 16u) w1 |= y << ((k - 8u) * 4u);
                    else if (k < 24u) w2 |= y << ((k - 16u) * 4u);
                    else              w3 |= y << ((k - 24u) * 4u);
                    pm = y & 7u; mx = wnr4_mx(mx, pm);
                }
                renorm_scalar(coded && x < WNR4_L);
            }
            if (is_tail) {
                pending = true; pw0 = w0; pw1 = w1; pw2 = w2; pw3 = w3; pcell = cell;
            } else if (active) {
                hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w0), w1), w2), w3);
                if ((flag_bits & 2u) && wnr4_has_neg_zero4(w0, w1, w2, w3)) err |= WNR4_ERR_CZ;
                if constexpr (!Kind) {
                    if ((flag_bits & 1u) && mx == 5u) {
                        if (!s0) err |= WNR4_ERR_CEIL;
                        else {
                            so[cell] = (unsigned char)(s0 - 1u);
                            w0 = wnr4_ceil_dbl8(w0); w1 = wnr4_ceil_dbl8(w1);
                            w2 = wnr4_ceil_dbl8(w2); w3 = wnr4_ceil_dbl8(w3);
                        }
                    }
                }
                wo[cell] = make_uint4(w0, w1, w2, w3);
            }
        }
        if (pending) {
            const uint32_t tail = x - WNR4_L;
            if (x < WNR4_L || tail > 0xffffu) err |= WNR4_ERR_STRUCT;
            pw3 |= tail << 16;
            x = WNR4_L;
            hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, pw0), pw1), pw2), pw3);
            if ((flag_bits & 2u) && wnr4_has_neg_zero4(pw0, pw1, pw2, pw3)) err |= WNR4_ERR_CZ;
            if constexpr (!Kind) {
                if (flag_bits & 1u) {
                    uint32_t pmx = 0u;
#pragma unroll
                    for (uint32_t q = 0u; q < 8u; q++) {
                        uint32_t a0 = (pw0 >> (4u * q)) & 7u, a1 = (pw1 >> (4u * q)) & 7u;
                        uint32_t a2 = (pw2 >> (4u * q)) & 7u, a3 = (pw3 >> (4u * q)) & 7u;
                        uint32_t m01 = a0 > a1 ? a0 : a1, m23 = a2 > a3 ? a2 : a3;
                        uint32_t m4 = m01 > m23 ? m01 : m23;
                        if (m4 > pmx) pmx = m4;
                    }
                    if (pmx == 5u) {
                        const uint32_t ps = (uint32_t)so[pcell];
                        if (!ps) err |= WNR4_ERR_CEIL;
                        else {
                            so[pcell] = (unsigned char)(ps - 1u);
                            pw0 = wnr4_ceil_dbl8(pw0); pw1 = wnr4_ceil_dbl8(pw1);
                            pw2 = wnr4_ceil_dbl8(pw2); pw3 = wnr4_ceil_dbl8(pw3);
                        }
                    }
                }
            }
            wo[pcell] = make_uint4(pw0, pw1, pw2, pw3);
        }
    }

    wnr4_cp_wait0();
    uint32_t H = wnr4_h32_final(hash, wnr4_lane_hlen(wnr4_lane_scales(n * SPC, lane), wnr4_lane_cells(n, lane)));
#pragma unroll
    for (uint32_t d = 16u; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);
    if (lane == 0u && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1u && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (lane == 0u && wr.ptr_b != wr.nbytes_adj) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(err_ptr, err);
}

/* ---- CUDA kernel entry points ------------------------------------------------ */
/* Static SMEM per block (all <= 48 KiB, zero cudaFuncSetAttribute calls):
 *   MX: 12,432 B wrows0 + 800 B scales + ring (2/4/8/8/16 KB for W=1/2/4/8/16) -> max 29.6 KB
 *   NV: 23,088 B wrows0 +  32 B dummy  + ring (2/4/8/8/16 KB for W=1/2/4/8/16) -> max 39.5 KB
 * Launch bounds: MX W=16 targets 2 blocks/SM (32 warps/SM, <= 64 regs, 0 spills); NV W=16 targets
 * 1 block/SM so 88 KB of L1 stays available for NV's 62 KB wrows14 + srows without L1 thrashing. */
template<uint32_t Kind, bool PB5, uint32_t W>
static __global__ void __launch_bounds__(W * 32u, (W == 8u || (!Kind && W == 16u)) ? 2u : 1u)
wnr4_cuda_kernel(const Wnr4CudaJob *jobs, uint32_t njobs, const Wnr4CudaRec *recs, Wnr4CudaTables tab, uint32_t *errors) {
    constexpr uint32_t ROWS0 = WNR4_NCLS(Kind) * 37u;
    constexpr uint32_t SLOT_W = wnr4_slot_w<Kind, W>();

    __shared__ uint4 sm[ROWS0 * 3u];
    __shared__ uint4 sm_ssum[Kind ? 1u : 2u];
    __shared__ uint4 sm_sblk[Kind ? 1u : 16u * 3u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0);
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t m = __shfl_sync(0xffffffffu, jobs[first].m, 0);   /* the host groups jobs so a block has one matrix */
    wnr4_load_mat_smem<Kind, PB5, W>(sm, sm_ssum, sm_sblk, tab, m);

    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4CudaJob &job = jobs[jid];
    const uint32_t job_n = __shfl_sync(0xffffffffu, job.n, 0);
    if (!job_n) return;   /* padding job */
    const uint32_t job_rec = __shfl_sync(0xffffffffu, job.rec, 0);
    if (__shfl_sync(0xffffffffu, job.m, 0) != m) { if (!lane) atomicOr(errors + job_rec, WNR4_ERR_STRUCT); return; }

    const Wnr4CudaRec &rec = recs[job_rec];
    const uint32_t scols_m = Kind ? wnr4_sel3(m, rec.scols[0], rec.scols[1], rec.scols[2]) : 0u;
    unsigned char *so = (m == 0u) ? rec.dst_s[0] : (m == 1u ? rec.dst_s[1] : rec.dst_s[2]);
    uint4 *wo = (uint4 *)((m == 0u) ? rec.dst_w[0] : (m == 1u ? rec.dst_w[1] : rec.dst_w[2]));
    wnr4_decode_tile_dev<Kind, PB5, W>(sm, sm_ssum, sm_sblk, sm_ring + warp * (SLOT_W * 4u),
                                       tab, errors + job_rec, job.tile, so, wo, job.nwords, job.t, m,
                                       job.c0, job_n, rec.owner, rec.flags, rec.G, rec.anchors, scols_m);
}

/* Record-level direct launch (<= 32 records inline in constant-memory args): zero per-tile job structs or job H2D DMA. */
template<uint32_t Kind, bool PB5, uint32_t W>
static __global__ void __launch_bounds__(W * 32u, (W == 8u || (!Kind && W == 16u)) ? 2u : 1u)
wnr4_cuda_recs_kernel(Wnr4CudaBatchGeo geo, Wnr4InlineRecs irecs, Wnr4CudaTables tab, uint32_t *errors) {
    constexpr uint32_t ROWS0 = WNR4_NCLS(Kind) * 37u;
    constexpr uint32_t SLOT_W = wnr4_slot_w<Kind, W>();

    __shared__ uint4 sm[ROWS0 * 3u];
    __shared__ uint4 sm_ssum[Kind ? 1u : 2u];
    __shared__ uint4 sm_sblk[Kind ? 1u : 16u * 3u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];

    if (blockIdx.x >= geo.mat_blocks[3]) return;
    const uint32_t m = blockIdx.x < geo.mat_blocks[1] ? 0u : (blockIdx.x < geo.mat_blocks[2] ? 1u : 2u);
    const uint32_t blk_in_m = blockIdx.x - wnr4_sel3(m, geo.mat_blocks[0], geo.mat_blocks[1], geo.mat_blocks[2]);
    wnr4_load_mat_smem<Kind, PB5, W>(sm, sm_ssum, sm_sblk, tab, m);

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0);
    const uint32_t tb0 = wnr4_sel3(m, geo.tile_base[0], geo.tile_base[1], geo.tile_base[2]);
    const uint32_t tb1 = wnr4_sel3(m, geo.tile_base[1], geo.tile_base[2], geo.tile_base[3]);
    const uint32_t mt = tb1 - tb0;
    if (!mt) return;
    const uint32_t local_tile = blk_in_m * W + warp;
    const uint32_t ridx = (geo.nrecs == 1u) ? (local_tile >= mt ? 1u : 0u) : (local_tile / mt);
    if (ridx >= geo.nrecs) return;
    const uint32_t lt = (geo.nrecs == 1u) ? local_tile : (local_tile - ridx * mt);
    const Wnr4CudaDirectRec &rec = irecs.r[ridx];
    const uint32_t t = tb0 + lt;
    const uint32_t *offs = (const uint32_t *)(rec.rec + WNR4_HEADER_BYTES);
    const uint32_t o0 = __ldg(offs + t), o1 = __ldg(offs + t + 1u);
    const uint32_t cells_m = wnr4_sel3(m, geo.cells[0], geo.cells[1], geo.cells[2]);
    const uint32_t c0 = lt * (32u * geo.G);
    const uint32_t rem = cells_m - c0;
    const uint32_t n = rem < (32u * geo.G) ? rem : (32u * geo.G);
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 1u) || o1 > rec.payload || !n) {
        if (!lane) atomicOr(errors + rec.rec_idx, WNR4_ERR_STRUCT);
        return;
    }
    const uint32_t scols_m = wnr4_sel3(m, geo.scols[0], geo.scols[1], geo.scols[2]);
    unsigned char *so = (m == 0u) ? rec.dst_s[0] : (m == 1u ? rec.dst_s[1] : rec.dst_s[2]);
    uint4 *wo = (uint4 *)((m == 0u) ? rec.dst_w[0] : (m == 1u ? rec.dst_w[1] : rec.dst_w[2]));
    wnr4_decode_tile_dev<Kind, PB5, W>(sm, sm_ssum, sm_sblk, sm_ring + warp * (SLOT_W * 4u),
                                       tab, errors + rec.rec_idx, rec.rec + rec.data_at + o0, so, wo,
                                       (o1 - o0 - WNR4_TILE_STATE_BYTES) >> 1, t, m, c0, n,
                                       rec.owner, rec.flags, geo.G, rec.anchors, scols_m);
}

/* Record-level direct launch for large batches (> 32 records): reads Wnr4CudaDirectRec array from device memory. */
template<uint32_t Kind, bool PB5, uint32_t W>
static __global__ void __launch_bounds__(W * 32u, (W == 8u || (!Kind && W == 16u)) ? 2u : 1u)
wnr4_cuda_recs_ptr_kernel(Wnr4CudaBatchGeo geo, const Wnr4CudaDirectRec *recs, Wnr4CudaTables tab, uint32_t *errors) {
    constexpr uint32_t ROWS0 = WNR4_NCLS(Kind) * 37u;
    constexpr uint32_t SLOT_W = wnr4_slot_w<Kind, W>();

    __shared__ uint4 sm[ROWS0 * 3u];
    __shared__ uint4 sm_ssum[Kind ? 1u : 2u];
    __shared__ uint4 sm_sblk[Kind ? 1u : 16u * 3u];
    __shared__ alignas(16) uint16_t sm_ring[W * SLOT_W * 4u];

    if (blockIdx.x >= geo.mat_blocks[3]) return;
    const uint32_t m = blockIdx.x < geo.mat_blocks[1] ? 0u : (blockIdx.x < geo.mat_blocks[2] ? 1u : 2u);
    const uint32_t blk_in_m = blockIdx.x - wnr4_sel3(m, geo.mat_blocks[0], geo.mat_blocks[1], geo.mat_blocks[2]);
    wnr4_load_mat_smem<Kind, PB5, W>(sm, sm_ssum, sm_sblk, tab, m);

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0);
    const uint32_t tb0 = wnr4_sel3(m, geo.tile_base[0], geo.tile_base[1], geo.tile_base[2]);
    const uint32_t tb1 = wnr4_sel3(m, geo.tile_base[1], geo.tile_base[2], geo.tile_base[3]);
    const uint32_t mt = tb1 - tb0;
    if (!mt) return;
    const uint32_t local_tile = blk_in_m * W + warp;
    const uint32_t ridx = local_tile / mt;
    if (ridx >= geo.nrecs) return;
    const uint32_t lt = local_tile - ridx * mt;
    const Wnr4CudaDirectRec rec = recs[ridx];
    const uint32_t t = tb0 + lt;
    const uint32_t *offs = (const uint32_t *)(rec.rec + WNR4_HEADER_BYTES);
    const uint32_t o0 = __ldg(offs + t), o1 = __ldg(offs + t + 1u);
    const uint32_t cells_m = wnr4_sel3(m, geo.cells[0], geo.cells[1], geo.cells[2]);
    const uint32_t c0 = lt * (32u * geo.G);
    const uint32_t rem = cells_m - c0;
    const uint32_t n = rem < (32u * geo.G) ? rem : (32u * geo.G);
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 1u) || o1 > rec.payload || !n) {
        if (!lane) atomicOr(errors + rec.rec_idx, WNR4_ERR_STRUCT);
        return;
    }
    const uint32_t scols_m = wnr4_sel3(m, geo.scols[0], geo.scols[1], geo.scols[2]);
    unsigned char *so = (m == 0u) ? rec.dst_s[0] : (m == 1u ? rec.dst_s[1] : rec.dst_s[2]);
    uint4 *wo = (uint4 *)((m == 0u) ? rec.dst_w[0] : (m == 1u ? rec.dst_w[1] : rec.dst_w[2]));
    wnr4_decode_tile_dev<Kind, PB5, W>(sm, sm_ssum, sm_sblk, sm_ring + warp * (SLOT_W * 4u),
                                       tab, errors + rec.rec_idx, rec.rec + rec.data_at + o0, so, wo,
                                       (o1 - o0 - WNR4_TILE_STATE_BYTES) >> 1, t, m, c0, n,
                                       rec.owner, rec.flags, geo.G, rec.anchors, scols_m);
}

/* ---- host side --------------------------------------------------------------- */
static inline void wnr4_swap_pairs_u16(uint16_t *dst, const uint16_t *src, size_t n16) {
    for (size_t i = 0; i < n16; i += 2u) { dst[i] = src[i + 1u]; dst[i + 1u] = src[i]; }
}
static inline uint32_t wnr4_cuda_rows0(uint32_t kind) { return WNR4_NCLS(kind) * 37u; }
static inline uint32_t wnr4_cuda_rows(const Wnr4Model *m) { return wnr4_cuda_rows0(m->kind) * m->pb; }
static inline size_t wnr4_cuda_table_bytes(const Wnr4Model *m) {
    const size_t r0 = (size_t)wnr4_cuda_rows0(m->kind);
    const size_t w0 = 3u * r0 * 32u;
    const size_t w14 = (m->pb == 5u) ? (3u * 4u * r0 * 32u) : 0u;
    const size_t sr = 3u * (size_t)m->sctx * 544u;
    return w0 + w14 + sr;
}
/* device image: bucket-0 pre-swapped weight rows, optional bucket-major buckets 1..4 (pb == 5),
 * then pre-swapped two-level 256-symbol scale rows. */
static inline void wnr4_cuda_table_image(const Wnr4Model *m, unsigned char *img) {
    const uint32_t r0 = wnr4_cuda_rows0(m->kind), ncls = WNR4_NCLS(m->kind);
    const size_t w0_bytes = 3u * (size_t)r0 * 32u;
    const size_t w14_bytes = (m->pb == 5u) ? (3u * 4u * (size_t)r0 * 32u) : 0u;
    uint16_t *w0_u16 = (uint16_t *)img;
    uint16_t *w14_u16 = (uint16_t *)(img + w0_bytes);
    for (uint32_t mm = 0u; mm < 3u; mm++)
        for (uint32_t cls = 0u; cls < ncls; cls++)
            for (uint32_t mx = 0u; mx <= 8u; mx++)
                for (uint32_t pm = 0u; pm <= 8u; pm++) {
                    if (!((pm == 8u && mx == 8u) || (pm <= mx && mx < 8u))) continue;
                    const uint32_t row = cls * 37u + wnr4_state(pm, mx);
                    const uint32_t src_base = mm * m->wctx + wnr4_wctx(cls, pm, mx) * m->pb;
                    wnr4_swap_pairs_u16(w0_u16 + ((size_t)mm * r0 + row) * 16u,
                                        m->wcdf + (size_t)src_base * 17u, 16u);
                    if (m->pb == 5u) {
                        for (uint32_t b = 1u; b < 5u; b++)
                            wnr4_swap_pairs_u16(w14_u16 + (((size_t)mm * 4u + (b - 1u)) * r0 + row) * 16u,
                                                m->wcdf + (size_t)(src_base + b) * 17u, 16u);
                    }
                }
    unsigned char *sr = img + w0_bytes + w14_bytes;
    for (uint32_t t = 0u; t < 3u * m->sctx; t++) {
        const uint16_t *cdf = m->scdf + (size_t)t * 257u;
        uint16_t *o = (uint16_t *)(sr + (size_t)t * 544u);
        uint16_t sum[16];
        for (uint32_t q = 0u; q < 16u; q++) sum[q] = cdf[16u * q];
        wnr4_swap_pairs_u16(o, sum, 16u);
        wnr4_swap_pairs_u16(o + 16u, cdf, 256u);
    }
}
static inline void wnr4_cuda_tables_view(const Wnr4Model *m, const unsigned char *dev, Wnr4CudaTables *t) {
    const uint32_t r0 = wnr4_cuda_rows0(m->kind);
    const size_t w0_bytes = 3u * (size_t)r0 * 32u;
    const size_t w14_bytes = (m->pb == 5u) ? (3u * 4u * (size_t)r0 * 32u) : 0u;
    t->kind = m->kind; t->id = m->id; t->rows = r0 * m->pb; t->rows0 = r0; t->sctx = m->sctx; t->pb = m->pb;
    t->wrows = (const uint4 *)dev;
    t->wrows0 = (const uint4 *)dev;
    t->wrows14 = (m->pb == 5u) ? (const uint4 *)(dev + w0_bytes) : nullptr;
    t->scdf = nullptr;
    t->srows = (const uint4 *)(dev + w0_bytes + w14_bytes);
}

/* Rollback-safe job planner (zero temporary allocations; nothing appended on failure). */
static inline int wnr4_cuda_plan(const Wnr4Geo *g, const Wnr4Model *model, const unsigned char *rec_host, size_t total,
                                 uint32_t owner, const unsigned char *rec_dev, uint32_t index,
                                 unsigned char *const dst_s[3], unsigned char *const dst_w[3],
                                 Wnr4CudaRec *info, std::vector<Wnr4CudaJob> by_m[3]) {
    Wnr4Record r;
    uint32_t t, m;
    if (model->kind != g->kind || wnr4_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner)
        return -1;
    info->model_id = r.model_id; info->owner = r.owner; info->flags = r.flags; info->G = r.G; info->kind = r.kind;
    info->anchors = (uint32_t)r.anchor[0] | (uint32_t)r.anchor[1] << 8 | (uint32_t)r.anchor[2] << 16;
    for (m = 0; m < 3u; m++) { info->scols[m] = g->scols[m]; info->dst_s[m] = dst_s[m]; info->dst_w[m] = dst_w[m]; }
    const size_t saved[3] = {by_m[0].size(), by_m[1].size(), by_m[2].size()};
    const size_t data_at = (size_t)(r.data - rec_host);
    for (t = 0; t < r.ntiles; t++) {   /* re-read offsets: check again before keeping any job */
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 1u) || o1 > r.payload) {
            for (m = 0; m < 3u; m++) by_m[m].resize(saved[m]);
            return -1;
        }
        Wnr4CudaJob j;
        wnr4_tile_cells(r.tile_base, g->cells, r.G, t, &j.m, &j.c0, &j.n);
        j.tile = rec_dev + data_at + o0;
        j.nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u;
        j.rec = index; j.t = t;
        by_m[j.m].push_back(j);
    }
    return 0;
}
static int wnr4_cuda_window = 1;           /* compatibility knob (v2 always uses the 4-slot async SMEM ring) */
static uint32_t wnr4_cuda_warps = 0u;      /* warps per block: 1, 2, 4, 8 or 16; 0 = SM-aware auto */
static int wnr4_cuda_rows_global = -1;     /* compatibility knob (v2 keeps bucket 0 in SMEM, buckets 1..4 in L1/L2) */
static int wnr4_cuda_lds = 1;              /* compatibility knob (v2 always uses pre-swapped 4-way search) */
static thread_local uint32_t wnr4_cuda_planned_warps = 16u;

static inline uint32_t wnr4_cuda_sms(void) {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void)cudaGetLastError(); return 128u; }
    int sms = cached[dev].load(std::memory_order_relaxed);
    if (sms > 0) return (uint32_t)sms;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) {
        (void)cudaGetLastError();
        return 128u;
    }
    cached[dev].store(sms, std::memory_order_relaxed);
    return (uint32_t)sms;
}

static inline uint32_t wnr4_cuda_choose_warps(size_t tiles) {
    uint32_t W = wnr4_cuda_warps;
    if (!W) {
        const size_t sms = (size_t)wnr4_cuda_sms();
        W = tiles <= sms ? 1u : tiles <= 2u * sms ? 2u : tiles <= 4u * sms ? 4u : tiles <= 8u * sms ? 8u : 16u;
    }
    return W;
}

/* Concatenates the per-matrix lists, padding each to a multiple of W (padding jobs have n = 0). */
static inline size_t wnr4_cuda_jobs_cap(const std::vector<Wnr4CudaJob> by_m[3], uint32_t *w_out) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    const uint32_t W = wnr4_cuda_choose_warps(tiles);
    if (w_out) *w_out = W;
    size_t n = 0;
    for (uint32_t m = 0; m < 3u; m++) n += ((by_m[m].size() + W - 1u) / W) * W;
    return n;
}
static inline uint32_t wnr4_cuda_jobs_into(const std::vector<Wnr4CudaJob> by_m[3], Wnr4CudaJob *out, size_t *njobs_out) {
    uint32_t W = 16u;
    (void)wnr4_cuda_jobs_cap(by_m, &W);
    size_t pos = 0;
    for (uint32_t m = 0; m < 3u; m++) {
        const size_t sz = by_m[m].size();
        if (sz) { std::memcpy(out + pos, by_m[m].data(), sz * sizeof(Wnr4CudaJob)); pos += sz; }
        Wnr4CudaJob pad = {};
        pad.m = m;
        while (pos % W) out[pos++] = pad;
    }
    if (njobs_out) *njobs_out = pos;
    wnr4_cuda_planned_warps = W;
    return W;
}
static inline uint32_t wnr4_cuda_jobs(std::vector<Wnr4CudaJob> by_m[3], std::vector<Wnr4CudaJob> &jobs) {
    uint32_t W = 16u;
    const size_t cap = wnr4_cuda_jobs_cap(by_m, &W);
    jobs.resize(cap);
    size_t nj = 0;
    if (cap) wnr4_cuda_jobs_into(by_m, jobs.data(), &nj);
    else wnr4_cuda_planned_warps = W;
    jobs.resize(nj);
    return W;
}

template<uint32_t Kind, bool PB5>
static inline cudaError_t wnr4_cuda_launch_typed(const Wnr4CudaTables &tab, const Wnr4CudaJob *jobs_dev, uint32_t njobs,
                                                 const Wnr4CudaRec *recs_dev, uint32_t W, uint32_t blocks,
                                                 uint32_t *errors_dev, cudaStream_t stream) {
    if (W == 1u)       wnr4_cuda_kernel<Kind, PB5,  1u><<<blocks,  32u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev);
    else if (W == 2u)  wnr4_cuda_kernel<Kind, PB5,  2u><<<blocks,  64u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev);
    else if (W == 4u)  wnr4_cuda_kernel<Kind, PB5,  4u><<<blocks, 128u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev);
    else if (W == 8u)  wnr4_cuda_kernel<Kind, PB5,  8u><<<blocks, 256u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev);
    else if (W == 16u) wnr4_cuda_kernel<Kind, PB5, 16u><<<blocks, 512u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_cuda_launch_w(const Wnr4CudaTables &tab, const Wnr4CudaJob *jobs_dev, uint32_t njobs,
                                             const Wnr4CudaRec *recs_dev, uint32_t *errors_dev, uint32_t W,
                                             cudaStream_t stream) {
    if (W != 1u && W != 2u && W != 4u && W != 8u && W != 16u) return cudaErrorInvalidValue;
    if (tab.kind > 1u || (tab.pb != 1u && tab.pb != 5u)) return cudaErrorInvalidValue;
    const uint32_t blocks = (njobs + W - 1u) / W;
    if (!blocks) return cudaSuccess;
    if (tab.kind) {
        if (tab.pb == 5u) return wnr4_cuda_launch_typed<1u, true>(tab, jobs_dev, njobs, recs_dev, W, blocks, errors_dev, stream);
        return wnr4_cuda_launch_typed<1u, false>(tab, jobs_dev, njobs, recs_dev, W, blocks, errors_dev, stream);
    }
    if (tab.pb == 5u) return wnr4_cuda_launch_typed<0u, true>(tab, jobs_dev, njobs, recs_dev, W, blocks, errors_dev, stream);
    return wnr4_cuda_launch_typed<0u, false>(tab, jobs_dev, njobs, recs_dev, W, blocks, errors_dev, stream);
}

static inline cudaError_t wnr4_cuda_launch(const Wnr4CudaTables &tab, const Wnr4CudaJob *jobs_dev, uint32_t njobs,
                                           const Wnr4CudaRec *recs_dev, uint32_t *errors_dev, cudaStream_t stream) {
    return wnr4_cuda_launch_w(tab, jobs_dev, njobs, recs_dev, errors_dev, wnr4_cuda_planned_warps, stream);
}

template<uint32_t Kind, bool PB5>
static inline cudaError_t wnr4_cuda_launch_recs_typed(const Wnr4CudaTables &tab, const Wnr4CudaBatchGeo &geo,
                                                      const Wnr4InlineRecs &irecs, uint32_t W,
                                                      uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t blocks = geo.mat_blocks[3];
    if (!blocks) return cudaSuccess;
    if (W == 1u)       wnr4_cuda_recs_kernel<Kind, PB5,  1u><<<blocks,  32u, 0, stream>>>(geo, irecs, tab, errors_dev);
    else if (W == 2u)  wnr4_cuda_recs_kernel<Kind, PB5,  2u><<<blocks,  64u, 0, stream>>>(geo, irecs, tab, errors_dev);
    else if (W == 4u)  wnr4_cuda_recs_kernel<Kind, PB5,  4u><<<blocks, 128u, 0, stream>>>(geo, irecs, tab, errors_dev);
    else if (W == 8u)  wnr4_cuda_recs_kernel<Kind, PB5,  8u><<<blocks, 256u, 0, stream>>>(geo, irecs, tab, errors_dev);
    else if (W == 16u) wnr4_cuda_recs_kernel<Kind, PB5, 16u><<<blocks, 512u, 0, stream>>>(geo, irecs, tab, errors_dev);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_cuda_launch_recs(const Wnr4CudaTables &tab, const Wnr4CudaBatchGeo &geo,
                                                const Wnr4InlineRecs &irecs, uint32_t W,
                                                uint32_t *errors_dev, cudaStream_t stream) {
    if (tab.kind > 1u || (tab.pb != 1u && tab.pb != 5u)) return cudaErrorInvalidValue;
    if (tab.kind) {
        if (tab.pb == 5u) return wnr4_cuda_launch_recs_typed<1u, true>(tab, geo, irecs, W, errors_dev, stream);
        return wnr4_cuda_launch_recs_typed<1u, false>(tab, geo, irecs, W, errors_dev, stream);
    }
    if (tab.pb == 5u) return wnr4_cuda_launch_recs_typed<0u, true>(tab, geo, irecs, W, errors_dev, stream);
    return wnr4_cuda_launch_recs_typed<0u, false>(tab, geo, irecs, W, errors_dev, stream);
}

template<uint32_t Kind, bool PB5>
static inline cudaError_t wnr4_cuda_launch_recs_ptr_typed(const Wnr4CudaTables &tab, const Wnr4CudaBatchGeo &geo,
                                                          const Wnr4CudaDirectRec *recs_dev, uint32_t W,
                                                          uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t blocks = geo.mat_blocks[3];
    if (!blocks) return cudaSuccess;
    if (W == 1u)       wnr4_cuda_recs_ptr_kernel<Kind, PB5,  1u><<<blocks,  32u, 0, stream>>>(geo, recs_dev, tab, errors_dev);
    else if (W == 2u)  wnr4_cuda_recs_ptr_kernel<Kind, PB5,  2u><<<blocks,  64u, 0, stream>>>(geo, recs_dev, tab, errors_dev);
    else if (W == 4u)  wnr4_cuda_recs_ptr_kernel<Kind, PB5,  4u><<<blocks, 128u, 0, stream>>>(geo, recs_dev, tab, errors_dev);
    else if (W == 8u)  wnr4_cuda_recs_ptr_kernel<Kind, PB5,  8u><<<blocks, 256u, 0, stream>>>(geo, recs_dev, tab, errors_dev);
    else if (W == 16u) wnr4_cuda_recs_ptr_kernel<Kind, PB5, 16u><<<blocks, 512u, 0, stream>>>(geo, recs_dev, tab, errors_dev);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

static inline cudaError_t wnr4_cuda_launch_recs_ptr(const Wnr4CudaTables &tab, const Wnr4CudaBatchGeo &geo,
                                                    const Wnr4CudaDirectRec *recs_dev, uint32_t W,
                                                    uint32_t *errors_dev, cudaStream_t stream) {
    if (tab.kind > 1u || (tab.pb != 1u && tab.pb != 5u)) return cudaErrorInvalidValue;
    if (tab.kind) {
        if (tab.pb == 5u) return wnr4_cuda_launch_recs_ptr_typed<1u, true>(tab, geo, recs_dev, W, errors_dev, stream);
        return wnr4_cuda_launch_recs_ptr_typed<1u, false>(tab, geo, recs_dev, W, errors_dev, stream);
    }
    if (tab.pb == 5u) return wnr4_cuda_launch_recs_ptr_typed<0u, true>(tab, geo, recs_dev, W, errors_dev, stream);
    return wnr4_cuda_launch_recs_ptr_typed<0u, false>(tab, geo, recs_dev, W, errors_dev, stream);
}

#endif
