/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-SWAR-8MAG CUDA decoder (sm_80+ 4-slot cp.async ring buffer, scalar
 * fallback < sm_80; W in {1, 2, 4, 8, 16}). Bit-exact with wnr4_swar_decode_tile.
 *
 * Key architecture:
 *  - Weight tables: 63 contexts x 16 B = 1,008 B SMEM per matrix (1 x uint4/ctx).
 *  - Carry-free 64-bit SWAR 8-symbol CDF search + zero-SMEM register (c_s, f_s)
 *    extraction directly from the single uint4 row, with next-context read-only
 *    ld.shared.v4.u32 hoisted ahead of renorm & sign extraction.
 *  - Phase 1 scales: per-record s_rec16 (2 x uint4 = 32 B/matrix held in
 *    registers, zero SMEM!) with warp-uniform fast-skip escape for raw 8-bit
 *    out-of-window scales.
 *  - Shared 16-bit forward rANS word stream fed via a 4-slot cp.async SMEM ring
 *    buffer (128 u16 words = 256 B/slot, 1,024 B/warp; refilled at most once
 *    every 4 nibbles).
 *  - Warp-coalesced 4-byte-aligned backward 32-bit raw sign stream via warp-
 *    uniform 64-bit bit-buffer (sbuf, sbits) + __ldg(gmem32 + --sw_pos) +
 *    ballot/popc.
 *  - MAGT 480-bit tail (up to 16 non-zero signs of lane 2..31's last cell in
 *    x0[2..31] = L + bits), ceil-form inverse, and full 32-bit MurmurHash3
 *    covering scales, magnitudes, and raw signs.
 * ------------------------------------------------------------------------- */
#ifndef WNR4_SWAR_CUDA_CUH
#define WNR4_SWAR_CUDA_CUH

#include <cuda_runtime.h>
#include <vector>
#include "../../wnr4/wnr4_cuda.cuh"
#include "wnr4_swar.h"

struct Wnr4SwarCudaTables {
    uint32_t id;
    const uint4 *wrows; /* [3][63]: [C1,C2], [C3,C4], [C5,C6], [C7,16384] */
};

struct Wnr4SwarCudaRec {
    uint32_t model_id, owner, flags, G, anchors, srec_hash;
    uint4 scdf[3][2]; /* per-record s_rec16 CDF[0..15] packed as 2 x uint4 per matrix */
    unsigned char *dst_s[3], *dst_w[3];
};

struct Wnr4SwarCudaJob {
    const unsigned char *tile; /* 4-byte aligned device tile: 128 B states + words + pad + backward signs */
    uint32_t tile_bytes, rec, t, m, c0, n;
};

/* Read-only 128-bit SMEM load (no memory clobber so ptxas hoists next-nibble CDF ahead of renorm/signs). */
static __device__ __forceinline__ uint4 wnr4_swar_lds128_ro(uint32_t saddr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(saddr));
    return v;
}

/* Predicated SMEM uint16 renorm: if xn < 65536, returns (xn << 16) | *(u16*)saddr, else xn. */
static __device__ __forceinline__ uint32_t wnr4_swar_lds_renorm(uint32_t xn, uint32_t saddr) {
    uint32_t out = xn;
    asm volatile(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        ".reg .u32 %%w;\n\t"
        "setp.lt.u32 %%p, %0, 65536;\n\t"
        "@%%p ld.shared.u16 %%w, [%1];\n\t"
        "@%%p shl.b32 %0, %0, 16;\n\t"
        "@%%p or.b32 %0, %0, %%w;\n\t"
        "}"
        : "+r"(out)
        : "r"(saddr));
    return out;
}

/* Carry-free 64-bit SWAR 8-symbol CDF search + zero-SMEM register (c_s, f_s) extraction.
 * vec packs [C1, C2] in x, [C3, C4] in y, [C5, C6] in z, [C7, 16384] in w. */
static __device__ __forceinline__ void wnr4_swar_find8_dev(uint4 vec, uint32_t slot, uint32_t &mag, uint32_t &c_s,
                                                           uint32_t &f_s) {
    const uint64_t cdf_lo64 = (uint64_t)vec.x | ((uint64_t)vec.y << 32);
    const uint64_t cdf_hi64 = (uint64_t)vec.z | ((uint64_t)vec.w << 32);
    const uint32_t s2 = (slot | (slot << 16)) | 0x80008000u;
    const uint64_t s4 = (uint64_t)s2 | ((uint64_t)s2 << 32);
    const uint64_t d_lo = (s4 - cdf_lo64) & 0x8000800080008000ULL;
    const uint64_t d_hi = (s4 - cdf_hi64) & 0x8000800080008000ULL;
    /* Pack the 8 disjoint sign bits {15,31,47,63} of d_lo and d_hi into {12..15,28..31} -> 1 POPC. */
    const uint32_t packed = (uint32_t)d_lo | ((uint32_t)(d_lo >> 32) >> 1) |
                            ((uint32_t)d_hi >> 2) | ((uint32_t)(d_hi >> 32) >> 3);
    mag = (uint32_t)__popc(packed);
    /* Extract (c_s, c_end) packed in 32 bits purely in registers (0 extra SMEM loads):
     * odd mag  (1,3,5,7) -> vec.{x,y,z,w}
     * even mag (0,2,4,6) -> funnelshift_r(prev, curr, 16) */
    const uint32_t curr = (mag & 4u) ? ((mag & 2u) ? vec.w : vec.z) : ((mag & 2u) ? vec.y : vec.x);
    const uint32_t prev = (mag & 4u) ? ((mag & 2u) ? vec.z : vec.y) : ((mag & 2u) ? vec.x : 0u);
    const uint32_t pair = (mag & 1u) ? curr : __funnelshift_r(prev, curr, 16u);
    c_s = pair & 0xffffu;
    f_s = (pair >> 16) - c_s;
}

/* ---- 4-slot cp.async SMEM ring buffer (128 u16 words = 256 B/slot) -------- */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void wnr4_swar_cp_slot(uint32_t smem_slot_addr, const uint32_t *gmem32,
                                                         uint32_t slot_idx, uint32_t max_u32, uint32_t lane) {
    const uint32_t idx0 = slot_idx * 64u + lane;
    if (idx0 < max_u32) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem_slot_addr + 4u * lane), "l"(gmem32 + idx0) : "memory");
    }
    const uint32_t idx1 = idx0 + 32u;
    if (idx1 < max_u32) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem_slot_addr + 128u + 4u * lane), "l"(gmem32 + idx1) : "memory");
    }
}
static __device__ __forceinline__ void wnr4_swar_cp_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int N>
static __device__ __forceinline__ void wnr4_swar_cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
static __device__ __forceinline__ void wnr4_swar_cp_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}
#endif

static __device__ __forceinline__ void wnr4_swar_ring_init(uint32_t smem_base, const uint32_t *gmem32,
                                                           uint32_t max_u32, uint32_t lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    wnr4_swar_cp_slot(smem_base + 0u * 256u, gmem32, 0u, max_u32, lane);
    wnr4_swar_cp_commit();
    wnr4_swar_cp_slot(smem_base + 1u * 256u, gmem32, 1u, max_u32, lane);
    wnr4_swar_cp_commit();
    wnr4_swar_cp_slot(smem_base + 2u * 256u, gmem32, 2u, max_u32, lane);
    wnr4_swar_cp_commit();
    wnr4_swar_cp_slot(smem_base + 3u * 256u, gmem32, 3u, max_u32, lane);
    wnr4_swar_cp_commit();
    wnr4_swar_cp_wait<2>();
    __syncwarp();
#else
    (void)smem_base; (void)gmem32; (void)max_u32; (void)lane;
#endif
}

static __device__ __forceinline__ void wnr4_swar_ring_drain() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    wnr4_swar_cp_wait_all();
    __syncwarp();
#endif
}

/* Refill ring if ptr crossed a 128-word slot boundary; safe when called at least once every 4 nibbles. */
static __device__ __forceinline__ void wnr4_swar_ring_refill(uint32_t cur, uint32_t ptr, uint32_t smem_base,
                                                             const uint32_t *gmem32, uint32_t max_u32,
                                                             uint32_t lane, bool &ok) {
    if (ptr > (max_u32 << 1)) { ok = false; return; }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if ((ptr >> 7) != cur) {
        __syncwarp();
        wnr4_swar_cp_slot(smem_base + (cur & 3u) * 256u, gmem32, cur + 4u, max_u32, lane);
        wnr4_swar_cp_commit();
        wnr4_swar_cp_wait<2>();
        __syncwarp();
    }
#else
    (void)cur; (void)smem_base; (void)gmem32; (void)lane;
#endif
}

static __device__ __forceinline__ bool wnr4_swar_renorm_checked(uint32_t &x, bool active, uint32_t smem_base,
                                                                const uint32_t *gmem32, uint32_t max_u32,
                                                                uint32_t &ptr, uint32_t lane, uint32_t lmask) {
    const bool need = active && (x < WNR4_L);
    const uint32_t vote = __ballot_sync(0xffffffffu, need);
    if (!vote) return true;
    const uint32_t total = __popc(vote);
    if (ptr + total > (max_u32 << 1)) return false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t cur = ptr >> 7;
    const uint32_t widx = ptr + __popc(vote & lmask);
    if (active) x = wnr4_swar_lds_renorm(x, smem_base + ((widx & 511u) << 1));
    ptr += total;
    if ((ptr >> 7) != cur) {
        __syncwarp();
        wnr4_swar_cp_slot(smem_base + (cur & 3u) * 256u, gmem32, cur + 4u, max_u32, lane);
        wnr4_swar_cp_commit();
        wnr4_swar_cp_wait<2>();
        __syncwarp();
    }
#else
    (void)smem_base; (void)lane;
    const uint16_t *words = reinterpret_cast<const uint16_t *>(gmem32);
    if (need) x = (x << 16) | (uint32_t)__ldg(words + ptr + __popc(vote & lmask));
    ptr += total;
#endif
    return true;
}

/* ---- CUDA kernel: W in {1, 2, 4, 8, 16} warps per block ------------------- */
template<uint32_t W>
static __global__ void __launch_bounds__(W * 32u, (32u / W > 16u) ? 16u : (32u / W))
wnr4_swar_cuda_kernel(const Wnr4SwarCudaJob *jobs, uint32_t njobs, const Wnr4SwarCudaRec *recs,
                      Wnr4SwarCudaTables tab, uint32_t *errors) {
    /* 1,008 B weight CDFs (64 uint4 = 1,024 B) + W x 1,024 B 4-slot cp.async ring */
    __shared__ uint4 sm_wcdf[64];
    __shared__ uint32_t sm_ring[W][256];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lmask = (1u << lane) - 1u;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;

    const uint32_t m = jobs[first].m;
    const uint4 *grows = tab.wrows + (size_t)m * WNR4_SWAR_WCTX;
    for (uint32_t i = threadIdx.x; i < WNR4_SWAR_WCTX; i += blockDim.x) sm_wcdf[i] = __ldg(grows + i);
    if constexpr (W > 1u) __syncthreads();
    else __syncwarp();

    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4SwarCudaJob job = jobs[jid];
    if (!job.n) return;
    if (job.m != m || job.tile_bytes < WNR4_TILE_STATE_BYTES || (job.tile_bytes & 3u)) {
        if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT);
        return;
    }

    const Wnr4SwarCudaRec *rec = recs + job.rec;
    const uint32_t n = job.n, c0 = job.c0;
    const uint32_t flags = __ldg(&rec->flags);
    const uint32_t anchors = __ldg(&rec->anchors);
    const uint32_t anchor = (anchors >> (8u * m)) & 0xffu;
    const bool ceil_f = ((flags >> m) & 1u) != 0u;
    unsigned char *so = rec->dst_s[m];
    uint4 *wo = reinterpret_cast<uint4 *>(rec->dst_w[m]);

    /* Initial rANS states + MurmurHash3 seed */
    uint32_t x = __ldg(reinterpret_cast<const uint32_t *>(job.tile) + lane);
    uint32_t ptr = 0u, err = 0u;
    bool ok = (x >= WNR4_L);
    uint32_t hash = wnr4_swar_lane_seed(__ldg(&rec->model_id), __ldg(&rec->owner), job.t, lane,
                                        wnr4_word(WNR4_KIND_MX, flags, __ldg(&rec->G)), anchors,
                                        __ldg(&rec->srec_hash));

    const uint32_t *gmem32 = reinterpret_cast<const uint32_t *>(job.tile + WNR4_TILE_STATE_BYTES);
    const uint32_t max_u32 = (job.tile_bytes - WNR4_TILE_STATE_BYTES) >> 2;
    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sm_ring[warp]));
    const uint32_t wcdf_base = static_cast<uint32_t>(__cvta_generic_to_shared(sm_wcdf));

    wnr4_swar_ring_init(smem_base, gmem32, max_u32, lane);

    /* Phase 1: s_rec16 in registers (2 x uint4 = 32 B/matrix) + escape */
    {
        const uint4 sa = __ldg(&rec->scdf[m][0]), sb = __ldg(&rec->scdf[m][1]);
        for (uint32_t k = 0u; 32u * k < n && __all_sync(0xffffffffu, ok); k++) {
            const uint32_t i = 32u * k + lane;
            const bool active = (i < n);
            uint32_t sym = 0u;
            if (active) {
                const uint32_t slot = x & (WNR4_TOTAL - 1u);
                uint32_t c_s, f_s;
                wnr4_row_find(sa, sb, slot, sym, c_s, f_s);
                x = f_s * (x >> WNR4_PREC) + slot - c_s;
            }
            if (!wnr4_swar_renorm_checked(x, active, smem_base, gmem32, max_u32, ptr, lane, lmask)) ok = false;

            const bool esc = active && (sym == 15u);
            uint32_t y = 0u;
            if (__any_sync(0xffffffffu, esc)) {
                if (esc) {
                    const uint32_t slot = x & (WNR4_TOTAL - 1u);
                    y = slot >> 6;
                    const int rel = (int)y - (int)anchor + 7;
                    if (rel >= 0 && rel <= 14) err |= WNR4_ERR_STRUCT;
                    x = ((x >> WNR4_PREC) << 6) | (slot & 63u);
                }
                if (!wnr4_swar_renorm_checked(x, esc, smem_base, gmem32, max_u32, ptr, lane, lmask)) ok = false;
            }
            if (active) {
                if (!esc) {
                    const int val = (int)anchor + (int)sym - 7;
                    if (val < 0 || val > 255) err |= WNR4_ERR_STRUCT;
                    else y = (uint32_t)val;
                }
                so[c0 + i] = (unsigned char)y;
                hash = wnr4_h32_block(hash, y);
            }
        }
    }
    __syncwarp();

    /* Phase 2: 63-ctx 8-mag carry-free SWAR + aligned backward 32-bit raw signs + MAGT.
     * Double-buffered warp register sign window (sw0, sw1): 32 lanes preload 256 B
     * (64 sign u32s = ~76 nibbles) via 1 coalesced cacheline load every ~38 nibbles;
     * per-step sign refill is a 2-cycle __shfl_sync with zero global stall. */
    auto sw_load = [&](uint32_t base) -> uint32_t {
        const uint32_t k = base + lane;
        return (k < max_u32) ? __ldg(gmem32 + (max_u32 - 1u - k)) : 0u;
    };
    uint32_t sw_base = 0u, sw_used = 0u;
    uint32_t sw0 = sw_load(0u), sw1 = sw_load(32u);
    uint64_t sbuf = 0ULL;
    uint32_t sbits = 0u;

    /* Fast full-warp rounds: all 32 lanes active, none on their last cell (no MAGT tail logic). */
    const uint32_t fast_rounds = (n > 32u) ? ((n - 33u) >> 5) : 0u;
    uint32_t j = 0u;
    for (; j < fast_rounds && __all_sync(0xffffffffu, ok); j++) {
        const uint32_t cell = c0 + 32u * j + lane;
        const uint32_t sc = (uint32_t)so[cell];
        const uint32_t cls_saddr = wcdf_base + wnr4_cls(sc, anchor, 3) * (9u * 16u);
        uint32_t mx = 8u;
        uint4 vec = wnr4_swar_lds128_ro(cls_saddr + 8u * 16u);
        uint32_t w[4];

#pragma unroll
        for (uint32_t q = 0u; q < 4u; q++) {
            uint32_t wq = 0u;
            /* Two 4-nibble halves per u32: ring refilled at most once every 4 nibbles (<= 128 words). */
#pragma unroll 1
            for (uint32_t h = 0u; h < 2u; h++) {
                const uint32_t cur = ptr >> 7;
                uint32_t sh = h * 16u;
#pragma unroll 1
                for (uint32_t s = 0u; s < 4u; s++, sh += 4u) {
                    const uint32_t slot = x & (WNR4_TOTAL - 1u);
                    uint32_t mag, c_s, f_s;
                    wnr4_swar_find8_dev(vec, slot, mag, c_s, f_s);
                    mx = wnr4_mx(mx, mag);
                    vec = wnr4_swar_lds128_ro(cls_saddr + mx * 16u);
                    const uint32_t xn = f_s * (x >> WNR4_PREC) + slot - c_s;

                    const uint32_t vote = __ballot_sync(0xffffffffu, xn < WNR4_L);
                    const uint32_t widx = ptr + __popc(vote & lmask);
                    ptr += __popc(vote);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                    x = wnr4_swar_lds_renorm(xn, smem_base + ((widx & 511u) << 1));
#else
                    if (xn < WNR4_L) {
                        if (widx < (max_u32 << 1))
                            x = (xn << 16) | (uint32_t)__ldg(reinterpret_cast<const uint16_t *>(gmem32) + widx);
                        else ok = false;
                    } else x = xn;
#endif
                    const uint32_t nz_mask = __ballot_sync(0xffffffffu, mag != 0u);
                    const uint32_t cnt = __popc(nz_mask);
                    if (sbits < cnt) {
                        if (sw_used >= max_u32) ok = false;
                        else {
                            const uint32_t sw_w = __shfl_sync(0xffffffffu, sw0, sw_used - sw_base);
                            sbuf |= (uint64_t)sw_w << sbits;
                            sbits += 32u;
                            sw_used++;
                            if (sw_used - sw_base == 32u) {
                                sw0 = sw1;
                                sw_base += 32u;
                                sw1 = sw_load(sw_base + 32u);
                            }
                        }
                    }
                    const uint32_t rank = __popc(nz_mask & lmask);
                    const uint32_t sign = (mag != 0u) ? (uint32_t)((sbuf >> rank) & 1u) : 0u;
                    sbuf >>= cnt;
                    sbits -= cnt;
                    wq |= ((sign << 3) | mag) << sh;
                }
                wnr4_swar_ring_refill(cur, ptr, smem_base, gmem32, max_u32, lane, ok);
            }
            w[q] = wq;
        }

        if (ok) {
            hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w[0]), w[1]), w[2]), w[3]);
            if (ceil_f && mx == 5u) {
                if (!sc) err |= WNR4_ERR_CEIL;
                else {
                    so[cell] = (unsigned char)(sc - 1u);
#pragma unroll
                    for (uint32_t q = 0u; q < 4u; q++) w[q] = wnr4_ceil_dbl8(w[q]);
                }
            }
            wo[cell] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }

    /* Final 1..2 rounds: partial warp + MAGT last-cell handling */
    for (; 32u * j < n && __all_sync(0xffffffffu, ok); j++) {
        const uint32_t off = 32u * j + lane;
        const uint32_t cell = c0 + off;
        const bool active = (off < n);
        const bool is_last = active && (lane >= 2u) && (off + 32u >= n);
        const uint32_t sc = active ? (uint32_t)so[cell] : 0u;
        const uint32_t cls_saddr = wcdf_base + wnr4_cls(sc, anchor, 3) * (9u * 16u);
        uint32_t mx = 8u;
        uint32_t tleft = is_last ? 16u : 0u;
        uint32_t w[4];

#pragma unroll
        for (uint32_t q = 0u; q < 4u; q++) {
            uint32_t wq = 0u;
#pragma unroll 1
            for (uint32_t r4 = 0u; r4 < 8u; r4++) {
                uint32_t mag = 0u;
                if (active) {
                    const uint32_t slot = x & (WNR4_TOTAL - 1u);
                    uint32_t c_s, f_s;
                    wnr4_swar_find8_dev(wnr4_swar_lds128_ro(cls_saddr + mx * 16u), slot, mag, c_s, f_s);
                    x = f_s * (x >> WNR4_PREC) + slot - c_s;
                    mx = wnr4_mx(mx, mag);
                }
                ok = wnr4_swar_renorm_checked(x, active, smem_base, gmem32, max_u32, ptr, lane, lmask) && ok;

                const bool is_nz = active && (mag != 0u);
                const bool take_tail = is_nz && (tleft != 0u);
                if (take_tail) tleft--;
                const bool need_sign = is_nz && !take_tail;

                const uint32_t nz_mask = __ballot_sync(0xffffffffu, need_sign);
                const uint32_t cnt = __popc(nz_mask);
                if (sbits < cnt) {
                    if (sw_used >= max_u32) ok = false;
                    else {
                        const uint32_t sw_w = __shfl_sync(0xffffffffu, sw0, sw_used - sw_base);
                        sbuf |= (uint64_t)sw_w << sbits;
                        sbits += 32u;
                        sw_used++;
                        if (sw_used - sw_base == 32u) {
                            sw0 = sw1;
                            sw_base += 32u;
                            sw1 = sw_load(sw_base + 32u);
                        }
                    }
                }
                const uint32_t rank = __popc(nz_mask & lmask);
                const uint32_t sign = need_sign ? (uint32_t)((sbuf >> rank) & 1u) : 0u;
                sbuf >>= cnt;
                sbits -= cnt;
                wq |= ((sign << 3) | mag) << (4u * r4);
            }
            w[q] = wq;
        }

        if (active && ok) {
            if (!is_last) {
                hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w[0]), w[1]), w[2]), w[3]);
                if (ceil_f && mx == 5u) {
                    if (!sc) err |= WNR4_ERR_CEIL;
                    else {
                        so[cell] = (unsigned char)(sc - 1u);
#pragma unroll
                        for (uint32_t q = 0u; q < 4u; q++) w[q] = wnr4_ceil_dbl8(w[q]);
                    }
                }
            }
            wo[cell] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }

    wnr4_swar_ring_drain();

    if (!__all_sync(0xffffffffu, ok)) {
        if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT);
        return;
    }

    /* Finalize MAGT last cell for lanes 2..31 with at least one cell */
    if (lane >= 2u && lane < n) {
        const uint32_t pcell = c0 + lane + 32u * ((n - lane - 1u) >> 5);
        const uint32_t tail = x - WNR4_L;
        if (x < WNR4_L || tail > 0xffffu) err |= WNR4_ERR_STRUCT;
        const uint4 pv = wo[pcell];
        uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
        uint32_t bit = 0u, pmx = 8u;
#pragma unroll
        for (uint32_t q = 0u; q < 4u; q++) {
            uint32_t wq = pw[q];
#pragma unroll
            for (uint32_t r4 = 0u; r4 < 8u; r4++) {
                const uint32_t mag = (wq >> (4u * r4)) & 7u;
                pmx = wnr4_mx(pmx, mag);
                if (mag != 0u && bit < 16u) {
                    wq |= ((tail >> bit) & 1u) << (4u * r4 + 3u);
                    bit++;
                }
            }
            pw[q] = wq;
        }
        if (tail >> bit) err |= WNR4_ERR_STRUCT;
        x = WNR4_L;
        hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, pw[0]), pw[1]), pw[2]), pw[3]);
        if (ceil_f && pmx == 5u) {
            const uint32_t ps = so[pcell];
            if (!ps) err |= WNR4_ERR_CEIL;
            else {
                so[pcell] = (unsigned char)(ps - 1u);
#pragma unroll
                for (uint32_t q = 0u; q < 4u; q++) pw[q] = wnr4_ceil_dbl8(pw[q]);
            }
        }
        wo[pcell] = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    }

    /* Exact forward/backward meet + zero pad + zero unused high sign bits */
    if (!lane) {
        const uint32_t fwd_end = 2u * ptr;
        const uint32_t sign_beg = 4u * (max_u32 - sw_used);
        const uint32_t pad2 = (ptr & 1u) ? 2u : 0u;
        if (fwd_end > sign_beg || sign_beg - fwd_end != pad2) {
            err |= WNR4_ERR_STRUCT;
        } else if (pad2) {
            const uint16_t zp = __ldg(reinterpret_cast<const uint16_t *>(gmem32) + ptr);
            if (zp != 0u) err |= WNR4_ERR_STRUCT;
        }
        if (sbits >= 32u || sbuf != 0ULL) err |= WNR4_ERR_STRUCT;
    }

    uint32_t H = wnr4_h32_final(hash, wnr4_swar_lane_hlen(wnr4_lane_scales(n, lane), wnr4_lane_cells(n, lane)));
#pragma unroll
    for (uint32_t d = 16u; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);

    if (lane == 0u && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1u && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + job.rec, err);
}

/* ---- Host side ------------------------------------------------------------ */
static inline size_t wnr4_swar_cuda_table_bytes(void) {
    return 3u * (size_t)WNR4_SWAR_WCTX * 16u; /* 3,024 B total */
}

static inline void wnr4_swar_cuda_table_image(const Wnr4SwarModel *m, unsigned char *img) {
    uint4 *dst = reinterpret_cast<uint4 *>(img);
    for (uint32_t mm = 0u; mm < 3u; mm++) {
        for (uint32_t ctx = 0u; ctx < WNR4_SWAR_WCTX; ctx++) {
            const uint16_t *c = m->wcdf[mm][ctx];
            uint4 v;
            v.x = (uint32_t)c[1] | ((uint32_t)c[2] << 16);
            v.y = (uint32_t)c[3] | ((uint32_t)c[4] << 16);
            v.z = (uint32_t)c[5] | ((uint32_t)c[6] << 16);
            v.w = (uint32_t)c[7] | ((uint32_t)WNR4_TOTAL << 16);
            dst[mm * WNR4_SWAR_WCTX + ctx] = v;
        }
    }
}

static inline void wnr4_swar_cuda_tables_view(const Wnr4SwarModel *m, const unsigned char *dev,
                                              Wnr4SwarCudaTables *t) {
    t->id = m->id;
    t->wrows = reinterpret_cast<const uint4 *>(dev);
}

static inline int wnr4_swar_cuda_plan(const Wnr4Geo *g, const Wnr4SwarModel *model, const unsigned char *rec_host,
                                      size_t total, uint32_t owner, const unsigned char *rec_dev, uint32_t index,
                                      unsigned char *const dst_s[3], unsigned char *const dst_w[3],
                                      Wnr4SwarCudaRec *info, std::vector<Wnr4SwarCudaJob> by_m[3]) {
    Wnr4SwarRecord r;
    if (wnr4_swar_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner) return -1;
    info->model_id = r.model_id;
    info->owner = r.owner;
    info->flags = r.flags;
    info->G = r.G;
    info->anchors = (uint32_t)r.anchor[0] | ((uint32_t)r.anchor[1] << 8) | ((uint32_t)r.anchor[2] << 16);
    info->srec_hash = r.srec_hash;
    for (uint32_t m = 0u; m < 3u; m++) {
        info->dst_s[m] = dst_s[m];
        info->dst_w[m] = dst_w[m];
        memcpy(&info->scdf[m][0], &r.scdf[m][0], 16u);
        memcpy(&info->scdf[m][1], &r.scdf[m][8], 16u);
    }
    const size_t data_at = (size_t)(r.data - rec_host);
    for (uint32_t t = 0u; t < r.ntiles; t++) {
        const uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || ((o1 - o0) & 3u) || o1 > r.payload) return -1;
    }
    for (uint32_t t = 0u; t < r.ntiles; t++) {
        const uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        Wnr4SwarCudaJob j;
        wnr4_tile_cells(r.tile_base, g->cells, r.G, t, &j.m, &j.c0, &j.n);
        j.tile = rec_dev + data_at + o0;
        j.tile_bytes = o1 - o0;
        j.rec = index;
        j.t = t;
        by_m[j.m].push_back(j);
    }
    return 0;
}

static inline uint32_t wnr4_swar_cuda_jobs(std::vector<Wnr4SwarCudaJob> by_m[3], std::vector<Wnr4SwarCudaJob> &jobs,
                                           uint32_t force_W = 0u) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    const uint32_t W = force_W ? force_W : (tiles < 400u ? 1u : tiles < 800u ? 2u : tiles < 1600u ? 4u : 8u);
    jobs.clear();
    for (uint32_t m = 0u; m < 3u; m++) {
        for (const auto &j : by_m[m]) jobs.push_back(j);
        Wnr4SwarCudaJob pad = {};
        pad.m = m;
        while (jobs.size() % W) jobs.push_back(pad);
    }
    return W;
}

static inline cudaError_t wnr4_swar_cuda_launch(const Wnr4SwarCudaTables &tab, const Wnr4SwarCudaJob *jobs_dev,
                                                uint32_t njobs, const Wnr4SwarCudaRec *recs_dev, uint32_t *errors_dev,
                                                uint32_t W, cudaStream_t stream) {
    const uint32_t blocks = (njobs + W - 1u) / W;
    if (!blocks) return cudaSuccess;
    switch (W) {
        case 1u:  wnr4_swar_cuda_kernel<1u><<<blocks, 32u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); break;
        case 2u:  wnr4_swar_cuda_kernel<2u><<<blocks, 64u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); break;
        case 4u:  wnr4_swar_cuda_kernel<4u><<<blocks, 128u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); break;
        case 8u:  wnr4_swar_cuda_kernel<8u><<<blocks, 256u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); break;
        case 16u: wnr4_swar_cuda_kernel<16u><<<blocks, 512u, 0, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

#endif
