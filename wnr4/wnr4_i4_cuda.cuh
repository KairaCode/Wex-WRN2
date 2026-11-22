/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-I4 CUDA decoder (kind 3, wnr4/wnr4_i4.h; bit-exact with wnr4_i4_decode_tile). One warp decodes one tile
 * with its 32 lanes in lockstep, as wnr4_cuda.cuh does for MX/NV: renormalising lanes take the next words of the
 * tile in lane order from a 64-word register window (ballot + popc + shuffle), lanes 2..31 carry their tail in the
 * state, lanes 0 and 1 end at the tile hash, every failure sets error bits of the record (fail closed: its
 * destination may hold partial data). A block holds 4, 8 or 16 tiles of one matrix; the matrix's nibble rows sit in
 * shared memory in compact order (13,248 B: per class the state (9, 9) and the 45 states pm <= mx); the scale rows
 * (two-level: a 16-value summary row, then the 16-value block row) are read through L1, phase 1 being 1/65 of the
 * symbols; the tile's lattice indices, the contexts of later steps, sit in shared memory (32 G bytes per warp,
 * hence G <= WNR4_I4_CUDA_MAX_G). The lattice is computed with __fmul_rn and __fdiv_rn: do not build with
 * --use_fast_math or -ftz=true. Scales are written in phase 1 and read back (same thread) for the classes of
 * phase 2. Destination: the record's decoded bytes, 4-byte aligned (codes and scales are written as 32-bit words).
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

/* symbol, start, frequency of slot in a 256-symbol two-level row */
static __device__ __forceinline__ uint32_t wnr4_i4_find256(const uint4 *row, uint32_t slot, uint32_t &c, uint32_t &f) {
    uint32_t kb, sS, fS, jb;
    wnr4_row_find(__ldg(row), __ldg(row + 1), slot, kb, sS, fS);
    wnr4_row_find(__ldg(row + 2u + 2u * kb), __ldg(row + 3u + 2u * kb), slot, jb, c, f);
    if (c + f == WNR4_TOTAL) f = sS + fS - c;   /* the block's last symbol: its end is the next summary value */
    return 16u * kb + jb;
}

template<uint32_t W>
static __global__ void __launch_bounds__(W * 32u, 32u / W) wnr4_i4_cuda_kernel(
    const Wnr4I4CudaJob *jobs, uint32_t njobs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    __shared__ uint4 sm[WNR4_I4_CUDA_NIB_U4];
    __shared__ unsigned char kbuf[W][32u * WNR4_I4_CUDA_MAX_G];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t m = jobs[first].m;   /* the host groups jobs so a block has one matrix */
    for (uint32_t i = threadIdx.x; i < WNR4_I4_CUDA_NIB_U4; i += blockDim.x) sm[i] = tab[(size_t)m * WNR4_I4_CUDA_MAT_U4 + i];
    __syncthreads();
    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4I4CudaJob job = jobs[jid];
    if (!job.n) return;   /* padding job */
    if (job.m != m || job.n > 32u * WNR4_I4_CUDA_MAX_G) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }
    const uint4 *srows = tab + (size_t)m * WNR4_I4_CUDA_MAT_U4 + WNR4_I4_CUDA_NIB_U4;
    const uint16_t *words = (const uint16_t *)(job.tile + WNR4_TILE_STATE_BYTES);
    unsigned char *kk = kbuf[warp];
    uint32_t x = (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane + 1u) << 16;
    uint32_t ptr = 0, err = 0;
    bool ok = x >= WNR4_L;
    Wnr4Win win;
    wnr4_win_init(win, words, job.nwords, lane);
    uint32_t hash = wnr4_lane_seed(model_id, job.owner, job.t, lane, wnr4_word(WNR4_KIND_I4, job.pairs, job.G),
                                   job.m | job.anchor << 16);
    const uint32_t n = job.n, c0 = job.c0;
    /* phase 1: scales, row-major across the lanes; the lattice index of group i at kk[i - c0] */
    for (uint32_t k = 0; 32u * k < n && __all_sync(0xffffffffu, ok); k++) {
        const uint32_t i = c0 + 32u * k + lane;
        const bool active = 32u * k + lane < n;
        bool esc = false;
        uint32_t val = 0;
        if (active) {
            const uint32_t u = wnr4_i4_ref(job.gc, i, c0), has = u != 0xffffffffu;
            const uint32_t kref = wnr4_i4_kref(has ? kk[u - c0] : 0u, (int)has);
            uint32_t slot = x & (WNR4_TOTAL - 1u), c, f, y = wnr4_i4_find256(srows + wnr4_i4_sctx(kref, (int)has) * 34u, slot, c, f);
            x = f * (x >> WNR4_PREC) + slot - c;
            uint32_t kv = (y + kref) & 255u;
            if (kv > WNR4_I4_ESC) { ok = false; kv = WNR4_I4_ESC; }
            kk[i - c0] = (unsigned char)kv;
            esc = kv == WNR4_I4_ESC;
            if (!esc) {
                const unsigned char *bp = job.bases + 4u * wnr4_i4_block(job.bbase, job.gc, job.nbc, i);
                val = wnr4_i4_lat(kv, (uint32_t)bp[0] | (uint32_t)bp[1] << 8 | (uint32_t)bp[2] << 16 | (uint32_t)bp[3] << 24);
            }
        }
        if (!wnr4_renorm_win(x, active, words, ptr, job.nwords, lane, win)) ok = false;
        for (uint32_t b = 0; b < 4u; b++) {   /* escaped scales: their 4 bytes, high byte first */
            if (esc) {
                uint32_t slot = x & (WNR4_TOTAL - 1u), c, f, y = wnr4_i4_find256(srows + (64u + b) * 34u, slot, c, f);
                x = f * (x >> WNR4_PREC) + slot - c;
                val |= y << (24u - 8u * b);
            }
            if (!wnr4_renorm_win(x, esc, words, ptr, job.nwords, lane, win)) ok = false;
        }
        if (active) { job.scales[i] = val; hash = wnr4_h32_block(hash, val); }
        __syncwarp();   /* kk of this step is read by other lanes at later steps */
    }
    /* phase 2: groups, lane l owning groups c0 + l + 32 j, their 64 nibbles in element order */
    uint32_t pv[8];   /* a tail lane's last group, finished at the end */
    unsigned char *pgrp = nullptr;
    for (uint32_t j = 0; 32u * j < n && __all_sync(0xffffffffu, ok); j++) {
        const uint32_t i = c0 + 32u * j + lane;
        const bool active = 32u * j + lane < n;
        const uint32_t cls = active ? wnr4_i4_cls(job.scales[i], job.anchor) : 0u;
        const uint4 *rows = sm + cls * WNR4_I4_CUDA_STATES * 2u;
        uint32_t v[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, pm = 9u, mx = 9u;
        const bool tail = active && wnr4_i4_tail_skip(lane, j, 63u, n);
        const uint32_t eend = tail ? 60u : 64u;   /* tail lanes stop at 60 */
#pragma unroll
        for (uint32_t e = 0; e < 64u; e++) {
            const bool coded = active && e < eend;
            if (coded) {
                const uint32_t st = wnr4_i4_state(pm, mx);
                const uint32_t slot = x & (WNR4_TOTAL - 1u), y = wnr4_row_sym(rows[2u * st], rows[2u * st + 1u], slot);
                const uint16_t *r16 = (const uint16_t *)(rows + 2u * st);
                const uint32_t c = r16[y], f = (y == 15u ? WNR4_TOTAL : (uint32_t)r16[y + 1u]) - c;
                x = f * (x >> WNR4_PREC) + slot - c;
                v[e / 8u] |= y << ((e % 8u) * 4u);
                const uint32_t a = wnr4_i4_mag(y);
                pm = a; mx = wnr4_i4_mx(mx, a);
            }
            ok = wnr4_renorm_win(x, coded, words, ptr, job.nwords, lane, win) && ok;   /* no break: convergent loop */
        }
        if (active && ok) {
            unsigned char *grp = job.codes + 32u * (size_t)i;
            if (tail) { for (uint32_t q = 0; q < 8u; q++) pv[q] = v[q]; pgrp = grp; }
            else {
                uint32_t w[8];
                wnr4_i4_pack(v, job.pairs, w);
                for (uint32_t q = 0; q < 8u; q++) { hash = wnr4_h32_block(hash, w[q]); ((uint32_t *)grp)[q] = w[q]; }
            }
        }
    }
    if (!__all_sync(0xffffffffu, ok)) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }
    if (pgrp) {   /* elements 60..63 of the lane's last group are its final state - L */
        const uint32_t tail = x - WNR4_L;
        uint32_t w[8];
        if (tail > 0xffffu) err |= WNR4_ERR_STRUCT;
        pv[7] |= tail << 16;
        x = WNR4_L;
        wnr4_i4_pack(pv, job.pairs, w);
        for (uint32_t q = 0; q < 8u; q++) { hash = wnr4_h32_block(hash, w[q]); ((uint32_t *)pgrp)[q] = w[q]; }
    }
    const uint32_t lc = wnr4_lane_cells(n, lane);
    uint32_t H = wnr4_h32_final(hash, wnr4_i4_hlen(lc, lc));
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);
    if (lane == 0 && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1 && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (lane == 0 && ptr != job.nwords) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + job.rec, err);
}

/* ---- host side --------------------------------------------------------------- */
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
/* Concatenates the per-matrix lists, each padded to a multiple of the block size (padding jobs have n = 0).
 * Returns the block size: wnr4_cuda_warps, or by the number of tiles (4, 8 or 16 warps). */
static inline uint32_t wnr4_i4_cuda_jobs(std::vector<Wnr4I4CudaJob> by_m[3], std::vector<Wnr4I4CudaJob> &jobs) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    const uint32_t W = wnr4_cuda_warps ? wnr4_cuda_warps : tiles < 1000u ? 4u : tiles < 2000u ? 8u : 16u;
    jobs.clear();
    for (uint32_t m = 0; m < 3u; m++) {
        for (auto &j : by_m[m]) jobs.push_back(j);
        Wnr4I4CudaJob pad = {};
        pad.m = m;
        while (jobs.size() % W) jobs.push_back(pad);
    }
    return W;
}
static inline cudaError_t wnr4_i4_cuda_launch(const unsigned char *tab_dev, uint32_t model_id, const Wnr4I4CudaJob *jobs_dev,
                                              uint32_t njobs, uint32_t W, uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t blocks = (njobs + W - 1u) / W;
    const uint4 *tab = (const uint4 *)tab_dev;
    if (!blocks) return cudaSuccess;
    if (W == 4u) wnr4_i4_cuda_kernel<4u><<<blocks, 128u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 8u) wnr4_i4_cuda_kernel<8u><<<blocks, 256u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 16u) wnr4_i4_cuda_kernel<16u><<<blocks, 512u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

#endif
