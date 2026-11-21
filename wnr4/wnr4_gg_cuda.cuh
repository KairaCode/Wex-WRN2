/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-GG CUDA decoder (kind 2, wnr4/wnr4_gg.h; bit-exact with wnr4_gg_decode_tile). One warp decodes one
 * tile with its 32 lanes in lockstep, as wnr4_cuda.cuh does for MX/NV: renormalising lanes take the next words
 * of the tile in lane order from a 64-word register window (ballot + popc + shuffle), lanes 2..31 carry their
 * tail in the state, lanes 0 and 1 end at the tile hash, every failure sets error bits of the record (fail
 * closed: its destination may hold partial data). A block holds 4, 8 or 16 tiles of one subtype; the
 * subtype's tables sit in shared memory (19,232 B): the high-byte table and the 32 low-byte tables as
 * two-level rows (a 16-value summary row, then the 16-value block row), the 32 nibble rows, the class map.
 * Output goes straight to the record's destination, interleaved as the GGUF slice: block c of a slice at
 * off + 18 c, byte 0 the low and byte 1 the high byte of d, bytes 2..17 the cell. The high bytes written in
 * phase 1 are read back (same thread) for the classes of phases 2 and 3. Verbatim slices are copied by
 * wnr4_gg_cuda_copy from the record's device copy.
 * Host side: wnr4_gg_cuda_table_image() (model), wnr4_gg_cuda_plan() (jobs and copies of a record, from its
 * host copy), wnr4_gg_cuda_jobs(), wnr4_gg_cuda_launch(), wnr4_gg_cuda_copy(). */
#ifndef WNR4_GG_CUDA_CUH
#define WNR4_GG_CUDA_CUH

#include "wnr4_cuda.cuh"
#include "wnr4_gg.h"

/* uint4 per subtype in the device image: hi rows (34), lo rows [32][34], nibble rows [32][2], map (16) */
#define WNR4_GG_CUDA_LO_AT 34u
#define WNR4_GG_CUDA_NIB_AT (WNR4_GG_CUDA_LO_AT + WNR4_GG_CLASSES * 34u)
#define WNR4_GG_CUDA_MAP_AT (WNR4_GG_CUDA_NIB_AT + WNR4_GG_CLASSES * 2u)
#define WNR4_GG_CUDA_SUB_U4 (WNR4_GG_CUDA_MAP_AT + 16u)

struct Wnr4GgCudaJob {           /* one tile */
    const unsigned char *tile;   /* device: its 32 states then words */
    unsigned char *out;          /* device: the decoded slice (block 0) */
    uint32_t nwords, rec, t, sub, type, c0, n, owner, G;
};
struct Wnr4GgCudaCopy {          /* one verbatim slice */
    const unsigned char *src;    /* device: in the record's copy */
    unsigned char *dst;
    size_t bytes;
};

/* symbol, start, frequency of slot in a 256-symbol two-level row (wnr4_find256's symbol) */
static __device__ __forceinline__ uint32_t wnr4_gg_find256(const uint4 *row, uint32_t slot, uint32_t &c, uint32_t &f) {
    uint32_t kb, sS, fS, jb;
    wnr4_row_find(row[0], row[1], slot, kb, sS, fS);
    wnr4_row_find(row[2u + 2u * kb], row[3u + 2u * kb], slot, jb, c, f);
    if (c + f == WNR4_TOTAL) f = sS + fS - c;   /* the block's last symbol: its end is the next summary value */
    return 16u * kb + jb;
}

template<uint32_t W>
static __global__ void __launch_bounds__(W * 32u, 32u / W) wnr4_gg_cuda_kernel(
    const Wnr4GgCudaJob *jobs, uint32_t njobs, const uint4 *tab, uint32_t model_id, uint32_t *errors) {
    __shared__ uint4 sm[WNR4_GG_CUDA_SUB_U4];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t sub = jobs[first].sub;   /* the host groups jobs so a block has one subtype */
    for (uint32_t i = threadIdx.x; i < WNR4_GG_CUDA_SUB_U4; i += blockDim.x) sm[i] = tab[(size_t)sub * WNR4_GG_CUDA_SUB_U4 + i];
    __syncthreads();
    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4GgCudaJob job = jobs[jid];
    if (!job.n) return;   /* padding job */
    if (job.sub != sub) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }   /* mis-planned block */
    const uint4 *lo_rows = sm + WNR4_GG_CUDA_LO_AT, *nib_rows = sm + WNR4_GG_CUDA_NIB_AT;
    const unsigned char *map = (const unsigned char *)(sm + WNR4_GG_CUDA_MAP_AT);
    const uint16_t *words = (const uint16_t *)(job.tile + WNR4_TILE_STATE_BYTES);
    uint32_t x = (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane + 1u) << 16;
    uint32_t ptr = 0, err = 0;
    bool ok = x >= WNR4_L;
    Wnr4Win win;
    wnr4_win_init(win, words, job.nwords, lane);
    uint32_t hash = wnr4_lane_seed(model_id, job.owner, job.t, lane, wnr4_word(WNR4_KIND_GG, 0u, job.G), job.type);
    const uint32_t n = job.n;
    unsigned char *blk = job.out + (size_t)job.c0 * 18u;   /* the tile's first block */
    /* phase 1: high bytes of the scales, row-major across the lanes */
    for (uint32_t k = 0; 32u * k < n && __all_sync(0xffffffffu, ok); k++) {
        const uint32_t i = 32u * k + lane;
        const bool active = i < n;
        if (active) {
            uint32_t slot = x & (WNR4_TOTAL - 1u), c, f, y = wnr4_gg_find256(sm, slot, c, f);
            x = f * (x >> WNR4_PREC) + slot - c;
            blk[(size_t)i * 18u + 1u] = (unsigned char)y;
            hash = wnr4_h32_block(hash, y);
        }
        if (!wnr4_renorm_win(x, active, words, ptr, job.nwords, lane, win)) ok = false;
    }
    /* phase 2: low bytes, in the class of their high byte (this lane's own phase-1 writes; next one prefetched) */
    uint32_t hnext = lane < n ? blk[(size_t)lane * 18u + 1u] : 0u;
    for (uint32_t k = 0; 32u * k < n && __all_sync(0xffffffffu, ok); k++) {
        const uint32_t i = 32u * k + lane, h = hnext;
        const bool active = i < n;
        hnext = i + 32u < n ? blk[(size_t)(i + 32u) * 18u + 1u] : 0u;
        if (active) {
            uint32_t slot = x & (WNR4_TOTAL - 1u), c, f, y = wnr4_gg_find256(lo_rows + map[h] * 34u, slot, c, f);
            x = f * (x >> WNR4_PREC) + slot - c;
            blk[(size_t)i * 18u] = (unsigned char)y;
            hash = wnr4_h32_block(hash, y);
        }
        if (!wnr4_renorm_win(x, active, words, ptr, job.nwords, lane, win)) ok = false;
    }
    /* phase 3: cells, lane l owning blocks l + 32 j; one nibble row per cell, held in registers */
    uint32_t pw[4] = {0u, 0u, 0u, 0u};   /* a tail lane's last cell, finished at the end */
    unsigned char *pcell = nullptr;
    hnext = lane < n ? blk[(size_t)lane * 18u + 1u] : 0u;
    for (uint32_t j = 0; 32u * j < n && __all_sync(0xffffffffu, ok); j++) {
        const uint32_t i = 32u * j + lane, cls = map[hnext];
        const bool active = i < n;
        hnext = i + 32u < n ? blk[(size_t)(i + 32u) * 18u + 1u] : 0u;
        const uint4 ra = nib_rows[2u * cls], rb = nib_rows[2u * cls + 1u];
        const uint16_t *r16 = (const uint16_t *)(nib_rows + 2u * cls);
        uint32_t w[4] = {0u, 0u, 0u, 0u};
        const bool tail = active && wnr4_tail_skip(lane, j, 31u, n);
        const uint32_t kend = tail ? 28u : 32u;   /* tail lanes stop at 28 */
#pragma unroll
        for (uint32_t k = 0; k < 32u; k++) {
            const bool coded = active && k < kend;
            if (coded) {
                const uint32_t slot = x & (WNR4_TOTAL - 1u), y = wnr4_row_sym(ra, rb, slot);
                const uint32_t c = r16[y], f = (y == 15u ? WNR4_TOTAL : (uint32_t)r16[y + 1u]) - c;
                x = f * (x >> WNR4_PREC) + slot - c;
                w[k / 8u] |= y << ((k % 8u) * 4u);
            }
            ok = wnr4_renorm_win(x, coded, words, ptr, job.nwords, lane, win) && ok;   /* no break: convergent loop */
        }
        if (active && ok) {
            unsigned char *cell = blk + (size_t)i * 18u + 2u;
            if (tail) { pw[0] = w[0]; pw[1] = w[1]; pw[2] = w[2]; pw[3] = w[3]; pcell = cell; }
            else {
                hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w[0]), w[1]), w[2]), w[3]);
#pragma unroll
                for (uint32_t q = 0; q < 16u; q++) cell[q] = (unsigned char)(w[q / 4u] >> (8u * (q % 4u)));
            }
        }
    }
    if (!__all_sync(0xffffffffu, ok)) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }
    if (pcell) {   /* nibbles 28..31 of the lane's last cell are its final state - L */
        const uint32_t tail = x - WNR4_L;
        if (tail > 0xffffu) err |= WNR4_ERR_STRUCT;
        pw[3] |= tail << 16;
        x = WNR4_L;
        hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, pw[0]), pw[1]), pw[2]), pw[3]);
#pragma unroll
        for (uint32_t q = 0; q < 16u; q++) pcell[q] = (unsigned char)(pw[q / 4u] >> (8u * (q % 4u)));
    }
    const uint32_t lc = wnr4_lane_cells(n, lane);
    uint32_t H = wnr4_h32_final(hash, wnr4_gg_hlen(lc, lc));
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);
    if (lane == 0 && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1 && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (lane == 0 && ptr != job.nwords) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + job.rec, err);
}

/* verbatim slices: blockIdx.y = slice, bytes in grid stride */
static __global__ void wnr4_gg_copy_kernel(const Wnr4GgCudaCopy *cp) {
    const Wnr4GgCudaCopy c = cp[blockIdx.y];
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < c.bytes; i += (size_t)gridDim.x * blockDim.x)
        c.dst[i] = c.src[i];
}

/* ---- host side --------------------------------------------------------------- */
static inline size_t wnr4_gg_cuda_table_bytes(void) { return (size_t)WNR4_GG_SUBTYPES * WNR4_GG_CUDA_SUB_U4 * 16u; }
/* device image of a model, per subtype: two-level rows (cdf[0], cdf[16], .., cdf[240], then cdf[0..255]) of
 * the hi table and the 32 lo tables, cdf[0..15] of the 32 nibble tables, the class map */
static inline void wnr4_gg_cuda_table_image(const Wnr4GgModel *m, unsigned char *img) {
    uint32_t s, c, q;
    for (s = 0; s < WNR4_GG_SUBTYPES; s++) {
        uint16_t *o = (uint16_t *)(img + (size_t)s * WNR4_GG_CUDA_SUB_U4 * 16u);
        for (c = 0; c <= WNR4_GG_CLASSES; c++) {   /* c = 0: hi; 1..32: lo of class c - 1 */
            const uint16_t *cdf = c ? m->lo + ((size_t)s * WNR4_GG_CLASSES + c - 1u) * 257u : m->hi + (size_t)s * 257u;
            for (q = 0; q < 16u; q++) o[q] = cdf[16u * q];
            for (q = 0; q < 256u; q++) o[16u + q] = cdf[q];
            o += 272;
        }
        for (c = 0; c < WNR4_GG_CLASSES; c++, o += 16)
            memcpy(o, m->nib + ((size_t)s * WNR4_GG_CLASSES + c) * 17u, 32u);
        memcpy(o, m->map[s], 256u);
    }
}

/* Jobs and verbatim copies of one record (host copy rec_host, device copy rec_dev of the same bytes), decoded
 * into dst (device, g->raw bytes); jobs appended per subtype to by_sub[2]. Returns 0 or -1 (invalid record:
 * nothing appended). */
static inline int wnr4_gg_cuda_plan(const Wnr4GgGeo *g, const Wnr4GgModel *model, const unsigned char *rec_host,
                                    size_t total, uint32_t owner, const unsigned char *rec_dev, uint32_t index,
                                    unsigned char *dst, std::vector<Wnr4GgCudaJob> by_sub[2],
                                    std::vector<Wnr4GgCudaCopy> &copies) {
    Wnr4GgRecord r;
    std::vector<Wnr4GgCudaJob> mine;
    uint32_t t, s;
    if (wnr4_gg_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner) return -1;
    const size_t data_at = (size_t)(r.data - rec_host), raw_at = (size_t)(r.raw - rec_host);
    for (t = 0; t < r.ntiles; t++) {   /* the offsets are read again here: check them again */
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        Wnr4GgCudaJob j;
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r.payload) return -1;
        wnr4_gg_tile_cells(r.tile_base, g->cells, g->nseg, r.G, t, &s, &j.c0, &j.n);
        if (g->sub[s] < 0) return -1;
        j.tile = rec_dev + data_at + o0; j.out = dst + g->off[s];
        j.nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u; j.rec = index; j.t = t;
        j.sub = (uint32_t)g->sub[s]; j.type = g->type[s]; j.owner = r.owner; j.G = r.G;
        mine.push_back(j);
    }
    for (auto &j : mine) by_sub[j.sub].push_back(j);
    size_t at = 0;
    for (s = 0; s < g->nseg; s++)
        if (g->sub[s] < 0) { copies.push_back({rec_dev + raw_at + at, dst + g->off[s], g->bytes[s]}); at += g->bytes[s]; }
    return 0;
}
/* Concatenates the per-subtype lists, each padded to a multiple of the block size (padding jobs have n = 0).
 * Returns the block size as wnr4_cuda_jobs: wnr4_cuda_warps, or by the number of tiles (4, 8 or 16 warps). */
static inline uint32_t wnr4_gg_cuda_jobs(std::vector<Wnr4GgCudaJob> by_sub[2], std::vector<Wnr4GgCudaJob> &jobs) {
    const size_t tiles = by_sub[0].size() + by_sub[1].size();
    const uint32_t W = wnr4_cuda_warps ? wnr4_cuda_warps : tiles < 1000u ? 4u : tiles < 2000u ? 8u : 16u;
    jobs.clear();
    for (uint32_t s = 0; s < WNR4_GG_SUBTYPES; s++) {
        for (auto &j : by_sub[s]) jobs.push_back(j);
        Wnr4GgCudaJob pad = {};
        pad.sub = s;
        while (jobs.size() % W) jobs.push_back(pad);
    }
    return W;
}
static inline cudaError_t wnr4_gg_cuda_launch(const unsigned char *tab_dev, uint32_t model_id, const Wnr4GgCudaJob *jobs_dev,
                                              uint32_t njobs, uint32_t W, uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t blocks = (njobs + W - 1u) / W;
    const uint4 *tab = (const uint4 *)tab_dev;
    if (!blocks) return cudaSuccess;
    if (W == 4u) wnr4_gg_cuda_kernel<4u><<<blocks, 128u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 8u) wnr4_gg_cuda_kernel<8u><<<blocks, 256u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else if (W == 16u) wnr4_gg_cuda_kernel<16u><<<blocks, 512u, 0, stream>>>(jobs_dev, njobs, tab, model_id, errors_dev);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}
/* the verbatim slices of a batch (copies_dev: ncopies entries, max_bytes = the largest) in one launch */
static inline cudaError_t wnr4_gg_cuda_copy(const Wnr4GgCudaCopy *copies_dev, uint32_t ncopies, size_t max_bytes,
                                            cudaStream_t stream) {
    if (!ncopies || !max_bytes) return cudaSuccess;
    if (ncopies > 65535u) return cudaErrorInvalidValue;
    const size_t bx = (max_bytes + 4095u) / 4096u;
    wnr4_gg_copy_kernel<<<dim3((unsigned)(bx < 256u ? bx : 256u), ncopies), 256u, 0, stream>>>(copies_dev);
    return cudaGetLastError();
}

#endif
