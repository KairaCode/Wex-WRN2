/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FAST_DECODE_H
#define WEX_FAST_DECODE_H
/* Fast CPU decoder for WNR3 records. NO format change: same bytes in, same bytes out, same accept/reject
 * as the reference wex_nr2_decode (c/wex_neural_rans.h).
 *
 *   int wex_fast_decode(view, layout, model, scales_out, weights_out);            tables built per call
 *   wex_fast_model_init(&tables, model);                                          once per model (224 KiB)
 *   int wex_fast_decode_with(&tables, view, layout, model, scales_out, weights_out, &options);
 *
 * Contract (the reference's): `view` comes from wex_nr3_parse with this `layout`, `model` from
 * wex_nr2_model_open. Returns 0 or -1; outputs may be partially written on -1. Views that are not WNR3 go to
 * the reference decoder unchanged.
 *
 * What is different from the reference:
 *  (a) streams are independent, so they run in parallel (OpenMP, compile with -fopenmp): the u16 lengths are
 *      prefix-summed once into at most WEXF_MAX_TASKS task offsets, no allocation;
 *  (b) symbol search without branches. Weights (kernel 1): each (matrix, srel, maximum, previous) context has
 *      a 128-byte table entry {upper[16], start[16], freq[16], next[16]}; one AVX2 compare of the slot with
 *      upper[] + tzcnt gives the symbol (NEON: two 8-lane compares narrowed to a 64-bit mask + ctz), next[]
 *      gives the following context (no arithmetic on previous / maximum). Scales: a 16384-entry slot ->
 *      symbol table per matrix;
 *  (c) N streams are decoded in one loop (kernel 1) so the long per-symbol dependency chain of one stream
 *      overlaps with the others';
 *  (d) kernel 2 decodes 8 streams per AVX2 vector with gathers (binary search over the 16 CDF entries of the
 *      model blob). AVX2 only: NEON has no gather load, a request for kernel 2 runs kernel 1 with 4 streams;
 *  (e) fuse = 1 computes the MurmurHash3 of each stream and undoes the ceil form inside the decode loop.
 *
 * Renormalisation is branchless and reads 4 bytes ahead; a stream that runs past its end is noticed at the
 * next 32-symbol boundary instead of at the byte. This is equivalent to the reference check: the read
 * pointer only grows, so "needs a byte at end" in the reference <=> "pointer beyond end" here, and both then
 * fail. Streams closer than WEXF_SLACK bytes to the end of the weight payload take the checked per-byte
 * loop, so nothing is ever read at or beyond that end. The state stays in [2^23, 2^31) for any input bytes
 * (valid model: every frequency >= 1), hence at most 2 renormalisation bytes per symbol; the code is
 * memory-safe (3 bytes per symbol at most) even if that did not hold.
 *
 * Build: -O3 -march=haswell (AVX2, BMI, BMI2, LZCNT) selects the AVX2 kernels; a little-endian AArch64 target
 * selects the NEON kernel 1 (NEON is part of the base AArch64 instruction set, no -march needed); on any other
 * target, or with -DWEX_FAST_SCALAR, only kernel 0 exists (the reference loop per stream, still parallel
 * across streams). WEXF_SIMD is 1 (AVX2), 2 (NEON) or 0; WEXF_KERNELS names the choice. The SIMD paths
 * assume a little-endian target. */
#include "wex_format.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__BMI__) && defined(__BMI2__) && defined(__LZCNT__) && !defined(WEX_FAST_SCALAR)
#include <immintrin.h>
#define WEXF_SIMD 1
#define WEXF_KERNELS "AVX2"
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__BYTE_ORDER__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && !defined(WEX_FAST_SCALAR)
#include <arm_neon.h>
#define WEXF_SIMD 2
#define WEXF_KERNELS "NEON"
#else
#define WEXF_SIMD 0
#define WEXF_KERNELS "scalar"
#endif

#define WEXF_MAX_TASKS 256u
#define WEXF_SLACK 104u         /* > 3 bytes x 32 symbols + a 4-byte read */
#define WEXF_ENTRY 128u         /* bytes per weight context: u16 upper[16], start[16], freq[16], next[16] */
#define WEXF_SREL_BYTES (65u * WEXF_ENTRY)   /* entries maximum * 8 + previous (0..63) and 64 = block start */
#define WEXF_CTX_BYTES (3u * 7u * WEXF_SREL_BYTES)
#define WEXF_INLINE static inline __attribute__((always_inline))

typedef struct {
    int threads;   /* OpenMP threads, 0 = the runtime's default; ignored without OpenMP */
    int kernel;    /* 0 checked scalar loop, 1 AVX2 or NEON search + interleaved streams, 2 AVX2 gathers */
    int lanes;     /* kernel 1: streams per loop (1, 2, 3, 4, 6, 8); kernel 2: vectors of 8 streams per loop (1, 2, 3) */
    int fuse;      /* kernels 1, 2: hash and ceil inverse inside the decode loop */
} WexFastOptions;

/* Decode tables of one model (AVX2 and NEON kernels). Read-only after wex_fast_model_init, shareable between threads.
 * The entries start 64-byte aligned inside `raw`; a byte copy of the struct still decodes correctly (unaligned). */
typedef struct {
    uint32_t id, skew;                         /* model id, offset of the first entry in raw */
    unsigned char lut[3u * WEX_NR2_TOTAL];     /* slot -> scale symbol */
    unsigned char raw[WEXF_CTX_BYTES + 64u];
} WexFastModel;

typedef struct {
    const WexNr2Record *v;
    const WexNr2Layout *layout;
    const WexNr2Model *model;
    unsigned char *const *scales, *const *weights;
    const unsigned char *hard_end;   /* end of the weight payload */
    const WexFastModel *fm;          /* NULL: kernel 0 only */
    WexFastOptions o;
    uint32_t cz, ceil;
} WexfJob;

typedef struct {
    const unsigned char *begin;
    uint32_t length, index, matrix, first, count;   /* index = record-wide stream number (hashed) */
} WexfStream;

WEXF_INLINE void wexf_ident(const WexfJob *J, uint32_t index, uint32_t ident[4]) {
    ident[0] = J->v->model_id; ident[1] = J->v->owner; ident[2] = index; ident[3] = J->v->word24;
}

/* Ceil inverse of the blocks of one stream (reference tables). */
static inline int wexf_ceil_blocks(unsigned char *q, unsigned char *scales, uint32_t blocks) {
    uint32_t b, k;
    for (b = 0; b < blocks; b++, q += 16) {
        if (wex_nr3_block_max(q, NULL) != 5u) continue;
        if (!scales[b]) return -1;
        for (k = 0; k < 16u; k++)
            q[k] = (unsigned char)((q[k] & 0x88u) | wex_nr3_double[q[k] & 7u] | wex_nr3_double[(q[k] >> 4) & 7u] << 4);
        scales[b]--;
    }
    return 0;
}

/* Kernel 0: one stream with the reference's loop and its per-byte end check. Portable C. */
static inline int wexf_stream_checked(const WexfJob *J, const WexfStream *s, int weights) {
    const WexNr2Model *model = J->model;
    const uint32_t m = s->matrix;
    const unsigned char *ptr, *end;
    uint32_t state, j, symbol, ident[4];
    if (wex_nr2_stream_open(s->begin, s->length, &ptr, &end, &state)) return -1;
    wexf_ident(J, s->index, ident);
    if (!weights) {
        const unsigned char *cdf = wex_nr2_scale_cdf(model, m);
        unsigned char *out = J->scales[m] + s->first;
        for (j = 0; j < s->count; j++) {
            if (wex_nr2_decode_symbol(cdf, 256u, &state, &ptr, end, &symbol)) return -1;
            out[j] = (unsigned char)symbol;
        }
        return ptr != end || state != wex_nr3_stream_x0(ident, out, s->count) ? -1 : 0;
    } else {
        unsigned char *out = J->weights[m] + s->first / 2u, *sc = J->scales[m] + s->first / 32u;
        uint32_t srel = 0, previous = 8u, maximum = 8u;
        for (j = 0; j < s->count; j++) {
            if (!(j % 32u)) {
                srel = wex_nr2_srel(sc[j / 32u], J->v->median[m]);
                previous = 8u; maximum = 8u;
            }
            if (wex_nr2_decode_symbol(wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum)),
                                      16u, &state, &ptr, end, &symbol) || (J->cz && symbol == 8u)) return -1;
            if (!(j & 1u)) out[j / 2u] = (unsigned char)symbol;
            else out[j / 2u] |= (unsigned char)(symbol << 4);
            previous = symbol & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
        if (ptr != end || state != wex_nr3_stream_x0(ident, out, s->count / 2u)) return -1;
        return J->ceil >> m & 1u ? wexf_ceil_blocks(out, sc, s->count / 32u) : 0;
    }
}

#if WEXF_SIMD
/* Branchless byte renormalisation with the next bytes at p: 0, 1 or 2 are used on a valid model (never more
 * than 3); *spare = 3 - bytes used. */
WEXF_INLINE uint32_t wexf_renorm(uint32_t x, const unsigned char *p, uint32_t *spare) {
#if WEXF_SIMD == 2
    const uint32_t keep = (0u - (x ? (uint32_t)__builtin_clz(x) : 32u)) & 24u;   /* one CLZ instruction */
#else
    const uint32_t keep = (0u - _lzcnt_u32(x)) & 24u;   /* leading zeros 1..8 -> 24, 9..16 -> 16, 17..24 -> 8 */
#endif
    uint32_t next;
    uint64_t wide;
    memcpy(&next, p, 4);
    *spare = keep >> 3;
    wide = ((uint64_t)x << 32 | __builtin_bswap32(next)) >> 8;
    return (uint32_t)(wide >> keep);
}

static inline void wex_fast_model_init(WexFastModel *f, const WexNr2Model *model) {
    uint32_t m, srel, e, symbol;
    f->id = model->id;
    f->skew = (uint32_t)(0u - (uintptr_t)f->raw) & 63u;
    memset(f->lut, 0, sizeof(f->lut));
    for (m = 0; m < 3u; m++)
        for (symbol = 0; symbol < 256u; symbol++) {
            const unsigned char *cdf = wex_nr2_scale_cdf(model, m) + symbol * 2u;
            const uint32_t lo = wex_nr_u16(cdf), hi = wex_nr_u16(cdf + 2u);
            if (lo < hi && hi <= WEX_NR2_TOTAL) memset(f->lut + (size_t)m * WEX_NR2_TOTAL + lo, (int)symbol, hi - lo);
        }
    for (m = 0; m < 3u; m++)
        for (srel = 0; srel < 7u; srel++)
            for (e = 0; e < 65u; e++) {
                const uint32_t previous = e == 64u ? 8u : e & 7u, maximum = e == 64u ? 8u : e >> 3;
                const unsigned char *cdf = wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum));
                uint16_t entry[64];
                for (symbol = 0; symbol < 16u; symbol++) {
                    const uint32_t a = symbol & 7u, mx = e == 64u || a > maximum ? a : maximum;
                    entry[symbol] = (uint16_t)wex_nr_u16(cdf + symbol * 2u + 2u);
                    entry[16u + symbol] = (uint16_t)wex_nr_u16(cdf + symbol * 2u);
                    entry[32u + symbol] = (uint16_t)(entry[symbol] - entry[16u + symbol]);
                    entry[48u + symbol] = (uint16_t)((mx * 8u + a) * (WEXF_ENTRY / 8u));   /* next entry, in 8-byte units */
                }
                entry[15] = 0x7fff;   /* cdf[16] = 16384 on a valid model: the search always stops by symbol 15 */
                memcpy(f->raw + f->skew + ((size_t)(m * 7u + srel) * 65u + e) * WEXF_ENTRY, entry, WEXF_ENTRY);
            }
}

/* D(a) = a + min(a, 2) on the magnitudes of 16 nibbles that are all <= 5; signs kept. */
WEXF_INLINE uint64_t wexf_double16(uint64_t w) {
    const uint64_t one = UINT64_C(0x1111111111111111), mag = w & 7u * one;
    const uint64_t ge2 = (mag >> 1 | mag >> 2) & one;
    return (w & 8u * one) | (mag + (ge2 << 1) + (mag & one & ~ge2));
}
/* Non-zero iff some nibble of w is 8. */
WEXF_INLINE uint64_t wexf_has8(uint64_t w) {
    const uint64_t one = UINT64_C(0x1111111111111111), t = w ^ 8u * one;
    return (t - one) & ~t & 8u * one;
}
WEXF_INLINE int wexf_ceil_fused(unsigned char *q, unsigned char *scale) {
    uint64_t w[2];
    if (!*scale) return -1;
    memcpy(w, q, 16);
    w[0] = wexf_double16(w[0]); w[1] = wexf_double16(w[1]);
    memcpy(q, w, 16);
    --*scale;
    return 0;
}

/* Kernel 1, weights: N streams of one matrix with the same symbol count (a multiple of 32), one loop. */
WEXF_INLINE int wexf_weights_n(const WexfJob *J, const WexfStream *s, const uint32_t N, const int fuse) {
    const uint32_t m = s[0].matrix, blocks = s[0].count / 32u, median = J->v->median[m], ceil = J->ceil >> m & 1u;
    const unsigned char *table = J->fm->raw + J->fm->skew + (size_t)m * (7u * WEXF_SREL_BYTES);
#if WEXF_SIMD == 1
    const uint64_t nibbles = UINT64_C(0x1e1e1e1e1e1e1e1e);   /* 2 x symbol in each byte of s2 */
#endif
    const unsigned char *p, *e[8], *base[8], *ctx[8];
    unsigned char *out[8], *sc[8], s2[8][16];
    uintptr_t q[8];   /* read position of a stream minus 3 x (symbols done in the current half block) */
    uint32_t x[8], h[8], ident[4], k, b, j, half;
    uint64_t bad = 0;
    for (k = 0; k < N; k++) {
        if (wex_nr2_stream_open(s[k].begin, s[k].length, &p, &e[k], &x[k])) return -1;
        q[k] = (uintptr_t)p;
        out[k] = J->weights[m] + s[k].first / 2u; sc[k] = J->scales[m] + s[k].first / 32u;
        wexf_ident(J, s[k].index, ident);
        h[k] = wex_nr3_ident_hash(ident);
    }
    for (b = 0; b < blocks; b++) {
        for (k = 0; k < N; k++) {
            if (q[k] > (uintptr_t)e[k]) return -1;
            base[k] = table + (size_t)wex_nr2_srel(sc[k][b], median) * WEXF_SREL_BYTES;
            ctx[k] = base[k] + 64u * WEXF_ENTRY;   /* block start: previous = 8, maximum = 8 */
        }
        for (half = 0; half < 2u; half++) {
            for (j = 0; j < 16u; j++)
                for (k = 0; k < N; k++) {
                    const uint32_t slot = x[k] & (WEX_NR2_TOTAL - 1u);
#if WEXF_SIMD == 2
                    /* the same signed compare as AVX2 on the 16 upper[] values, in two halves; one byte of each
                     * 16-bit lane is kept (uzp1), then 4 bits of each byte (shrn): bit 4 x symbol is the lowest
                     * set bit of the mask, which is never 0 because upper[15] = 0x7fff > slot */
                    const int16x8_t slots = vdupq_n_s16((int16_t)slot);
                    const uint8x16_t gt_lo = vreinterpretq_u8_u16(vcgtq_s16(vreinterpretq_s16_u8(vld1q_u8(ctx[k])), slots));
                    const uint8x16_t gt_hi = vreinterpretq_u8_u16(vcgtq_s16(vreinterpretq_s16_u8(vld1q_u8(ctx[k] + 16)), slots));
                    const uint64_t found = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vuzp1q_u8(gt_lo, gt_hi)), 4)), 0);
                    const uint32_t t = (uint32_t)__builtin_ctzll(found) >> 1;
#else
                    const __m256i upper = _mm256_loadu_si256((const __m256i *)(const void *)ctx[k]);
                    const uint32_t t = _tzcnt_u32((uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi16(upper, _mm256_set1_epi16((short)slot))));
#endif
                    uint16_t start, freq, next;   /* t = 2 x symbol <= 30 */
                    uint32_t spare;
                    memcpy(&start, ctx[k] + 32u + t, 2); memcpy(&freq, ctx[k] + 64u + t, 2); memcpy(&next, ctx[k] + 96u + t, 2);
                    s2[k][j] = (unsigned char)t;
                    x[k] = wexf_renorm(freq * (x[k] >> WEX_NR2_PRECISION) + (slot - start), (const unsigned char *)(q[k] + 3u * j), &spare);
                    q[k] -= spare;
                    ctx[k] = base[k] + 8u * (size_t)next;
                }
            for (k = 0; k < N; k++) {
#if WEXF_SIMD == 2
                /* a symbol per byte, then the two symbols of each 16-bit lane into its low byte: a + (a >> 4) */
                const uint16x8_t pair = vreinterpretq_u16_u8(vshrq_n_u8(vld1q_u8(s2[k]), 1));
                const uint64_t acc = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vsraq_n_u16(pair, pair, 4))), 0);
#else
                uint64_t lo, hi, acc;
                memcpy(&lo, s2[k], 8); memcpy(&hi, s2[k] + 8, 8);
                acc = _pext_u64(lo, nibbles) | _pext_u64(hi, nibbles) << 32;
#endif
                memcpy(out[k] + (size_t)b * 16u + half * 8u, &acc, 8);
                bad |= wexf_has8(acc);
                if (fuse) h[k] = wex_nr3_h32_block(wex_nr3_h32_block(h[k], (uint32_t)acc), (uint32_t)(acc >> 32));
                q[k] += 48u;
            }
        }
        if (fuse && ceil)
            for (k = 0; k < N; k++)   /* entry index = maximum * 8 + previous */
                if ((size_t)(ctx[k] - base[k]) / (8u * WEXF_ENTRY) == 5u && wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b)) return -1;
    }
    if (bad && J->cz) return -1;
    for (k = 0; k < N; k++) {
        uint32_t x0;
        if (fuse) x0 = WEX_NR_L + (wex_nr3_h32_final(h[k], 16u + blocks * 16u) & WEX_NR3_HASH_MASK);
        else { wexf_ident(J, s[k].index, ident); x0 = wex_nr3_stream_x0(ident, out[k], blocks * 16u); }
        if (q[k] != (uintptr_t)e[k] || x[k] != x0) return -1;
        if (!fuse && ceil && wexf_ceil_blocks(out[k], sc[k], blocks)) return -1;
    }
    return 0;
}

/* Kernel 1, scales: N streams of one matrix with the same count; symbol = table[slot]. */
WEXF_INLINE int wexf_scales_n(const WexfJob *J, const WexfStream *s, const uint32_t N) {
    const uint32_t m = s[0].matrix, count = s[0].count;
    const unsigned char *cdf = wex_nr2_scale_cdf(J->model, m), *lut = J->fm->lut + (size_t)m * WEX_NR2_TOTAL;
    const unsigned char *p[8], *e[8];
    unsigned char *out[8];
    uint32_t x[8], ident[4], k, j;
    for (k = 0; k < N; k++) {
        if (wex_nr2_stream_open(s[k].begin, s[k].length, &p[k], &e[k], &x[k])) return -1;
        out[k] = J->scales[m] + s[k].first;
    }
    for (j = 0; j < count; j++) {
        if (!(j & 31u))
            for (k = 0; k < N; k++) if (p[k] > e[k]) return -1;
        for (k = 0; k < N; k++) {
            const uint32_t slot = x[k] & (WEX_NR2_TOTAL - 1u), symbol = lut[slot];
            uint32_t sf, spare;   /* sf = cdf[symbol] | cdf[symbol + 1] << 16 */
            memcpy(&sf, cdf + 2u * symbol, 4);
            x[k] = wexf_renorm(((sf >> 16) - (sf & 0xffffu)) * (x[k] >> WEX_NR2_PRECISION) + (slot - (sf & 0xffffu)), p[k], &spare);
            p[k] += 3u - spare;
            out[k][j] = (unsigned char)symbol;
        }
    }
    for (k = 0; k < N; k++) {
        wexf_ident(J, s[k].index, ident);
        if (p[k] != e[k] || x[k] != wex_nr3_stream_x0(ident, out[k], count)) return -1;
    }
    return 0;
}

#if WEXF_SIMD == 1
/* Kernel 2, weights: G vectors of 8 streams (one matrix, same count). Per symbol: 4 gathers binary-search the
 * 16-entry CDF and leave cdf[symbol] / cdf[symbol + 1] in start / end, 1 gather fetches the next bytes. */
WEXF_INLINE int wexf_weights_g8(const WexfJob *J, const WexfStream *s, const uint32_t G, const int fuse) {
    const uint32_t m = s[0].matrix, blocks = s[0].count / 32u, median = J->v->median[m], ceil = J->ceil >> m & 1u;
    const unsigned char *cdf = wex_nr2_weight_cdf(J->model, m, 0), *pay = s[0].begin;
    const __m256i low16 = _mm256_set1_epi32(0xffff), low14 = _mm256_set1_epi32((int)WEX_NR2_TOTAL - 1);
    const __m256i be16 = _mm256_setr_epi8(1, 0, -1, -1, 5, 4, -1, -1, 9, 8, -1, -1, 13, 12, -1, -1,
                                          1, 0, -1, -1, 5, 4, -1, -1, 9, 8, -1, -1, 13, 12, -1, -1);
    const __m256i c1 = _mm256_set1_epi32((int)UINT32_C(0xcc9e2d51)), c2 = _mm256_set1_epi32(0x1b873593);
    const __m256i ones = _mm256_set1_epi32(0x11111111), eights = _mm256_set1_epi32((int)UINT32_C(0x88888888));
    __m256i x[3], p[3], e[3], h[3], base[3], ctx[3], mx[3], acc[3], bad = _mm256_setzero_si256();
    unsigned char *out[24], *sc[24];
    uint32_t t[8], ident[4], g, k, b, w, j;
    for (g = 0; g < G; g++) {
        uint32_t st[8], pp[8], ee[8], hh[8];
        for (k = 0; k < 8u; k++) {
            const WexfStream *q = s + g * 8u + k;
            const unsigned char *ptr, *end;
            if (wex_nr2_stream_open(q->begin, q->length, &ptr, &end, &st[k])) return -1;
            pp[k] = (uint32_t)(ptr - pay); ee[k] = (uint32_t)(end - pay);
            out[g * 8u + k] = J->weights[m] + q->first / 2u; sc[g * 8u + k] = J->scales[m] + q->first / 32u;
            wexf_ident(J, q->index, ident);
            hh[k] = wex_nr3_ident_hash(ident);
        }
        x[g] = _mm256_loadu_si256((const __m256i *)(const void *)st); p[g] = _mm256_loadu_si256((const __m256i *)(const void *)pp);
        e[g] = _mm256_loadu_si256((const __m256i *)(const void *)ee); h[g] = _mm256_loadu_si256((const __m256i *)(const void *)hh);
    }
    for (b = 0; b < blocks; b++) {
        for (g = 0; g < G; g++) {
            if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(p[g], e[g]))) return -1;
            for (k = 0; k < 8u; k++) t[k] = wex_nr2_srel(sc[g * 8u + k][b], median) * (81u * 34u);
            base[g] = _mm256_loadu_si256((const __m256i *)(const void *)t);
            ctx[g] = _mm256_add_epi32(base[g], _mm256_set1_epi32(80 * 34));
            mx[g] = _mm256_setzero_si256();
        }
        for (w = 0; w < 4u; w++) {
            for (g = 0; g < G; g++) acc[g] = _mm256_setzero_si256();
            for (j = 0; j < 8u; j++)
                for (g = 0; g < G; g++) {
                    const __m256i slot = _mm256_and_si256(x[g], low14);
                    __m256i lo = _mm256_setzero_si256(), start = lo, end = _mm256_set1_epi32((int)WEX_NR2_TOTAL);
                    __m256i n, bytes, shift, a, c;
                    uint32_t step;
                    for (step = 8u; step; step >>= 1) {
                        const __m256i mid = _mm256_add_epi32(lo, _mm256_set1_epi32((int)step));
                        const __m256i value = _mm256_and_si256(low16, _mm256_i32gather_epi32((const int *)(const void *)cdf,
                                                               _mm256_add_epi32(ctx[g], _mm256_add_epi32(mid, mid)), 1));
                        const __m256i gt = _mm256_cmpgt_epi32(value, slot);
                        lo = _mm256_blendv_epi8(mid, lo, gt);
                        start = _mm256_blendv_epi8(value, start, gt);
                        end = _mm256_blendv_epi8(end, value, gt);
                    }
                    x[g] = _mm256_sub_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(end, start),
                                            _mm256_srli_epi32(x[g], WEX_NR2_PRECISION)), slot), start);
                    /* n = (x < 2^23) + (x < 2^15); x < 2^31, so signed compares are right */
                    n = _mm256_sub_epi32(_mm256_setzero_si256(),
                            _mm256_add_epi32(_mm256_cmpgt_epi32(_mm256_set1_epi32((int)WEX_NR_L), x[g]),
                                             _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(WEX_NR_L >> 8)), x[g])));
                    bytes = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int *)(const void *)pay, p[g], 1), be16);
                    shift = _mm256_slli_epi32(n, 3);
                    x[g] = _mm256_or_si256(_mm256_sllv_epi32(x[g], shift),
                                           _mm256_srlv_epi32(bytes, _mm256_sub_epi32(_mm256_set1_epi32(16), shift)));
                    p[g] = _mm256_add_epi32(p[g], n);
                    acc[g] = _mm256_or_si256(acc[g], _mm256_sll_epi32(lo, _mm_cvtsi32_si128((int)(4u * j))));
                    a = _mm256_and_si256(lo, _mm256_set1_epi32(7));
                    mx[g] = _mm256_max_epu32(mx[g], a);
                    c = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(a, 3), a), mx[g]);       /* a * 9 + mx */
                    ctx[g] = _mm256_add_epi32(base[g], _mm256_add_epi32(_mm256_slli_epi32(c, 5), _mm256_add_epi32(c, c)));
                }
            for (g = 0; g < G; g++) {
                const __m256i z = _mm256_xor_si256(acc[g], eights);   /* a zero nibble of z = a nibble 8 */
                bad = _mm256_or_si256(bad, _mm256_and_si256(_mm256_andnot_si256(z, _mm256_sub_epi32(z, ones)), eights));
                _mm256_storeu_si256((__m256i *)(void *)t, acc[g]);
                for (k = 0; k < 8u; k++) memcpy(out[g * 8u + k] + (size_t)b * 16u + w * 4u, &t[k], 4);
                if (fuse) {   /* wex_nr3_h32_block on 8 streams */
                    __m256i q = _mm256_mullo_epi32(acc[g], c1);
                    q = _mm256_mullo_epi32(_mm256_or_si256(_mm256_slli_epi32(q, 15), _mm256_srli_epi32(q, 17)), c2);
                    q = _mm256_xor_si256(h[g], q);
                    q = _mm256_or_si256(_mm256_slli_epi32(q, 13), _mm256_srli_epi32(q, 19));
                    h[g] = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(q, 2), q), _mm256_set1_epi32((int)UINT32_C(0xe6546b64)));
                }
            }
        }
        if (fuse && ceil)
            for (g = 0; g < G; g++) {
                uint32_t five = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(mx[g], _mm256_set1_epi32(5))));
                for (; five; five &= five - 1u) {
                    k = g * 8u + _tzcnt_u32(five);
                    if (wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b)) return -1;
                }
            }
    }
    if (J->cz && !_mm256_testz_si256(bad, bad)) return -1;
    for (g = 0; g < G; g++) {
        uint32_t st[8], pp[8], ee[8], hh[8];
        _mm256_storeu_si256((__m256i *)(void *)st, x[g]); _mm256_storeu_si256((__m256i *)(void *)pp, p[g]);
        _mm256_storeu_si256((__m256i *)(void *)ee, e[g]); _mm256_storeu_si256((__m256i *)(void *)hh, h[g]);
        for (k = 0; k < 8u; k++) {
            uint32_t x0;
            if (fuse) x0 = WEX_NR_L + (wex_nr3_h32_final(hh[k], 16u + blocks * 16u) & WEX_NR3_HASH_MASK);
            else { wexf_ident(J, s[g * 8u + k].index, ident); x0 = wex_nr3_stream_x0(ident, out[g * 8u + k], blocks * 16u); }
            if (pp[k] != ee[k] || st[k] != x0) return -1;
            if (!fuse && ceil && wexf_ceil_blocks(out[g * 8u + k], sc[g * 8u + k], blocks)) return -1;
        }
    }
    return 0;
}
#endif

/* Out-of-line instances: the group size and the fuse flag are compile-time constants inside each. */
static int wexf_group(const WexfJob *J, const WexfStream *s, int weights, uint32_t n) {
    const int fuse = J->o.fuse;
    if (!weights) switch (n) {
        case 1: return wexf_scales_n(J, s, 1);
        case 2: return wexf_scales_n(J, s, 2);
        case 3: return wexf_scales_n(J, s, 3);
        case 4: return wexf_scales_n(J, s, 4);
        case 6: return wexf_scales_n(J, s, 6);
        default: return wexf_scales_n(J, s, 8);
    }
#if WEXF_SIMD == 1
    if (J->o.kernel == 2 && n >= 8u) switch (n * 2u + (uint32_t)fuse) {
        case 16: return wexf_weights_g8(J, s, 1, 0);
        case 17: return wexf_weights_g8(J, s, 1, 1);
        case 32: return wexf_weights_g8(J, s, 2, 0);
        case 33: return wexf_weights_g8(J, s, 2, 1);
        case 48: return wexf_weights_g8(J, s, 3, 0);
        default: return wexf_weights_g8(J, s, 3, 1);
    }
#endif
    switch (n * 2u + (uint32_t)fuse) {
        case 2: return wexf_weights_n(J, s, 1, 0);
        case 3: return wexf_weights_n(J, s, 1, 1);
        case 4: return wexf_weights_n(J, s, 2, 0);
        case 5: return wexf_weights_n(J, s, 2, 1);
        case 6: return wexf_weights_n(J, s, 3, 0);
        case 7: return wexf_weights_n(J, s, 3, 1);
        case 8: return wexf_weights_n(J, s, 4, 0);
        case 9: return wexf_weights_n(J, s, 4, 1);
        case 12: return wexf_weights_n(J, s, 6, 0);
        case 13: return wexf_weights_n(J, s, 6, 1);
        case 16: return wexf_weights_n(J, s, 8, 0);
        default: return wexf_weights_n(J, s, 8, 1);
    }
}
#endif

/* Streams [s0, s1) of a section (all scale streams or all weight streams); stream s0 starts at `at`. */
static int wexf_run(const WexfJob *J, int weights, uint32_t s0, uint32_t s1, const unsigned char *at) {
    const uint32_t *total = weights ? J->layout->symbol_count : J->layout->scale_count;
    const uint32_t chunk = weights ? J->v->weight_chunk : J->v->scale_chunk;
    const uint32_t index0 = weights ? J->v->scale_streams : 0u;
    const uint32_t n0 = wex_nr2_streams(total[0], chunk), n1 = n0 + wex_nr2_streams(total[1], chunk);
    uint32_t want = 1u;
#if WEXF_SIMD
    if (J->o.kernel == 1) want = (uint32_t)J->o.lanes;
    if (J->o.kernel == 2) want = weights ? 8u * (uint32_t)J->o.lanes : 8u;
#endif
    while (s0 < s1) {
        WexfStream g[24];
        const unsigned char *q = at;
        uint32_t n, fast = 0;
        for (n = 0; n < want && s0 + n < s1; n++) {
            const uint32_t t = s0 + n, m = t < n0 ? 0u : t < n1 ? 1u : 2u, local = t - (m == 0u ? 0u : m == 1u ? n0 : n1);
            WexfStream *s = g + n;
            s->begin = q; s->index = index0 + t; s->matrix = m; s->first = local * chunk;
            s->length = wex_nr_u16(J->v->lengths + (size_t)s->index * 2u);
            s->count = total[m] - s->first < chunk ? total[m] - s->first : chunk;
            if ((size_t)(J->hard_end - q) < s->length) return -1;   /* cannot happen on a parsed record */
            fast = (size_t)(J->hard_end - q) >= (size_t)s->length + WEXF_SLACK && !(weights && s->count % 32u);
            if (!fast || m != g[0].matrix || s->count != g[0].count) break;
            q += s->length;
        }
#if WEXF_SIMD
        if (J->o.kernel && n == want) {
            if (wexf_group(J, g, weights, n)) return -1;
            s0 += n; at = q;
            continue;
        }
        if (J->o.kernel && n) {   /* stream s0 alone is fast */
            if (wexf_group(J, g, weights, 1u)) return -1;
            s0++; at += g[0].length;
            continue;
        }
#endif
        (void)fast;
        if (wexf_stream_checked(J, g, weights)) return -1;
        s0++; at += g[0].length;
    }
    return 0;
}

static int wexf_section(const WexfJob *J, int weights) {
    const uint32_t *total = weights ? J->layout->symbol_count : J->layout->scale_count;
    const uint32_t chunk = weights ? J->v->weight_chunk : J->v->scale_chunk;
    const uint32_t index0 = weights ? J->v->scale_streams : 0u;
    const uint64_t streams = (uint64_t)wex_nr2_streams(total[0], chunk) + wex_nr2_streams(total[1], chunk) +
                             wex_nr2_streams(total[2], chunk);
    /* tasks of `per` streams: a multiple of 48 so that full groups of any kernel fit, at most WEXF_MAX_TASKS */
    const uint64_t per = ((streams + WEXF_MAX_TASKS - 1u) / WEXF_MAX_TASKS + 47u) / 48u * 48u;
    const int tasks = per ? (int)((streams + per - 1u) / per) : 0;
    const unsigned char *payload = weights ? J->v->weight_payload : J->v->scale_payload;
    uint64_t offset[WEXF_MAX_TASKS], at = 0, s = 0;
    int bad = 0, t;
    for (t = 0; t < tasks; t++) {   /* prefix sum of the u16 lengths, one offset per task */
        const uint64_t stop = s + per < streams ? s + per : streams;
        offset[t] = at;
        for (; s < stop; s++) at += wex_nr_u16(J->v->lengths + (size_t)(index0 + s) * 2u);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:bad) num_threads(J->o.threads > 0 ? J->o.threads : omp_get_max_threads())
#endif
    for (t = 0; t < tasks; t++) {
        const uint64_t s0 = (uint64_t)t * per, s1 = s0 + per < streams ? s0 + per : streams;
        bad |= wexf_run(J, weights, (uint32_t)s0, (uint32_t)s1, payload + offset[t]) != 0;
    }
    return bad ? -1 : 0;
}

/* tables: from wex_fast_model_init for this model, or NULL (kernel 0). */
static inline int wex_fast_decode_with(const WexFastModel *tables, const WexNr2Record *v, const WexNr2Layout *layout,
                                       const WexNr2Model *model, unsigned char *const scales_out[3],
                                       unsigned char *const weights_out[3], const WexFastOptions *options) {
    WexfJob J;
    if (v->magic != WEX_NR3_MAGIC) return wex_nr2_decode(v, layout, model, scales_out, weights_out);
    if (v->model_id != model->id) return -1;
    J.v = v; J.layout = layout; J.model = model; J.scales = scales_out; J.weights = weights_out;
    J.hard_end = v->weight_payload + v->weight_payload_bytes;
    J.o = *options;
    J.cz = v->flags & WEX_NR3_FLAG_CZ; J.ceil = v->flags & WEX_NR3_FLAG_CEIL_MASK;
    J.fm = tables;
#if WEXF_SIMD
    if (!tables || tables->id != model->id) J.o.kernel = 0;
    if (J.o.kernel < 0 || J.o.kernel > 2) J.o.kernel = 1;
#if WEXF_SIMD == 2
    if (J.o.kernel == 2) { J.o.kernel = 1; J.o.lanes = 4; }   /* no gather kernel on NEON */
#endif
    if (J.o.kernel == 1 && J.o.lanes != 1 && J.o.lanes != 2 && J.o.lanes != 3 && J.o.lanes != 6 && J.o.lanes != 8) J.o.lanes = 4;
    if (J.o.kernel == 2 && (J.o.lanes < 1 || J.o.lanes > 3)) J.o.lanes = 1;
    J.o.fuse = J.o.fuse != 0;
#else
    J.o.kernel = 0;
#endif
    if (wexf_section(&J, 0)) return -1;
    return wexf_section(&J, 1);
}

/* Same, with the tables built for this call (heap; the checked kernel if there is no memory). */
static inline int wex_fast_decode_opt(const WexNr2Record *v, const WexNr2Layout *layout, const WexNr2Model *model,
                                      unsigned char *const scales_out[3], unsigned char *const weights_out[3],
                                      const WexFastOptions *options) {
    WexFastModel *tables = NULL;
    int status;
#if WEXF_SIMD
    if (options->kernel && v->magic == WEX_NR3_MAGIC && v->model_id == model->id) {
        tables = (WexFastModel *)malloc(sizeof(*tables));
        if (tables) wex_fast_model_init(tables, model);
    }
#endif
    status = wex_fast_decode_with(tables, v, layout, model, scales_out, weights_out, options);
    free(tables);
    return status;
}

/* Default: the fastest configuration measured (see RESULTS.md). */
static inline int wex_fast_decode(const WexNr2Record *v, const WexNr2Layout *layout, const WexNr2Model *model,
                                  unsigned char *const scales_out[3], unsigned char *const weights_out[3]) {
    static const WexFastOptions best = {0, 1, 4, 1};
    return wex_fast_decode_opt(v, layout, model, scales_out, weights_out, &best);
}
#endif
