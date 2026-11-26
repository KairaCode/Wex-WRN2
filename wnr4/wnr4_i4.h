/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Ludovico Zanichelli */
/*
 * wnr4_i4.h - WNR4 records of int4-g64 experts (kind 3, "I4"). Header only, C99, needs wnr4.h.
 * docs/WNR4-SPEC.md section 6.
 *
 * An int4-g64 expert record is 1 to 3 matrices: first the codes of every matrix (signed int4 stored as v + 8, one
 * nibble per weight, 32 bytes per group of 64 inputs along the row, "planar" or "pairs" packing), then the f32
 * scales (one per group) of every matrix, back to back. Dequant: (nibble - 8) * scale.
 *
 * Per matrix the groups are cut into tiles of 32 lanes x G groups, coded as WNR4 tiles (32-bit rANS lanes sharing
 * one 16-bit word stream, 14-bit tables, at most one word per symbol, decoded by one warp in lockstep):
 *   phase 1: the tile's scales, lane l owning group c0 + 32 k + l at step k. A scale is an index k into a lattice
 *            fl(fl(E4M3(k) * B) / 7) (B = the f32 base of its 128-row x 128-input block, stored in the record),
 *            coded as a delta from the index of the group r rows above, or the escape index 127 followed by the
 *            scale's 4 bytes;
 *   phase 2: the groups' 64 nibbles in element order, each in the context of (class of its group scale, previous
 *            magnitude, running maximum magnitude).
 * Lanes 2..31 carry the last 4 nibbles of their last group in their initial state; lanes 0 and 1 start at L + the
 * halves of a 32-bit tile hash (MurmurHash3 per lane over the tile ident, the decoded scales and the group words).
 *
 * Model blob WNM4 kind 3: 32-byte header {WNM4, 1, 3, 14, 900, 68, 0, 0}, then 3 x 900 nibble tables (17 u16),
 * then 3 x 68 scale tables (257 u16). Id = FNV-1a of the blob.
 *
 * The scale lattice is binary32 arithmetic with round to nearest even and subnormals kept: do not build this file
 * with -ffast-math (refused below) or a CUDA flag that flushes subnormals (-ftz=true, --use_fast_math).
 */
#ifndef WNR4_I4_H
#define WNR4_I4_H

#include "wnr4.h"

#if defined(__FAST_MATH__)
#error "wnr4_i4.h needs IEEE binary32 arithmetic: build without -ffast-math"
#endif

#define WNR4_KIND_I4 3u
#define WNR4_I4_FLAG_PAIRS 0x01u
#define WNR4_I4_NCLS 9u
#define WNR4_I4_WROWS 900u                 /* 9 classes x 10 previous magnitudes x 10 running maxima */
#define WNR4_I4_SROWS 68u                  /* 64 index contexts + 4 escape bytes */
#define WNR4_I4_ESC 127u
#define WNR4_I4_MODEL_BYTES (32u + 3u * WNR4_I4_WROWS * 34u + 3u * WNR4_I4_SROWS * 514u)
#define WNR4_I4_BLOCK_ROWS 128u
#define WNR4_I4_BLOCK_GROUPS 2u

/* ---- scale lattice -------------------------------------------------------------- */
static WNR4_HD inline float wnr4_i4_f(uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f; memcpy(&f, &b, 4); return f;
#endif
}
static WNR4_HD inline uint32_t wnr4_i4_b(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t b; memcpy(&b, &f, 4); return b;
#endif
}
/* E4M3 code k (0..126) as f32 bits: k < 8 subnormal k * 2^-9, else (1 + (k & 7) / 8) * 2^((k >> 3) - 7) */
static WNR4_HD inline uint32_t wnr4_i4_e4m3(uint32_t k) {
    return k < 8u ? wnr4_i4_b((float)k * 0.001953125f) : ((k >> 3) + 120u) << 23 | (k & 7u) << 20;   /* k * 2^-9 exact */
}
/* lattice value k under base B (both f32 bits): fl(fl(E4M3(k) * B) / 7) */
static WNR4_HD inline uint32_t wnr4_i4_lat(uint32_t k, uint32_t B) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(__fdiv_rn(__fmul_rn(__uint_as_float(wnr4_i4_e4m3(k)), __uint_as_float(B)), 7.0f));
#else
    volatile float p = wnr4_i4_f(wnr4_i4_e4m3(k)) * wnr4_i4_f(B);   /* volatile: one rounding per operation */
    volatile float q = p / 7.0f;
    return wnr4_i4_b(q);
#endif
}
/* a stored base: 0, or a positive normal finite f32 */
static WNR4_HD inline int wnr4_i4_base_ok(uint32_t B) { return B == 0u || (B >= 0x00800000u && B < 0x7f800000u); }

/* ---- geometry ----------------------------------------------------------------------- */
typedef struct {
    uint32_t nmat, pairs;
    uint32_t rows[3], cols[3], gc[3];  /* gc = groups per row */
    uint32_t cells[3];                 /* groups of matrix m (0 for m >= nmat) */
    uint32_t coff[3], soff[3];         /* byte offsets of the codes and of the scales of matrix m */
    uint32_t nbc[3], bbase[3], nbases; /* base blocks per block row, first base of m, bases in the record */
    uint64_t raw;                      /* decoded bytes = 36 x groups */
} Wnr4I4Geo;

static inline int wnr4_i4_geo(Wnr4I4Geo *g, uint32_t nmat, const uint32_t rows[], const uint32_t cols[], uint32_t pairs) {
    uint32_t m;
    uint64_t at = 0, groups = 0, nb = 0;
    memset(g, 0, sizeof(*g));
    if (!nmat || nmat > 3u || pairs > 1u) return -1;
    g->nmat = nmat; g->pairs = pairs;
    for (m = 0; m < nmat; m++) {
        if (!rows[m] || !cols[m] || cols[m] % 64u) return -1;
        g->rows[m] = rows[m]; g->cols[m] = cols[m]; g->gc[m] = cols[m] / 64u;
        if ((uint64_t)rows[m] * g->gc[m] >= (1u << 27)) return -1;
        g->cells[m] = rows[m] * g->gc[m];
        groups += g->cells[m];
        g->nbc[m] = (g->gc[m] + 1u) / 2u; g->bbase[m] = (uint32_t)nb;
        nb += (uint64_t)((rows[m] + WNR4_I4_BLOCK_ROWS - 1u) / WNR4_I4_BLOCK_ROWS) * g->nbc[m];
    }
    if (36u * groups > 0xffffffffu) return -1;
    for (m = 0; m < nmat; m++) { g->coff[m] = (uint32_t)at; at += 32u * (uint64_t)g->cells[m]; }
    for (m = 0; m < nmat; m++) { g->soff[m] = (uint32_t)at; at += 4u * (uint64_t)g->cells[m]; }
    g->nbases = (uint32_t)nb; g->raw = at;
    return 0;
}
/* base index of group i of matrix m */
static WNR4_HD inline uint32_t wnr4_i4_block(uint32_t bbase, uint32_t gc, uint32_t nbc, uint32_t i) {
    return bbase + (i / gc / WNR4_I4_BLOCK_ROWS) * nbc + (i % gc) / WNR4_I4_BLOCK_GROUPS;
}
/* element e (0..63) of a group's 32 code bytes */
static WNR4_HD inline uint32_t wnr4_i4_nib(const unsigned char *grp, uint32_t pairs, uint32_t e) {
    return pairs ? (grp[e >> 1] >> ((e & 1u) * 4u)) & 15u : (grp[e & 31u] >> ((e >> 5) * 4u)) & 15u;
}
/* the 64 nibbles as 8 words (element e at bits 4 (e % 8) of word e / 8) -> the 32 code bytes as 8 LE words */
static WNR4_HD inline void wnr4_i4_pack(const uint32_t v[8], uint32_t pairs, uint32_t w[8]) {
    uint32_t b, e;
    if (pairs) { for (b = 0; b < 8u; b++) w[b] = v[b]; return; }
    for (b = 0; b < 8u; b++) w[b] = 0;
    for (e = 0; e < 32u; e++) {   /* byte e = element e | element e + 32 << 4 */
        uint32_t lo = (v[e >> 3] >> ((e & 7u) * 4u)) & 15u, hi = (v[(e + 32u) >> 3] >> ((e & 7u) * 4u)) & 15u;
        w[e >> 2] |= (lo | hi << 4) << ((e & 3u) * 8u);
    }
}
/* reference group of group i (r rows above, inside the tile and decoded at an earlier step), or UINT32_MAX */
static WNR4_HD inline uint32_t wnr4_i4_ref(uint32_t gc, uint32_t i, uint32_t c0) {
    uint32_t d = gc >= 32u ? gc : ((32u + gc - 1u) / gc) * gc;
    return i >= c0 + d ? i - d : 0xffffffffu;
}
/* scale table row (0..63) and the reference index from the reference's k (or none) */
static WNR4_HD inline uint32_t wnr4_i4_kref(uint32_t kr, int has) { return has ? (kr == WNR4_I4_ESC ? 126u : kr) : 120u; }
static WNR4_HD inline uint32_t wnr4_i4_sctx(uint32_t kref, int has) { return (has ? 32u : 0u) + kref / 4u; }
/* class of a group scale: clamp((bits >> 21) - anchor, -4, 4) + 4 */
static WNR4_HD inline uint32_t wnr4_i4_cls(uint32_t bits, uint32_t anchor) {
    int d = (int)(bits >> 21) - (int)anchor;
    d = d < -4 ? -4 : d > 4 ? 4 : d;
    return (uint32_t)(d + 4);
}
static WNR4_HD inline uint32_t wnr4_i4_wrow(uint32_t cls, uint32_t pm, uint32_t mx) { return (cls * 10u + pm) * 10u + mx; }
static WNR4_HD inline uint32_t wnr4_i4_mag(uint32_t y) { return y > 8u ? y - 8u : 8u - y; }
static WNR4_HD inline uint32_t wnr4_i4_mx(uint32_t mx, uint32_t a) { return mx == 9u ? a : (a > mx ? a : mx); }
static WNR4_HD inline uint32_t wnr4_i4_hlen(uint32_t nsl, uint32_t ncl) { return 24u + 4u * nsl + 32u * ncl; }
/* tail in the state: elements 60..63 of the last group of a lane l >= 2 */
static WNR4_HD inline int wnr4_i4_tail_skip(uint32_t l, uint32_t j, uint32_t e, uint32_t n) {
    return l >= 2u && e >= 60u && l < n && j == (n - l - 1u) / 32u;
}

/* ---- model ----------------------------------------------------------------------------- */
typedef struct {
    uint32_t id;
    uint16_t *w;   /* [3][900][17] */
    uint16_t *s;   /* [3][68][257] */
} Wnr4I4Model;

static inline void wnr4_i4_model_free(Wnr4I4Model *m) { free(m->w); free(m->s); m->w = m->s = NULL; }
static inline int wnr4_i4_model_open(const unsigned char *blob, size_t bytes, Wnr4I4Model *m) {
    uint32_t i;
    const unsigned char *p;
    memset(m, 0, sizeof(*m));
    if (!blob || bytes != WNR4_I4_MODEL_BYTES || wnr4_u32(blob) != WNM4_MAGIC || wnr4_u32(blob + 4) != 1u ||
        wnr4_u32(blob + 8) != WNR4_KIND_I4 || wnr4_u32(blob + 12) != WNR4_PREC || wnr4_u32(blob + 16) != WNR4_I4_WROWS ||
        wnr4_u32(blob + 20) != WNR4_I4_SROWS || wnr4_u32(blob + 24) || wnr4_u32(blob + 28)) return -1;
    m->w = (uint16_t *)malloc(3u * WNR4_I4_WROWS * 17u * 2u);
    m->s = (uint16_t *)malloc(3u * WNR4_I4_SROWS * 257u * 2u);
    if (!m->w || !m->s) { wnr4_i4_model_free(m); return -1; }
    p = blob + 32;
    for (i = 0; i < 3u * WNR4_I4_WROWS * 17u; i++, p += 2) m->w[i] = (uint16_t)(p[0] | p[1] << 8);
    for (i = 0; i < 3u * WNR4_I4_SROWS * 257u; i++, p += 2) m->s[i] = (uint16_t)(p[0] | p[1] << 8);
    for (i = 0; i < 3u * WNR4_I4_WROWS; i++) if (!wnr4_cdf_ok(m->w + (size_t)i * 17u, 16u)) { wnr4_i4_model_free(m); return -1; }
    for (i = 0; i < 3u * WNR4_I4_SROWS; i++) if (!wnr4_cdf_ok(m->s + (size_t)i * 257u, 256u)) { wnr4_i4_model_free(m); return -1; }
    m->id = wnr4_fnv1a(blob, bytes);
    return 0;
}

/* ---- encoder side info (not normative: any anchors, bases and indices decode) ------------------ */
typedef struct {
    uint32_t anchor[3];
    uint32_t *base;      /* [nbases] */
    unsigned char *k;    /* per group of every matrix, at gfirst[m] + i: lattice index or WNR4_I4_ESC */
    uint32_t gfirst[3];
} Wnr4I4Side;

static inline void wnr4_i4_side_free(Wnr4I4Side *s) { free(s->base); free(s->k); s->base = NULL; s->k = NULL; }
static inline uint32_t wnr4_i4_scale(const Wnr4I4Geo *g, const unsigned char *slab, uint32_t m, uint32_t i) {
    return wnr4_u32(slab + g->soff[m] + 4u * (size_t)i);
}
/* leftmost k (0..126) with lat(k, B) == v, else WNR4_I4_ESC (lat is nondecreasing in k for a valid B) */
static inline uint32_t wnr4_i4_find_k(uint32_t B, uint32_t v) {
    uint32_t lo = 0, hi = 127;
    float f = wnr4_i4_f(v);
    if (!(f >= 0.0f)) return WNR4_I4_ESC;   /* negative or NaN: never on the lattice */
    while (lo < hi) { uint32_t mid = (lo + hi) / 2u; if (wnr4_i4_f(wnr4_i4_lat(mid, B)) < f) lo = mid + 1u; else hi = mid; }
    return lo < 127u && wnr4_i4_lat(lo, B) == v ? lo : WNR4_I4_ESC;
}
/* base of one block (n scale bit patterns): the candidate under which the most scales lie on the lattice. The
 * candidates put the block maximum on one of the codes 96..126, the converter's amax / 7 of an FP8 block-scaled
 * source; a block that fits nowhere gets 0 (its scales escape unless they are 0). */
static inline uint32_t wnr4_i4_block_base(const uint32_t *v, uint32_t n) {
    float smax = 0.0f;
    uint32_t i, best = 0, bestB = 0;
    int km, u;
    for (i = 0; i < n; i++) { float f = wnr4_i4_f(v[i]); if (f > smax) smax = f; }
    if (!(smax > 0.0f) || smax > 3.0e38f) return 0;
    for (i = 0; i < n; i++) best += v[i] == 0u;   /* base 0: only the zero scales fit */
    for (km = 126; km >= 96 && best < n; km--) {
        float B0 = (float)((double)smax * 7.0 / (double)wnr4_i4_f(wnr4_i4_e4m3((uint32_t)km)));
        uint32_t b0 = wnr4_i4_b(B0);
        for (u = -6; u <= 6 && best < n; u++) {
            uint32_t bb = b0 + (uint32_t)u, cov = 0;
            if (!wnr4_i4_base_ok(bb) || !bb || wnr4_i4_lat((uint32_t)km, bb) != wnr4_i4_b(smax)) continue;
            for (i = 0; i < n; i++) cov += wnr4_i4_find_k(bb, v[i]) != WNR4_I4_ESC;
            if (cov > best) { best = cov; bestB = bb; }
        }
    }
    return bestB;
}
/* anchors (median of bits >> 21 per matrix), block bases and every group's index */
static inline int wnr4_i4_side(const Wnr4I4Geo *g, const unsigned char *slab, Wnr4I4Side *s) {
    uint32_t m, i, total = 0, *v;
    memset(s, 0, sizeof(*s));
    for (m = 0; m < g->nmat; m++) { s->gfirst[m] = total; total += g->cells[m]; }
    s->base = (uint32_t *)calloc(g->nbases ? g->nbases : 1u, 4);
    s->k = (unsigned char *)malloc(total ? total : 1u);
    v = (uint32_t *)malloc(WNR4_I4_BLOCK_ROWS * WNR4_I4_BLOCK_GROUPS * 4u);
    if (!s->base || !s->k || !v) { free(v); wnr4_i4_side_free(s); return -1; }
    for (m = 0; m < g->nmat; m++) {
        uint32_t h[2048] = {0}, c = 0, a = 0, br, bc, nbr = (g->rows[m] + WNR4_I4_BLOCK_ROWS - 1u) / WNR4_I4_BLOCK_ROWS;
        for (i = 0; i < g->cells[m]; i++) h[wnr4_i4_scale(g, slab, m, i) >> 21]++;
        while (c + h[a] <= g->cells[m] / 2u) c += h[a++];
        s->anchor[m] = a;
        for (br = 0; br < nbr; br++)
            for (bc = 0; bc < g->nbc[m]; bc++) {
                uint32_t n = 0, o, cc;
                for (o = br * WNR4_I4_BLOCK_ROWS; o < g->rows[m] && o < (br + 1u) * WNR4_I4_BLOCK_ROWS; o++)
                    for (cc = bc * 2u; cc < g->gc[m] && cc < bc * 2u + 2u; cc++) v[n++] = wnr4_i4_scale(g, slab, m, o * g->gc[m] + cc);
                s->base[g->bbase[m] + br * g->nbc[m] + bc] = wnr4_i4_block_base(v, n);
            }
        for (i = 0; i < g->cells[m]; i++)
            s->k[s->gfirst[m] + i] = (unsigned char)wnr4_i4_find_k(s->base[wnr4_i4_block(g->bbase[m], g->gc[m], g->nbc[m], i)],
                                                                    wnr4_i4_scale(g, slab, m, i));
    }
    free(v);
    return 0;
}

/* tile t -> matrix, first group, groups (wnr4_tile_cells over the groups) */
static inline void wnr4_i4_tile_base(const Wnr4I4Geo *g, uint32_t G, uint32_t tb[4]) {
    uint32_t m, t = 0;
    for (m = 0; m < 3u; m++) { tb[m] = t; t += wnr4_tiles(g->cells[m], G); }
    tb[3] = t;
}

/* Symbols of tile t in decode order (with tails != 0 also the tail nibbles, for counting). Entry: bit 31 =
 * 256-symbol table, bits 24..28 lane, bits 8..23 table row (m * 68 + row, or m * 900 + row), bits 0..7 symbol.
 * Returns the count. */
static inline size_t wnr4_i4_symbols(const Wnr4I4Geo *g, const Wnr4I4Side *sd, const unsigned char *slab, const uint32_t tb[4],
                                     uint32_t G, uint32_t t, int tails, uint32_t *sym) {
    uint32_t m, c0, n, k, l, j, e, b;
    size_t q = 0;
    const unsigned char *kk;
    wnr4_tile_cells(tb, g->cells, G, t, &m, &c0, &n);
    kk = sd->k + sd->gfirst[m];
    for (k = 0; 32u * k < n; k++) {
        uint32_t esc = 0;
        for (l = 0; l < 32u && 32u * k + l < n; l++) {
            uint32_t i = c0 + 32u * k + l, u = wnr4_i4_ref(g->gc[m], i, c0), has = u != 0xffffffffu;
            uint32_t kref = wnr4_i4_kref(has ? kk[u] : 0u, (int)has);
            sym[q++] = 0x80000000u | l << 24 | (m * WNR4_I4_SROWS + wnr4_i4_sctx(kref, (int)has)) << 8 | ((kk[i] - kref) & 255u);
            if (kk[i] == WNR4_I4_ESC) esc |= 1u << l;
        }
        for (b = 0; b < 4u; b++)
            for (l = 0; l < 32u; l++)
                if (esc >> l & 1u)
                    sym[q++] = 0x80000000u | l << 24 | (m * WNR4_I4_SROWS + 64u + b) << 8 |
                               ((wnr4_i4_scale(g, slab, m, c0 + 32u * k + l) >> (24u - 8u * b)) & 255u);
    }
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, cls[32], pm[32], mx[32];
        for (l = 0; l < act; l++) { cls[l] = wnr4_i4_cls(wnr4_i4_scale(g, slab, m, c0 + 32u * j + l), sd->anchor[m]); pm[l] = mx[l] = 9u; }
        for (e = 0; e < 64u; e++)
            for (l = 0; l < act; l++) {
                uint32_t y = wnr4_i4_nib(slab + g->coff[m] + 32u * (size_t)(c0 + 32u * j + l), g->pairs, e), a = wnr4_i4_mag(y);
                if (tails || !wnr4_i4_tail_skip(l, j, e, n)) sym[q++] = l << 24 | (m * WNR4_I4_WROWS + wnr4_i4_wrow(cls[l], pm[l], mx[l])) << 8 | y;
                pm[l] = a; mx[l] = wnr4_i4_mx(mx[l], a);
            }
    }
    return q;
}

/* ---- training (not normative: any tables that pass the checks decode) -------------------------- */
/* counts: nibbles [3][900][16], then scale symbols [3][68][256] (u64) */
static inline size_t wnr4_i4_counts_words(void) { return 3u * (size_t)WNR4_I4_WROWS * 16u + 3u * (size_t)WNR4_I4_SROWS * 256u; }
/* adds one record (counted under tiles of G = 32, every nibble, the tail nibbles included; the escape-byte tables
 * also count the bytes of every scale, so they fit scales that miss the lattice). Returns 0 or -1 (out of memory). */
static inline int wnr4_i4_count(const Wnr4I4Geo *g, const unsigned char *slab, uint64_t *counts) {
    Wnr4I4Side sd;
    uint32_t tb[4], t, m, i, b, *sym = (uint32_t *)malloc(((size_t)32u * 32u * 69u + 64u) * 4u);
    uint64_t *sc = counts + 3u * (size_t)WNR4_I4_WROWS * 16u;
    if (!sym || wnr4_i4_side(g, slab, &sd)) { free(sym); return -1; }
    wnr4_i4_tile_base(g, 32u, tb);
    for (t = 0; t < tb[3]; t++) {
        size_t q = wnr4_i4_symbols(g, &sd, slab, tb, 32u, t, 1, sym), p;
        for (p = 0; p < q; p++) {
            uint32_t e = sym[p], row = (e >> 8) & 0xffffu;
            if (e >> 31) { if (row % WNR4_I4_SROWS < 64u) sc[(size_t)row * 256u + (e & 255u)]++; }
            else counts[(size_t)row * 16u + (e & 15u)]++;
        }
    }
    for (m = 0; m < g->nmat; m++)
        for (i = 0; i < g->cells[m]; i++)
            for (b = 0; b < 4u; b++)
                sc[((size_t)m * WNR4_I4_SROWS + 64u + b) * 256u + ((wnr4_i4_scale(g, slab, m, i) >> (24u - 8u * b)) & 255u)]++;
    wnr4_i4_side_free(&sd); free(sym);
    return 0;
}
static inline void wnr4_i4_model_build(const uint64_t *counts, unsigned char *blob) {
    uint32_t hdr[8] = {WNM4_MAGIC, 1u, WNR4_KIND_I4, WNR4_PREC, WNR4_I4_WROWS, WNR4_I4_SROWS, 0, 0}, i, k;
    uint16_t cdf[257];
    unsigned char *p = blob + 32;
    for (i = 0; i < 8u; i++) wnr4_put32(blob + 4 * i, hdr[i]);
    for (i = 0; i < 3u * WNR4_I4_WROWS; i++) {
        wnr4_quantize(counts + (size_t)i * 16u, 16u, cdf);
        for (k = 0; k < 17u; k++) { *p++ = (unsigned char)cdf[k]; *p++ = (unsigned char)(cdf[k] >> 8); }
    }
    for (i = 0; i < 3u * WNR4_I4_SROWS; i++) {
        wnr4_quantize(counts + 3u * (size_t)WNR4_I4_WROWS * 16u + (size_t)i * 256u, 256u, cdf);
        for (k = 0; k < 257u; k++) { *p++ = (unsigned char)cdf[k]; *p++ = (unsigned char)(cdf[k] >> 8); }
    }
}

/* ---- record ------------------------------------------------------------------------------------- */
typedef struct {
    uint32_t model_id, owner, G, ntiles, payload, anchor[3];
    uint32_t tile_base[4];
    const unsigned char *offsets, *bases, *data;
} Wnr4I4Record;

static inline int wnr4_i4_parse(const unsigned char *rec, size_t total, const Wnr4I4Geo *g, Wnr4I4Record *r) {
    uint32_t m, t, prev = 0;
    size_t used;
    memset(r, 0, sizeof(*r));
    if (!rec || total < WNR4_HEADER_BYTES || (total & 4095u) || wnr4_u32(rec) != WNR4_MAGIC) return -1;
    r->model_id = wnr4_u32(rec + 4); r->owner = wnr4_u32(rec + 8);
    r->G = (uint32_t)rec[14] | (uint32_t)rec[15] << 8;
    r->ntiles = wnr4_u32(rec + 20); r->payload = wnr4_u32(rec + 24);
    if (!r->model_id || rec[12] != WNR4_KIND_I4 || rec[13] != g->pairs || r->G < 1u || r->G > WNR4_MAX_G) return -1;
    for (t = 16; t < 20; t++) if (rec[t]) return -1;
    for (m = 0; m < 3u; m++) {
        r->anchor[m] = (uint32_t)rec[28 + 2 * m] | (uint32_t)rec[29 + 2 * m] << 8;
        if (m >= g->nmat && r->anchor[m]) return -1;
    }
    for (t = 34; t < 64; t++) if (rec[t]) return -1;
    wnr4_i4_tile_base(g, r->G, r->tile_base);
    if (r->tile_base[3] != r->ntiles) return -1;
    used = WNR4_HEADER_BYTES + 4u * ((size_t)r->ntiles + 1u) + 4u * (size_t)g->nbases + r->payload;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    r->offsets = rec + WNR4_HEADER_BYTES;
    r->bases = r->offsets + 4u * ((size_t)r->ntiles + 1u);
    r->data = r->bases + 4u * (size_t)g->nbases;
    for (t = 0; t < g->nbases; t++) if (!wnr4_i4_base_ok(wnr4_u32(r->bases + 4u * t))) return -1;
    if (wnr4_u32(r->offsets)) return -1;
    for (t = 1; t <= r->ntiles; t++) {
        uint32_t o = wnr4_u32(r->offsets + 4u * t);
        if (o < prev || o - prev < WNR4_TILE_STATE_BYTES || ((o - prev) & 1u)) return -1;
        prev = o;
    }
    if (prev != r->payload) return -1;
    while (used < total && (used & 7u)) { if (rec[used++]) return -1; }
    for (; used + 8u <= total; used += 8u) {
        uint64_t z;
        memcpy(&z, rec + used, 8u);
        if (z) return -1;
    }
    for (; used < total; used++) if (rec[used]) return -1;
    return 0;
}

/* Decodes tile t into out (the record's decoded bytes). Returns 0 or error bits; out may hold partial data. */
static inline uint32_t wnr4_i4_decode_tile(const Wnr4I4Model *model, const Wnr4I4Geo *g, const Wnr4I4Record *r,
                                           uint32_t t, unsigned char *out) {
    uint32_t m, c0, n, o0 = wnr4_u32(r->offsets + 4u * t), o1 = wnr4_u32(r->offsets + 4u * t + 4u);
    const unsigned char *tile = r->data + o0;
    uint32_t x[32], hash[32], ptr = 0, l, j, k, e, b, H = 0, nwords, anchor;
    unsigned char *kk, *so, *co;
    const uint16_t *wt, *st;
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r->payload) return WNR4_ERR_STRUCT;
    nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u;
    wnr4_tile_cells(r->tile_base, g->cells, r->G, t, &m, &c0, &n);
    anchor = r->anchor[m];
    so = out + g->soff[m]; co = out + g->coff[m];
    wt = model->w + (size_t)m * WNR4_I4_WROWS * 17u; st = model->s + (size_t)m * WNR4_I4_SROWS * 257u;
    for (l = 0; l < 32u; l++) {
        x[l] = wnr4_u32(tile + 4u * l);
        if (x[l] < WNR4_L) return WNR4_ERR_STRUCT;
        hash[l] = wnr4_lane_seed(r->model_id, r->owner, t, l, wnr4_word(WNR4_KIND_I4, g->pairs, r->G), m | anchor << 16);
    }
    kk = (unsigned char *)malloc(n);   /* the tile's lattice indices, the contexts of later steps */
    if (!kk) return WNR4_ERR_STRUCT;
#define WNR4_I4_RENORM(L_)                                                           \
    if (x[L_] < WNR4_L) {                                                            \
        if (ptr >= nwords) { free(kk); return WNR4_ERR_STRUCT; }                     \
        x[L_] = x[L_] << 16 | (uint32_t)(tile[128u + 2u * ptr] | tile[129u + 2u * ptr] << 8); ptr++; \
    }
#define WNR4_I4_DEC(L_, CDF, FIND, Y)                                                \
    do { uint32_t slot_ = x[L_] & (WNR4_TOTAL - 1u); Y = FIND(CDF, slot_);           \
         x[L_] = (uint32_t)((CDF)[Y + 1] - (CDF)[Y]) * (x[L_] >> WNR4_PREC) + slot_ - (CDF)[Y]; WNR4_I4_RENORM(L_) } while (0)
    /* phase 1: scales */
    for (k = 0; 32u * k < n; k++) {
        uint32_t esc = 0, val[32];
        for (l = 0; l < 32u && 32u * k + l < n; l++) {
            uint32_t i = c0 + 32u * k + l, u = wnr4_i4_ref(g->gc[m], i, c0), has = u != 0xffffffffu, y, kv;
            uint32_t kref = wnr4_i4_kref(has ? kk[u - c0] : 0u, (int)has);
            const uint16_t *cdf = st + (size_t)wnr4_i4_sctx(kref, (int)has) * 257u;
            WNR4_I4_DEC(l, cdf, wnr4_find256, y);
            kv = (y + kref) & 255u;
            if (kv > WNR4_I4_ESC) { free(kk); return WNR4_ERR_STRUCT; }
            kk[i - c0] = (unsigned char)kv;
            if (kv == WNR4_I4_ESC) { esc |= 1u << l; val[l] = 0; }
            else val[l] = wnr4_i4_lat(kv, wnr4_u32(r->bases + 4u * wnr4_i4_block(g->bbase[m], g->gc[m], g->nbc[m], i)));
        }
        for (b = 0; b < 4u; b++)
            for (l = 0; l < 32u; l++)
                if (esc >> l & 1u) {
                    const uint16_t *cdf = st + (size_t)(64u + b) * 257u;
                    uint32_t y;
                    WNR4_I4_DEC(l, cdf, wnr4_find256, y);
                    val[l] |= y << (24u - 8u * b);
                }
        for (l = 0; l < 32u && 32u * k + l < n; l++) {
            wnr4_put32(so + 4u * (size_t)(c0 + 32u * k + l), val[l]);
            hash[l] = wnr4_h32_block(hash[l], val[l]);
        }
    }
    /* phase 2: groups */
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, cls[32], pm[32], mx[32], v[32][8];
        for (l = 0; l < act; l++) {
            cls[l] = wnr4_i4_cls(wnr4_u32(so + 4u * (size_t)(c0 + 32u * j + l)), anchor); pm[l] = mx[l] = 9u;
            for (b = 0; b < 8u; b++) v[l][b] = 0;
        }
        for (e = 0; e < 64u; e++)
            for (l = 0; l < act; l++) {
                const uint16_t *cdf;
                uint32_t y, a;
                if (wnr4_i4_tail_skip(l, j, e, n)) continue;
                cdf = wt + (size_t)wnr4_i4_wrow(cls[l], pm[l], mx[l]) * 17u;
                WNR4_I4_DEC(l, cdf, wnr4_find16, y);
                v[l][e >> 3] |= y << ((e & 7u) * 4u);
                a = wnr4_i4_mag(y); pm[l] = a; mx[l] = wnr4_i4_mx(mx[l], a);
            }
        for (l = 0; l < act; l++) {
            uint32_t w[8];
            unsigned char *grp = co + 32u * (size_t)(c0 + 32u * j + l);
            if (wnr4_i4_tail_skip(l, j, 63u, n)) {
                uint32_t tail = x[l] - WNR4_L;
                if (x[l] < WNR4_L || tail > 0xffffu) { free(kk); return WNR4_ERR_STRUCT; }
                v[l][7] |= tail << 16;
                x[l] = WNR4_L;
            }
            wnr4_i4_pack(v[l], g->pairs, w);
            for (b = 0; b < 8u; b++) { hash[l] = wnr4_h32_block(hash[l], w[b]); wnr4_put32(grp + 4u * b, w[b]); }
        }
    }
#undef WNR4_I4_DEC
#undef WNR4_I4_RENORM
    free(kk);
    for (l = 0; l < 32u; l++) H ^= wnr4_h32_final(hash[l], wnr4_i4_hlen(wnr4_lane_cells(n, l), wnr4_lane_cells(n, l)));
    if (ptr != nwords) return WNR4_ERR_STRUCT;
    if (x[0] != WNR4_L + (H & 0xffffu) || x[1] != WNR4_L + (H >> 16)) return WNR4_ERR_HASH;
    for (l = 2; l < 32u; l++) if (x[l] != WNR4_L) return WNR4_ERR_STRUCT;
    return 0;
}

/* one record into out (g->raw bytes); 0 or error bits, out may hold partial data on failure */
static inline uint32_t wnr4_i4_decode(const Wnr4I4Model *model, const Wnr4I4Geo *g, const unsigned char *rec,
                                      size_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4I4Record r;
    uint32_t err = 0;
    int t;
    if (wnr4_i4_parse(rec, total, g, &r) || r.model_id != model->id || r.owner != owner) return WNR4_ERR_STRUCT;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(| : err)
#endif
    for (t = 0; t < (int)r.ntiles; t++) err |= wnr4_i4_decode_tile(model, g, &r, (uint32_t)t, out);
    (void)threads;
    return err;
}

/* ---- encode ------------------------------------------------------------------------------------- */
/* Encodes one record's decoded bytes (g->raw) into rec (capacity cap). Returns the record bytes (a multiple of
 * 4096), or 0 on error or overflow. */
static inline size_t wnr4_i4_encode(const Wnr4I4Model *model, const Wnr4I4Geo *g, const unsigned char *slab,
                                    uint32_t owner, uint32_t G, unsigned char *rec, size_t cap) {
    Wnr4I4Side sd;
    uint32_t tb[4], t, l, m, b;
    size_t at, payload0, nsym = (size_t)32u * G * 69u + 64u, total;
    uint16_t *stack;
    uint32_t *sym;
    if (!G || G > WNR4_MAX_G) return 0;
    wnr4_i4_tile_base(g, G, tb);
    payload0 = WNR4_HEADER_BYTES + 4u * ((size_t)tb[3] + 1u) + 4u * (size_t)g->nbases;
    if (payload0 > cap) return 0;
    if (wnr4_i4_side(g, slab, &sd)) return 0;
    stack = (uint16_t *)malloc(nsym * 2u);
    sym = (uint32_t *)malloc(nsym * 4u);
    if (!stack || !sym) { free(stack); free(sym); wnr4_i4_side_free(&sd); return 0; }
    memset(rec, 0, payload0);
    wnr4_put32(rec, WNR4_MAGIC); wnr4_put32(rec + 4, model->id); wnr4_put32(rec + 8, owner);
    rec[12] = (unsigned char)WNR4_KIND_I4; rec[13] = (unsigned char)g->pairs;
    rec[14] = (unsigned char)G; rec[15] = (unsigned char)(G >> 8);
    wnr4_put32(rec + 20, tb[3]);
    for (m = 0; m < g->nmat; m++) { rec[28 + 2 * m] = (unsigned char)sd.anchor[m]; rec[29 + 2 * m] = (unsigned char)(sd.anchor[m] >> 8); }
    for (b = 0; b < g->nbases; b++) wnr4_put32(rec + WNR4_HEADER_BYTES + 4u * ((size_t)tb[3] + 1u) + 4u * b, sd.base[b]);
    at = payload0;
    for (t = 0; t < tb[3]; t++) {
        uint32_t c0, n, x[32], hash[32], H = 0, j, k;
        size_t sp = 0, p, q;
        const unsigned char *co;
        wnr4_tile_cells(tb, g->cells, G, t, &m, &c0, &n);
        co = slab + g->coff[m];
        q = wnr4_i4_symbols(g, &sd, slab, tb, G, t, 0, sym);
        for (l = 0; l < 32u; l++)
            hash[l] = wnr4_lane_seed(model->id, owner, t, l, wnr4_word(WNR4_KIND_I4, g->pairs, G), m | sd.anchor[m] << 16);
        for (k = 0; 32u * k < n; k++)
            for (l = 0; l < 32u && 32u * k + l < n; l++) hash[l] = wnr4_h32_block(hash[l], wnr4_i4_scale(g, slab, m, c0 + 32u * k + l));
        for (j = 0; 32u * j < n; j++)
            for (l = 0; l < 32u && 32u * j + l < n; l++)
                for (b = 0; b < 8u; b++) hash[l] = wnr4_h32_block(hash[l], wnr4_u32(co + 32u * (size_t)(c0 + 32u * j + l) + 4u * b));
        for (l = 0; l < 32u; l++) H ^= wnr4_h32_final(hash[l], wnr4_i4_hlen(wnr4_lane_cells(n, l), wnr4_lane_cells(n, l)));
        for (l = 0; l < 32u; l++) {
            x[l] = l == 0 ? WNR4_L + (H & 0xffffu) : l == 1 ? WNR4_L + (H >> 16) : WNR4_L;
            if (l >= 2u && l < n) {   /* elements 60..63 of the lane's last group */
                const unsigned char *last = co + 32u * (size_t)(c0 + l + 32u * ((n - l - 1u) / 32u));
                for (k = 0; k < 4u; k++) x[l] += wnr4_i4_nib(last, g->pairs, 60u + k) << (4u * k);
            }
        }
        for (p = q; p-- > 0;) {
            uint32_t e = sym[p], y = e & 0xffu, ln = (e >> 24) & 31u, row = (e >> 8) & 0xffffu, f, s0;
            const uint16_t *cdf = e >> 31 ? model->s + (size_t)row * 257u : model->w + (size_t)row * 17u;
            s0 = cdf[y]; f = (uint32_t)cdf[y + 1] - s0;
            if (x[ln] >= (f << (32u - WNR4_PREC))) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
            x[ln] = ((x[ln] / f) << WNR4_PREC) + x[ln] % f + s0;
        }
        if (at + WNR4_TILE_STATE_BYTES + 2u * sp > cap || at + WNR4_TILE_STATE_BYTES + 2u * sp - payload0 > 0xffffffffu) {
            free(stack); free(sym); wnr4_i4_side_free(&sd); return 0;
        }
        for (l = 0; l < 32u; l++) wnr4_put32(rec + at + 4u * l, x[l]);
        for (p = 0; p < sp; p++) {
            uint16_t w = stack[sp - 1u - p];
            rec[at + 128u + 2u * p] = (unsigned char)w; rec[at + 129u + 2u * p] = (unsigned char)(w >> 8);
        }
        at += WNR4_TILE_STATE_BYTES + 2u * sp;
        wnr4_put32(rec + WNR4_HEADER_BYTES + 4u * (t + 1u), (uint32_t)(at - payload0));
    }
    free(stack); free(sym); wnr4_i4_side_free(&sd);
    wnr4_put32(rec + 24, (uint32_t)(at - payload0));
    total = (at + 4095u) & ~(size_t)4095u;
    if (total > cap) return 0;
    memset(rec + at, 0, total - at);
    return total;
}

#endif
