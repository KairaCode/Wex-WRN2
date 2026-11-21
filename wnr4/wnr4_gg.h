/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Ludovico Zanichelli */
/*
 * wnr4_gg.h - WNR4 records of GGUF expert slices (kind 2, "GG"). Header only, C99, needs wnr4.h.
 *
 * A GGUF routed-expert record (wex4/gguf2wex4.py) is the expert's slices of the gate, gate_up, up and down
 * tensors, one after the other. Slices of ggml type Q2_0 (42) or IQ4_NL (20) are arrays of 18-byte blocks:
 * an f16 scale d (2 bytes, little endian) and 16 code bytes. WNR4-GG codes those blocks; every other slice is
 * stored verbatim in the record.
 *
 * Per coded slice the blocks are cut into tiles of 32 lanes x G blocks, coded as WNR4 tiles (32-bit rANS lanes
 * sharing one 16-bit word stream, 14-bit tables, at most one word per symbol, decoded by one warp in lockstep):
 *   phase 1: the high bytes of the tile's n scales (order 0), row-major across the lanes;
 *   phase 2: their low bytes, each in the context of its high byte's class;
 *   phase 3: the cells (the 16 code bytes = 32 nibbles of a block), lane l owning blocks c0 + l + 32 j, each
 *            nibble in the context of its block's class.
 * The class of a high byte comes from a 256-entry map of the model (the 31 most frequent high bytes of the
 * training slices get classes 1..31, all others class 0). Lanes 2..31 carry the last 4 nibbles of their last
 * cell in their initial state; lanes 0 and 1 start at L + the halves of a 32-bit tile hash (MurmurHash3 per lane
 * over the tile ident, the decoded scale bytes and the cell words), as in wnr4.h.
 *
 * Model blob WNM4 kind 2: 32-byte header {WNM4, 1, 2, 14, classes (32), subtypes (2), 0, 0}, then per subtype
 * (0 = Q2_0, 1 = IQ4_NL): u8 class map[256], hi table (257 u16), lo tables [32][257] u16, nibble tables
 * [32][17] u16. Id = FNV-1a of the blob.
 *
 * Record: the 64-byte WNR4 header (kind 2, flags 0, anchors 0), the u32 tile offsets (ntiles + 1), the verbatim
 * slices in slice order zero-padded to a multiple of 4 bytes, then the tiles; zero padding to 4096.
 */
#ifndef WNR4_GG_H
#define WNR4_GG_H

#include "wnr4.h"

#define WNR4_KIND_GG 2u
#define WNR4_GG_CLASSES 32u
#define WNR4_GG_SUBTYPES 2u
#define WNR4_GG_MAX_SEG 4u
#define WNR4_GG_Q2_0 42u              /* ggml type ids */
#define WNR4_GG_IQ4_NL 20u
#define WNR4_GG_SUB_BYTES (256u + 514u + WNR4_GG_CLASSES * 514u + WNR4_GG_CLASSES * 34u)
#define WNR4_GG_MODEL_BYTES (32u + WNR4_GG_SUBTYPES * WNR4_GG_SUB_BYTES)

/* subtype of a ggml type, or -1 for a slice stored verbatim */
static WNR4_HD inline int wnr4_gg_sub(uint32_t ggml_type) {
    return ggml_type == WNR4_GG_Q2_0 ? 0 : ggml_type == WNR4_GG_IQ4_NL ? 1 : -1;
}

/* ---- geometry ---------------------------------------------------------------- */
typedef struct {
    uint32_t nseg, type[WNR4_GG_MAX_SEG], bytes[WNR4_GG_MAX_SEG], off[WNR4_GG_MAX_SEG];
    int sub[WNR4_GG_MAX_SEG];        /* coded subtype, -1 = verbatim */
    uint32_t cells[WNR4_GG_MAX_SEG]; /* blocks of a coded slice, else 0 */
    uint32_t raw_bytes;              /* verbatim bytes in the record, before padding */
    uint64_t raw;                    /* record's decoded size = sum of the slices */
} Wnr4GgGeo;

/* nseg slices of the given ggml types and byte sizes, back to back from offset 0 */
static inline int wnr4_gg_geo(Wnr4GgGeo *g, uint32_t nseg, const uint32_t type[], const uint32_t bytes[]) {
    uint32_t i;
    uint64_t at = 0, raw = 0;
    memset(g, 0, sizeof(*g));
    if (!nseg || nseg > WNR4_GG_MAX_SEG) return -1;
    g->nseg = nseg;
    for (i = 0; i < nseg; i++) {
        if (!bytes[i]) return -1;
        g->type[i] = type[i]; g->bytes[i] = bytes[i]; g->off[i] = (uint32_t)at;
        g->sub[i] = wnr4_gg_sub(type[i]);
        if (g->sub[i] >= 0) {
            if (bytes[i] % 18u || bytes[i] / 18u >= (1u << 27)) return -1;
            g->cells[i] = bytes[i] / 18u;
        } else {
            raw += bytes[i];
        }
        at += bytes[i];
        if (at > 0xffffffffu) return -1;
    }
    if (raw > 0x7fffffffu) return -1;
    g->raw_bytes = (uint32_t)raw; g->raw = at;
    return 0;
}

/* ---- model ------------------------------------------------------------------- */
typedef struct {
    uint32_t id;
    unsigned char map[WNR4_GG_SUBTYPES][256];
    uint16_t *hi;   /* [sub][257] */
    uint16_t *lo;   /* [sub][classes][257] */
    uint16_t *nib;  /* [sub][classes][17] */
} Wnr4GgModel;

static inline void wnr4_gg_model_free(Wnr4GgModel *m) {
    free(m->hi); free(m->lo); free(m->nib); m->hi = m->lo = m->nib = NULL;
}
static inline int wnr4_gg_model_open(const unsigned char *blob, size_t bytes, Wnr4GgModel *m) {
    uint32_t s, i, c;
    const unsigned char *p;
    memset(m, 0, sizeof(*m));
    if (!blob || bytes != WNR4_GG_MODEL_BYTES || wnr4_u32(blob) != WNM4_MAGIC || wnr4_u32(blob + 4) != 1u ||
        wnr4_u32(blob + 8) != WNR4_KIND_GG || wnr4_u32(blob + 12) != WNR4_PREC ||
        wnr4_u32(blob + 16) != WNR4_GG_CLASSES || wnr4_u32(blob + 20) != WNR4_GG_SUBTYPES || wnr4_u32(blob + 24) ||
        wnr4_u32(blob + 28)) return -1;
    m->hi = (uint16_t *)malloc(WNR4_GG_SUBTYPES * 257u * 2u);
    m->lo = (uint16_t *)malloc(WNR4_GG_SUBTYPES * WNR4_GG_CLASSES * 257u * 2u);
    m->nib = (uint16_t *)malloc(WNR4_GG_SUBTYPES * WNR4_GG_CLASSES * 17u * 2u);
    if (!m->hi || !m->lo || !m->nib) { wnr4_gg_model_free(m); return -1; }
    p = blob + 32;
    for (s = 0; s < WNR4_GG_SUBTYPES; s++) {
        for (i = 0; i < 256u; i++) {
            if (p[i] >= WNR4_GG_CLASSES) { wnr4_gg_model_free(m); return -1; }
            m->map[s][i] = p[i];
        }
        p += 256;
        for (i = 0; i < 257u; i++, p += 2) m->hi[s * 257u + i] = (uint16_t)(p[0] | p[1] << 8);
        for (c = 0; c < WNR4_GG_CLASSES; c++)
            for (i = 0; i < 257u; i++, p += 2) m->lo[(s * WNR4_GG_CLASSES + c) * 257u + i] = (uint16_t)(p[0] | p[1] << 8);
        for (c = 0; c < WNR4_GG_CLASSES; c++)
            for (i = 0; i < 17u; i++, p += 2) m->nib[(s * WNR4_GG_CLASSES + c) * 17u + i] = (uint16_t)(p[0] | p[1] << 8);
    }
    for (s = 0; s < WNR4_GG_SUBTYPES; s++) {
        if (!wnr4_cdf_ok(m->hi + s * 257u, 256u)) { wnr4_gg_model_free(m); return -1; }
        for (c = 0; c < WNR4_GG_CLASSES; c++)
            if (!wnr4_cdf_ok(m->lo + (s * WNR4_GG_CLASSES + c) * 257u, 256u) ||
                !wnr4_cdf_ok(m->nib + (s * WNR4_GG_CLASSES + c) * 17u, 16u)) { wnr4_gg_model_free(m); return -1; }
    }
    m->id = wnr4_fnv1a(blob, bytes);
    return 0;
}

/* Training counts, per subtype: hi[256] (u64), the hi histogram is also what picks the classes; then
 * per high byte (not per class: the classes are only known when the model is built) lo[256][256] and
 * nib[256][16]. */
#define WNR4_GG_COUNTS_SUB (256u + 256u * 256u + 256u * 16u)
static inline size_t wnr4_gg_counts_words(void) { return (size_t)WNR4_GG_SUBTYPES * WNR4_GG_COUNTS_SUB; }

static inline uint32_t wnr4_gg_nib(const unsigned char *cell, uint32_t i) { return (cell[i / 2u] >> ((i & 1u) * 4u)) & 15u; }

/* adds the counts of one record's decoded bytes (g->raw of them) */
static inline void wnr4_gg_count(const Wnr4GgGeo *g, const unsigned char *slab, uint64_t *counts) {
    uint32_t s, c, i;
    for (s = 0; s < g->nseg; s++) {
        uint64_t *k;
        if (g->sub[s] < 0) continue;
        k = counts + (size_t)g->sub[s] * WNR4_GG_COUNTS_SUB;
        for (c = 0; c < g->cells[s]; c++) {
            const unsigned char *b = slab + g->off[s] + (size_t)c * 18u;
            uint32_t hi = b[1];
            k[hi]++;
            k[256u + hi * 256u + b[0]]++;
            for (i = 0; i < 32u; i++) k[256u + 65536u + hi * 16u + wnr4_gg_nib(b + 2, i)]++;
        }
    }
}

/* classes: the 31 most frequent high bytes (ties: the smaller byte) get 1..31 in that order, the rest 0 */
static inline void wnr4_gg_classes(const uint64_t *hi_counts, unsigned char map[256]) {
    unsigned char used[256] = {0};
    uint32_t c, b;
    memset(map, 0, 256);
    for (c = 1; c < WNR4_GG_CLASSES; c++) {
        int best = -1;
        for (b = 0; b < 256u; b++)
            if (!used[b] && hi_counts[b] && (best < 0 || hi_counts[b] > hi_counts[best])) best = (int)b;
        if (best < 0) break;
        used[best] = 1; map[best] = (unsigned char)c;
    }
}

/* writes the WNR4_GG_MODEL_BYTES blob; returns 0, or -1 (out of memory: nothing usable written) */
static inline int wnr4_gg_model_build(const uint64_t *counts, unsigned char *blob) {
    uint32_t hdr[8] = {WNM4_MAGIC, 1u, WNR4_KIND_GG, WNR4_PREC, WNR4_GG_CLASSES, WNR4_GG_SUBTYPES, 0, 0};
    uint32_t s, i, c, b;
    uint16_t cdf[257];
    unsigned char *p = blob + 32;
    uint64_t *lo = (uint64_t *)malloc((size_t)WNR4_GG_CLASSES * 256u * 8u);
    if (!lo) return -1;
    for (i = 0; i < 8u; i++) wnr4_put32(blob + 4 * i, hdr[i]);
    for (s = 0; s < WNR4_GG_SUBTYPES; s++) {
        const uint64_t *k = counts + (size_t)s * WNR4_GG_COUNTS_SUB;
        unsigned char map[256];
        uint64_t nib[WNR4_GG_CLASSES * 16u];
        wnr4_gg_classes(k, map);
        memset(lo, 0, (size_t)WNR4_GG_CLASSES * 256u * 8u);
        memset(nib, 0, sizeof(nib));
        for (b = 0; b < 256u; b++) {
            for (i = 0; i < 256u; i++) lo[map[b] * 256u + i] += k[256u + b * 256u + i];
            for (i = 0; i < 16u; i++) nib[map[b] * 16u + i] += k[256u + 65536u + b * 16u + i];
        }
        memcpy(p, map, 256); p += 256;
        wnr4_quantize(k, 256u, cdf);
        for (i = 0; i < 257u; i++) { *p++ = (unsigned char)cdf[i]; *p++ = (unsigned char)(cdf[i] >> 8); }
        for (c = 0; c < WNR4_GG_CLASSES; c++) {
            wnr4_quantize(lo + c * 256u, 256u, cdf);
            for (i = 0; i < 257u; i++) { *p++ = (unsigned char)cdf[i]; *p++ = (unsigned char)(cdf[i] >> 8); }
        }
        for (c = 0; c < WNR4_GG_CLASSES; c++) {
            wnr4_quantize(nib + c * 16u, 16u, cdf);
            for (i = 0; i < 17u; i++) { *p++ = (unsigned char)cdf[i]; *p++ = (unsigned char)(cdf[i] >> 8); }
        }
    }
    free(lo);
    return 0;
}

/* ---- record ------------------------------------------------------------------- */
typedef struct {
    uint32_t model_id, owner, G, ntiles, payload;
    uint32_t tile_base[WNR4_GG_MAX_SEG + 1u];   /* first tile of each slice; [nseg] = ntiles */
    const unsigned char *offsets, *raw, *data;  /* tile offsets, verbatim bytes, tiles */
} Wnr4GgRecord;

static inline int wnr4_gg_parse(const unsigned char *rec, size_t total, const Wnr4GgGeo *g, Wnr4GgRecord *r) {
    uint32_t s, t, prev = 0;
    size_t used;
    memset(r, 0, sizeof(*r));
    if (!rec || total < WNR4_HEADER_BYTES || (total & 4095u) || wnr4_u32(rec) != WNR4_MAGIC) return -1;
    r->model_id = wnr4_u32(rec + 4); r->owner = wnr4_u32(rec + 8);
    r->G = (uint32_t)rec[14] | (uint32_t)rec[15] << 8;
    r->ntiles = wnr4_u32(rec + 20); r->payload = wnr4_u32(rec + 24);
    if (!r->model_id || rec[12] != WNR4_KIND_GG || rec[13] || r->G < 1u || r->G > WNR4_MAX_G) return -1;
    for (t = 16; t < 20; t++) if (rec[t]) return -1;
    for (t = 28; t < 64; t++) if (rec[t]) return -1;
    for (s = 0, t = 0; s < g->nseg; s++) { r->tile_base[s] = t; t += g->sub[s] >= 0 ? wnr4_tiles(g->cells[s], r->G) : 0u; }
    r->tile_base[g->nseg] = t;
    if (t != r->ntiles) return -1;
    used = WNR4_HEADER_BYTES + 4u * ((size_t)r->ntiles + 1u) + (((size_t)g->raw_bytes + 3u) & ~(size_t)3u) + r->payload;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    r->offsets = rec + WNR4_HEADER_BYTES;
    r->raw = r->offsets + 4u * ((size_t)r->ntiles + 1u);
    r->data = r->raw + (((size_t)g->raw_bytes + 3u) & ~(size_t)3u);
    for (used = g->raw_bytes; r->raw + used < r->data; used++) if (r->raw[used]) return -1;
    if (wnr4_u32(r->offsets)) return -1;
    for (t = 1; t <= r->ntiles; t++) {
        uint32_t o = wnr4_u32(r->offsets + 4u * t);
        if (o < prev || o - prev < WNR4_TILE_STATE_BYTES || ((o - prev) & 1u)) return -1;
        prev = o;
    }
    if (prev != r->payload) return -1;
    for (used = (size_t)(r->data - rec) + r->payload; used < total; used++) if (rec[used]) return -1;
    return 0;
}

/* tile t -> slice, first block, blocks of the tile */
static WNR4_HD inline void wnr4_gg_tile_cells(const uint32_t *tile_base, const uint32_t *cells, uint32_t nseg, uint32_t G,
                                              uint32_t t, uint32_t *s, uint32_t *c0, uint32_t *n) {
    uint32_t ss = 0, first, left;
    while (ss + 1u < nseg && t >= tile_base[ss + 1u]) ss++;
    first = (t - tile_base[ss]) * 32u * G; left = cells[ss] - first;
    *s = ss; *c0 = first; *n = left < 32u * G ? left : 32u * G;
}
/* hashed lane length: 4 bytes per scale symbol (hi and lo), 16 per cell */
static WNR4_HD inline uint32_t wnr4_gg_hlen(uint32_t nsl, uint32_t ncl) { return 24u + 8u * nsl + 16u * ncl; }

/* Decodes tile t into out (the record's decoded bytes). Returns 0 or error bits. */
static inline uint32_t wnr4_gg_decode_tile(const Wnr4GgModel *model, const Wnr4GgGeo *g, const Wnr4GgRecord *r,
                                           uint32_t t, unsigned char *out) {
    uint32_t s, c0, n, o0 = wnr4_u32(r->offsets + 4u * t), o1 = wnr4_u32(r->offsets + 4u * t + 4u);
    const unsigned char *tile = r->data + o0;
    uint32_t x[32], hash[32], ptr = 0, l, j, k, err = 0, H = 0, nwords;
    unsigned char *base;
    const uint16_t *hi_t, *lo_t, *nib_t;
    const unsigned char *map;
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r->payload) return WNR4_ERR_STRUCT;
    nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u;
    wnr4_gg_tile_cells(r->tile_base, g->cells, g->nseg, r->G, t, &s, &c0, &n);
    if (g->sub[s] < 0) return WNR4_ERR_STRUCT;
    base = out + g->off[s];
    hi_t = model->hi + (size_t)g->sub[s] * 257u;
    lo_t = model->lo + (size_t)g->sub[s] * WNR4_GG_CLASSES * 257u;
    nib_t = model->nib + (size_t)g->sub[s] * WNR4_GG_CLASSES * 17u;
    map = model->map[g->sub[s]];
    for (l = 0; l < 32u; l++) {
        x[l] = wnr4_u32(tile + 4u * l);
        if (x[l] < WNR4_L) return WNR4_ERR_STRUCT;
        hash[l] = wnr4_lane_seed(r->model_id, r->owner, t, l, wnr4_word(WNR4_KIND_GG, 0u, r->G), g->type[s]);
    }
#define WNR4_GG_RENORM(L_)                                                           \
    if (x[L_] < WNR4_L) {                                                            \
        if (ptr >= nwords) return WNR4_ERR_STRUCT;                                   \
        x[L_] = x[L_] << 16 | (uint32_t)(tile[128u + 2u * ptr] | tile[129u + 2u * ptr] << 8); ptr++; \
    }
    /* phases 1 and 2: high then low scale bytes, row-major across the lanes */
    for (k = 0; 32u * k < n; k++)
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l, slot, y;
            if (i >= n) continue;
            slot = x[l] & (WNR4_TOTAL - 1u); y = wnr4_find256(hi_t, slot);
            x[l] = (uint32_t)(hi_t[y + 1] - hi_t[y]) * (x[l] >> WNR4_PREC) + slot - hi_t[y];
            base[(size_t)(c0 + i) * 18u + 1u] = (unsigned char)y;
            hash[l] = wnr4_h32_block(hash[l], y);
            WNR4_GG_RENORM(l)
        }
    for (k = 0; 32u * k < n; k++)
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l, slot, y;
            const uint16_t *cdf;
            if (i >= n) continue;
            cdf = lo_t + (size_t)map[base[(size_t)(c0 + i) * 18u + 1u]] * 257u;
            slot = x[l] & (WNR4_TOTAL - 1u); y = wnr4_find256(cdf, slot);
            x[l] = (uint32_t)(cdf[y + 1] - cdf[y]) * (x[l] >> WNR4_PREC) + slot - cdf[y];
            base[(size_t)(c0 + i) * 18u] = (unsigned char)y;
            hash[l] = wnr4_h32_block(hash[l], y);
            WNR4_GG_RENORM(l)
        }
    /* phase 3: cells */
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, wd[32][4];
        const uint16_t *cdfs[32];
        for (l = 0; l < act; l++) {
            wd[l][0] = wd[l][1] = wd[l][2] = wd[l][3] = 0;
            cdfs[l] = nib_t + (size_t)map[base[(size_t)(c0 + 32u * j + l) * 18u + 1u]] * 17u;
        }
        for (k = 0; k < 32u; k++)
            for (l = 0; l < act; l++) {
                uint32_t slot, y;
                if (wnr4_tail_skip(l, j, k, n)) continue;
                slot = x[l] & (WNR4_TOTAL - 1u); y = wnr4_find16(cdfs[l], slot);
                x[l] = (uint32_t)(cdfs[l][y + 1] - cdfs[l][y]) * (x[l] >> WNR4_PREC) + slot - cdfs[l][y];
                wd[l][k / 8u] |= y << ((k % 8u) * 4u);
                WNR4_GG_RENORM(l)
            }
        for (l = 0; l < act; l++) {
            unsigned char *cell = base + (size_t)(c0 + 32u * j + l) * 18u + 2u;
            if (wnr4_tail_skip(l, j, 31u, n)) {
                uint32_t tail = x[l] - WNR4_L;
                if (x[l] < WNR4_L || tail > 0xffffu) return err | WNR4_ERR_STRUCT;
                wd[l][3] |= tail << 16;
                x[l] = WNR4_L;
            }
            for (k = 0; k < 4u; k++) { hash[l] = wnr4_h32_block(hash[l], wd[l][k]); wnr4_put32(cell + 4u * k, wd[l][k]); }
        }
    }
#undef WNR4_GG_RENORM
    for (l = 0; l < 32u; l++)
        H ^= wnr4_h32_final(hash[l], wnr4_gg_hlen(wnr4_lane_cells(n, l), wnr4_lane_cells(n, l)));
    if (ptr != nwords) err |= WNR4_ERR_STRUCT;
    if (x[0] != WNR4_L + (H & 0xffffu) || x[1] != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    for (l = 2; l < 32u; l++) if (x[l] != WNR4_L) err |= WNR4_ERR_STRUCT;
    return err;
}

/* the verbatim slices into out */
static inline void wnr4_gg_copy_raw(const Wnr4GgGeo *g, const Wnr4GgRecord *r, unsigned char *out) {
    uint32_t s;
    size_t at = 0;
    for (s = 0; s < g->nseg; s++)
        if (g->sub[s] < 0) { memcpy(out + g->off[s], r->raw + at, g->bytes[s]); at += g->bytes[s]; }
}

static inline uint32_t wnr4_gg_decode(const Wnr4GgModel *model, const Wnr4GgGeo *g, const unsigned char *rec,
                                      size_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4GgRecord r;
    uint32_t err = 0;
    int t;
    if (wnr4_gg_parse(rec, total, g, &r) || r.model_id != model->id || r.owner != owner) return WNR4_ERR_STRUCT;
    wnr4_gg_copy_raw(g, &r, out);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(| : err)
#endif
    for (t = 0; t < (int)r.ntiles; t++) err |= wnr4_gg_decode_tile(model, g, &r, (uint32_t)t, out);
    (void)threads;
    return err;
}

/* ---- encode ------------------------------------------------------------------- */
/* Encodes one record's decoded bytes (g->raw) into rec (capacity cap). Returns the record bytes (a multiple of
 * 4096), or 0 on error or overflow. */
static inline size_t wnr4_gg_encode(const Wnr4GgModel *model, const Wnr4GgGeo *g, const unsigned char *slab,
                                    uint32_t owner, uint32_t G, unsigned char *rec, size_t cap) {
    uint32_t ntiles = 0, s, t, l, k, j, tile_base[WNR4_GG_MAX_SEG + 1u];
    size_t at, payload0, raw_end, nsym = (size_t)32u * G * 34u + 64u, total;
    uint16_t *stack;
    uint32_t *sym;   /* per symbol in decode order: bits 30..31 table kind (1 hi, 2 lo, 0 nib), 24..28 lane,
                        8..15 class, 0..7 symbol */
    if (!G || G > WNR4_MAX_G) return 0;
    for (s = 0; s < g->nseg; s++) { tile_base[s] = ntiles; ntiles += g->sub[s] >= 0 ? wnr4_tiles(g->cells[s], G) : 0u; }
    tile_base[g->nseg] = ntiles;
    payload0 = WNR4_HEADER_BYTES + 4u * ((size_t)ntiles + 1u);
    raw_end = payload0 + (((size_t)g->raw_bytes + 3u) & ~(size_t)3u);
    if (raw_end > cap) return 0;
    memset(rec, 0, raw_end);
    wnr4_put32(rec, WNR4_MAGIC); wnr4_put32(rec + 4, model->id); wnr4_put32(rec + 8, owner);
    rec[12] = (unsigned char)WNR4_KIND_GG; rec[14] = (unsigned char)G; rec[15] = (unsigned char)(G >> 8);
    wnr4_put32(rec + 20, ntiles);
    for (s = 0, at = payload0; s < g->nseg; s++)
        if (g->sub[s] < 0) { memcpy(rec + at, slab + g->off[s], g->bytes[s]); at += g->bytes[s]; }
    stack = (uint16_t *)malloc(nsym * 2u);
    sym = (uint32_t *)malloc(nsym * 4u);
    if (!stack || !sym) { free(stack); free(sym); return 0; }
    at = raw_end;
    for (t = 0; t < ntiles; t++) {
        uint32_t c0, n, x[32], hash[32], H = 0, sub;
        size_t sp = 0, p, q = 0;
        const unsigned char *base, *map;
        wnr4_gg_tile_cells(tile_base, g->cells, g->nseg, G, t, &s, &c0, &n);
        sub = (uint32_t)g->sub[s]; base = slab + g->off[s]; map = model->map[sub];
        for (l = 0; l < 32u; l++)
            hash[l] = wnr4_lane_seed(model->id, owner, t, l, wnr4_word(WNR4_KIND_GG, 0u, G), g->type[s]);
        for (k = 0; 32u * k < n; k++)
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l, y;
                if (i >= n) continue;
                y = base[(size_t)(c0 + i) * 18u + 1u];
                sym[q++] = 0x40000000u | l << 24 | y;
                hash[l] = wnr4_h32_block(hash[l], y);
            }
        for (k = 0; 32u * k < n; k++)
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l, y;
                const unsigned char *b;
                if (i >= n) continue;
                b = base + (size_t)(c0 + i) * 18u; y = b[0];
                sym[q++] = 0x80000000u | l << 24 | (uint32_t)map[b[1]] << 8 | y;
                hash[l] = wnr4_h32_block(hash[l], y);
            }
        for (j = 0; 32u * j < n; j++) {
            uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u;
            for (k = 0; k < 32u; k++)
                for (l = 0; l < act; l++) {
                    const unsigned char *b = base + (size_t)(c0 + 32u * j + l) * 18u;
                    if (!wnr4_tail_skip(l, j, k, n)) sym[q++] = l << 24 | (uint32_t)map[b[1]] << 8 | wnr4_gg_nib(b + 2, k);
                }
            for (l = 0; l < act; l++)
                for (k = 0; k < 4u; k++)
                    hash[l] = wnr4_h32_block(hash[l], wnr4_u32(base + (size_t)(c0 + 32u * j + l) * 18u + 2u + 4u * k));
        }
        for (l = 0; l < 32u; l++)
            H ^= wnr4_h32_final(hash[l], wnr4_gg_hlen(wnr4_lane_cells(n, l), wnr4_lane_cells(n, l)));
        for (l = 0; l < 32u; l++) {
            x[l] = l == 0 ? WNR4_L + (H & 0xffffu) : l == 1 ? WNR4_L + (H >> 16) : WNR4_L;
            if (l >= 2u && l < n) {   /* the lane's last cell: nibbles 28..31 = code bytes 14..15 */
                const unsigned char *last = base + (size_t)(c0 + l + 32u * ((n - l - 1u) / 32u)) * 18u + 2u;
                x[l] += (uint32_t)last[14] | (uint32_t)last[15] << 8;
            }
        }
        for (p = q; p-- > 0;) {
            uint32_t e = sym[p], y = e & 0xffu, ln = (e >> 24) & 31u, cl = (e >> 8) & 0xffu, f, s0;
            const uint16_t *cdf = (e >> 30) == 1u ? model->hi + (size_t)sub * 257u
                                : (e >> 30) == 2u ? model->lo + ((size_t)sub * WNR4_GG_CLASSES + cl) * 257u
                                                  : model->nib + ((size_t)sub * WNR4_GG_CLASSES + cl) * 17u;
            s0 = cdf[y]; f = (uint32_t)cdf[y + 1] - s0;
            if (x[ln] >= (f << (32u - WNR4_PREC))) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
            x[ln] = ((x[ln] / f) << WNR4_PREC) + x[ln] % f + s0;
        }
        if (at + WNR4_TILE_STATE_BYTES + 2u * sp > cap) { free(stack); free(sym); return 0; }
        for (l = 0; l < 32u; l++) wnr4_put32(rec + at + 4u * l, x[l]);
        for (p = 0; p < sp; p++) {
            uint16_t w = stack[sp - 1u - p];
            rec[at + 128u + 2u * p] = (unsigned char)w; rec[at + 129u + 2u * p] = (unsigned char)(w >> 8);
        }
        at += WNR4_TILE_STATE_BYTES + 2u * sp;
        wnr4_put32(rec + WNR4_HEADER_BYTES + 4u * (t + 1u), (uint32_t)(at - raw_end));
    }
    free(stack); free(sym);
    wnr4_put32(rec + 24, (uint32_t)(at - raw_end));
    total = (at + 4095u) & ~(size_t)4095u;
    if (total > cap) return 0;
    memset(rec + at, 0, total - at);
    return total;
}

#endif
