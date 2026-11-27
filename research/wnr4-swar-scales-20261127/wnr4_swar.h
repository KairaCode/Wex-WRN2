/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 MXFP4 research prototypes (Item 3a + Item 3b), header-only C99:
 *
 *  - Prototype 3a (wnr4_srec16_*): standard WNR4-MX weights (16-symbol CDFs,
 *    pb = 1 or 5, 32-lane rANS, tail 4 nibbles in x0[2..31]) paired with a
 *    per-record 16-symbol anchor-relative UE8M0 scale table (s_rec16: u16
 *    presence mask for rel = s - anchor[m] + 7 in 0..14 plus bit 15 escape,
 *    quantized over present symbols only -> zero floor-1 waste; escaped scales
 *    carry their raw 8-bit byte as a uniform 256-symbol rANS step, f = 64).
 *
 *  - Prototype 3b (wnr4_swar_*): 8-magnitude carry-free 64-bit SWAR CDF search
 *    (1 x uint4 = 16 B/ctx) with 63 contexts/matrix (7 srel x 9 running_max =
 *    1,008 B SMEM/matrix, no previous_magnitude) + shared 16-bit forward rANS
 *    word stream + warp-coalesced 4-byte-aligned backward 32-bit raw sign
 *    stream (0 sign bits on zeros -> canonical zero by construction) + MAGT
 *    480-bit sign tail (up to 16 non-zero signs of each lane 2..31's last cell
 *    packed in x0[2..31] = L + bits) + s_rec16 Phase 1 + ceil-form + full
 *    32-bit MurmurHash3 covering scales, magnitudes, and raw sign bits.
 * ------------------------------------------------------------------------- */
#ifndef WNR4_SWAR_H
#define WNR4_SWAR_H

#include "../../wnr4/wnr4.h"

#define WNR4_SREC16_MAGIC UINT32_C(0x53345257)   /* "WR4S": WNR4-MX + s_rec16 */
#define WNR4_SWAR_MAGIC   UINT32_C(0x38345257)   /* "WR48": WNR4-SWAR-8MAG record */
#define WNM4_SWAR_MAGIC   UINT32_C(0x34575357)   /* "WSW4": WNR4-SWAR-8MAG model */

#define WNR4_SWAR_NCLS 7u
#define WNR4_SWAR_WCTX 63u                       /* 7 srel x 9 running_max (8 = block start, 0..7 = max) */
#define WNR4_SWAR_NSYM 8u
#define WNR4_SWAR_SREC16_MAX_BYTES 104u          /* max 3 * (1 + 15) * 2 = 96 B, aligned to 4 */

/* ---- shared helpers ------------------------------------------------------- */
static WNR4_HD inline uint32_t wnr4_swar_wctx(uint32_t cls, uint32_t mx) {
    return cls * 9u + mx;
}
static WNR4_HD inline uint32_t wnr4_swar_lane_seed(uint32_t model, uint32_t owner, uint32_t tile, uint32_t lane,
                                                   uint32_t word, uint32_t anchors, uint32_t srec_hash) {
    uint32_t h = wnr4_lane_seed(model, owner, tile, lane, word, anchors);
    return wnr4_h32_block(h, srec_hash);
}
static WNR4_HD inline uint32_t wnr4_swar_lane_hlen(uint32_t nsl, uint32_t ncl) {
    return 28u + 4u * nsl + 16u * ncl;
}
static inline uint16_t wnr4_u16(const unsigned char *p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}
static inline void wnr4_put16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
}

/* ---- s_rec16: per-record 16-symbol anchor-relative scale table ------------ */
/* Build 14-bit CDF over present symbols only (absent symbols have freq = 0)
 * and serialize into out[0..*out_bytes-1] (padded to a multiple of 4 bytes).
 * Returns 0 on success, -1 on error. */
static inline int wnr4_srec16_build(const Wnr4Geo *g, const unsigned char *coded, const unsigned char anchor[3],
                                    uint16_t scdf[3][17], unsigned char *out, uint32_t *out_bytes) {
    uint32_t m, i, b, at = 0;
    if (g->kind != WNR4_KIND_MX) return -1;
    for (m = 0; m < 3u; m++) {
        const unsigned char *s = coded + g->soff[m];
        const uint32_t n = g->scount[m];
        uint64_t cnt[16] = {0}, rem[16] = {0};
        uint32_t f[16] = {0}, present = 0, used = 0, spare, extra, last_b = 0;
        uint16_t mask = 0;
        unsigned char taken[16] = {0};
        if (!n) return -1;
        for (i = 0; i < n; i++) {
            int rel = (int)s[i] - (int)anchor[m] + 7;
            uint32_t sym = (rel >= 0 && rel <= 14) ? (uint32_t)rel : 15u;
            cnt[sym]++;
        }
        for (b = 0; b < 16u; b++) {
            if (cnt[b] > 0u) {
                mask |= (uint16_t)(1u << b);
                present++;
                last_b = b;
            }
        }
        if (!present) return -1;
        spare = WNR4_TOTAL - present;
        for (b = 0; b < 16u; b++) {
            if (!cnt[b]) continue;
            uint64_t scaled = cnt[b] * (uint64_t)spare;
            f[b] = 1u + (uint32_t)(scaled / (uint64_t)n);
            rem[b] = scaled % (uint64_t)n;
            used += f[b];
        }
        for (extra = WNR4_TOTAL - used; extra; extra--) {
            uint32_t best = 16u;
            for (b = 0; b < 16u; b++) {
                if (cnt[b] && !taken[b] && (best == 16u || rem[b] > rem[best])) best = b;
            }
            taken[best] = 1u;
            f[best]++;
        }
        used = 0;
        for (b = 0; b <= 16u; b++) {
            scdf[m][b] = (uint16_t)used;
            if (b < 16u) used += f[b];
        }
        wnr4_put16(out + at, mask);
        at += 2u;
        for (b = 0; b < 16u; b++) {
            if ((mask & (1u << b)) && b != last_b) {
                wnr4_put16(out + at, (uint16_t)f[b]);
                at += 2u;
            }
        }
    }
    if (at & 2u) {
        wnr4_put16(out + at, 0u);
        at += 2u;
    }
    *out_bytes = at;
    return 0;
}

/* Parse serialized s_rec16 block of exact length srec_bytes (must be a multiple
 * of 4). Validates mask != 0, in-window range 0..255, positive freqs, exact sum
 * to 16384, and zero alignment padding. */
static inline int wnr4_srec16_parse(const unsigned char *buf, uint32_t srec_bytes, const unsigned char anchor[3],
                                    uint16_t scdf[3][17]) {
    uint32_t m, b, at = 0;
    if (!srec_bytes || (srec_bytes & 3u) || srec_bytes > WNR4_SWAR_SREC16_MAX_BYTES) return -1;
    for (m = 0; m < 3u; m++) {
        uint16_t mask;
        uint32_t present = 0, last_b = 0, sum = 0, f[16] = {0}, acc = 0;
        if (at + 2u > srec_bytes) return -1;
        mask = wnr4_u16(buf + at);
        at += 2u;
        if (!mask) return -1;
        for (b = 0; b < 16u; b++) {
            if (mask & (1u << b)) {
                if (b < 15u) {
                    int val = (int)anchor[m] + (int)b - 7;
                    if (val < 0 || val > 255) return -1;
                }
                present++;
                last_b = b;
            }
        }
        if (at + 2u * (present - 1u) > srec_bytes) return -1;
        for (b = 0; b < 16u; b++) {
            if ((mask & (1u << b)) && b != last_b) {
                uint32_t fb = wnr4_u16(buf + at);
                at += 2u;
                if (!fb || sum + fb >= WNR4_TOTAL) return -1;
                f[b] = fb;
                sum += fb;
            }
        }
        f[last_b] = WNR4_TOTAL - sum;
        if (!f[last_b]) return -1;
        for (b = 0; b <= 16u; b++) {
            scdf[m][b] = (uint16_t)acc;
            if (b < 16u) acc += f[b];
        }
    }
    if (at + 2u == srec_bytes) {
        if (wnr4_u16(buf + at) != 0u) return -1;
        at += 2u;
    }
    return (at == srec_bytes) ? 0 : -1;
}

/* ==========================================================================
 * Prototype 3a: WNR4-MX + s_rec16 (isolated scale-table upgrade over WNR4-MX)
 * ========================================================================== */
typedef struct {
    uint32_t model_id, owner, kind, flags, G, ntiles, payload, srec_bytes, srec_hash;
    unsigned char anchor[3];
    uint16_t scdf[3][17];
    const unsigned char *offsets, *data;
    uint32_t tile_base[4];
} Wnr4Srec16Record;

static inline int wnr4_srec16_parse_rec(const unsigned char *rec, size_t total, const Wnr4Geo *g,
                                        Wnr4Srec16Record *r) {
    uint32_t m, t, prev = 0;
    size_t used;
    memset(r, 0, sizeof(*r));
    if (!rec || total < WNR4_HEADER_BYTES || (total & 4095u) || g->kind != WNR4_KIND_MX) return -1;
    if (wnr4_u32(rec) != WNR4_SREC16_MAGIC) return -1;
    r->model_id = wnr4_u32(rec + 4);
    r->owner = wnr4_u32(rec + 8);
    r->kind = rec[12];
    r->flags = rec[13];
    r->G = wnr4_u16(rec + 14);
    r->anchor[0] = rec[16];
    r->anchor[1] = rec[17];
    r->anchor[2] = rec[18];
    r->ntiles = wnr4_u32(rec + 20);
    r->payload = wnr4_u32(rec + 24);
    r->srec_bytes = wnr4_u16(rec + 28);
    if (!r->model_id || r->kind != WNR4_KIND_MX || rec[19] || r->G < 1u || r->G > WNR4_MAX_G) return -1;
    if (r->flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ)) return -1;
    for (t = 30; t < 64; t++) if (rec[t]) return -1;
    for (m = 0, t = 0; m < 3u; m++) { r->tile_base[m] = t; t += wnr4_tiles(g->cells[m], r->G); }
    r->tile_base[3] = t;
    if (t != r->ntiles) return -1;
    if (WNR4_HEADER_BYTES + (size_t)r->srec_bytes > total) return -1;
    if (wnr4_srec16_parse(rec + WNR4_HEADER_BYTES, r->srec_bytes, r->anchor, r->scdf)) return -1;
    r->srec_hash = wnr4_fnv1a(rec + WNR4_HEADER_BYTES, r->srec_bytes);
    used = WNR4_HEADER_BYTES + (size_t)r->srec_bytes + 4u * ((size_t)r->ntiles + 1u) + (size_t)r->payload;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    r->offsets = rec + WNR4_HEADER_BYTES + r->srec_bytes;
    r->data = r->offsets + 4u * ((size_t)r->ntiles + 1u);
    if (wnr4_u32(r->offsets)) return -1;
    for (t = 1; t <= r->ntiles; t++) {
        uint32_t o = wnr4_u32(r->offsets + 4u * t);
        if (o < prev || o - prev < WNR4_TILE_STATE_BYTES || ((o - prev) & 1u)) return -1;
        prev = o;
    }
    if (prev != r->payload) return -1;
    for (; used < total; used++) if (rec[used]) return -1;
    return 0;
}

static inline uint32_t wnr4_srec16_decode_tile(const Wnr4Model *model, const Wnr4Geo *g, const Wnr4Srec16Record *r,
                                               uint32_t t, unsigned char *out) {
    const uint32_t B = 32u, wctx = model->wctx;
    uint32_t m, c0, n, o0 = wnr4_u32(r->offsets + 4u * t), o1 = wnr4_u32(r->offsets + 4u * t + 4u);
    const unsigned char *tile = r->data + o0;
    const uint32_t nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u;
    uint32_t x[32], hash[32], ptr = 0, l, j, k, err = 0, H = 0;
    const uint32_t anchors = (uint32_t)r->anchor[0] | ((uint32_t)r->anchor[1] << 8) | ((uint32_t)r->anchor[2] << 16);
    wnr4_tile_cells(r->tile_base, g->cells, r->G, t, &m, &c0, &n);
    const uint16_t *wt = model->wcdf + (size_t)m * wctx * 17u;
    const uint16_t *scdf = r->scdf[m];
    const int ceil_f = (r->flags >> m) & 1u, cz = (r->flags & WNR4_FLAG_CZ) != 0;
    unsigned char *so = out + g->soff[m], *wo = out + g->woff[m];
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r->payload) return WNR4_ERR_STRUCT;
    for (l = 0; l < 32u; l++) {
        x[l] = wnr4_u32(tile + 4u * l);
        if (x[l] < WNR4_L) return WNR4_ERR_STRUCT;
        hash[l] = wnr4_swar_lane_seed(r->model_id, r->owner, t, l, wnr4_word(WNR4_KIND_MX, r->flags, r->G), anchors,
                                      r->srec_hash);
    }
#define WNR4_S16_RENORM(L_)                                                               \
    if (x[L_] < WNR4_L) {                                                                 \
        if (ptr >= nwords) return WNR4_ERR_STRUCT;                                        \
        x[L_] = (x[L_] << 16) | (uint32_t)wnr4_u16(tile + 128u + 2u * ptr);               \
        ptr++;                                                                            \
    }
    /* Phase 1: s_rec16 scales + flat 8-bit escape */
    for (k = 0; 32u * k < n; k++) {
        uint32_t sym[32], any_esc = 0;
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l;
            if (i >= n) { sym[l] = 0; continue; }
            uint32_t slot = x[l] & (WNR4_TOTAL - 1u);
            uint32_t s = wnr4_find16(scdf, slot);
            uint32_t f = (uint32_t)scdf[s + 1u] - scdf[s];
            x[l] = f * (x[l] >> WNR4_PREC) + slot - scdf[s];
            sym[l] = s;
            if (s == 15u) any_esc = 1u;
            WNR4_S16_RENORM(l)
        }
        if (any_esc) {
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l;
                if (i >= n || sym[l] != 15u) continue;
                uint32_t slot = x[l] & (WNR4_TOTAL - 1u);
                uint32_t raw_s = slot >> 6;
                int rel = (int)raw_s - (int)r->anchor[m] + 7;
                if (rel >= 0 && rel <= 14) err |= WNR4_ERR_STRUCT; /* non-canonical escape */
                x[l] = ((x[l] >> WNR4_PREC) << 6) | (slot & 63u);
                so[c0 + i] = (unsigned char)raw_s;
                hash[l] = wnr4_h32_block(hash[l], raw_s);
                WNR4_S16_RENORM(l)
            }
        }
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l;
            if (i >= n || sym[l] == 15u) continue;
            int val = (int)r->anchor[m] + (int)sym[l] - 7;
            if (val < 0 || val > 255) { err |= WNR4_ERR_STRUCT; continue; }
            so[c0 + i] = (unsigned char)val;
            hash[l] = wnr4_h32_block(hash[l], (uint32_t)val);
        }
    }
    /* Phase 2: identical to standard WNR4-MX */
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u;
        uint32_t pm[32], mx[32], cls[32], wd[32][4];
        for (l = 0; l < act; l++) {
            wd[l][0] = wd[l][1] = wd[l][2] = wd[l][3] = 0;
            cls[l] = wnr4_cls(so[c0 + 32u * j + l], r->anchor[m], 3);
            pm[l] = mx[l] = 8u;
        }
        for (k = 0; k < 32u; k++) {
            for (l = 0; l < act; l++) {
                if (wnr4_tail_skip(l, j, k, n)) continue;
                const uint16_t *cdf = wt + (size_t)wnr4_wrow(wnr4_wctx(cls[l], pm[l], mx[l]), model->pb, k, B) * 17u;
                uint32_t slot = x[l] & (WNR4_TOTAL - 1u), y = wnr4_find16(cdf, slot);
                x[l] = (uint32_t)(cdf[y + 1u] - cdf[y]) * (x[l] >> WNR4_PREC) + slot - cdf[y];
                if (cz && y == 8u) err |= WNR4_ERR_CZ;
                wd[l][k / 8u] |= y << ((k % 8u) * 4u);
                pm[l] = y & 7u;
                mx[l] = wnr4_mx(mx[l], pm[l]);
                WNR4_S16_RENORM(l)
            }
        }
        for (l = 0; l < act; l++) {
            uint32_t cell = c0 + 32u * j + l;
            if (wnr4_tail_skip(l, j, 31u, n)) {
                uint32_t tail = x[l] - WNR4_L, q;
                if (x[l] < WNR4_L || tail > 0xffffu) return err | WNR4_ERR_STRUCT;
                for (q = 0; q < 4u; q++) {
                    uint32_t y = (tail >> (4u * q)) & 15u;
                    if (cz && y == 8u) err |= WNR4_ERR_CZ;
                    mx[l] = wnr4_mx(mx[l], y & 7u);
                }
                wd[l][3] |= tail << 16;
                x[l] = WNR4_L;
            }
            for (k = 0; k < 4u; k++) hash[l] = wnr4_h32_block(hash[l], wd[l][k]);
            if (ceil_f && mx[l] == 5u) {
                if (!so[cell]) err |= WNR4_ERR_CEIL;
                else {
                    so[cell]--;
                    for (k = 0; k < 4u; k++) wd[l][k] = wnr4_ceil_dbl8(wd[l][k]);
                }
            }
            for (k = 0; k < 4u; k++) wnr4_put32(wo + (size_t)cell * 16u + 4u * k, wd[l][k]);
        }
    }
#undef WNR4_S16_RENORM
    for (l = 0; l < 32u; l++)
        H ^= wnr4_h32_final(hash[l], wnr4_swar_lane_hlen(wnr4_lane_scales(n, l), wnr4_lane_cells(n, l)));
    if (ptr != nwords) err |= WNR4_ERR_STRUCT;
    if (x[0] != WNR4_L + (H & 0xffffu) || x[1] != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    for (l = 2; l < 32u; l++) if (x[l] != WNR4_L) err |= WNR4_ERR_STRUCT;
    return err;
}

static inline uint32_t wnr4_srec16_decode(const Wnr4Model *model, const Wnr4Geo *g, const unsigned char *rec,
                                          size_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4Srec16Record r;
    uint32_t err = 0;
    int t;
    if (model->kind != WNR4_KIND_MX || wnr4_srec16_parse_rec(rec, total, g, &r) || r.model_id != model->id ||
        r.owner != owner)
        return WNR4_ERR_STRUCT;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(| : err)
#endif
    for (t = 0; t < (int)r.ntiles; t++) err |= wnr4_srec16_decode_tile(model, g, &r, (uint32_t)t, out);
    (void)threads;
    return err;
}

/* Encodes WNR4-MX + s_rec16; if scale_bytes_out != NULL, writes total scale
 * bytes (s_rec16 header + Phase 1 rANS words). */
static inline size_t wnr4_srec16_encode(const Wnr4Model *model, const Wnr4Geo *g, const unsigned char *coded,
                                        uint32_t flags, uint32_t owner, uint32_t G, unsigned char *rec, size_t cap,
                                        size_t *scale_bytes_out) {
    const uint32_t B = 32u, wctx = model->wctx;
    unsigned char anchor[3], srec_buf[WNR4_SWAR_SREC16_MAX_BYTES];
    uint16_t scdf[3][17];
    uint32_t srec_bytes = 0, srec_hash, ntiles = 0, m, t, l, k, j, tile_base[4];
    size_t at, payload0, nsym = (size_t)32u * G * 34u + 64u, total, scale_words_total = 0;
    uint16_t *stack;
    uint32_t *sym; /* bits 30..31: 0=weight, 1=srec16, 2=raw8; 24..28: lane; 8..23: table; 0..7: sym */
    if (!G || G > WNR4_MAX_G || g->kind != WNR4_KIND_MX || model->kind != WNR4_KIND_MX ||
        (flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ)))
        return 0;
    for (m = 0; m < 3u; m++) {
        anchor[m] = wnr4_anchor(coded + g->soff[m], g->scount[m]);
        tile_base[m] = ntiles;
        ntiles += wnr4_tiles(g->cells[m], G);
    }
    tile_base[3] = ntiles;
    if (wnr4_srec16_build(g, coded, anchor, scdf, srec_buf, &srec_bytes)) return 0;
    srec_hash = wnr4_fnv1a(srec_buf, srec_bytes);
    payload0 = WNR4_HEADER_BYTES + (size_t)srec_bytes + 4u * ((size_t)ntiles + 1u);
    if (payload0 > cap) return 0;
    memset(rec, 0, payload0);
    wnr4_put32(rec, WNR4_SREC16_MAGIC);
    wnr4_put32(rec + 4, model->id);
    wnr4_put32(rec + 8, owner);
    rec[12] = WNR4_KIND_MX;
    rec[13] = (unsigned char)flags;
    wnr4_put16(rec + 14, (uint16_t)G);
    rec[16] = anchor[0];
    rec[17] = anchor[1];
    rec[18] = anchor[2];
    wnr4_put32(rec + 20, ntiles);
    wnr4_put16(rec + 28, (uint16_t)srec_bytes);
    memcpy(rec + WNR4_HEADER_BYTES, srec_buf, srec_bytes);
    const uint32_t anchors = (uint32_t)anchor[0] | ((uint32_t)anchor[1] << 8) | ((uint32_t)anchor[2] << 16);
    stack = (uint16_t *)malloc(nsym * 2u);
    sym = (uint32_t *)malloc(nsym * 4u);
    if (!stack || !sym) { free(stack); free(sym); return 0; }
    at = payload0;
    for (t = 0; t < ntiles; t++) {
        uint32_t c0, n, x[32], hash[32], H = 0;
        size_t sp = 0, p, q = 0, q_scale_end = 0;
        wnr4_tile_cells(tile_base, g->cells, G, t, &m, &c0, &n);
        const unsigned char *sc = coded + g->soff[m], *wt = coded + g->woff[m];
        const uint32_t wbase = m * wctx;
        for (l = 0; l < 32u; l++)
            hash[l] = wnr4_swar_lane_seed(model->id, owner, t, l, wnr4_word(WNR4_KIND_MX, flags, G), anchors,
                                          srec_hash);
        /* Phase 1: s_rec16 scales + flat 8-bit escape */
        for (k = 0; 32u * k < n; k++) {
            uint32_t any_esc = 0;
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l;
                if (i >= n) continue;
                uint32_t y = sc[c0 + i];
                int rel = (int)y - (int)anchor[m] + 7;
                uint32_t s = (rel >= 0 && rel <= 14) ? (uint32_t)rel : 15u;
                sym[q++] = (1u << 30) | (l << 24) | s;
                if (s == 15u) any_esc = 1u;
                hash[l] = wnr4_h32_block(hash[l], y);
            }
            if (any_esc) {
                for (l = 0; l < 32u; l++) {
                    uint32_t i = 32u * k + l;
                    if (i >= n) continue;
                    uint32_t y = sc[c0 + i];
                    int rel = (int)y - (int)anchor[m] + 7;
                    if (rel < 0 || rel > 14) sym[q++] = (2u << 30) | (l << 24) | y;
                }
            }
        }
        q_scale_end = q;
        /* Phase 2: identical to standard WNR4-MX */
        for (j = 0; 32u * j < n; j++) {
            uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, pm[32], mx[32], cls[32];
            for (l = 0; l < act; l++) { cls[l] = wnr4_cls(sc[c0 + 32u * j + l], anchor[m], 3); pm[l] = mx[l] = 8u; }
            for (k = 0; k < 32u; k++) {
                for (l = 0; l < act; l++) {
                    uint32_t y = wnr4_nib(wt, c0 + 32u * j + l, k);
                    if (!wnr4_tail_skip(l, j, k, n))
                        sym[q++] = (l << 24) |
                                   ((wbase + wnr4_wrow(wnr4_wctx(cls[l], pm[l], mx[l]), model->pb, k, B)) << 8) | y;
                    pm[l] = y & 7u;
                    mx[l] = wnr4_mx(mx[l], pm[l]);
                }
            }
            for (l = 0; l < act; l++)
                for (k = 0; k < 4u; k++)
                    hash[l] = wnr4_h32_block(hash[l], wnr4_u32(wt + (size_t)(c0 + 32u * j + l) * 16u + 4u * k));
        }
        for (l = 0; l < 32u; l++)
            H ^= wnr4_h32_final(hash[l], wnr4_swar_lane_hlen(wnr4_lane_scales(n, l), wnr4_lane_cells(n, l)));
        for (l = 0; l < 32u; l++) {
            x[l] = l == 0 ? WNR4_L + (H & 0xffffu) : l == 1 ? WNR4_L + (H >> 16) : WNR4_L;
            if (l >= 2u && l < n) {
                const unsigned char *last = wt + (size_t)(c0 + l + 32u * ((n - l - 1u) / 32u)) * 16u;
                x[l] += (uint32_t)last[14] | ((uint32_t)last[15] << 8);
            }
        }
        size_t sp_after_weights = 0;
        for (p = q; p-- > 0;) {
            uint32_t e = sym[p], tag = e >> 30, ln = (e >> 24) & 31u, y = e & 0xffu;
            if (p + 1u == q_scale_end) sp_after_weights = sp;
            if (tag == 2u) {
                if ((x[ln] >> 18u) >= 64u) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
                x[ln] = ((x[ln] >> 6) << WNR4_PREC) | (y << 6) | (x[ln] & 63u);
            } else {
                const uint16_t *cdf = (tag == 1u) ? scdf[m] : model->wcdf + (size_t)((e >> 8) & 0xffffu) * 17u;
                uint32_t s0 = cdf[y], f = (uint32_t)cdf[y + 1u] - s0;
                if ((x[ln] >> 18u) >= f) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
                x[ln] = ((x[ln] / f) << WNR4_PREC) + (x[ln] % f) + s0;
            }
        }
        scale_words_total += (sp - sp_after_weights);
        if (at + WNR4_TILE_STATE_BYTES + 2u * sp > cap ||
            at + WNR4_TILE_STATE_BYTES + 2u * sp - payload0 > 0xffffffffu) {
            free(stack); free(sym); return 0;
        }
        for (l = 0; l < 32u; l++) wnr4_put32(rec + at + 4u * l, x[l]);
        for (p = 0; p < sp; p++) wnr4_put16(rec + at + 128u + 2u * p, stack[sp - 1u - p]);
        at += WNR4_TILE_STATE_BYTES + 2u * sp;
        wnr4_put32(rec + WNR4_HEADER_BYTES + srec_bytes + 4u * (t + 1u), (uint32_t)(at - payload0));
    }
    free(stack); free(sym);
    wnr4_put32(rec + 24, (uint32_t)(at - payload0));
    total = (at + 4095u) & ~(size_t)4095u;
    if (total > cap) return 0;
    memset(rec + at, 0, total - at);
    if (scale_bytes_out) *scale_bytes_out = (size_t)srec_bytes + 2u * scale_words_total;
    return total;
}

/* ==========================================================================
 * Prototype 3b: WNR4-SWAR-8MAG (63-ctx 8-mag SWAR + aligned raw signs +
 *               MAGT 480-bit tail + s_rec16 + 32-bit MurmurHash3)
 * ========================================================================== */
typedef struct {
    uint32_t id;
    uint16_t wcdf[3][WNR4_SWAR_WCTX][9]; /* [m][ctx][0..8], cdf[0]=0 .. cdf[8]=16384 */
} Wnr4SwarModel;

static inline size_t wnr4_swar_model_bytes(void) {
    return 32u + 3u * (size_t)WNR4_SWAR_WCTX * 9u * 2u; /* 3,434 B */
}
static inline size_t wnr4_swar_counts_words(void) {
    return 3u * (size_t)WNR4_SWAR_WCTX * WNR4_SWAR_NSYM; /* 1,512 u64s */
}
static inline void wnr4_swar_count(const Wnr4Geo *g, const unsigned char *coded, const unsigned char anchor[3],
                                   uint64_t *counts) {
    uint32_t m, c, i;
    if (g->kind != WNR4_KIND_MX) return;
    for (m = 0; m < 3u; m++) {
        const unsigned char *s = coded + g->soff[m], *w = coded + g->woff[m];
        for (c = 0; c < g->cells[m]; c++) {
            uint32_t cls = wnr4_cls(s[c], anchor[m], 3), mx = 8u;
            for (i = 0; i < 32u; i++) {
                uint32_t mag = wnr4_nib(w, c, i) & 7u;
                counts[((size_t)m * WNR4_SWAR_WCTX + wnr4_swar_wctx(cls, mx)) * 8u + mag]++;
                mx = wnr4_mx(mx, mag);
            }
        }
    }
}
static inline void wnr4_swar_model_build(const uint64_t *counts, unsigned char *blob) {
    uint32_t hdr[8] = {WNM4_SWAR_MAGIC, 1u, WNR4_KIND_MX, WNR4_PREC, WNR4_SWAR_WCTX, WNR4_SWAR_NSYM, 0u, 0u};
    uint16_t cdf[9];
    uint32_t i, k;
    unsigned char *p = blob + 32u;
    for (i = 0; i < 8u; i++) wnr4_put32(blob + 4u * i, hdr[i]);
    for (i = 0; i < 3u * WNR4_SWAR_WCTX; i++) {
        wnr4_quantize(counts + (size_t)i * 8u, 8u, cdf);
        for (k = 0; k < 9u; k++) { wnr4_put16(p, cdf[k]); p += 2; }
    }
}
static inline int wnr4_swar_model_open(const unsigned char *blob, size_t bytes, Wnr4SwarModel *m) {
    uint32_t mm, ctx, k;
    const unsigned char *p;
    memset(m, 0, sizeof(*m));
    if (!blob || bytes != wnr4_swar_model_bytes()) return -1;
    if (wnr4_u32(blob) != WNM4_SWAR_MAGIC || wnr4_u32(blob + 4) != 1u || wnr4_u32(blob + 8) != WNR4_KIND_MX ||
        wnr4_u32(blob + 12) != WNR4_PREC || wnr4_u32(blob + 16) != WNR4_SWAR_WCTX ||
        wnr4_u32(blob + 20) != WNR4_SWAR_NSYM || wnr4_u32(blob + 24) || wnr4_u32(blob + 28))
        return -1;
    p = blob + 32u;
    for (mm = 0; mm < 3u; mm++) {
        for (ctx = 0; ctx < WNR4_SWAR_WCTX; ctx++) {
            for (k = 0; k < 9u; k++, p += 2) m->wcdf[mm][ctx][k] = wnr4_u16(p);
            if (!wnr4_cdf_ok(m->wcdf[mm][ctx], 8u)) return -1;
        }
    }
    m->id = wnr4_fnv1a(blob, bytes);
    return 0;
}

/* Carry-free 64-bit SWAR 8-symbol CDF search (exact same formula as CUDA). */
static inline uint32_t wnr4_swar_find8(const uint16_t *cdf, uint32_t slot) {
    const uint64_t lo = (uint64_t)cdf[1] | ((uint64_t)cdf[2] << 16) | ((uint64_t)cdf[3] << 32) |
                        ((uint64_t)cdf[4] << 48);
    const uint64_t hi = (uint64_t)cdf[5] | ((uint64_t)cdf[6] << 16) | ((uint64_t)cdf[7] << 32) |
                        ((uint64_t)WNR4_TOTAL << 48);
    const uint64_t s4 = ((uint64_t)slot * UINT64_C(0x0001000100010001)) | UINT64_C(0x8000800080008000);
    const uint64_t d0 = (s4 - lo) & UINT64_C(0x8000800080008000);
    const uint64_t d1 = (s4 - hi) & UINT64_C(0x8000800080008000);
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)(__builtin_popcountll(d0) + __builtin_popcountll(d1));
#else
    const uint64_t c0 = ((d0 >> 15) * UINT64_C(0x0001000100010001)) >> 48;
    const uint64_t c1 = ((d1 >> 15) * UINT64_C(0x0001000100010001)) >> 48;
    return (uint32_t)((c0 & 7u) + (c1 & 7u));
#endif
}

typedef struct {
    uint32_t model_id, owner, kind, flags, G, ntiles, payload, srec_bytes, srec_hash;
    unsigned char anchor[3];
    uint16_t scdf[3][17];
    const unsigned char *offsets, *data;
    uint32_t tile_base[4];
} Wnr4SwarRecord;

static inline int wnr4_swar_parse(const unsigned char *rec, size_t total, const Wnr4Geo *g, Wnr4SwarRecord *r) {
    uint32_t m, t, prev = 0;
    size_t used;
    memset(r, 0, sizeof(*r));
    if (!rec || total < WNR4_HEADER_BYTES || (total & 4095u) || g->kind != WNR4_KIND_MX) return -1;
    if (wnr4_u32(rec) != WNR4_SWAR_MAGIC) return -1;
    r->model_id = wnr4_u32(rec + 4);
    r->owner = wnr4_u32(rec + 8);
    r->kind = rec[12];
    r->flags = rec[13];
    r->G = wnr4_u16(rec + 14);
    r->anchor[0] = rec[16];
    r->anchor[1] = rec[17];
    r->anchor[2] = rec[18];
    r->ntiles = wnr4_u32(rec + 20);
    r->payload = wnr4_u32(rec + 24);
    r->srec_bytes = wnr4_u16(rec + 28);
    if (!r->model_id || r->kind != WNR4_KIND_MX || rec[19] || r->G < 1u || r->G > WNR4_MAX_G) return -1;
    if (!(r->flags & WNR4_FLAG_CZ) || (r->flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ))) return -1;
    if (r->payload & 3u) return -1;
    for (t = 30; t < 64; t++) if (rec[t]) return -1;
    for (m = 0, t = 0; m < 3u; m++) { r->tile_base[m] = t; t += wnr4_tiles(g->cells[m], r->G); }
    r->tile_base[3] = t;
    if (t != r->ntiles) return -1;
    if (WNR4_HEADER_BYTES + (size_t)r->srec_bytes > total) return -1;
    if (wnr4_srec16_parse(rec + WNR4_HEADER_BYTES, r->srec_bytes, r->anchor, r->scdf)) return -1;
    r->srec_hash = wnr4_fnv1a(rec + WNR4_HEADER_BYTES, r->srec_bytes);
    used = WNR4_HEADER_BYTES + (size_t)r->srec_bytes + 4u * ((size_t)r->ntiles + 1u) + (size_t)r->payload;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    r->offsets = rec + WNR4_HEADER_BYTES + r->srec_bytes;
    r->data = r->offsets + 4u * ((size_t)r->ntiles + 1u);
    if (wnr4_u32(r->offsets)) return -1;
    for (t = 1; t <= r->ntiles; t++) {
        uint32_t o = wnr4_u32(r->offsets + 4u * t);
        if (o < prev || o - prev < WNR4_TILE_STATE_BYTES || ((o - prev) & 3u)) return -1;
        prev = o;
    }
    if (prev != r->payload) return -1;
    for (; used < total; used++) if (rec[used]) return -1;
    return 0;
}

static inline uint32_t wnr4_swar_decode_tile(const Wnr4SwarModel *model, const Wnr4Geo *g, const Wnr4SwarRecord *r,
                                             uint32_t t, unsigned char *out) {
    uint32_t m, c0, n, o0 = wnr4_u32(r->offsets + 4u * t), o1 = wnr4_u32(r->offsets + 4u * t + 4u);
    const unsigned char *tile = r->data + o0;
    const uint32_t tile_bytes = o1 - o0;
    const uint32_t max_words = (tile_bytes - WNR4_TILE_STATE_BYTES) / 2u;
    const uint32_t max_swords = (tile_bytes - WNR4_TILE_STATE_BYTES) / 4u;
    uint32_t x[32], hash[32], ptr = 0, sw_read = 0, sbits = 0, l, j, k, err = 0, H = 0;
    uint64_t sbuf = 0;
    const uint32_t anchors = (uint32_t)r->anchor[0] | ((uint32_t)r->anchor[1] << 8) | ((uint32_t)r->anchor[2] << 16);
    wnr4_tile_cells(r->tile_base, g->cells, r->G, t, &m, &c0, &n);
    const uint16_t *scdf = r->scdf[m];
    const int ceil_f = (r->flags >> m) & 1u;
    unsigned char *so = out + g->soff[m], *wo = out + g->woff[m];
    uint32_t pw[32][4], pmx[32], pcell[32];
    unsigned char pending[32] = {0}, ps[32] = {0};
    if (o1 < o0 || tile_bytes < WNR4_TILE_STATE_BYTES || (tile_bytes & 3u) || o1 > r->payload)
        return WNR4_ERR_STRUCT;
    for (l = 0; l < 32u; l++) {
        x[l] = wnr4_u32(tile + 4u * l);
        if (x[l] < WNR4_L) return WNR4_ERR_STRUCT;
        hash[l] = wnr4_swar_lane_seed(r->model_id, r->owner, t, l, wnr4_word(WNR4_KIND_MX, r->flags, r->G), anchors,
                                      r->srec_hash);
    }
#define WNR4_SW_RENORM(L_)                                                                \
    if (x[L_] < WNR4_L) {                                                                 \
        if (ptr >= max_words) return WNR4_ERR_STRUCT;                                     \
        x[L_] = (x[L_] << 16) | (uint32_t)wnr4_u16(tile + 128u + 2u * ptr);               \
        ptr++;                                                                            \
    }
    /* Phase 1: s_rec16 scales + flat 8-bit escape */
    for (k = 0; 32u * k < n; k++) {
        uint32_t sym[32], any_esc = 0;
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l;
            if (i >= n) { sym[l] = 0; continue; }
            uint32_t slot = x[l] & (WNR4_TOTAL - 1u);
            uint32_t s = wnr4_find16(scdf, slot);
            uint32_t f = (uint32_t)scdf[s + 1u] - scdf[s];
            x[l] = f * (x[l] >> WNR4_PREC) + slot - scdf[s];
            sym[l] = s;
            if (s == 15u) any_esc = 1u;
            WNR4_SW_RENORM(l)
        }
        if (any_esc) {
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l;
                if (i >= n || sym[l] != 15u) continue;
                uint32_t slot = x[l] & (WNR4_TOTAL - 1u);
                uint32_t raw_s = slot >> 6;
                int rel = (int)raw_s - (int)r->anchor[m] + 7;
                if (rel >= 0 && rel <= 14) err |= WNR4_ERR_STRUCT;
                x[l] = ((x[l] >> WNR4_PREC) << 6) | (slot & 63u);
                so[c0 + i] = (unsigned char)raw_s;
                hash[l] = wnr4_h32_block(hash[l], raw_s);
                WNR4_SW_RENORM(l)
            }
        }
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l;
            if (i >= n || sym[l] == 15u) continue;
            int val = (int)r->anchor[m] + (int)sym[l] - 7;
            if (val < 0 || val > 255) { err |= WNR4_ERR_STRUCT; continue; }
            so[c0 + i] = (unsigned char)val;
            hash[l] = wnr4_h32_block(hash[l], (uint32_t)val);
        }
    }
    /* Phase 2: 63-ctx 8-magnitude SWAR + backward aligned 32-bit sign stream + MAGT */
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u;
        uint32_t mx[32], cls[32], wd[32][4], tleft[32];
        unsigned char is_last[32];
        for (l = 0; l < act; l++) {
            wd[l][0] = wd[l][1] = wd[l][2] = wd[l][3] = 0;
            cls[l] = wnr4_cls(so[c0 + 32u * j + l], r->anchor[m], 3);
            mx[l] = 8u;
            is_last[l] = (unsigned char)(l >= 2u && 32u * j + l + 32u >= n);
            tleft[l] = is_last[l] ? 16u : 0u;
        }
        for (k = 0; k < 32u; k++) {
            uint32_t mags[32], need_sign[32], cnt = 0;
            for (l = 0; l < act; l++) {
                const uint16_t *cdf = model->wcdf[m][wnr4_swar_wctx(cls[l], mx[l])];
                uint32_t slot = x[l] & (WNR4_TOTAL - 1u);
                uint32_t mag = wnr4_swar_find8(cdf, slot);
                uint32_t f = (uint32_t)cdf[mag + 1u] - cdf[mag];
                x[l] = f * (x[l] >> WNR4_PREC) + slot - cdf[mag];
                mags[l] = mag;
                mx[l] = wnr4_mx(mx[l], mag);
                if (mag != 0u) {
                    if (tleft[l] != 0u) { tleft[l]--; need_sign[l] = 0u; }
                    else { need_sign[l] = 1u; cnt++; }
                } else {
                    need_sign[l] = 0u;
                }
                WNR4_SW_RENORM(l)
            }
            if (sbits < cnt) {
                if (sw_read >= max_swords) return WNR4_ERR_STRUCT;
                sw_read++;
                sbuf |= (uint64_t)wnr4_u32(tile + tile_bytes - 4u * sw_read) << sbits;
                sbits += 32u;
            }
            for (l = 0; l < act; l++) {
                uint32_t sign = 0u;
                if (need_sign[l]) {
                    sign = (uint32_t)(sbuf & 1u);
                    sbuf >>= 1;
                    sbits--;
                }
                wd[l][k / 8u] |= ((sign << 3) | mags[l]) << ((k % 8u) * 4u);
            }
        }
        for (l = 0; l < act; l++) {
            uint32_t cell = c0 + 32u * j + l;
            if (is_last[l]) {
                pending[l] = 1u;
                pw[l][0] = wd[l][0]; pw[l][1] = wd[l][1]; pw[l][2] = wd[l][2]; pw[l][3] = wd[l][3];
                pmx[l] = mx[l];
                ps[l] = so[cell];
                pcell[l] = cell;
            } else {
                for (k = 0; k < 4u; k++) hash[l] = wnr4_h32_block(hash[l], wd[l][k]);
                if (ceil_f && mx[l] == 5u) {
                    if (!so[cell]) err |= WNR4_ERR_CEIL;
                    else {
                        so[cell]--;
                        for (k = 0; k < 4u; k++) wd[l][k] = wnr4_ceil_dbl8(wd[l][k]);
                    }
                }
                for (k = 0; k < 4u; k++) wnr4_put32(wo + (size_t)cell * 16u + 4u * k, wd[l][k]);
            }
        }
    }
#undef WNR4_SW_RENORM
    /* Finalize MAGT pending cells from x[2..31] */
    for (l = 2u; l < 32u; l++) {
        if (!pending[l]) continue;
        uint32_t tail = x[l] - WNR4_L, bit = 0, q, r4;
        if (x[l] < WNR4_L || tail > 0xffffu) return err | WNR4_ERR_STRUCT;
        for (q = 0; q < 4u; q++) {
            uint32_t wq = pw[l][q];
            for (r4 = 0; r4 < 8u; r4++) {
                uint32_t mag = (wq >> (4u * r4)) & 7u;
                if (mag != 0u && bit < 16u) {
                    wq |= ((tail >> bit) & 1u) << (4u * r4 + 3u);
                    bit++;
                }
            }
            pw[l][q] = wq;
        }
        if (tail >> bit) err |= WNR4_ERR_STRUCT; /* unused high tail bits must be 0 */
        x[l] = WNR4_L;
        for (q = 0; q < 4u; q++) hash[l] = wnr4_h32_block(hash[l], pw[l][q]);
        if (ceil_f && pmx[l] == 5u) {
            if (!ps[l]) err |= WNR4_ERR_CEIL;
            else {
                so[pcell[l]] = (unsigned char)(ps[l] - 1u);
                for (q = 0; q < 4u; q++) pw[l][q] = wnr4_ceil_dbl8(pw[l][q]);
            }
        }
        for (q = 0; q < 4u; q++) wnr4_put32(wo + (size_t)pcell[l] * 16u + 4u * q, pw[l][q]);
    }
    /* Exact forward/backward pointer meet + zero pad + zero unused high sign bits */
    {
        uint32_t fwd_end = WNR4_TILE_STATE_BYTES + 2u * ptr;
        uint32_t sign_beg = tile_bytes - 4u * sw_read;
        uint32_t pad2 = (ptr & 1u) ? 2u : 0u;
        if (fwd_end > sign_beg || sign_beg - fwd_end != pad2) err |= WNR4_ERR_STRUCT;
        else if (pad2 && wnr4_u16(tile + fwd_end) != 0u) err |= WNR4_ERR_STRUCT;
        if (sbits >= 32u || sbuf != 0ULL) err |= WNR4_ERR_STRUCT;
    }
    for (l = 0; l < 32u; l++)
        H ^= wnr4_h32_final(hash[l], wnr4_swar_lane_hlen(wnr4_lane_scales(n, l), wnr4_lane_cells(n, l)));
    if (x[0] != WNR4_L + (H & 0xffffu) || x[1] != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    for (l = 2u; l < 32u; l++) if (x[l] != WNR4_L) err |= WNR4_ERR_STRUCT;
    return err;
}

static inline uint32_t wnr4_swar_decode(const Wnr4SwarModel *model, const Wnr4Geo *g, const unsigned char *rec,
                                        size_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4SwarRecord r;
    uint32_t err = 0;
    int t;
    if (wnr4_swar_parse(rec, total, g, &r) || r.model_id != model->id || r.owner != owner) return WNR4_ERR_STRUCT;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(| : err)
#endif
    for (t = 0; t < (int)r.ntiles; t++) err |= wnr4_swar_decode_tile(model, g, &r, (uint32_t)t, out);
    (void)threads;
    return err;
}

static inline int wnr4_swar_decode_batch(const Wnr4SwarModel *model, const Wnr4Geo *g, int n,
                                         const unsigned char *const *recs, const size_t *totals,
                                         const uint32_t *owners, unsigned char *const *outs, uint32_t *status,
                                         int threads) {
    Wnr4SwarRecord *r = (Wnr4SwarRecord *)calloc((size_t)(n > 0 ? n : 1), sizeof(Wnr4SwarRecord));
    long long *first = (long long *)calloc((size_t)(n > 0 ? n : 1) + 1u, sizeof(long long));
    long long t, total = 0;
    int i, bad = 0;
    if (!r || !first) { free(r); free(first); for (i = 0; i < n; i++) status[i] = WNR4_ERR_STRUCT; return n; }
    for (i = 0; i < n; i++) {
        status[i] = wnr4_swar_parse(recs[i], totals[i], g, &r[i]) || r[i].model_id != model->id ||
                    r[i].owner != owners[i] ? WNR4_ERR_STRUCT : 0u;
        first[i] = total;
        if (!status[i]) total += r[i].ntiles;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1)
#endif
    for (t = 0; t < total; t++) {
        int lo = 0, hi = n;
        while (hi - lo > 1) { int mid = (lo + hi) / 2; if (first[mid] <= t) lo = mid; else hi = mid; }
        uint32_t e = wnr4_swar_decode_tile(model, g, &r[lo], (uint32_t)(t - first[lo]), outs[lo]);
        if (e) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            status[lo] |= e;
        }
    }
    for (i = 0; i < n; i++) bad += status[i] != 0;
    free(r); free(first);
    (void)threads;
    return bad;
}

/* Optional breakdown stats for research instrumentation */
typedef struct {
    size_t srec_header_bytes;
    size_t scale_rans_bytes;
    size_t mag_rans_bytes;
    size_t sign_stream_bytes;
    size_t magt_saved_bits;
    size_t tile_pad2_bytes;
} Wnr4SwarStats;

static inline size_t wnr4_swar_encode(const Wnr4SwarModel *model, const Wnr4Geo *g, const unsigned char *coded,
                                      uint32_t flags, uint32_t owner, uint32_t G, unsigned char *rec, size_t cap,
                                      Wnr4SwarStats *stats_out) {
    unsigned char anchor[3], srec_buf[WNR4_SWAR_SREC16_MAX_BYTES];
    uint16_t scdf[3][17];
    uint32_t srec_bytes = 0, srec_hash, ntiles = 0, m, t, l, k, j, tile_base[4];
    size_t at, payload0, nsym = (size_t)32u * G * 34u + 64u, total;
    uint16_t *stack;
    uint32_t *sym, *sign_words;
    Wnr4SwarStats st;
    memset(&st, 0, sizeof(st));
    if (!G || G > WNR4_MAX_G || g->kind != WNR4_KIND_MX || !(flags & WNR4_FLAG_CZ) ||
        (flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ)))
        return 0;
    for (m = 0; m < 3u; m++) {
        anchor[m] = wnr4_anchor(coded + g->soff[m], g->scount[m]);
        tile_base[m] = ntiles;
        ntiles += wnr4_tiles(g->cells[m], G);
    }
    tile_base[3] = ntiles;
    if (wnr4_srec16_build(g, coded, anchor, scdf, srec_buf, &srec_bytes)) return 0;
    srec_hash = wnr4_fnv1a(srec_buf, srec_bytes);
    st.srec_header_bytes = srec_bytes;
    payload0 = WNR4_HEADER_BYTES + (size_t)srec_bytes + 4u * ((size_t)ntiles + 1u);
    if (payload0 > cap) return 0;
    memset(rec, 0, payload0);
    wnr4_put32(rec, WNR4_SWAR_MAGIC);
    wnr4_put32(rec + 4, model->id);
    wnr4_put32(rec + 8, owner);
    rec[12] = WNR4_KIND_MX;
    rec[13] = (unsigned char)flags;
    wnr4_put16(rec + 14, (uint16_t)G);
    rec[16] = anchor[0];
    rec[17] = anchor[1];
    rec[18] = anchor[2];
    wnr4_put32(rec + 20, ntiles);
    wnr4_put16(rec + 28, (uint16_t)srec_bytes);
    memcpy(rec + WNR4_HEADER_BYTES, srec_buf, srec_bytes);
    const uint32_t anchors = (uint32_t)anchor[0] | ((uint32_t)anchor[1] << 8) | ((uint32_t)anchor[2] << 16);
    stack = (uint16_t *)malloc(nsym * 2u);
    sym = (uint32_t *)malloc(nsym * 4u);
    sign_words = (uint32_t *)malloc(((size_t)32u * G + 4u) * 4u);
    if (!stack || !sym || !sign_words) { free(stack); free(sym); free(sign_words); return 0; }
    at = payload0;
    for (t = 0; t < ntiles; t++) {
        uint32_t c0, n, x[32], hash[32], H = 0;
        uint32_t tail_bits[32] = {0}, tail_cnt[32] = {0};
        uint32_t cur_sw = 0, cur_sb = 0, nsw = 0, u;
        size_t sp = 0, p, q = 0, q_scale_end = 0, sp_after_mags = 0;
        wnr4_tile_cells(tile_base, g->cells, G, t, &m, &c0, &n);
        const unsigned char *sc = coded + g->soff[m], *wt = coded + g->woff[m];
        for (l = 0; l < 32u; l++)
            hash[l] = wnr4_swar_lane_seed(model->id, owner, t, l, wnr4_word(WNR4_KIND_MX, flags, G), anchors,
                                          srec_hash);
        /* Phase 1: s_rec16 scales + flat 8-bit escape */
        for (k = 0; 32u * k < n; k++) {
            uint32_t any_esc = 0;
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l;
                if (i >= n) continue;
                uint32_t y = sc[c0 + i];
                int rel = (int)y - (int)anchor[m] + 7;
                uint32_t s = (rel >= 0 && rel <= 14) ? (uint32_t)rel : 15u;
                sym[q++] = (1u << 30) | (l << 24) | s;
                if (s == 15u) any_esc = 1u;
                hash[l] = wnr4_h32_block(hash[l], y);
            }
            if (any_esc) {
                for (l = 0; l < 32u; l++) {
                    uint32_t i = 32u * k + l;
                    if (i >= n) continue;
                    uint32_t y = sc[c0 + i];
                    int rel = (int)y - (int)anchor[m] + 7;
                    if (rel < 0 || rel > 14) sym[q++] = (2u << 30) | (l << 24) | y;
                }
            }
        }
        q_scale_end = q;
        /* Phase 2: magnitudes + raw signs + MAGT */
        for (j = 0; 32u * j < n; j++) {
            uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, mx[32], cls[32];
            unsigned char is_last[32];
            for (l = 0; l < act; l++) {
                cls[l] = wnr4_cls(sc[c0 + 32u * j + l], anchor[m], 3);
                mx[l] = 8u;
                is_last[l] = (unsigned char)(l >= 2u && 32u * j + l + 32u >= n);
            }
            for (k = 0; k < 32u; k++) {
                for (l = 0; l < act; l++) {
                    uint32_t nib = wnr4_nib(wt, c0 + 32u * j + l, k);
                    if (nib == 8u) { free(stack); free(sym); free(sign_words); return 0; } /* non-CZ refused */
                    uint32_t mag = nib & 7u, sign = nib >> 3;
                    sym[q++] = (l << 24) | (wnr4_swar_wctx(cls[l], mx[l]) << 8) | mag;
                    mx[l] = wnr4_mx(mx[l], mag);
                    if (mag != 0u) {
                        if (is_last[l] && tail_cnt[l] < 16u) {
                            tail_bits[l] |= (sign << tail_cnt[l]);
                            tail_cnt[l]++;
                        } else {
                            cur_sw |= (sign << cur_sb);
                            cur_sb++;
                            if (cur_sb == 32u) {
                                sign_words[nsw++] = cur_sw;
                                cur_sw = 0;
                                cur_sb = 0;
                            }
                        }
                    }
                }
            }
            for (l = 0; l < act; l++)
                for (k = 0; k < 4u; k++)
                    hash[l] = wnr4_h32_block(hash[l], wnr4_u32(wt + (size_t)(c0 + 32u * j + l) * 16u + 4u * k));
        }
        if (cur_sb > 0u) sign_words[nsw++] = cur_sw;
        for (l = 0; l < 32u; l++)
            H ^= wnr4_h32_final(hash[l], wnr4_swar_lane_hlen(wnr4_lane_scales(n, l), wnr4_lane_cells(n, l)));
        x[0] = WNR4_L + (H & 0xffffu);
        x[1] = WNR4_L + (H >> 16);
        for (l = 2u; l < 32u; l++) {
            x[l] = WNR4_L + tail_bits[l];
            st.magt_saved_bits += tail_cnt[l];
        }
        /* Backward rANS encode */
        for (p = q; p-- > 0;) {
            uint32_t e = sym[p], tag = e >> 30, ln = (e >> 24) & 31u, y = e & 0xffu;
            if (p + 1u == q_scale_end) sp_after_mags = sp;
            if (tag == 2u) {
                if ((x[ln] >> 18u) >= 64u) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
                x[ln] = ((x[ln] >> 6) << WNR4_PREC) | (y << 6) | (x[ln] & 63u);
            } else {
                const uint16_t *cdf = (tag == 1u) ? scdf[m] : model->wcdf[m][(e >> 8) & 0xffu];
                uint32_t s0 = cdf[y], f = (uint32_t)cdf[y + 1u] - s0;
                if ((x[ln] >> 18u) >= f) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
                x[ln] = ((x[ln] / f) << WNR4_PREC) + (x[ln] % f) + s0;
            }
        }
        st.mag_rans_bytes += 2u * sp_after_mags;
        st.scale_rans_bytes += 2u * (sp - sp_after_mags);
        st.sign_stream_bytes += 4u * (size_t)nsw;
        {
            uint32_t pad2 = (sp & 1u) ? 2u : 0u;
            size_t tile_bytes = WNR4_TILE_STATE_BYTES + 2u * sp + pad2 + 4u * (size_t)nsw;
            st.tile_pad2_bytes += pad2;
            if (at + tile_bytes > cap || at + tile_bytes - payload0 > 0xffffffffu) {
                free(stack); free(sym); free(sign_words); return 0;
            }
            for (l = 0; l < 32u; l++) wnr4_put32(rec + at + 4u * l, x[l]);
            for (p = 0; p < sp; p++) wnr4_put16(rec + at + 128u + 2u * p, stack[sp - 1u - p]);
            if (pad2) wnr4_put16(rec + at + 128u + 2u * sp, 0u);
            for (u = 0; u < nsw; u++) wnr4_put32(rec + at + tile_bytes - 4u * (u + 1u), sign_words[u]);
            at += tile_bytes;
            wnr4_put32(rec + WNR4_HEADER_BYTES + srec_bytes + 4u * (t + 1u), (uint32_t)(at - payload0));
        }
    }
    free(stack); free(sym); free(sign_words);
    wnr4_put32(rec + 24, (uint32_t)(at - payload0));
    total = (at + 4095u) & ~(size_t)4095u;
    if (total > cap) return 0;
    memset(rec + at, 0, total - at);
    if (stats_out) *stats_out = st;
    return total;
}

#endif
