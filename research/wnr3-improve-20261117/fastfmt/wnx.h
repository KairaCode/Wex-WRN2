/* WNX: a prototype record coding for MXFP4 expert records designed for decode speed (research only).
 *
 * Same coded domain as WNR3-CZ (canonical zero, ceil form, rule-g anchors; wex_neural_rans.h), other entropy
 * coding:
 *   - every stream is 32 interleaved rANS lanes that share ONE word stream (lane k reads its renormalisation
 *     words in lane order), 32-bit state, L = 2^16, 16-bit words, at most one word per lane per step;
 *   - weights: one step = one 32-nibble block; lane k carries block position pos(k) = 2k (k < 16) or
 *     2(k - 16) + 1 (k >= 16), so the context (srel of the block's coded scale) is the same for all 32 lanes;
 *   - variant MAG (C1): 8-symbol magnitude alphabet, 12-bit CDF per (matrix, srel), the sign of every
 *     non-zero magnitude as a raw bit; sign bits are packed in u32 words stored backwards from the stream end;
 *     a -0 cannot be represented (the format is canonical-zero by construction);
 *   - variant JOINT (C2): 15-symbol alphabet (all nibbles but 8) per (matrix, srel), no sign region;
 *   - scales: 32 interleaved lanes, lane k = scale 32t + k of the stream, order-0 256-symbol 12-bit CDF;
 *   - integrity: lane k keeps a MurmurHash3 of its own coded symbols (words of 8 nibbles / 4 scale bytes);
 *     G = MurmurHash3 of the 32 lane hashes; the encoder starts lane 0 at L + (G & 0xffff), lane 1 at
 *     L + (G >> 16), lane k >= 2 at L + (h_k & 0xffff); the decoder requires those final states, the word
 *     pointer to meet the sign pointer exactly and zero unused sign bits.
 *
 * Record: 64-byte header, u32 end offset of every stream (scale streams, then weight streams, relative to the
 * payload start), payload, zero padding to 4096.
 *   0 magic "WNX1"  4 model id  8 flags (bits 0-2 ceil mask, bit 4 = JOINT)  12 owner  16 anchors[3], 0
 *  20 Bw (blocks per weight stream)  24 Bs (steps of 32 scales per scale stream)  28 streams  32 payload bytes
 *  36..63 zero.
 * Model blob "WNXM": 32-byte header {magic, 1, Pw = 12, Ps = 12, variant, symbols, 0, 0}, weight CDFs
 * [3][7][symbols + 1] u16, scale CDFs [3][257] u16. Model id = FNV-1a of the blob. */
#ifndef WNX_H
#define WNX_H
#include "wex_format.h"
#include <stdlib.h>

#define WNX_MAGIC 0x31584e57u
#define WNX_MODEL_MAGIC 0x4d584e57u
#define WNX_SEED 0x31584e57u
#define WNX_L (1u << 16)
#define WNX_P 12u
#define WNX_T (1u << WNX_P)
#ifndef WNX_PS
#define WNX_PS 14u   /* scale CDF precision (12 costs +0.043% of raw held-out, eval_test.txt) */
#endif
#define WNX_TS (1u << WNX_PS)
#define WNX_LANES 32u
#define WNX_HDR 64u
#define WNX_FLAG_JOINT 0x10u
#define WNX_FLAG_TAIL 0x20u   /* MAG only: the sign bits of the last WNX_TAIL_STEPS blocks go first into lanes 2..31's
                                 initial states (16 bits each, 480 bits), the rest after the region's bits */
#define WNX_TAIL_STEPS 24u
#define WNX_POOL_BITS 480u
#define WNX_SLACK 192u   /* fast kernels: readable bytes needed after a stream's end */

typedef struct {
    int joint;                 /* 0 = MAG (C1), 1 = JOINT (C2) */
    uint32_t nsym;             /* 8 or 15 */
    uint16_t wc[3][7][16];     /* weight CDFs, wc[m][srel][0..nsym] */
    uint16_t sc[3][257];       /* scale CDFs */
    uint32_t id;
    unsigned char blob[32 + 3 * 7 * 16 * 2 + 3 * 257 * 2];
    uint32_t blob_bytes;
} WnxModel;

static inline uint32_t wnx_nib_of(uint32_t joint_symbol) { return joint_symbol < 8u ? joint_symbol : joint_symbol + 1u; }
static inline uint32_t wnx_jsym_of(uint32_t nibble) { return nibble < 8u ? nibble : nibble - 1u; }
static inline uint32_t wnx_pos(uint32_t k) { return k < 16u ? 2u * k : 2u * (k - 16u) + 1u; }

/* wex_nr2_quantize generalised to any total: every symbol >= 1, remainders largest first, ties lower symbol */
static inline void wnx_quantize(const uint64_t *counts, uint32_t symbols, uint32_t total_f, uint16_t *cdf) {
    uint64_t total = 0, rem[256];
    uint32_t f[256], spare = total_f - symbols, used = 0, s, extra;
    unsigned char taken[256] = {0};
    for (s = 0; s < symbols; s++) total += counts[s];
    for (s = 0; s < symbols; s++) {
        uint64_t scaled = total ? counts[s] * spare : spare, div = total ? total : symbols;
        f[s] = 1u + (uint32_t)(scaled / div); rem[s] = scaled % div; used += f[s];
    }
    for (extra = total_f - used; extra; extra--) {
        uint32_t best = symbols;
        for (s = 0; s < symbols; s++) if (!taken[s] && (best == symbols || rem[s] > rem[best])) best = s;
        taken[best] = 1; f[best]++;
    }
    for (used = 0, s = 0; s <= symbols; s++) { cdf[s] = (uint16_t)used; if (s < symbols) used += f[s]; }
}

/* counts: wcount[3][7][16] (nibble-indexed: mags for MAG are summed from nibbles), scount[3][256] */
static inline void wnx_model_build(WnxModel *M, int joint, const uint64_t wcount[3][7][16], const uint64_t scount[3][256]) {
    uint32_t m, r, s, at = 32;
    uint32_t hdr[8] = {WNX_MODEL_MAGIC, 1u, WNX_P, WNX_PS, (uint32_t)joint, joint ? 15u : 8u, 0, 0};
    memset(M, 0, sizeof(*M));
    M->joint = joint; M->nsym = joint ? 15u : 8u;
    for (m = 0; m < 3u; m++) {
        for (r = 0; r < 7u; r++) {
            uint64_t c[16] = {0};
            for (s = 0; s < 16u; s++) {
                if (joint) { if (s != 8u) c[wnx_jsym_of(s)] += wcount[m][r][s]; }
                else c[s & 7u] += wcount[m][r][s];
            }
            wnx_quantize(c, M->nsym, WNX_T, M->wc[m][r]);
        }
        wnx_quantize(scount[m], 256u, WNX_TS, M->sc[m]);
    }
    memcpy(M->blob, hdr, 32);
    for (m = 0; m < 3u; m++) for (r = 0; r < 7u; r++) { memcpy(M->blob + at, M->wc[m][r], 2u * (M->nsym + 1u)); at += 2u * (M->nsym + 1u); }
    for (m = 0; m < 3u; m++) { memcpy(M->blob + at, M->sc[m], 514u); at += 514u; }
    M->blob_bytes = at;
    M->id = wex_nr2_fnv1a(M->blob, at);
}

/* ---- hashing ---------------------------------------------------------------------------------------- */
static inline uint32_t wnx_ident_hash(uint32_t model_id, uint32_t owner, uint32_t g, uint32_t word16) {
    uint32_t h = WNX_SEED;
    h = wex_nr3_h32_block(h, model_id); h = wex_nr3_h32_block(h, owner);
    h = wex_nr3_h32_block(h, g); return wex_nr3_h32_block(h, word16);
}
static inline uint32_t wnx_lane_seed(uint32_t s0, uint32_t j) { return s0 ^ (0x9e3779b9u * (j + 1u)); }
/* expected final states from the 16 finished chain hashes (chain j covers lanes 2j and 2j + 1) */
static inline void wnx_expect(uint32_t s0, const uint32_t h[16], uint32_t x0[32]) {
    uint32_t G = s0, k;
    for (k = 0; k < 16u; k++) G = wex_nr3_h32_block(G, h[k]);
    G = wex_nr3_h32_final(G, 64u);
    x0[0] = WNX_L + (G & 0xffffu); x0[1] = WNX_L + (G >> 16);
    for (k = 2; k < 32u; k++) x0[k] = WNX_L + ((h[k >> 1] >> (16u * (k & 1u))) & 0xffffu);
}

/* ---- record view ------------------------------------------------------------------------------------ */
typedef struct {
    uint32_t model_id, flags, owner, Bw, Bs, streams, payload_bytes, scale_streams, weight_streams;
    unsigned char anchor[3];
    const uint32_t *ends;            /* u32 LE end offsets (little-endian host assumed) */
    const unsigned char *payload;
    uint32_t word16;                 /* anchors | flags << 24 */
} WnxView;

static inline uint32_t wnx_div_up(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static inline int wnx_parse(const unsigned char *rec, size_t total, const WexNr2Layout *lay, const WnxModel *M, WnxView *v) {
    uint32_t m, s, prev = 0;
    size_t used;
    if (total < WNX_HDR || (total & 4095u) || wex_nr_u32(rec) != WNX_MAGIC) return -1;
    v->model_id = wex_nr_u32(rec + 4); v->flags = wex_nr_u32(rec + 8); v->owner = wex_nr_u32(rec + 12);
    v->anchor[0] = rec[16]; v->anchor[1] = rec[17]; v->anchor[2] = rec[18];
    v->Bw = wex_nr_u32(rec + 20); v->Bs = wex_nr_u32(rec + 24); v->streams = wex_nr_u32(rec + 28);
    v->payload_bytes = wex_nr_u32(rec + 32);
    if (rec[19] || (v->flags & ~0x37u) || v->model_id != M->id || ((v->flags & WNX_FLAG_JOINT) != 0) != (M->joint != 0) ||
        ((v->flags & WNX_FLAG_TAIL) && M->joint)) return -1;
    for (s = 36; s < WNX_HDR; s++) if (rec[s]) return -1;
    if (!v->Bw || !v->Bs || v->Bw > 65536u || v->Bs > 65536u) return -1;
    v->scale_streams = v->weight_streams = 0;
    for (m = 0; m < 3u; m++) {
        if (!lay->scale_count[m] || (uint64_t)lay->scale_count[m] * 32u != lay->symbol_count[m]) return -1;
        v->scale_streams += wnx_div_up(lay->scale_count[m], 32u * v->Bs);
        v->weight_streams += wnx_div_up(lay->scale_count[m], v->Bw);
    }
    if (v->streams != v->scale_streams + v->weight_streams) return -1;
    used = WNX_HDR + 4u * (size_t)v->streams + v->payload_bytes;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    v->ends = (const uint32_t *)(const void *)(rec + WNX_HDR);
    v->payload = rec + WNX_HDR + 4u * (size_t)v->streams;
    for (s = 0; s < v->streams; s++) {
        uint32_t e = v->ends[s];
        if (e < prev || e - prev < 128u) return -1;
        prev = e;
    }
    if (prev != v->payload_bytes) return -1;
    for (; used < total; used++) if (rec[used]) return -1;
    v->word16 = (uint32_t)v->anchor[0] | (uint32_t)v->anchor[1] << 8 | (uint32_t)v->anchor[2] << 16 | v->flags << 24;
    return 0;
}

/* ---- encoder ---------------------------------------------------------------------------------------- */
typedef struct { unsigned char *buf; size_t cap; } WnxScratch;

/* Encode one interleaved stream: steps T, lane k at step t has symbol sym[t*32+k] (active[t] lanes active,
 * lanes 0..active-1), CDF cdfs[t] (nsym+1 entries). x0 = initial (encoder) states. Writes [states][words] to
 * out, returns bytes. Words are produced backwards into tmp. */
static inline size_t wnx_rans_encode(const uint8_t *sym, const uint16_t *const *cdfs, const uint8_t *active, uint32_t T,
                                     const uint32_t x0[32], unsigned char *out, uint16_t *tmp, size_t tmp_words, uint32_t P) {
    uint32_t x[32], k, t;
    size_t w = tmp_words;
    for (k = 0; k < 32u; k++) x[k] = x0[k];
    for (t = T; t-- > 0;) {
        const uint16_t *c = cdfs[t];
        for (k = active[t]; k-- > 0;) {
            uint32_t s = sym[(size_t)t * 32u + k], start = c[s], f = c[s + 1u] - start;
            if ((uint64_t)x[k] >= ((uint64_t)f << (32u - P))) { tmp[--w] = (uint16_t)x[k]; x[k] >>= 16; }
            x[k] = ((x[k] / f) << P) + x[k] % f + start;
        }
    }
    for (k = 0; k < 32u; k++) memcpy(out + 4u * k, &x[k], 4);
    memcpy(out + 128, tmp + w, 2u * (tmp_words - w));
    return 128u + 2u * (tmp_words - w);
}

/* 16 hash chains; chain j takes lanes 2j (bits 0..15) and 2j + 1 (bits 16..31) of `per` = 16 / bits steps per
 * word: the symbol of lane k at step t sits at bit bits * (t % per) + 16 * (k & 1). Inactive lanes put 0. */
typedef struct { uint32_t h[16], acc[16]; } WnxLaneHash;
static inline void wnx_lh_init(WnxLaneHash *H, uint32_t s0) {
    uint32_t j;
    for (j = 0; j < 16u; j++) { H->h[j] = wnx_lane_seed(s0, j); H->acc[j] = 0; }
}
static inline void wnx_lh_put(WnxLaneHash *H, uint32_t k, uint32_t t, uint32_t v, uint32_t bits) {
    H->acc[k >> 1] |= v << (bits * (t % (16u / bits)) + 16u * (k & 1u));
}
static inline void wnx_lh_step(WnxLaneHash *H, uint32_t t, uint32_t bits) {   /* after step t */
    uint32_t j;
    if ((t + 1u) % (16u / bits)) return;
    for (j = 0; j < 16u; j++) { H->h[j] = wex_nr3_h32_block(H->h[j], H->acc[j]); H->acc[j] = 0; }
}
static inline void wnx_lh_final(WnxLaneHash *H, uint32_t T, uint32_t bits) {
    uint32_t j;
    for (j = 0; j < 16u; j++) {
        if (T % (16u / bits)) H->h[j] = wex_nr3_h32_block(H->h[j], H->acc[j]);
        H->h[j] = wex_nr3_h32_final(H->h[j], T);
    }
}

/* Encode a coded expert (scales cs[m], packed coded nibbles cw[m]; flags = ceil mask). Returns used bytes
 * (unpadded) or 0 on error; out is zero-padded to the 4096 multiple, *total set. */
static inline size_t wnx_encode(const WnxModel *M, const WexNr2Layout *lay, unsigned char *const cs[3], unsigned char *const cw[3],
                                uint32_t ceil_flags, uint32_t owner, uint32_t Bw, uint32_t Bs, int tail,
                                unsigned char *out, size_t cap, size_t *total) {
    uint32_t m, streams = 0, g = 0, flags = ceil_flags | (M->joint ? WNX_FLAG_JOINT : 0u) | (tail && !M->joint ? WNX_FLAG_TAIL : 0u);
    unsigned char anchor[3];
    size_t at, pay0;
    uint32_t maxT = Bw > Bs ? Bw : Bs;
    uint8_t *sym = (uint8_t *)malloc((size_t)maxT * 32u), *act = (uint8_t *)malloc(maxT);
    const uint16_t **cdfs = (const uint16_t **)malloc(sizeof(*cdfs) * maxT);
    size_t tmpw = (size_t)maxT * 32u + 64u;
    uint16_t *tmp = (uint16_t *)malloc(tmpw * 2u);
    uint32_t *signw = (uint32_t *)malloc(((size_t)maxT + 1u) * 4u);
    uint8_t *sbit = (uint8_t *)malloc((size_t)maxT * 32u + 32u);
    for (m = 0; m < 3u; m++) {
        anchor[m] = wex_nr3_anchor(cs[m], lay->scale_count[m]);
        streams += wnx_div_up(lay->scale_count[m], 32u * Bs) + wnx_div_up(lay->scale_count[m], Bw);
    }
    pay0 = WNX_HDR + 4u * (size_t)streams; at = pay0;
    if (cap < pay0) return 0;
    memset(out, 0, pay0);
    {
        uint32_t word16 = (uint32_t)anchor[0] | (uint32_t)anchor[1] << 8 | (uint32_t)anchor[2] << 16 | flags << 24;
        uint32_t pass;
        for (pass = 0; pass < 2u; pass++)
            for (m = 0; m < 3u; m++) {
                const uint32_t nsc = lay->scale_count[m];
                uint32_t first;
                const uint32_t per = pass == 0 ? 32u * Bs : Bw;   /* scales or blocks per stream */
                for (first = 0; first < nsc; first += per, g++) {
                    const uint32_t cnt = nsc - first < per ? nsc - first : per;
                    const uint32_t s0 = wnx_ident_hash(M->id, owner, g, word16);
                    WnxLaneHash H;
                    uint32_t x0[32], t, k, T, nsw = 0, nsb = 0, nmain = 0;
                    size_t len;
                    wnx_lh_init(&H, s0);
                    if (pass == 0) {
                        T = wnx_div_up(cnt, 32u);
                        for (t = 0; t < T; t++) {
                            act[t] = (uint8_t)(cnt - 32u * t < 32u ? cnt - 32u * t : 32u);
                            cdfs[t] = M->sc[m];
                            for (k = 0; k < act[t]; k++) {
                                uint32_t v = cs[m][first + 32u * t + k];
                                sym[t * 32u + k] = (uint8_t)v;
                                wnx_lh_put(&H, k, t, v, 8u);
                            }
                            wnx_lh_step(&H, t, 8u);
                        }
                        wnx_lh_final(&H, T, 8u);
                    } else {
                        T = cnt;
                        for (t = 0; t < T; t++) {
                            const uint32_t b = first + t;
                            const unsigned char *q = cw[m] + (size_t)b * 16u;
                            act[t] = 32u;
                            cdfs[t] = M->wc[m][wex_nr2_srel(cs[m][b], anchor[m])];
                            for (k = 0; k < 32u; k++) {
                                const uint32_t p = wnx_pos(k), nib = (q[p / 2u] >> ((p & 1u) * 4u)) & 15u;
                                if (nib == 8u) goto fail;   /* the coded domain is canonical zero */
                                wnx_lh_put(&H, k, t, nib, 4u);
                                if (M->joint) sym[t * 32u + k] = (uint8_t)wnx_jsym_of(nib);
                                else {
                                    sym[t * 32u + k] = (uint8_t)(nib & 7u);
                                    if (nib & 7u) sbit[nsb++] = (uint8_t)(nib >> 3);
                                }
                            }
                            wnx_lh_step(&H, t, 4u);
                            if (t + 1u == (T > WNX_TAIL_STEPS ? T - WNX_TAIL_STEPS : 0u)) nmain = nsb;
                        }
                        wnx_lh_final(&H, T, 4u);
                    }
                    wnx_expect(s0, H.h, x0);
                    if (pass == 1 && !M->joint) {   /* sign bits -> region words (and the pool of the TAIL form) */
                        uint32_t i, j = 0, bits = 0, sacc = 0, npool = 0;
                        if (flags & WNX_FLAG_TAIL) {
                            npool = nsb - nmain < WNX_POOL_BITS ? nsb - nmain : WNX_POOL_BITS;
                            for (k = 2; k < 32u; k++) x0[k] = WNX_L;
                            for (i = 0; i < npool; i++) x0[2u + i / 16u] |= (uint32_t)sbit[nmain + i] << (i % 16u);
                        }
                        for (i = 0; i < nsb; i++) {
                            if (i >= nmain && i < nmain + npool) continue;
                            sacc |= (uint32_t)sbit[i] << bits;
                            if (++bits == 32u) { signw[nsw++] = sacc; sacc = 0; bits = 0; }
                            j++;
                        }
                        if (bits) signw[nsw++] = sacc;
                    }
                    if (cap - at < 128u + (size_t)T * 64u + 4u * nsw) goto fail;
                    len = wnx_rans_encode(sym, cdfs, act, T, x0, out + at, tmp, tmpw, pass == 0 ? WNX_PS : WNX_P);
                    for (k = 0; k < nsw; k++) memcpy(out + at + len + 4u * (nsw - 1u - k), &signw[k], 4);
                    len += 4u * nsw;
                    at += len;
                    {
                        uint32_t e = (uint32_t)(at - pay0);
                        memcpy(out + WNX_HDR + 4u * g, &e, 4);
                    }
                }
            }
        {
            uint32_t hdr[9] = {WNX_MAGIC, M->id, flags, owner, word16 & 0xffffffu, Bw, Bs, streams, (uint32_t)(at - pay0)};
            memcpy(out, hdr, 36);
        }
    }
    *total = (at + 4095u) & ~(size_t)4095u;
    if (*total > cap) goto fail;
    memset(out + at, 0, *total - at);
    free(sym); free(act); free(cdfs); free(tmp); free(signw); free(sbit);
    return at;
fail:
    free(sym); free(act); free(cdfs); free(tmp); free(signw); free(sbit);
    return 0;
}

/* ---- reference decoder (scalar, every check per byte) ------------------------------------------------- */
static inline uint32_t wnx_find(const uint16_t *c, uint32_t nsym, uint32_t slot) {
    uint32_t s = 0;
    while (s + 1u < nsym && c[s + 1u] <= slot) s++;
    return s;
}

/* One stream with every check per byte. pass 0 = scale stream (cnt scales from `first`), 1 = weight stream
 * (cnt blocks from block `first`). so[m] holds the decoded coded scales of the matrix (weights: read for the
 * contexts, decremented by the ceil inverse). Returns 0 / -1. */
static inline int wnx_stream_ref(const WnxView *v, const WnxModel *M, uint32_t pass, uint32_t m, uint32_t first, uint32_t cnt,
                                 uint32_t g, const unsigned char *b0, const unsigned char *end,
                                 unsigned char *const so[3], unsigned char *const wo[3]) {
    const unsigned char *ptr = b0 + 128, *sp = end;
    const uint32_t s0 = wnx_ident_hash(v->model_id, v->owner, g, v->word16), ceil = v->flags & 7u;
    uint32_t x[32], x0[32], k, t, T;
    uint64_t bb = 0;
    uint32_t nb = 0;
    WnxLaneHash H;
    if (end - b0 < 128) return -1;
    for (k = 0; k < 32u; k++) { memcpy(&x[k], b0 + 4u * k, 4); if (x[k] < WNX_L) return -1; }
    wnx_lh_init(&H, s0);
    if (pass == 0) {
        const uint16_t *c = M->sc[m];
        T = wnx_div_up(cnt, 32u);
        for (t = 0; t < T; t++) {
            const uint32_t a = cnt - 32u * t < 32u ? cnt - 32u * t : 32u;
            for (k = 0; k < a; k++) {
                const uint32_t slot = x[k] & (WNX_TS - 1u), s = wnx_find(c, 256u, slot);
                x[k] = (uint32_t)(c[s + 1u] - c[s]) * (x[k] >> WNX_PS) + slot - c[s];
                if (x[k] < WNX_L) {
                    uint16_t w;
                    if (sp - ptr < 2) return -1;
                    memcpy(&w, ptr, 2); ptr += 2; x[k] = x[k] << 16 | w;
                }
                so[m][first + 32u * t + k] = (unsigned char)s;
                wnx_lh_put(&H, k, t, s, 8u);
            }
            wnx_lh_step(&H, t, 8u);
        }
        wnx_lh_final(&H, T, 8u);
    } else {
        const uint32_t tail = (v->flags & WNX_FLAG_TAIL) != 0, Tm = tail && cnt > WNX_TAIL_STEPS ? cnt - WNX_TAIL_STEPS : tail ? 0u : cnt;
        uint8_t dbuf[WNX_TAIL_STEPS][32];
        uint32_t pool_used = 0, pass2;
        T = cnt;
        for (pass2 = 0; pass2 < 2u; pass2++)   /* pass2 0: rANS for every step (signs of steps < Tm); 1: deferred steps */
            for (t = pass2 ? Tm : 0; t < T; t++) {
                const uint32_t b = first + t;
                const uint16_t *c = M->wc[m][wex_nr2_srel(so[m][b], v->anchor[m])];
                unsigned char *q = wo[m] + (size_t)b * 16u;
                uint32_t nib[32], mx = 0;
                if (pass2 == 0 && tail && t == Tm) {   /* nothing of the last steps is known yet but their magnitudes */
                    for (; t < T; t++) {
                        const uint16_t *cc = M->wc[m][wex_nr2_srel(so[m][first + t], v->anchor[m])];
                        for (k = 0; k < 32u; k++) {
                            const uint32_t slot = x[k] & (WNX_T - 1u), s = wnx_find(cc, M->nsym, slot);
                            x[k] = (uint32_t)(cc[s + 1u] - cc[s]) * (x[k] >> WNX_P) + slot - cc[s];
                            if (x[k] < WNX_L) {
                                uint16_t w;
                                if (sp - ptr < 2) return -1;
                                memcpy(&w, ptr, 2); ptr += 2; x[k] = x[k] << 16 | w;
                            }
                            dbuf[t - Tm][k] = (uint8_t)s;
                        }
                    }
                    for (k = 2; k < 32u; k++) if (x[k] >= 2u * WNX_L) return -1;   /* lanes 2..31 end at L + pool bits */
                    break;
                }
                if (pass2 == 0) {
                    for (k = 0; k < 32u; k++) {
                        const uint32_t slot = x[k] & (WNX_T - 1u), s = wnx_find(c, M->nsym, slot);
                        x[k] = (uint32_t)(c[s + 1u] - c[s]) * (x[k] >> WNX_P) + slot - c[s];
                        if (x[k] < WNX_L) {
                            uint16_t w;
                            if (sp - ptr < 2) return -1;
                            memcpy(&w, ptr, 2); ptr += 2; x[k] = x[k] << 16 | w;
                        }
                        nib[k] = M->joint ? wnx_nib_of(s) : s;
                    }
                } else for (k = 0; k < 32u; k++) nib[k] = dbuf[t - Tm][k];
                if (!M->joint)
                    for (k = 0; k < 32u; k++)
                        if (nib[k]) {
                            if (pass2 && pool_used < WNX_POOL_BITS) {
                                const uint32_t i = pool_used++;
                                nib[k] |= (((x[2u + i / 16u] - WNX_L) >> (i % 16u)) & 1u) << 3;
                                continue;
                            }
                            if (!nb) {
                                uint32_t w;
                                if (sp - ptr < 4) return -1;
                                sp -= 4; memcpy(&w, sp, 4); bb = w; nb = 32;
                            }
                            nib[k] |= (uint32_t)(bb & 1u) << 3; bb >>= 1; nb--;
                        }
                memset(q, 0, 16);
                for (k = 0; k < 32u; k++) {
                    const uint32_t p = wnx_pos(k);
                    q[p / 2u] |= (unsigned char)(nib[k] << ((p & 1u) * 4u));
                    wnx_lh_put(&H, k, t, nib[k], 4u);
                    if ((nib[k] & 7u) > mx) mx = nib[k] & 7u;
                }
                wnx_lh_step(&H, t, 4u);
                if ((ceil >> m & 1u) && mx == 5u) {
                    if (!so[m][b]) return -1;
                    for (k = 0; k < 16u; k++)
                        q[k] = (unsigned char)((q[k] & 0x88u) | wex_nr3_double[q[k] & 7u] | wex_nr3_double[(q[k] >> 4) & 7u] << 4);
                    so[m][b]--;   /* after its context was taken: the next block reads its own scale */
                }
                if (!tail && pass2 == 0 && t + 1u == T) break;
            }
        wnx_lh_final(&H, T, 4u);
        if (nb && bb) return -1;   /* unused sign bits must be 0 */
        if (tail) {
            for (; pool_used < WNX_POOL_BITS; pool_used++)   /* unused pool bits must be 0 */
                if (((x[2u + pool_used / 16u] - WNX_L) >> (pool_used % 16u)) & 1u) return -1;
            if (ptr != sp) return -1;
            wnx_expect(s0, H.h, x0);
            return x[0] == x0[0] && x[1] == x0[1] ? 0 : -1;
        }
    }
    if (ptr != sp) return -1;
    wnx_expect(s0, H.h, x0);
    for (k = 0; k < 32u; k++) if (x[k] != x0[k]) return -1;
    return 0;
}

/* Stream table of a parsed record: g order, pass, matrix, first, count. */
typedef struct { uint32_t pass, m, first, cnt, g, begin, end; } WnxStreamInfo;
static inline uint32_t wnx_streams(const WnxView *v, const WexNr2Layout *lay, WnxStreamInfo *si) {
    uint32_t m, g = 0, pass, prev = 0;
    for (pass = 0; pass < 2u; pass++)
        for (m = 0; m < 3u; m++) {
            const uint32_t nsc = lay->scale_count[m], per = pass == 0 ? 32u * v->Bs : v->Bw;
            uint32_t first;
            for (first = 0; first < nsc; first += per, g++) {
                si[g].pass = pass; si[g].m = m; si[g].first = first; si[g].cnt = nsc - first < per ? nsc - first : per;
                si[g].g = g; si[g].begin = prev; si[g].end = v->ends[g]; prev = v->ends[g];
            }
        }
    return g;
}

/* scales_out[m] / weights_out[m] as wex_nr2_decode (decoded slab layout). Returns 0 / -1. */
static inline int wnx_decode_ref(const WnxView *v, const WexNr2Layout *lay, const WnxModel *M,
                                 unsigned char *const so[3], unsigned char *const wo[3]) {
    WnxStreamInfo *si = (WnxStreamInfo *)malloc(sizeof(*si) * v->streams);
    uint32_t n = wnx_streams(v, lay, si), s;
    int r = 0;
    for (s = 0; s < n && !r; s++)
        r = wnx_stream_ref(v, M, si[s].pass, si[s].m, si[s].first, si[s].cnt, si[s].g, v->payload + si[s].begin,
                           v->payload + si[s].end, so, wo);
    free(si);
    return r;
}
#endif
