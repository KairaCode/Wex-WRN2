/* Prototype WNR3-skeleton codec with a generalised static model (research only).
 * Same record skeleton as WNR3 (32-B header, u16 lengths, 4-B big-endian flush, x0 = L + h23 MurmurHash3,
 * 32-bit state, L = 2^23, byte renormalisation), but:
 *   - CDF precision P (bits, 10..16; tables are u32 here),
 *   - K weight table sets and KS scale table sets, the encoder picks per stream the set that gives the
 *     shortest real stream; selector = ceil(log2 K) bits per weight stream (+ ceil(log2 KS) per scale stream),
 *     bit-packed after the length table,
 *   - scale symbol mode smode: 0 = absolute scale (WNR3), 1 = (scale - anchor[m]) mod 256.
 * With P = 14, K = KS = 1, smode 0 and the WNM2 tables the record is byte-identical to WNR3 (harness check).
 * Build: gcc -O2 -shared -fPIC -I/workspace/Wex-WRN2/c -o mlib.so mlib.c */
#include "wex_format.h"
#include <stdlib.h>

#define NC 567u
typedef struct {
    uint32_t P, K, KS, smode;
    const uint32_t *wcdf;   /* [K][3][567][17] */
    const uint32_t *scdf;   /* [KS][3][257] */
    uint32_t ncls;          /* >= 1: selector-free classes from the stream's decoded scales; weight sets = ncls * K */
    const int32_t *thr;     /* ncls - 1 ascending thresholds on f = sum of srel over the stream's blocks */
    uint32_t ns;            /* smode 2: scale alphabet size including the escape (last index) */
    const uint8_t *smap;    /* smode 2: [256] relative value -> index, 255 = escape (then 8 raw bits, uniform) */
    const uint8_t *sval;    /* smode 2: [ns - 1] index -> relative value */
} Mdl;
static uint32_t stream_class(const Mdl *M, const unsigned char *s, uint32_t b0, uint32_t nb, uint32_t anchor) {
    int32_t f = 0; uint32_t b, c = 0;
    if (M->ncls <= 1) return 0;
    for (b = 0; b < nb; b++) f += (int32_t)wex_nr2_srel(s[b0 + b], anchor);
    while (c + 1u < M->ncls && f >= M->thr[c]) c++;
    return c;
}

static uint32_t ceil_log2(uint32_t k) { uint32_t b = 0; while ((1u << b) < k) b++; return b; }

static void views(const WexEntry *e, const unsigned char *slab, WexNr2Layout *lay, const unsigned char *sc[3], const unsigned char *wt[3]) {
    uint32_t m;
    wex_nr2_layout(e, lay);
    for (m = 0; m < 3; m++) { sc[m] = slab + e->scale_offsets[m]; wt[m] = slab + e->weight_offsets[m]; }
}
static inline uint32_t ssym(uint32_t s, uint32_t anchor, uint32_t smode) { return smode ? (s - anchor) & 255u : s; }
static inline uint32_t sinv(uint32_t y, uint32_t anchor, uint32_t smode) { return smode ? (y + anchor) & 255u : y; }
static inline int dsym(const uint32_t *c, uint32_t nsym, uint32_t P, uint32_t *x, const unsigned char **p,
                       const unsigned char *end, uint32_t *out);

/* counts: wc[3][567][16], sc[3][256] (scale symbols in smode), anchors = rule g of the coded scales. */
int m_count(const WexEntry *e, const unsigned char *slab, uint32_t smode, uint64_t *wc, uint64_t *scnt) {
    WexNr2Layout lay; const unsigned char *sc[3], *wt[3]; uint32_t m, i;
    views(e, slab, &lay, sc, wt);
    for (m = 0; m < 3; m++) {
        uint32_t a = wex_nr3_anchor(sc[m], lay.scale_count[m]), srel = 0, pm = 8, mx = 8;
        for (i = 0; i < lay.scale_count[m]; i++) scnt[m * 256u + ssym(sc[m][i], a, smode)]++;
        for (i = 0; i < lay.symbol_count[m]; i++) {
            uint32_t x = (wt[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
            if (!(i % 32u)) { srel = wex_nr2_srel(sc[m][i / 32u], a); pm = 8; mx = 8; }
            wc[((size_t)m * NC + wex_nr2_context(srel, pm, mx)) * 16u + x]++;
            pm = x & 7u; if (mx == 8u || pm > mx) mx = pm;
        }
    }
    return 0;
}

/* Lloyd step for K weight table sets: cost tables cw[K][3][567][16] (bits, double); every weight stream of
 * chunk wchunk is assigned to the cheapest set; its counts are added to acc[K][3][567][16]; returns total bits
 * and writes assign[] (per weight stream, g order) and per-stream bits under set 0 (bits0) if non-NULL. */
double m_lloyd(const WexEntry *e, const unsigned char *slab, uint32_t wchunk, uint32_t K, const double *cw,
               uint64_t *acc, uint8_t *assign, double *bits0) {
    WexNr2Layout lay; const unsigned char *sc[3], *wt[3]; uint32_t m, first, ws = 0; double total = 0;
    uint16_t *ctx = malloc(sizeof(uint16_t) * wchunk); uint8_t *sym = malloc(wchunk);
    views(e, slab, &lay, sc, wt);
    for (m = 0; m < 3; m++) {
        uint32_t a = wex_nr3_anchor(sc[m], lay.scale_count[m]);
        for (first = 0; first < lay.symbol_count[m]; first += wchunk, ws++) {
            uint32_t n = lay.symbol_count[m] - first < wchunk ? lay.symbol_count[m] - first : wchunk, j, k, best = 0;
            uint32_t srel = 0, pm = 8, mx = 8; double bc = 1e300;
            for (j = 0; j < n; j++) {
                uint32_t i = first + j, x = (wt[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                if (!(i % 32u)) { srel = wex_nr2_srel(sc[m][i / 32u], a); pm = 8; mx = 8; }
                ctx[j] = (uint16_t)wex_nr2_context(srel, pm, mx); sym[j] = (uint8_t)x;
                pm = x & 7u; if (mx == 8u || pm > mx) mx = pm;
            }
            for (k = 0; k < K; k++) {
                const double *t = cw + ((size_t)k * 3 + m) * NC * 16u; double c = 0;
                for (j = 0; j < n; j++) c += t[ctx[j] * 16u + sym[j]];
                if (k == 0 && bits0) bits0[ws] = c;
                if (c < bc) { bc = c; best = k; }
            }
            total += bc; if (assign) assign[ws] = (uint8_t)best;
            if (acc) for (j = 0; j < n; j++) acc[(((size_t)best * 3 + m) * NC + ctx[j]) * 16u + sym[j]]++;
        }
    }
    free(ctx); free(sym);
    return total;
}

static inline int put(uint32_t *x, unsigned char **cur, const unsigned char *floor, uint32_t P, uint32_t start, uint32_t f) {
    uint32_t maximum = ((WEX_NR_L >> P) << 8) * f;
    if (!f) return -1;
    while (*x >= maximum) { if (*cur == floor) return -1; *--*cur = (unsigned char)*x; *x >>= 8; }
    *x = ((*x / f) << P) + *x % f + start;
    return 0;
}

#define TMP (4u * WEX_NR2_MAX_CHUNK + 16u)
/* Encode one coded slab. out receives the unpadded record (*used). selk (optional): chosen set per stream
 * (g order, scale streams first). Returns 0 / -1. */
int m_encode(const Mdl *M, const WexEntry *e, const unsigned char *slab, uint32_t wchunk, uint32_t schunk,
             uint32_t flags, uint32_t owner, uint32_t model_id, unsigned char *out, size_t cap, size_t *used, uint8_t *selk) {
    WexNr2Layout lay; const unsigned char *sc[3], *wt[3]; unsigned char anchor[3];
    uint32_t m, S = 0, W = 0, g = 0, ident[4], bw = ceil_log2(M->K), bs = ceil_log2(M->KS);
    size_t meta, selbytes, cursor; uint64_t sp = 0, wp = 0;
    unsigned char *tmp = malloc(TMP), *best = malloc(TMP); uint16_t *ctx = malloc(2u * WEX_NR2_MAX_CHUNK);
    int rc = -1;
    views(e, slab, &lay, sc, wt);
    for (m = 0; m < 3; m++) {
        anchor[m] = wex_nr3_anchor(sc[m], lay.scale_count[m]);
        S += wex_nr2_streams(lay.scale_count[m], schunk); W += wex_nr2_streams(lay.symbol_count[m], wchunk);
    }
    selbytes = ((uint64_t)S * bs + (uint64_t)W * bw + 7u) / 8u;
    meta = 32u + 2u * (size_t)(S + W) + selbytes;
    if (cap < meta) goto done;
    memset(out, 0, meta); cursor = meta;
    ident[0] = model_id; ident[1] = owner;
    ident[3] = (uint32_t)anchor[0] | (uint32_t)anchor[1] << 8 | (uint32_t)anchor[2] << 16 | flags << 24;
    {
        uint64_t bitpos = 0; uint32_t pass;
        for (pass = 0; pass < 6; pass++) {
            uint32_t mm = pass % 3u, isS = pass < 3u, first, chunk = isS ? schunk : wchunk;
            uint32_t all = isS ? lay.scale_count[mm] : lay.symbol_count[mm], nb = isS ? bs : bw, KK = isS ? M->KS : M->K;
            for (first = 0; first < all; first += chunk, g++) {
                uint32_t n = all - first < chunk ? all - first : chunk, j, k, x0, bestlen = 0xffffffffu, bk = 0;
                if (!isS) {
                    uint32_t srel = 0, pm = 8, mx = 8;
                    for (j = 0; j < n; j++) {
                        uint32_t i = first + j, x = (wt[mm][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                        if (!(i % 32u)) { srel = wex_nr2_srel(sc[mm][i / 32u], anchor[mm]); pm = 8; mx = 8; }
                        ctx[j] = (uint16_t)wex_nr2_context(srel, pm, mx);
                        pm = x & 7u; if (mx == 8u || pm > mx) mx = pm;
                    }
                }
                uint32_t cls = isS ? 0 : stream_class(M, sc[mm], first / 32u, n / 32u, anchor[mm]);
                ident[2] = g;
                x0 = isS ? wex_nr3_stream_x0(ident, sc[mm] + first, n) : wex_nr3_stream_x0(ident, wt[mm] + first / 2u, n / 2u);
                for (k = 0; k < KK; k++) {
                    unsigned char *end = tmp + TMP, *p = end; uint32_t x = x0, len; int b;
                    for (j = n; j-- > 0;) {
                        uint32_t i = first + j;
                        const uint32_t *c;
                        uint32_t s;
                        if (isS) {
                            c = M->scdf + ((size_t)k * 3 + mm) * 257u; s = ssym(sc[mm][i], anchor[mm], M->smode);
                            if (M->smode == 2) {
                                uint32_t y = s; s = M->smap[y];
                                if (s == 255u) {   /* escape: raw byte (decoded after the escape, so put first) */
                                    if (put(&x, &p, tmp, M->P, y << (M->P - 8u), 1u << (M->P - 8u))) goto done;
                                    s = M->ns - 1u;
                                }
                            }
                        }
                        else { c = M->wcdf + ((((size_t)cls * M->K + k) * 3 + mm) * NC + ctx[j]) * 17u; s = (wt[mm][i / 2u] >> ((i & 1u) * 4u)) & 15u; }
                        if (put(&x, &p, tmp, M->P, c[s], c[s + 1] - c[s])) goto done;
                    }
                    for (b = 0; b < 4; b++) { *--p = (unsigned char)x; x >>= 8; }
                    len = (uint32_t)(end - p);
                    if (len < bestlen) { bestlen = len; bk = k; memcpy(best, p, len); }
                }
                if (bestlen > 65535u || cap - cursor < bestlen) goto done;
                memcpy(out + cursor, best, bestlen); cursor += bestlen;
                out[32u + g * 2u] = (unsigned char)bestlen; out[32u + g * 2u + 1u] = (unsigned char)(bestlen >> 8);
                if (isS) sp += bestlen; else wp += bestlen;
                for (j = 0; j < nb; j++, bitpos++)
                    if (bk >> j & 1u) out[32u + 2u * (S + W) + bitpos / 8u] |= (unsigned char)(1u << (bitpos % 8u));
                if (selk) selk[g] = (uint8_t)bk;
            }
        }
    }
    {
        uint32_t f[8] = {0x34524e57u, model_id, (uint32_t)sp, (uint32_t)wp, wchunk, schunk, ident[3], owner}, i;
        if (M->P == 14 && M->K == 1 && M->KS == 1 && M->smode == 0) f[0] = WEX_NR3_MAGIC;   /* WNR3-identical profile */
        for (i = 0; i < 32u; i++) out[i] = (unsigned char)(f[i / 4u] >> ((i % 4u) * 8u));
    }
    *used = cursor; rc = 0;
done:
    free(tmp); free(best); free(ctx);
    return rc;
}

static inline int dsym(const uint32_t *c, uint32_t nsym, uint32_t P, uint32_t *x, const unsigned char **p,
                       const unsigned char *end, uint32_t *out) {
    /* binary search over cdf[0..nsym] */
    uint32_t slot = *x & ((1u << P) - 1u), lo = 0, hi = nsym, f;
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (c[mid] <= slot) lo = mid; else hi = mid; }
    f = c[lo + 1] - c[lo];
    if (!f) return -1;
    *x = f * (*x >> P) + slot - c[lo];
    while (*x < WEX_NR_L) { if (*p == end) return -1; *x = (*x << 8) | *(*p)++; }
    *out = lo;
    return 0;
}

/* Decode a record of m_encode into a coded slab (scales + weights at the entry's offsets), checking every
 * stream hash and CZ, then undo the ceil form (so out = Z(source) when everything is right). */
int m_decode(const Mdl *M, const WexEntry *e, const unsigned char *rec, size_t total, uint32_t model_id, unsigned char *slab) {
    WexNr2Layout lay; unsigned char *sc[3], *wt[3];
    uint32_t m, S = 0, W = 0, g = 0, ident[4], bw = ceil_log2(M->K), bs = ceil_log2(M->KS), pass;
    uint32_t wchunk = wex_nr_u32(rec + 16), schunk = wex_nr_u32(rec + 20), flags = rec[27], cz = flags & 8u;
    const unsigned char *payload; uint64_t bitpos = 0; size_t meta;
    wex_nr2_layout(e, &lay);
    for (m = 0; m < 3; m++) {
        sc[m] = slab + e->scale_offsets[m]; wt[m] = slab + e->weight_offsets[m];
        S += wex_nr2_streams(lay.scale_count[m], schunk); W += wex_nr2_streams(lay.symbol_count[m], wchunk);
    }
    meta = 32u + 2u * (size_t)(S + W) + ((uint64_t)S * bs + (uint64_t)W * bw + 7u) / 8u;
    if (meta > total) return -1;
    payload = rec + meta;
    ident[0] = model_id; ident[1] = wex_nr_u32(rec + 28); ident[3] = wex_nr_u32(rec + 24);
    for (pass = 0; pass < 6; pass++) {
        uint32_t mm = pass % 3u, isS = pass < 3u, first, chunk = isS ? schunk : wchunk;
        uint32_t all = isS ? lay.scale_count[mm] : lay.symbol_count[mm], nb = isS ? bs : bw, anchor = rec[24 + mm];
        for (first = 0; first < all; first += chunk, g++) {
            uint32_t n = all - first < chunk ? all - first : chunk, j, k = 0, x, s, len = wex_nr_u16(rec + 32u + g * 2u);
            uint32_t srel = 0, pm = 8, mx = 8;
            const unsigned char *p, *end;
            for (j = 0; j < nb; j++, bitpos++) k |= (uint32_t)(rec[32u + 2u * (S + W) + bitpos / 8u] >> (bitpos % 8u) & 1u) << j;
            if (k >= (isS ? M->KS : M->K) || len < 4u || payload + len > rec + total) return -1;
            x = (uint32_t)payload[0] << 24 | (uint32_t)payload[1] << 16 | (uint32_t)payload[2] << 8 | payload[3];
            if (x < WEX_NR_L || x >= (WEX_NR_L << 8)) return -1;
            p = payload + 4; end = payload + len;
            if (!isS) k += stream_class(M, sc[mm], first / 32u, n / 32u, anchor) * M->K;
            for (j = 0; j < n; j++) {
                uint32_t i = first + j;
                if (isS) {
                    if (dsym(M->scdf + ((size_t)k * 3 + mm) * 257u, M->smode == 2 ? M->ns : 256u, M->P, &x, &p, end, &s)) return -1;
                    if (M->smode == 2) {
                        if (s == M->ns - 1u) {   /* escape: 8 raw bits */
                            uint32_t sh = M->P - 8u, slot = x & ((1u << M->P) - 1u);
                            s = slot >> sh;
                            x = (1u << sh) * (x >> M->P) + slot - (s << sh);
                            while (x < WEX_NR_L) { if (p == end) return -1; x = (x << 8) | *p++; }
                            if (M->smap[s] != 255u) return -1;   /* non-canonical escape */
                        } else s = M->sval[s];
                    }
                    sc[mm][i] = (unsigned char)sinv(s, anchor, M->smode);
                } else {
                    if (!(i % 32u)) { srel = wex_nr2_srel(sc[mm][i / 32u], anchor); pm = 8; mx = 8; }
                    if (dsym(M->wcdf + (((size_t)k * 3 + mm) * NC + wex_nr2_context(srel, pm, mx)) * 17u, 16u, M->P, &x, &p, end, &s)
                        || (cz && s == 8u)) return -1;
                    if (!(i & 1u)) wt[mm][i / 2u] = (unsigned char)s; else wt[mm][i / 2u] |= (unsigned char)(s << 4);
                    pm = s & 7u; if (mx == 8u || pm > mx) mx = pm;
                }
            }
            ident[2] = g;
            if (p != end || x != (isS ? wex_nr3_stream_x0(ident, sc[mm] + first, n) : wex_nr3_stream_x0(ident, wt[mm] + first / 2u, n / 2u)))
                return -1;
            payload += len;
        }
    }
    if (payload != rec + total) return -1;
    return wex_nr3_ceil_inverse(&lay, flags & 7u, sc, wt);
}

/* Weight counts by a per-stream class given by the caller (cls[ws] < K, weight streams in g order). */
int m_count_by_class(const WexEntry *e, const unsigned char *slab, uint32_t wchunk, uint32_t K, const uint8_t *cls, uint64_t *acc) {
    WexNr2Layout lay; const unsigned char *sc[3], *wt[3]; uint32_t m, i, ws = 0;
    views(e, slab, &lay, sc, wt);
    for (m = 0; m < 3; m++) {
        uint32_t a = wex_nr3_anchor(sc[m], lay.scale_count[m]), srel = 0, pm = 8, mx = 8, k = 0;
        for (i = 0; i < lay.symbol_count[m]; i++) {
            uint32_t x = (wt[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
            if (!(i % wchunk)) { k = cls[ws++]; if (k >= K) return -1; }
            if (!(i % 32u)) { srel = wex_nr2_srel(sc[m][i / 32u], a); pm = 8; mx = 8; }
            acc[(((size_t)k * 3 + m) * NC + wex_nr2_context(srel, pm, mx)) * 16u + x]++;
            pm = x & 7u; if (mx == 8u || pm > mx) mx = pm;
        }
    }
    return 0;
}

/* Lloyd step with selector-free classes: weight stream of class c (from its scales) picks the cheapest of the
 * K sets c*K..c*K+K-1; cw[ncls*K][3][567][16]; acc likewise. */
double m_lloyd2(const WexEntry *e, const unsigned char *slab, uint32_t wchunk, uint32_t K, uint32_t ncls, const int32_t *thr,
                const double *cw, uint64_t *acc, uint8_t *assign) {
    WexNr2Layout lay; const unsigned char *sc[3], *wt[3]; uint32_t m, first, ws = 0; double total = 0;
    Mdl M; uint16_t *ctx = malloc(sizeof(uint16_t) * wchunk); uint8_t *sym = malloc(wchunk);
    M.ncls = ncls; M.thr = thr;
    views(e, slab, &lay, sc, wt);
    for (m = 0; m < 3; m++) {
        uint32_t a = wex_nr3_anchor(sc[m], lay.scale_count[m]);
        for (first = 0; first < lay.symbol_count[m]; first += wchunk, ws++) {
            uint32_t n = lay.symbol_count[m] - first < wchunk ? lay.symbol_count[m] - first : wchunk, j, k, best = 0;
            uint32_t srel = 0, pm = 8, mx = 8, cls = stream_class(&M, sc[m], first / 32u, n / 32u, a); double bc = 1e300;
            for (j = 0; j < n; j++) {
                uint32_t i = first + j, x = (wt[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                if (!(i % 32u)) { srel = wex_nr2_srel(sc[m][i / 32u], a); pm = 8; mx = 8; }
                ctx[j] = (uint16_t)wex_nr2_context(srel, pm, mx); sym[j] = (uint8_t)x;
                pm = x & 7u; if (mx == 8u || pm > mx) mx = pm;
            }
            for (k = 0; k < K; k++) {
                const double *t = cw + (((size_t)cls * K + k) * 3 + m) * NC * 16u; double c = 0;
                for (j = 0; j < n; j++) c += t[ctx[j] * 16u + sym[j]];
                if (c < bc) { bc = c; best = k; }
            }
            total += bc; if (assign) assign[ws] = (uint8_t)best;
            if (acc) for (j = 0; j < n; j++) acc[((((size_t)cls * K + best) * 3 + m) * NC + ctx[j]) * 16u + sym[j]]++;
        }
    }
    free(ctx); free(sym);
    return total;
}
