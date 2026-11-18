/* Scales, skeleton and integrity of the WNR3 expert record: CPU prototype (research only).
 *
 * Modes
 *   ovh_proto train RECS OUTDIR L:E ...          decode the listed (TRAIN) records with the reference decoder,
 *                                                build a WNM2 model of their coded data (model_train.bin) and the
 *                                                relative scale tables of this file (scale_train.bin)
 *   ovh_proto run RECS OUTDIR OUT.json L:E ...   for each (TEST) record: reference decode, harness check (the
 *                                                production model re-encodes the stored record byte for byte),
 *                                                then every profile below with the TRAIN model: encode, decode with
 *                                                this file's decoder, compare with the reference slab Z(source)
 *   ovh_proto audit RECS OUTDIR OUT.json N SEED L:E ...   corruption audit of the audit profiles (N trials per
 *                                                corruption type, profile and record)
 *
 * The entropy coder is a copy of WNR3's (wex_nr2_put / decode_symbol, 14-bit CDFs, 32-bit state) with these
 * switches per profile:
 *   wnr3   1: exact WNR3 record (32-B header, u16 lengths, x0 = L + h23, 4-B flush): must reproduce production.
 *   lenc   coded length table (copy of wnr4-record: groups of 32, u16 min, u8 width, packed residuals).
 *   sd     minimal 3/4-byte start state (byte renormalisation only).
 *   r16    16-bit renormalisation, L = 2^16 (state in [2^16, 2^32)), 4-B flush; x0 = 2^k + payload.
 *   k      x0 = 2^k + (tail nibbles | hash bits << 4*tail). Byte renormalisation needs k = 23.
 *   tail   last `tail` weight nibbles of each weight stream stored raw in the low bits of x0.
 *   digest 32-bit record digest in the header = XOR of every stream's full MurmurHash3 (same hash as WNR3).
 *   smode  scale coding: 0 = today (absolute byte, the model's order-0 table of the matrix);
 *          1 = rel16 global: symbol = scale - anchor[m] + 7 when in [0,14], else 15 (escape) followed by the
 *              raw byte as two flat 4-bit symbols; one 16-symbol table per matrix in the model;
 *          2 = rel16 per record: as 1, but the record carries its own 16-symbol table per matrix
 *              (u16 presence mask + u16 frequency of every present symbol but the last);
 *          3 = rel16 | mean8 global: as 1 with 10 contexts per matrix: the mean of the up to 8 previous scales of
 *              the same row and the same stream, in half steps relative to the anchor, clamped to +-4, or "none".
 *   inl    scales inline: no scale streams; each weight stream carries the scale symbol of every block just
 *          before the block's 32 nibbles (smode 1..3 for that symbol; mean8 is limited to the weight stream).
 * Everything a profile changes is in the record; the TRAIN model supplies the weight tables and scale tables.
 */
#include "/workspace/Wex-WRN2/c/tools/wex_nr3_tools.c"
#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef struct {
    const char *name;
    uint32_t wchunk, schunk, wnr3, lenc, sd, r16, k, tail, digest, smode, inl;
} Cfg;

static const Cfg cfgs[] = {
    /* name                 Cw     Cs   wnr3 lenc sd r16  k tail dig smode inl */
    {"wnr3_train",          2048,  512, 1, 0, 0, 0, 23, 0, 0, 0, 0},   /* WNR3 bytes with the TRAIN model: baseline */
    {"hdr40",               2048,  512, 0, 0, 0, 0, 23, 0, 0, 0, 0},   /* WNR4 40-B header, else WNR3 */
    {"lenc",                2048,  512, 0, 1, 0, 0, 23, 0, 0, 0, 0},
    {"sd",                  2048,  512, 0, 0, 1, 0, 23, 0, 0, 0, 0},
    {"r16_k23",             2048,  512, 0, 0, 0, 1, 23, 0, 0, 0, 0},
    {"r16_k16_dig",         2048,  512, 0, 0, 0, 1, 16, 0, 1, 0, 0},
    {"tail2_dig",           2048,  512, 0, 0, 0, 0, 23, 2, 1, 0, 0},
    {"tail5_dig",           2048,  512, 0, 0, 0, 0, 23, 5, 1, 0, 0},
    {"s_rel16",             2048,  512, 0, 0, 0, 0, 23, 0, 0, 1, 0},
    {"s_rec16",             2048,  512, 0, 0, 0, 0, 23, 0, 0, 2, 0},
    {"s_mean8",             2048,  512, 0, 0, 0, 0, 23, 0, 0, 3, 0},
    {"s_rel16_cs2048",      2048, 2048, 0, 0, 0, 0, 23, 0, 0, 1, 0},
    {"s_rec16_cs2048",      2048, 2048, 0, 0, 0, 0, 23, 0, 0, 2, 0},
    {"inl_rel16",           2048,  512, 0, 0, 0, 0, 23, 0, 0, 1, 1},
    {"inl_rec16",           2048,  512, 0, 0, 0, 0, 23, 0, 0, 2, 1},
    {"inl_mean8",           2048,  512, 0, 0, 0, 0, 23, 0, 0, 3, 1},
    {"w4096_s1024",         4096, 1024, 0, 0, 0, 0, 23, 0, 0, 0, 0},
    {"w8192_s2048",         8192, 2048, 0, 0, 0, 0, 23, 0, 0, 0, 0},
    /* combinations */
    {"A_lenc_sd",           2048,  512, 0, 1, 1, 0, 23, 0, 0, 0, 0},   /* = wnr4-record's lenc_sd (hash unchanged) */
    {"B_A_rec16",           2048,  512, 0, 1, 1, 0, 23, 0, 0, 2, 0},   /* + per-record scale tables */
    {"C_A_inl_rec16",       2048,  512, 0, 1, 1, 0, 23, 0, 0, 2, 1},   /* + scales inline */
    {"D_C_tail2_dig",       2048,  512, 0, 1, 1, 0, 23, 2, 1, 2, 1},   /* + tail 2 (15-bit stream hash) + digest */
    {"E_C_tail5_dig",       2048,  512, 0, 1, 1, 0, 23, 5, 1, 2, 1},   /* + tail 5 (3-bit stream hash) + digest */
    {"F_r16k16_inl_rec16",  2048,  512, 0, 1, 0, 1, 16, 0, 1, 2, 1},   /* 16-bit renorm, 16-bit stream hash + digest */
    {"G_C_w4096",           4096, 1024, 0, 1, 1, 0, 23, 0, 0, 2, 1},
    {"H_E_w4096",           4096, 1024, 0, 1, 1, 0, 23, 5, 1, 2, 1},
    {"I_C_w8192",           8192, 2048, 0, 1, 1, 0, 23, 0, 0, 2, 1},
    {"J_E_w8192",           8192, 2048, 0, 1, 1, 0, 23, 5, 1, 2, 1},
};
#define NCFG (sizeof(cfgs) / sizeof(cfgs[0]))
#define HDR4 40u
#define MAGIC4 UINT32_C(0x34524e57)
#define NCTX_S 10u

/* ---- scale tables of the TRAIN model ---------------------------------------- */
typedef struct {
    unsigned char rel[3][17 * 2];            /* smode 1: one 16-symbol CDF per matrix (u16 LE like the WNM2 blob) */
    unsigned char mean8[3][NCTX_S][17 * 2];  /* smode 3 */
} STables;
static unsigned char flat16[17 * 2];         /* escape payload: 16 x 1024 */

static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }
static uint32_t maskb(uint32_t bits) { return bits >= 32u ? UINT32_MAX : (1u << bits) - 1u; }

/* quantise over the present symbols only (absent ones get 0); used for per-record tables */
static void quantize_present(const uint64_t *counts, uint32_t symbols, unsigned char *cdf) {
    uint64_t c2[16];
    uint32_t idx[16], n = 0, s, at = 0;
    unsigned char tmp[17 * 2];
    for (s = 0; s < symbols; s++) if (counts[s]) { c2[n] = counts[s]; idx[n++] = s; }
    wex_nr2_quantize(c2, n, tmp);
    for (s = 0; s <= symbols; s++) {
        uint32_t i;
        for (i = 0; i < n && idx[i] < s; i++) ;
        at = wex_nr_u16(tmp + i * 2u);   /* cdf[s] = sum of freq of present symbols < s */
        cdf[s * 2u] = (unsigned char)at; cdf[s * 2u + 1u] = (unsigned char)(at >> 8);
    }
}

static inline uint32_t rel_sym(uint32_t s, uint32_t anchor) { int r = (int)s - (int)anchor + 7; return r >= 0 && r <= 14 ? (uint32_t)r : 15u; }

/* mean8 context of scale index i (matrix m, row length C), first index of its stream f0 */
static inline uint32_t mean8_ctx(const unsigned char *sc, uint32_t i, uint32_t C, uint32_t f0, uint32_t anchor) {
    uint32_t rowstart = i - i % C, st = i >= 8u ? i - 8u : 0u, j;
    int sum = 0, cnt, num, den, v;
    if (st < rowstart) st = rowstart;
    if (st < f0) st = f0;
    cnt = (int)(i - st);
    if (!cnt) return 9u;
    for (j = st; j < i; j++) sum += (int)sc[j] - (int)anchor;
    num = 4 * sum + cnt; den = 2 * cnt;
    v = num >= 0 ? num / den : -((-num + den - 1) / den);   /* floor */
    v = v < -4 ? -4 : v > 4 ? 4 : v;
    return (uint32_t)(v + 4);
}

/* ---- length section (copy of wnr4-record) ---------------------------------- */
static size_t lenc_pack(const uint32_t *len, uint32_t n, unsigned char *out) {
    size_t at = 0;
    uint32_t g, i;
    for (g = 0; g < n; g += 32u) {
        uint32_t cnt = n - g < 32u ? n - g : 32u, lo = len[g], hi = len[g], w = 0, have = 0;
        uint64_t acc = 0;
        for (i = 1; i < cnt; i++) { if (len[g + i] < lo) lo = len[g + i]; if (len[g + i] > hi) hi = len[g + i]; }
        while ((hi - lo) >> w) w++;
        out[at++] = (unsigned char)lo; out[at++] = (unsigned char)(lo >> 8); out[at++] = (unsigned char)w;
        for (i = 0; i < cnt; i++) {
            acc |= (uint64_t)(len[g + i] - lo) << have; have += w;
            while (have >= 8u) { out[at++] = (unsigned char)acc; acc >>= 8; have -= 8u; }
        }
        if (have) out[at++] = (unsigned char)acc;
    }
    return at;
}
static size_t lenc_unpack(const unsigned char *in, size_t cap, uint32_t n, uint32_t *len) {
    size_t at = 0;
    uint32_t g, i;
    for (g = 0; g < n; g += 32u) {
        uint32_t cnt = n - g < 32u ? n - g : 32u, lo, w, have = 0;
        uint64_t acc = 0;
        if (cap - at < 3u) return 0;
        lo = wex_nr_u16(in + at); w = in[at + 2u]; at += 3u;
        if (w > 16u || cap - at < (cnt * w + 7u) / 8u) return 0;
        for (i = 0; i < cnt; i++) {
            while (have < w) { acc |= (uint64_t)in[at++] << have; have += 8u; }
            len[g + i] = lo + (uint32_t)(acc & maskb(w)); acc >>= w; have -= w;
        }
        if (acc) return 0;
    }
    return at;
}

/* ---- stream geometry ---------------------------------------------------------- */
typedef struct {
    uint32_t n, matrix, scale_pass, first, count;   /* count = symbols (scales or nibbles) of the stream */
} SInfo;
typedef struct {
    uint32_t n;            /* streams */
    uint32_t nscale;       /* scale streams */
    SInfo *s;
} Geo;
static void geo_build(const Cfg *c, const WexNr2Layout *L, Geo *g) {
    uint32_t m, pass, first, n = 0;
    g->n = 0; g->nscale = 0;
    for (pass = 0; pass < 2u; pass++)
        for (m = 0; m < 3u; m++) {
            if (!pass && c->inl) continue;
            n += wex_nr2_streams(pass ? L->symbol_count[m] : L->scale_count[m], pass ? c->wchunk : c->schunk);
        }
    g->s = (SInfo *)malloc(n * sizeof(SInfo));
    for (pass = 0; pass < 2u; pass++)
        for (m = 0; m < 3u; m++) {
            uint32_t all = pass ? L->symbol_count[m] : L->scale_count[m], chunk = pass ? c->wchunk : c->schunk;
            if (!pass && c->inl) continue;
            for (first = 0; first < all; first += chunk) {
                SInfo *s = &g->s[g->n];
                s->n = g->n; s->matrix = m; s->scale_pass = !pass; s->first = first;
                s->count = all - first < chunk ? all - first : chunk;
                g->n++; if (!pass) g->nscale++;
            }
        }
}

/* ---- per-record context ------------------------------------------------------ */
typedef struct {
    const WexNr2Model *model;
    const STables *st;
    const WexNr2Layout *L;
    uint32_t C[3];                          /* scale columns (row length) per matrix */
    unsigned char anchor[3];
    unsigned char rec_cdf[3][17 * 2];       /* smode 2 */
    uint32_t ident[4];
    uint32_t flags;
} Ctx;

static const unsigned char *scale_cdf(const Cfg *c, const Ctx *x, uint32_t m, const unsigned char *sc, uint32_t i, uint32_t f0) {
    if (c->smode == 1u) return x->st->rel[m];
    if (c->smode == 2u) return x->rec_cdf[m];
    return x->st->mean8[m][mean8_ctx(sc, i, x->C[m], f0, x->anchor[m])];
}

/* stream hash over ident || data (data = scales of the stream, then (weights) packed bytes) */
static uint32_t stream_hash(const Cfg *c, const Ctx *x, const SInfo *s, unsigned char *const sc[3], unsigned char *const w[3]) {
    uint32_t id[4] = {x->ident[0], x->ident[1], s->n, x->ident[3]};
    uint32_t h0 = wex_nr3_ident_hash(id);
    if (s->scale_pass) return wex_nr3_h32_bytes(h0, sc[s->matrix] + s->first, s->count, 16u);
    if (!c->inl) return wex_nr3_h32_bytes(h0, w[s->matrix] + s->first / 2u, s->count / 2u, 16u);
    {   /* inline: blocks' scales then packed weights, as one key */
        static _Thread_local unsigned char buf[WEX_NR2_MAX_CHUNK / 32u + WEX_NR2_MAX_CHUNK / 2u];
        uint32_t nb = s->count / 32u;
        memcpy(buf, sc[s->matrix] + s->first / 32u, nb);
        memcpy(buf + nb, w[s->matrix] + s->first / 2u, s->count / 2u);
        return wex_nr3_h32_bytes(h0, buf, nb + s->count / 2u, 16u);
    }
}

/* ---- encoder ------------------------------------------------------------------ */
typedef struct { const unsigned char *cdf; uint8_t sym; } Sym;

static int put_sym(const Cfg *c, uint32_t *x, unsigned char **p, const unsigned char *floor, const unsigned char *cdf, uint32_t s) {
    uint32_t start = wex_nr_u16(cdf + s * 2u), f = wex_nr_u16(cdf + s * 2u + 2u) - start;
    if (!f) return -1;
    if (c->r16) {
        uint64_t mx = (uint64_t)f << 18;
        while (*x >= mx) { if (*p - floor < 2) return -1; *p -= 2; (*p)[0] = (unsigned char)(*x >> 8); (*p)[1] = (unsigned char)*x; *x >>= 16; }
    } else {
        uint32_t mx = ((WEX_NR_L >> WEX_NR2_PRECISION) << 8) * f;
        while (*x >= mx) { if (*p == floor) return -1; *--*p = (unsigned char)*x; *x >>= 8; }
    }
    *x = ((*x / f) << WEX_NR2_PRECISION) + *x % f + start;
    return 0;
}

typedef struct { uint64_t used, len_bytes, table_bytes, scale_bytes, weight_bytes, padded4096, padded64; uint32_t streams, short_flush; } Sizes;

/* Encode one record. sc/w = coded scales/weights. Returns 0. */
static int v_encode(const Cfg *c, Ctx *x, unsigned char *const sc[3], unsigned char *const w[3],
                    unsigned char *out, size_t cap, Sizes *sz, uint32_t *hashes_out) {
    const WexNr2Layout *L = x->L;
    Geo geo;
    uint32_t g, m, digest = 0, header = c->wnr3 ? WEX_NR2_HEADER_BYTES : HDR4;
    unsigned char tmp[2u * WEX_NR2_MAX_CHUNK + 64u], *payload, tables[128];
    uint32_t *len;
    size_t cursor = 0, tsize = 0;
    static _Thread_local Sym syms[WEX_NR2_MAX_CHUNK + 3u * (WEX_NR2_MAX_CHUNK / 32u) + 8u];
    memset(sz, 0, sizeof(*sz));
    geo_build(c, L, &geo);
    len = (uint32_t *)malloc(geo.n * sizeof(uint32_t));
    payload = (unsigned char *)malloc(cap);
    if (c->smode == 2u) {   /* per-record tables */
        for (m = 0; m < 3u; m++) {
            uint64_t cnt[16] = {0};
            uint32_t i, mask = 0, s, np = 0, last = 0;
            for (i = 0; i < L->scale_count[m]; i++) {
                uint32_t r = rel_sym(sc[m][i], x->anchor[m]);
                cnt[r]++;   /* r == 15: escape (its raw byte uses the flat table) */
            }
            quantize_present(cnt, 16u, x->rec_cdf[m]);
            for (s = 0; s < 16u; s++) if (cnt[s]) { mask |= 1u << s; np++; last = s; }
            tables[tsize++] = (unsigned char)mask; tables[tsize++] = (unsigned char)(mask >> 8);
            for (s = 0; s < 16u; s++) if (cnt[s] && s != last) {
                uint32_t f = wex_nr_u16(x->rec_cdf[m] + s * 2u + 2u) - wex_nr_u16(x->rec_cdf[m] + s * 2u);
                tables[tsize++] = (unsigned char)f; tables[tsize++] = (unsigned char)(f >> 8);
            }
            (void)np;
        }
    }
    for (g = 0; g < geo.n; g++) {
        const SInfo *s = &geo.s[g];
        uint32_t ns = 0, j, h, state, tail = s->scale_pass ? 0u : c->tail, kw = tail < s->count ? tail : s->count, tv = 0;
        uint32_t coded_n, length, hb;
        unsigned char *end = tmp + sizeof(tmp), *p = end;
        m = s->matrix;
        /* forward symbol list */
        if (s->scale_pass) {
            for (j = 0; j < s->count; j++) {
                uint32_t i = s->first + j, v = sc[m][i];
                if (c->smode == 0u) { syms[ns].cdf = wex_nr2_scale_cdf(x->model, m); syms[ns++].sym = (uint8_t)v; }
                else {
                    uint32_t r = rel_sym(v, x->anchor[m]);
                    syms[ns].cdf = scale_cdf(c, x, m, sc[m], i, s->first); syms[ns++].sym = (uint8_t)r;
                    if (r == 15u) { syms[ns].cdf = flat16; syms[ns++].sym = (uint8_t)(v >> 4); syms[ns].cdf = flat16; syms[ns++].sym = (uint8_t)(v & 15u); }
                }
            }
        } else {
            uint32_t srel = 0, prev = 8, mx = 8;
            for (j = 0; j < s->count; j++) {
                uint32_t i = s->first + j, q = (w[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                if (!(i % 32u)) {
                    uint32_t b = i / 32u;
                    if (c->inl) {
                        uint32_t v = sc[m][b], r = rel_sym(v, x->anchor[m]);
                        syms[ns].cdf = scale_cdf(c, x, m, sc[m], b, s->first / 32u); syms[ns++].sym = (uint8_t)r;
                        if (r == 15u) { syms[ns].cdf = flat16; syms[ns++].sym = (uint8_t)(v >> 4); syms[ns].cdf = flat16; syms[ns++].sym = (uint8_t)(v & 15u); }
                    }
                    srel = wex_nr2_srel(sc[m][b], x->anchor[m]); prev = 8; mx = 8;
                }
                syms[ns].cdf = wex_nr2_weight_cdf(x->model, m, wex_nr2_context(srel, prev, mx)); syms[ns++].sym = (uint8_t)q;
                prev = q & 7u; if (mx == 8u || prev > mx) mx = prev;
            }
        }
        h = stream_hash(c, x, s, sc, w);
        if (hashes_out) hashes_out[g] = h;
        digest ^= h;
        coded_n = ns - kw;
        for (j = 0; j < kw; j++) tv |= (uint32_t)syms[coded_n + j].sym << (4u * j);
        hb = c->k - 4u * kw;
        state = (1u << c->k) + (tv | (h & maskb(hb)) << (4u * kw));
        for (j = coded_n; j-- > 0;) if (put_sym(c, &state, &p, tmp, syms[j].cdf, syms[j].sym)) { fprintf(stderr, "put fail\n"); exit(2); }
        if (c->sd) {
            uint32_t nb = state >> 24 ? 4u : 3u, b;   /* byte renorm: state >= 2^23 */
            for (b = 0; b < nb; b++) { *--p = (unsigned char)state; state >>= 8; }
            sz->short_flush += nb < 4u;
        } else { uint32_t b; for (b = 0; b < 4u; b++) { *--p = (unsigned char)state; state >>= 8; } }
        length = (uint32_t)(end - p);
        if (length > 65535u) { fprintf(stderr, "stream too long\n"); exit(2); }
        memcpy(payload + cursor, p, length); cursor += length; len[g] = length;
        if (s->scale_pass) sz->scale_bytes += length; else sz->weight_bytes += length;
    }
    memcpy(out + header, tables, tsize); sz->table_bytes = tsize;
    if (c->lenc) sz->len_bytes = lenc_pack(len, geo.n, out + header + tsize);
    else {
        for (g = 0; g < geo.n; g++) { out[header + tsize + g * 2u] = (unsigned char)len[g]; out[header + tsize + g * 2u + 1u] = (unsigned char)(len[g] >> 8); }
        sz->len_bytes = 2u * (size_t)geo.n;
    }
    memcpy(out + header + tsize + sz->len_bytes, payload, cursor);
    put32(out, c->wnr3 ? WEX_NR3_MAGIC : MAGIC4); put32(out + 4, x->model->id);
    put32(out + 8, (uint32_t)sz->scale_bytes); put32(out + 12, (uint32_t)sz->weight_bytes);
    put32(out + 16, c->wchunk); put32(out + 20, c->schunk); put32(out + 24, x->ident[3]); put32(out + 28, x->ident[1]);
    if (!c->wnr3) { put32(out + 32, c->digest ? digest : 0u); put32(out + 36, (uint32_t)(sz->len_bytes)); }
    sz->streams = geo.n; sz->used = header + tsize + sz->len_bytes + cursor;
    sz->padded4096 = (sz->used + 4095u) & ~(uint64_t)4095u; sz->padded64 = (sz->used + 63u) & ~(uint64_t)63u;
    free(len); free(payload); free(geo.s);
    return 0;
}

/* ---- decoder ------------------------------------------------------------------ */
typedef struct {
    Geo geo;
    uint32_t *len;
    size_t *off;       /* offset of each stream in the record */
    uint32_t digest, flags;
    Ctx x;
} Parsed;

static void parsed_free(Parsed *P) { free(P->len); free(P->off); free(P->geo.s); }

/* structural parse: header, tables, lengths. -1 on any failure. */
static int v_parse(const Cfg *c, const WexNr2Model *model, const STables *st, const WexNr2Layout *L, const uint32_t C[3],
                   const unsigned char *rec, size_t used, uint32_t owner, Parsed *P) {
    uint32_t header = c->wnr3 ? WEX_NR2_HEADER_BYTES : HDR4, g, m;
    size_t at, lsec, total = 0;
    memset(P, 0, sizeof(*P));
    if (used < header || wex_nr_u32(rec) != (c->wnr3 ? WEX_NR3_MAGIC : MAGIC4) || wex_nr_u32(rec + 4) != model->id ||
        wex_nr_u32(rec + 16) != c->wchunk || wex_nr_u32(rec + 20) != c->schunk || wex_nr_u32(rec + 28) != owner) return -1;
    P->flags = rec[27];
    if (P->flags & 0xF0u) return -1;
    geo_build(c, L, &P->geo);
    P->len = (uint32_t *)malloc(P->geo.n * sizeof(uint32_t));
    P->off = (size_t *)malloc(P->geo.n * sizeof(size_t));
    P->x.model = model; P->x.st = st; P->x.L = L;
    for (m = 0; m < 3u; m++) { P->x.C[m] = C[m]; P->x.anchor[m] = rec[24 + m]; }
    P->x.ident[0] = model->id; P->x.ident[1] = owner; P->x.ident[3] = wex_nr_u32(rec + 24); P->x.flags = P->flags;
    at = header;
    if (c->smode == 2u) {
        for (m = 0; m < 3u; m++) {
            uint32_t mask, s, cum = 0, last = 16;
            if (used - at < 2u) return -1;
            mask = wex_nr_u16(rec + at); at += 2;
            if (!mask) return -1;
            for (s = 0; s < 16u; s++) if (mask >> s & 1u) last = s;
            for (s = 0; s <= 16u; s++) {
                P->x.rec_cdf[m][s * 2u] = (unsigned char)cum; P->x.rec_cdf[m][s * 2u + 1u] = (unsigned char)(cum >> 8);
                if (s < 16u && (mask >> s & 1u)) {
                    uint32_t f;
                    if (s == last) f = WEX_NR2_TOTAL - cum;
                    else { if (used - at < 2u) return -1; f = wex_nr_u16(rec + at); at += 2; }
                    if (!f || cum + f > WEX_NR2_TOTAL) return -1;
                    cum += f;
                }
            }
            if (cum != WEX_NR2_TOTAL) return -1;
        }
    }
    if (c->wnr3) lsec = 2u * (size_t)P->geo.n;
    else lsec = wex_nr_u32(rec + 36);
    if (lsec > used - at) return -1;
    if (c->lenc) { if (lenc_unpack(rec + at, lsec, P->geo.n, P->len) != lsec) return -1; }
    else { if (lsec != 2u * (size_t)P->geo.n) return -1; for (g = 0; g < P->geo.n; g++) P->len[g] = wex_nr_u16(rec + at + 2u * g); }
    at += lsec;
    {
        uint64_t sp = 0, wp = 0;
        for (g = 0; g < P->geo.n; g++) {
            P->off[g] = at + total; total += P->len[g];
            if (P->geo.s[g].scale_pass) sp += P->len[g]; else wp += P->len[g];
            if (P->len[g] < (c->sd ? 3u : 4u)) return -1;
        }
        if (at + total != used || sp != wex_nr_u32(rec + 8) || wp != wex_nr_u32(rec + 12)) return -1;
    }
    P->digest = c->wnr3 ? 0u : wex_nr_u32(rec + 32);
    if (!c->wnr3 && !c->digest && P->digest) return -1;
    return 0;
}

static inline int dec_sym(const Cfg *c, const unsigned char *cdf, uint32_t nsym, uint32_t *x, const unsigned char **p, const unsigned char *end, uint32_t *out) {
    uint32_t slot = *x & (WEX_NR2_TOTAL - 1u), lo = 0, hi = nsym, start, f;
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
    start = wex_nr_u16(cdf + lo * 2u); f = wex_nr_u16(cdf + lo * 2u + 2u) - start;
    if (!f) return -1;   /* cannot happen with a valid cdf (search lands on f > 0); kept as a guard */
    *x = f * (*x >> WEX_NR2_PRECISION) + slot - start;
    if (c->r16) {
        if (*x < (1u << 16)) { if (end - *p < 2) return -1; *x = *x << 16 | (uint32_t)(*p)[0] << 8 | (*p)[1]; *p += 2; }
    } else {
        while (*x < WEX_NR_L) { if (*p == end) return -1; *x = *x << 8 | *(*p)++; }
    }
    *out = lo;
    return 0;
}

/* decode stream g into sc/w (coded domain). Returns 0 and its full hash in *hash, -1 on failure. */
static int v_decode_stream(const Cfg *c, const Parsed *P, const unsigned char *rec, uint32_t g,
                           unsigned char *const sc[3], unsigned char *const w[3], uint32_t *hash) {
    const SInfo *s = &P->geo.s[g];
    const Ctx *x = &P->x;
    const unsigned char *p = rec + P->off[g], *end = p + P->len[g];
    uint32_t m = s->matrix, state = 0, j, cz = P->flags & WEX_NR3_FLAG_CZ, tail = s->scale_pass ? 0u : c->tail;
    uint32_t kw = tail < s->count ? tail : s->count, h, v, hb;
    /* open */
    if (c->sd) { uint32_t nb = 0; while (state < WEX_NR_L && p < end && nb < 4u) { state = state << 8 | *p++; nb++; } if (state < WEX_NR_L) return -1; }
    else { if (end - p < 4) return -1; state = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; p += 4; }
    if (c->r16) { if (state < (1u << 16)) return -1; }
    else if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) return -1;
    if (s->scale_pass) {
        for (j = 0; j < s->count; j++) {
            uint32_t i = s->first + j, sym;
            if (c->smode == 0u) { if (dec_sym(c, wex_nr2_scale_cdf(x->model, m), 256u, &state, &p, end, &sym)) return -1; sc[m][i] = (unsigned char)sym; }
            else {
                if (dec_sym(c, scale_cdf(c, x, m, sc[m], i, s->first), 16u, &state, &p, end, &sym)) return -1;
                if (sym == 15u) { uint32_t a, b; if (dec_sym(c, flat16, 16u, &state, &p, end, &a) || dec_sym(c, flat16, 16u, &state, &p, end, &b)) return -1; sc[m][i] = (unsigned char)(a << 4 | b); if (rel_sym(sc[m][i], x->anchor[m]) != 15u) return -1; }
                else sc[m][i] = (unsigned char)(sym + x->anchor[m] - 7u);
            }
        }
    } else {
        uint32_t srel = 0, prev = 8, mx = 8, coded_n = s->count - kw;
        for (j = 0; j < s->count; j++) {
            uint32_t i = s->first + j, q;
            if (!(i % 32u)) {
                uint32_t b = i / 32u;
                if (c->inl) {
                    uint32_t sym;
                    if (dec_sym(c, scale_cdf(c, x, m, sc[m], b, s->first / 32u), 16u, &state, &p, end, &sym)) return -1;
                    if (sym == 15u) { uint32_t a, bb; if (dec_sym(c, flat16, 16u, &state, &p, end, &a) || dec_sym(c, flat16, 16u, &state, &p, end, &bb)) return -1; sc[m][b] = (unsigned char)(a << 4 | bb); if (rel_sym(sc[m][b], x->anchor[m]) != 15u) return -1; }
                    else sc[m][b] = (unsigned char)(sym + x->anchor[m] - 7u);
                }
                srel = wex_nr2_srel(sc[m][b], x->anchor[m]); prev = 8; mx = 8;
            }
            if (j < coded_n) { if (dec_sym(c, wex_nr2_weight_cdf(x->model, m, wex_nr2_context(srel, prev, mx)), 16u, &state, &p, end, &q)) return -1; }
            else {
                if (j == coded_n && (p != end || state >> c->k != 1u)) return -1;
                q = ((state - (1u << c->k)) >> (4u * (j - coded_n))) & 15u;
            }
            if (cz && q == 8u) return -1;
            if (!(i & 1u)) w[m][i / 2u] = (unsigned char)q; else w[m][i / 2u] |= (unsigned char)(q << 4);
            prev = q & 7u; if (mx == 8u || prev > mx) mx = prev;
        }
    }
    if (p != end || state >> c->k != 1u) return -1;
    v = state - (1u << c->k);
    h = stream_hash(c, x, s, sc, w);
    hb = c->k - 4u * kw;
    if (v >> (4u * kw) != (h & maskb(hb))) return -1;
    *hash = h;
    return 0;
}

/* full decode into sc/w (coded) + ceil inverse; returns 0 on accept */
static int v_decode(const Cfg *c, const WexNr2Model *model, const STables *st, const WexNr2Layout *L, const uint32_t C[3],
                    const unsigned char *rec, size_t used, uint32_t owner, unsigned char *const sc[3], unsigned char *const w[3]) {
    Parsed P;
    uint32_t g, digest = 0, h;
    int r = -1;
    if (v_parse(c, model, st, L, C, rec, used, owner, &P)) goto done;
    for (g = 0; g < P.geo.n; g++) { if (v_decode_stream(c, &P, rec, g, sc, w, &h)) goto done; digest ^= h; }
    if (c->digest && digest != P.digest) goto done;
    r = wex_nr3_ceil_inverse(L, P.flags & WEX_NR3_FLAG_CEIL_MASK, sc, w);
done:
    parsed_free(&P);
    return r;
}

/* ---- I/O helpers --------------------------------------------------------------- */
static unsigned char *slurp(const char *dir, const char *name, size_t *n) {
    char path[1024];
    FILE *f;
    unsigned char *p;
    long size;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    p = (unsigned char *)malloc((size_t)size + 1u);
    if (!p || fread(p, 1, (size_t)size, f) != (size_t)size) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    fclose(f);
    *n = (size_t)size;
    return p;
}
static void spit(const char *dir, const char *name, const void *p, size_t n) {
    char path[1024];
    FILE *f;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fclose(f);
}

typedef struct {
    unsigned char *page, *prod_blob, *table;
    WexHeader hdr;
    WexNr2Model prod;
} Container;
static void open_recs(const char *dir, Container *K) {
    size_t n, nm;
    K->page = slurp(dir, "header.bin", &n);
    K->prod_blob = slurp(dir, "model.bin", &nm);
    K->table = slurp(dir, "entries.bin", &n);
    memcpy(&K->hdr, K->page, sizeof(K->hdr));
    if (wex_nr2_model_open(K->prod_blob, nm, &K->prod)) { fprintf(stderr, "bad model\n"); exit(1); }
}

/* reference decode + prepare: slab = Z(source) (output domain), coded = coded domain. Returns flags or -1. */
typedef struct {
    int layer, expert;
    WexEntry e;
    WexNr2Layout L;
    uint32_t C[3], owner, flags;
    unsigned char *rec, *slab, *coded;
    size_t rec_bytes;
    uint64_t raw;
    unsigned char *sc[3], *w[3];
    unsigned char anchor[3];
} Expert;
static int load_expert(const Container *K, const char *dir, const char *spec, Expert *X) {
    char name[64];
    uint64_t aligned;
    uint32_t stats[16], m;
    int flags;
    sscanf(spec, "%d:%d", &X->layer, &X->expert);
    memcpy(&X->e, K->table + ((size_t)X->layer * K->hdr.num_experts + X->expert) * sizeof(WexEntry), sizeof(WexEntry));
    snprintf(name, sizeof(name), "rec_%02d_%03d.bin", X->layer, X->expert);
    X->rec = slurp(dir, name, &X->rec_bytes);
    wex_entry_raw_sizes(&X->e, &X->raw, &aligned);
    wex_nr2_layout(&X->e, &X->L);
    X->slab = (unsigned char *)malloc((size_t)X->raw); X->coded = (unsigned char *)malloc((size_t)X->raw);
    X->owner = (uint32_t)X->layer << 16 | (uint32_t)X->expert;
    if (X->rec_bytes != X->e.total_bytes || wex_nr3_decode(K->prod_blob, &X->e, X->rec, (uint32_t)X->layer, (uint32_t)X->expert, X->slab)) return -1;
    flags = wex_nr3_prepare(&X->e, X->slab, 1, X->coded, stats);
    if (flags < 0 || (uint32_t)flags != X->rec[27] || stats[13] || stats[14]) return -1;
    X->flags = (uint32_t)flags;
    for (m = 0; m < 3u; m++) {
        X->sc[m] = X->coded + X->e.scale_offsets[m]; X->w[m] = X->coded + X->e.weight_offsets[m];
        X->C[m] = X->e.scale_shapes[m][1];
        X->anchor[m] = X->rec[24 + m];
        if (X->anchor[m] != wex_nr3_anchor(X->sc[m], X->L.scale_count[m])) return -1;
    }
    return 0;
}
static void free_expert(Expert *X) { free(X->rec); free(X->slab); free(X->coded); }

static void ctx_init(Ctx *x, const WexNr2Model *model, const STables *st, const Expert *X) {
    uint32_t m;
    memset(x, 0, sizeof(*x));
    x->model = model; x->st = st; x->L = &X->L; x->flags = X->flags;
    for (m = 0; m < 3u; m++) { x->C[m] = X->C[m]; x->anchor[m] = X->anchor[m]; }
    x->ident[0] = model->id; x->ident[1] = X->owner;
    x->ident[3] = (uint32_t)X->anchor[0] | (uint32_t)X->anchor[1] << 8 | (uint32_t)X->anchor[2] << 16 | X->flags << 24;
}

/* ---- train ------------------------------------------------------------------------ */
static int do_train(int argc, char **argv) {
    Container K;
    int n = argc - 4, i;
    uint64_t *wc = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), sc[3 * 256] = {0};
    uint64_t rel[3][16] = {{0}}, m8[3][NCTX_S][16];
    unsigned char blob[WEX_NR2_MODEL_BYTES];
    STables st;
    uint32_t id;
    int bad = 0;
    memset(m8, 0, sizeof(m8));
    open_recs(argv[2], &K);
#pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < n; i++) {
        Expert X;
        uint64_t *wl = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), sl[3 * 256] = {0}, rl[3][16] = {{0}}, ml[3][NCTX_S][16];
        uint32_t m, j, k;
        memset(ml, 0, sizeof(ml));
        if (load_expert(&K, argv[2], argv[4 + i], &X)) {
#pragma omp atomic
            bad++;
            continue;
        }
        wex_nr3_count(&X.L, (const unsigned char *const *)X.sc, (const unsigned char *const *)X.w, wl, sl);
        for (m = 0; m < 3u; m++)
            for (j = 0; j < X.L.scale_count[m]; j++) {
                uint32_t r = rel_sym(X.sc[m][j], X.anchor[m]);
                rl[m][r]++;
                /* mean8 for the separate-stream geometry at Cs = 512 (the context is defined per stream; training
                   with Cs = 512 is used for every geometry) */
                ml[m][mean8_ctx(X.sc[m], j, X.C[m], j - j % 512u, X.anchor[m])][r]++;
            }
#pragma omp critical
        {
            for (k = 0; k < 3u * WEX_NR2_CONTEXTS * 16u; k++) wc[k] += wl[k];
            for (k = 0; k < 768u; k++) sc[k] += sl[k];
            for (m = 0; m < 3u; m++) for (k = 0; k < 16u; k++) { rel[m][k] += rl[m][k]; for (j = 0; j < NCTX_S; j++) m8[m][j][k] += ml[m][j][k]; }
            fprintf(stderr, "train %s\n", argv[4 + i]);
        }
        free(wl); free_expert(&X);
    }
    if (bad) { fprintf(stderr, "%d records failed\n", bad); return 1; }
    if (wex_nr3_build_model(wc, sc, blob, &id)) return 1;
    {
        uint32_t m, j;
        for (m = 0; m < 3u; m++) {
            wex_nr2_quantize(rel[m], 16u, st.rel[m]);
            for (j = 0; j < NCTX_S; j++) wex_nr2_quantize(m8[m][j], 16u, st.mean8[m][j]);
        }
    }
    spit(argv[3], "model_train.bin", blob, sizeof(blob));
    spit(argv[3], "scale_train.bin", &st, sizeof(st));
    fprintf(stderr, "model id %08x from %d records\n", id, n);
    return 0;
}

static void load_train(const char *dir, unsigned char **blob, WexNr2Model *model, STables *st) {
    size_t n;
    unsigned char *s;
    *blob = slurp(dir, "model_train.bin", &n);
    if (wex_nr2_model_open(*blob, n, model)) { fprintf(stderr, "bad train model\n"); exit(1); }
    s = slurp(dir, "scale_train.bin", &n);
    if (n != sizeof(*st)) { fprintf(stderr, "bad scale tables\n"); exit(1); }
    memcpy(st, s, n); free(s);
}

/* ---- run --------------------------------------------------------------------------- */
typedef struct {
    int layer, expert, harness_ok, ok[NCFG];
    uint32_t stored_total, stored_used, SP, WP;
    Sizes sz[NCFG];
} RunRes;

static int do_run(int argc, char **argv) {
    Container K;
    unsigned char *tblob;
    WexNr2Model tmodel;
    STables st;
    int n = argc - 5, i, bad = 0;
    RunRes *res = (RunRes *)calloc((size_t)n, sizeof(RunRes));
    FILE *out;
    open_recs(argv[2], &K);
    load_train(argv[3], &tblob, &tmodel, &st);
#pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < n; i++) {
        RunRes *r = &res[i];
        Expert X;
        size_t cap, used = 0, total = 0;
        unsigned char *buf, *back, *bs[3], *bw[3];
        uint32_t m;
        unsigned int c;
        if (load_expert(&K, argv[2], argv[5 + i], &X)) { r->layer = -1; continue; }
        r->layer = X.layer; r->expert = X.expert; r->stored_total = (uint32_t)X.rec_bytes;
        r->SP = wex_nr_u32(X.rec + 8); r->WP = wex_nr_u32(X.rec + 12);
        cap = (size_t)X.raw * 2u;
        buf = (unsigned char *)malloc(cap); back = (unsigned char *)malloc((size_t)X.raw);
        /* harness: production model through the shipped encoder, and through this file's encoder */
        r->harness_ok = !wex_nr3_encode_record(K.prod_blob, &X.e, X.coded, X.flags, X.owner, wex_nr_u32(X.rec + 16), wex_nr_u32(X.rec + 20),
                                               buf, cap, &used, &total) && total == X.rec_bytes && !memcmp(buf, X.rec, X.rec_bytes);
        r->stored_used = (uint32_t)used;
        {
            Ctx x; Sizes s0;
            ctx_init(&x, &K.prod, &st, &X);
            v_encode(&cfgs[0], &x, X.sc, X.w, buf, cap, &s0, NULL);
            r->harness_ok = r->harness_ok && s0.used == used && !memcmp(buf, X.rec, used) &&
                            !v_decode(&cfgs[0], &K.prod, &st, &X.L, X.C, X.rec, used, X.owner, (bs[0] = back + X.e.scale_offsets[0], bs[1] = back + X.e.scale_offsets[1], bs[2] = back + X.e.scale_offsets[2], bs),
                                      (bw[0] = back + X.e.weight_offsets[0], bw[1] = back + X.e.weight_offsets[1], bw[2] = back + X.e.weight_offsets[2], bw)) &&
                            !memcmp(back, X.slab, (size_t)X.raw);
        }
        for (m = 0; m < 3u; m++) { bs[m] = back + X.e.scale_offsets[m]; bw[m] = back + X.e.weight_offsets[m]; }
        for (c = 0; c < NCFG; c++) {
            Ctx x;
            ctx_init(&x, &tmodel, &st, &X);
            v_encode(&cfgs[c], &x, X.sc, X.w, buf, cap, &r->sz[c], NULL);
            memset(back, 0xA5, (size_t)X.raw);
            r->ok[c] = !v_decode(&cfgs[c], &tmodel, &st, &X.L, X.C, buf, (size_t)r->sz[c].used, X.owner, bs, bw) && !memcmp(back, X.slab, (size_t)X.raw);
        }
        free(buf); free(back); free_expert(&X);
#pragma omp critical
        fprintf(stderr, "L%02d E%03d harness %s\n", r->layer, r->expert, r->harness_ok ? "ok" : "FAIL");
    }
    out = fopen(argv[4], "w");
    fprintf(out, "{\"raw_bytes\": 18800640, \"configs\": [");
    for (i = 0; i < (int)NCFG; i++)
        fprintf(out, "%s{\"name\": \"%s\", \"wchunk\": %u, \"schunk\": %u, \"wnr3\": %u, \"lenc\": %u, \"sd\": %u, \"r16\": %u, \"k\": %u, \"tail\": %u, \"digest\": %u, \"smode\": %u, \"inl\": %u}",
                i ? ", " : "", cfgs[i].name, cfgs[i].wchunk, cfgs[i].schunk, cfgs[i].wnr3, cfgs[i].lenc, cfgs[i].sd, cfgs[i].r16, cfgs[i].k, cfgs[i].tail, cfgs[i].digest, cfgs[i].smode, cfgs[i].inl);
    fprintf(out, "],\n \"cfg_columns\": [\"round_trip_ok\", \"used\", \"len_bytes\", \"table_bytes\", \"scale_bytes\", \"weight_bytes\", \"padded4096\", \"padded64\", \"streams\", \"short_flush\"],\n \"experts\": [\n");
    for (i = 0; i < n; i++) {
        RunRes *r = &res[i];
        unsigned int c;
        if (r->layer < 0) { bad++; continue; }
        fprintf(out, "  %s{\"layer\": %d, \"expert\": %d, \"harness_ok\": %d, \"stored_total\": %u, \"stored_used\": %u, \"SP\": %u, \"WP\": %u, \"cfg\": [",
                i ? "," : "", r->layer, r->expert, r->harness_ok, r->stored_total, r->stored_used, r->SP, r->WP);
        for (c = 0; c < NCFG; c++) {
            Sizes *s = &r->sz[c];
            fprintf(out, "%s[%d, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %u, %u]", c ? ", " : "", r->ok[c], (unsigned long long)s->used,
                    (unsigned long long)s->len_bytes, (unsigned long long)s->table_bytes, (unsigned long long)s->scale_bytes,
                    (unsigned long long)s->weight_bytes, (unsigned long long)s->padded4096, (unsigned long long)s->padded64, s->streams, s->short_flush);
            bad += !r->ok[c];
        }
        fprintf(out, "]}\n");
        bad += !r->harness_ok;
    }
    fprintf(out, " ]}\n");
    fclose(out);
    fprintf(stderr, "%d experts, %d failures\n", n, bad);
    return bad != 0;
}

#include "audit.inc"
#include "bench.inc"

int main(int argc, char **argv) {
    int i;
    for (i = 0; i < 16; i++) { flat16[i * 2] = (unsigned char)(i * 1024); flat16[i * 2 + 1] = (unsigned char)((i * 1024) >> 8); }
    flat16[32] = 0; flat16[33] = 0x40;   /* 16384 */
    if (argc >= 5 && !strcmp(argv[1], "train")) return do_train(argc, argv);
    if (argc >= 6 && !strcmp(argv[1], "run")) return do_run(argc, argv);
    if (argc >= 8 && !strcmp(argv[1], "audit")) return do_audit(argc, argv);
    if (argc >= 6 && !strcmp(argv[1], "bench")) return do_bench(argc, argv);
    fprintf(stderr, "usage: see the comment at the top of ovh_proto.c\n");
    return 1;
}
