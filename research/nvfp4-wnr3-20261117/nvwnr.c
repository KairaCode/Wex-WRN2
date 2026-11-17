/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Research prototype of a WNR3-style coder for NVFP4 (and MXFP4) expert slabs. Not a format: no hashes, no
 * owner, no model blob format; the record layout only mirrors WNR3 to count the same overheads.
 *
 *   nvwnr RUNS [options] --train A.nvs ... --test B.nvs ...
 *
 * Input: .nvs files of analyze.py (64-byte header "NVS1", block B, bytes of trailing scalars, 3 x (rows,
 * nibbles per row); then the scale bytes of the three matrices, the packed weights of the three matrices, the
 * trailing scalars). Training counts every --train file in the coded domain (after canonical zero and ceil form
 * when asked), builds one model (14-bit tables, the quantizer of wex_nr2_quantize), then every --test file is
 * encoded, decoded RUNS + 1 times (the first not timed) on one thread, and compared with its source (or with
 * Z(source) under --cz) byte for byte.
 *
 * Record as counted: 32-byte header (anchors and flags), u16 length per stream, rANS payloads (WNR2 coder:
 * 32-bit state, L = 2^23, 14-bit tables, byte renormalisation, 4-byte big-endian flush), the trailing scalars
 * stored raw, zero padding to a multiple of 4096. Streams: scales in chunks of CS, then nibbles in chunks of CW
 * (CW a multiple of B), per matrix, scale streams of all matrices first, as in WNR3.
 *
 * Options:
 *   --wctx srel7     class = clamp(s - anchor, -3, 3), s the scale byte (WNR3, for MXFP4)
 *   --wctx exp7      class = clamp(e - anchor, -3, 3), e = exponent field of the FP8 E4M3 scale (bits 3..6)
 *   --wctx codeK     class = clamp(s - anchor, -K, K), K given (for example code12): 2K + 1 classes
 *   --wctx none      one class
 *   (contexts = classes x 81: the class, then pm and mx as WNR3 section 4.1; anchors by rule g, section 4.3,
 *   on the coded scale bytes, or on the exponent fields for exp7)
 *   --sctx o0        scales order 0 per matrix (WNR3)
 *   --sctx o1b       context = row start, or clamp(previous scale in the row - anchor, -8, 7): 17 tables
 *   --sctx o1        context = row start, or the previous scale byte: 257 tables
 *   --sctx o1p       context = row start, or (previous scale byte, parity of the position in the row): 513
 *   --cz             canonical zero: nibble 8 coded as 0, output Z(source)
 *   --ceil           ceil form of WNR3 section 4.2 (scale step 1 for B = 32, 8 = one octave of E4M3 for B = 16)
 *   --cw N --cs N    chunk sizes (default 2048 and 512)
 * Output: one "json {...}" line per test file and one summary line.
 * Build: gcc -std=c99 -O2 -Wall -Wextra nvwnr.c -o nvwnr */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define TOTAL 16384u
#define RANS_L (1u << 23)
#define MAXF 512

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

typedef struct {
    char name[512];
    uint32_t B, extra, rows[3], cols[3];
    size_t scount[3], wcount[3], soff[3], woff[3], body;
    unsigned char *data;   /* body: scales m0..m2, weights m0..m2, extra */
} Slab;

static void die(const char *msg) { fprintf(stderr, "nvwnr: %s\n", msg); exit(2); }

static void load(const char *path, Slab *s) {
    FILE *f = fopen(path, "rb");
    unsigned char head[64];
    long n;
    uint32_t m;
    size_t at = 0;
    if (!f || fread(head, 1, 64, f) != 64 || memcmp(head, "NVS1", 4)) die("bad input");
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 64, SEEK_SET);
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", path);
    memcpy(&s->B, head + 4, 4); memcpy(&s->extra, head + 8, 4);
    for (m = 0; m < 3; m++) { memcpy(&s->rows[m], head + 16 + 8 * m, 4); memcpy(&s->cols[m], head + 20 + 8 * m, 4); }
    for (m = 0; m < 3; m++) {
        s->wcount[m] = (size_t)s->rows[m] * s->cols[m];
        s->scount[m] = s->wcount[m] / s->B;
        s->soff[m] = at; at += s->scount[m];
    }
    for (m = 0; m < 3; m++) { s->woff[m] = at; at += s->wcount[m] / 2; }
    s->body = at + s->extra;
    if ((long)s->body != n - 64) die("size does not match header");
    s->data = (unsigned char *)malloc(s->body);
    if (!s->data || fread(s->data, 1, s->body, f) != s->body) die("read");
    fclose(f);
}

/* ---- options and contexts ---- */
static int WMODE = 0, WK = 3, SMODE = 0, CZ = 0, CEIL = 0;   /* WMODE 0 srel/code, 1 exp7, 2 none */
static uint32_t CW = 2048, CS = 512;
static uint32_t nclass(void) { return WMODE == 2 ? 1u : 2u * (uint32_t)WK + 1u; }
static uint32_t nsctx(void) { return SMODE == 0 ? 1u : SMODE == 1 ? 17u : SMODE == 2 ? 257u : 513u; }

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static uint32_t wclass(unsigned s, unsigned anchor) {
    if (WMODE == 2) return 0;
    if (WMODE == 1) return (uint32_t)(clampi((int)((s >> 3) & 15u) - (int)anchor, -WK, WK) + WK);
    return (uint32_t)(clampi((int)s - (int)anchor, -WK, WK) + WK);
}
static uint32_t sctx(int prev, unsigned sanchor, size_t column) {   /* prev < 0: row start */
    uint32_t PARITY = (uint32_t)(column & 1u);
    if (SMODE == 0) return 0;
    if (prev < 0) return 0;
    if (SMODE == 1) return 1u + (uint32_t)(clampi(prev - (int)sanchor, -8, 7) + 8);
    if (SMODE == 3) return 1u + (uint32_t)prev * 2u + PARITY;
    return 1u + (uint32_t)prev;
}
static unsigned rule_g(const uint32_t *h, unsigned n) {
    unsigned v, best = 0;
    uint64_t bv = (uint64_t)h[0] + h[1];
    for (v = 1; v + 1 < n; v++) if ((uint64_t)h[v] + h[v + 1] > bv) { bv = (uint64_t)h[v] + h[v + 1]; best = v; }
    return best;
}

/* ---- coded domain: canonical zero, ceil form ---- */
static const unsigned char H_[8] = {0, 0, 1, 0, 2, 3, 4, 5};   /* halve on {0,2,4,5,6,7} */
static const unsigned char D_[8] = {0, 2, 4, 5, 6, 7, 7, 7};
static unsigned nib(const unsigned char *w, size_t i) { return (w[i >> 1] >> ((i & 1u) * 4u)) & 15u; }
static void setnib(unsigned char *w, size_t i, unsigned v) {
    unsigned sh = (unsigned)(i & 1u) * 4u;
    w[i >> 1] = (unsigned char)((w[i >> 1] & ~(15u << sh)) | (v << sh));
}
static int ceil_ok(unsigned s, uint32_t B) { return B == 32 ? s <= 254u : (s < 0x80u && s + 8u <= 0x7Eu); }
static uint32_t ceil_step(uint32_t B) { return B == 32 ? 1u : 8u; }

/* coded copy of a slab; returns the ceil mask. cz: count of mapped nibbles in *mapped */
static uint32_t prepare(const Slab *s, unsigned char *coded, uint64_t *mapped) {
    uint32_t m, mask = 0;
    size_t i, b, j;
    memcpy(coded, s->data, s->body);
    *mapped = 0;
    for (m = 0; m < 3; m++) {
        unsigned char *w = coded + s->woff[m], *sc = coded + s->soff[m];
        int gate = 1;
        if (CZ) for (i = 0; i < s->wcount[m] / 2; i++) {
            if ((w[i] & 15u) == 8u) { w[i] &= 0xF0u; (*mapped)++; }
            if ((w[i] >> 4) == 8u) { w[i] &= 0x0Fu; (*mapped)++; }
        }
        if (!CEIL) continue;
        for (b = 0; b < s->scount[m] && gate; b++) {
            unsigned mx = 0;
            for (j = 0; j < s->B; j++) { unsigned a = nib(w, b * s->B + j) & 7u; if (a > mx) mx = a; }
            if (mx == 5) gate = 0;
        }
        if (!gate) continue;
        mask |= 1u << m;
        for (b = 0; b < s->scount[m]; b++) {
            unsigned mx = 0, odd = 0;
            for (j = 0; j < s->B; j++) { unsigned a = nib(w, b * s->B + j) & 7u; if (a > mx) mx = a; if (a == 1 || a == 3) odd = 1; }
            if (mx == 7 && !odd && ceil_ok(sc[b], s->B)) {
                for (j = 0; j < s->B; j++) { unsigned q = nib(w, b * s->B + j); setnib(w, b * s->B + j, (q & 8u) | H_[q & 7u]); }
                sc[b] = (unsigned char)(sc[b] + ceil_step(s->B));
            }
        }
    }
    return mask;
}

static int ceil_inverse(const Slab *s, unsigned char *out, uint32_t mask) {
    uint32_t m;
    size_t b, j;
    for (m = 0; m < 3; m++) {
        unsigned char *w = out + s->woff[m], *sc = out + s->soff[m];
        if (!(mask >> m & 1u)) continue;
        for (b = 0; b < s->scount[m]; b++) {
            unsigned mx = 0;
            for (j = 0; j < s->B; j++) { unsigned a = nib(w, b * s->B + j) & 7u; if (a > mx) mx = a; }
            if (mx != 5) continue;
            if (sc[b] < ceil_step(s->B)) return -1;
            for (j = 0; j < s->B; j++) { unsigned q = nib(w, b * s->B + j); setnib(w, b * s->B + j, (q & 8u) | D_[q & 7u]); }
            sc[b] = (unsigned char)(sc[b] - ceil_step(s->B));
        }
    }
    return 0;
}

/* anchors of a coded slab: a[m] weight-class anchor, a[3 + m] scale-context anchor */
static void anchors(const Slab *s, const unsigned char *coded, unsigned char a[6]) {
    uint32_t m;
    size_t i;
    for (m = 0; m < 3; m++) {
        uint32_t h[256] = {0}, he[16] = {0};
        const unsigned char *sc = coded + s->soff[m];
        for (i = 0; i < s->scount[m]; i++) { h[sc[i]]++; he[(sc[i] >> 3) & 15u]++; }
        a[3 + m] = (unsigned char)rule_g(h, 256);
        a[m] = WMODE == 1 ? (unsigned char)rule_g(he, 16) : a[3 + m];
    }
}

/* ---- model ---- */
typedef struct { uint16_t *wcdf; uint16_t *scdf; uint64_t *wcnt, *scnt; } Model;   /* [3][ctx][17], [3][sctx][257] */

static void quantize(const uint64_t *counts, uint32_t symbols, uint16_t *cdf) {   /* = wex_nr2_quantize */
    uint64_t total = 0, remainder[256];
    uint32_t frequency[256], spare = TOTAL - symbols, used = 0, s, extra;
    unsigned char taken[256] = {0};
    for (s = 0; s < symbols; s++) total += counts[s];
    for (s = 0; s < symbols; s++) {
        uint64_t scaled = total ? counts[s] * spare : spare, divisor = total ? total : symbols;
        frequency[s] = 1u + (uint32_t)(scaled / divisor);
        remainder[s] = scaled % divisor;
        used += frequency[s];
    }
    for (extra = TOTAL - used; extra; extra--) {
        uint32_t best = symbols;
        for (s = 0; s < symbols; s++) if (!taken[s] && (best == symbols || remainder[s] > remainder[best])) best = s;
        taken[best] = 1; frequency[best]++;
    }
    used = 0;
    for (s = 0; s <= symbols; s++) { cdf[s] = (uint16_t)used; if (s < symbols) used += frequency[s]; }
}
static uint32_t cdf_at(const uint16_t *cdf, uint32_t s, uint32_t symbols) { (void)symbols; return cdf[s]; }

static void count_slab(Model *M, const Slab *s, const unsigned char *coded, const unsigned char a[6]) {
    uint32_t m, nc = nclass() * 81u, ns = nsctx();
    size_t i, r, c;
    for (m = 0; m < 3; m++) {
        const unsigned char *sc = coded + s->soff[m], *w = coded + s->woff[m];
        size_t per = s->scount[m] / s->rows[m];
        for (r = 0; r < s->rows[m]; r++) for (c = 0; c < per; c++) {
            int prev = c ? sc[r * per + c - 1] : -1;
            M->scnt[((size_t)m * ns + sctx(prev, a[3 + m], c)) * 256u + sc[r * per + c]]++;
        }
        unsigned pm = 8, mx = 8, cls = 0;
        for (i = 0; i < s->wcount[m]; i++) {
            unsigned q = nib(w, i);
            if (i % s->B == 0) { pm = mx = 8; cls = wclass(sc[i / s->B], a[m]); }
            M->wcnt[((size_t)m * nc + (cls * 9u + pm) * 9u + mx) * 16u + q]++;
            pm = q & 7u; if (mx == 8u || pm > mx) mx = pm;
        }
    }
}

static void build(Model *M) {
    uint32_t i, nc = nclass() * 81u, ns = nsctx();
    for (i = 0; i < 3u * nc; i++) quantize(M->wcnt + (size_t)i * 16u, 16u, M->wcdf + (size_t)i * 17u);
    for (i = 0; i < 3u * ns; i++) quantize(M->scnt + (size_t)i * 256u, 256u, M->scdf + (size_t)i * 257u);
}

/* ---- rANS (the WNR2 coder) ---- */
typedef struct { unsigned char *buf; size_t cap, pos; } Rev;   /* bytes written backwards from cap */
static void enc_put(uint32_t *x, Rev *o, uint32_t start, uint32_t freq) {
    uint32_t xmax = ((RANS_L >> 14) << 8) * freq;
    while (*x >= xmax) { o->buf[--o->pos] = (unsigned char)(*x & 255u); *x >>= 8; }
    *x = ((*x / freq) << 14) + (*x % freq) + start;
}
static void enc_flush(uint32_t x, Rev *o) {
    o->buf[--o->pos] = (unsigned char)x; o->buf[--o->pos] = (unsigned char)(x >> 8);
    o->buf[--o->pos] = (unsigned char)(x >> 16); o->buf[--o->pos] = (unsigned char)(x >> 24);
}

typedef struct { const unsigned char *p, *end; uint32_t x; } Dec;
static int dec_open(Dec *d, const unsigned char *p, size_t n) {
    if (n < 4) return -1;
    d->x = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    d->p = p + 4; d->end = p + n;
    return d->x >= RANS_L ? 0 : -1;
}
static inline int dec_sym(Dec *d, const uint16_t *cdf, uint32_t symbols) {
    uint32_t slot = d->x & (TOTAL - 1u), lo = 0, hi = symbols, s, a, b;
    while (hi - lo > 1u) { uint32_t mid = (lo + hi) >> 1; if (cdf[mid] <= slot) lo = mid; else hi = mid; }
    s = lo; a = cdf[s]; b = cdf[s + 1u];
    d->x = (b - a) * (d->x >> 14) + slot - a;
    while (d->x < RANS_L) { if (d->p >= d->end) return -1; d->x = d->x << 8 | *d->p++; }
    return (int)s;
}

/* ---- record ---- */
typedef struct { size_t used, total, sp, wp, lens, streams; uint32_t mask; } Info;

static size_t encode(const Model *M, const Slab *s, const unsigned char *coded, uint32_t mask, const unsigned char a[6],
                     unsigned char *out, size_t cap, Info *info) {
    uint32_t m, nc = nclass() * 81u, ns = nsctx(), g = 0;
    size_t nstreams = 0, at, i, first;
    unsigned char *tmp = (unsigned char *)malloc(1u << 20);
    uint32_t *ctx = (uint32_t *)malloc(sizeof(uint32_t) * (CW > CS ? CW : CS));
    for (m = 0; m < 3; m++) nstreams += (s->scount[m] + CS - 1) / CS + (s->wcount[m] + CW - 1) / CW;
    memset(out, 0, 32);
    memcpy(out, "NVR0", 4); out[4] = (unsigned char)s->B; memcpy(out + 8, a, 6); out[14] = (unsigned char)mask;
    out[15] = (unsigned char)(CZ | WMODE << 1 | SMODE << 3); out[16] = (unsigned char)WK;
    memcpy(out + 20, &CW, 4); memcpy(out + 24, &CS, 4);
    at = 32 + 2 * nstreams;
    info->sp = info->wp = 0;
    for (int pass = 0; pass < 2; pass++) for (m = 0; m < 3; m++) {
        const unsigned char *sc = coded + s->soff[m], *w = coded + s->woff[m];
        size_t count = pass ? s->wcount[m] : s->scount[m], chunk = pass ? CW : CS, per = s->scount[m] / s->rows[m];
        for (first = 0; first < count; first += chunk) {
            size_t n = count - first < chunk ? count - first : chunk;
            Rev o = {tmp, 1u << 20, 1u << 20};
            uint32_t x = RANS_L;
            if (!pass) {
                for (i = 0; i < n; i++) { size_t k = first + i; int prev = k % per ? sc[k - 1] : -1; ctx[i] = sctx(prev, a[3 + m], k % per); }
                for (i = n; i-- > 0;) {
                    const uint16_t *cdf = M->scdf + ((size_t)m * ns + ctx[i]) * 257u;
                    unsigned v = sc[first + i];
                    enc_put(&x, &o, cdf[v], cdf_at(cdf, v + 1u, 256u) - cdf[v]);
                }
            } else {
                unsigned pm = 8, mx = 8, cls = 0;
                for (i = 0; i < n; i++) {
                    size_t k = first + i;
                    unsigned q = nib(w, k);
                    if (k % s->B == 0) { pm = mx = 8; cls = wclass(sc[k / s->B], a[m]); }
                    ctx[i] = (cls * 9u + pm) * 9u + mx;
                    pm = q & 7u; if (mx == 8u || pm > mx) mx = pm;
                }
                for (i = n; i-- > 0;) {
                    const uint16_t *cdf = M->wcdf + ((size_t)m * nc + ctx[i]) * 17u;
                    unsigned v = nib(w, first + i);
                    enc_put(&x, &o, cdf[v], cdf_at(cdf, v + 1u, 16u) - cdf[v]);
                }
            }
            enc_flush(x, &o);
            size_t len = o.cap - o.pos;
            if (len > 65535u || at + len > cap) die("stream too long or record too large");
            memcpy(out + at, o.buf + o.pos, len);
            out[32 + 2 * g] = (unsigned char)len; out[33 + 2 * g] = (unsigned char)(len >> 8);
            at += len; g++;
            if (pass) info->wp += len; else info->sp += len;
        }
    }
    memcpy(out + at, s->data + s->body - s->extra, s->extra);   /* trailing scalars, raw */
    at += s->extra;
    info->used = at; info->total = (at + 4095u) & ~(size_t)4095u; info->lens = 2 * nstreams; info->streams = nstreams;
    info->mask = mask;
    memset(out + at, 0, info->total - at);
    free(tmp); free(ctx);
    return at;
}

static int decode(const Model *M, const Slab *s, const unsigned char *rec, unsigned char *out) {
    uint32_t m, nc = nclass() * 81u, ns = nsctx(), g = 0, mask = rec[14];
    size_t nstreams = 0, at, i, first;
    const unsigned char *a = rec + 8;
    for (m = 0; m < 3; m++) nstreams += (s->scount[m] + CS - 1) / CS + (s->wcount[m] + CW - 1) / CW;
    at = 32 + 2 * nstreams;
    for (int pass = 0; pass < 2; pass++) for (m = 0; m < 3; m++) {
        unsigned char *sc = out + s->soff[m], *w = out + s->woff[m];
        size_t count = pass ? s->wcount[m] : s->scount[m], chunk = pass ? CW : CS, per = s->scount[m] / s->rows[m];
        for (first = 0; first < count; first += chunk) {
            size_t n = count - first < chunk ? count - first : chunk, len = rec[32 + 2 * g] | (size_t)rec[33 + 2 * g] << 8;
            Dec d;
            if (dec_open(&d, rec + at, len)) return -1;
            if (!pass) {
                for (i = 0; i < n; i++) {
                    size_t k = first + i;
                    int prev = k % per ? sc[k - 1] : -1, v = dec_sym(&d, M->scdf + ((size_t)m * ns + sctx(prev, a[3 + m], k % per)) * 257u, 256u);
                    if (v < 0) return -1;
                    sc[k] = (unsigned char)v;
                }
            } else {
                unsigned pm = 8, mx = 8, cls = 0;
                const uint16_t *base = M->wcdf + (size_t)m * nc * 17u;
                for (i = 0; i < n; i += 2) {
                    size_t k = first + i;
                    int lo, hi;
                    if (k % s->B == 0) { pm = mx = 8; cls = wclass(sc[k / s->B], a[m]); }
                    lo = dec_sym(&d, base + ((cls * 9u + pm) * 9u + mx) * 17u, 16u);
                    if (lo < 0) return -1;
                    pm = (unsigned)lo & 7u; if (mx == 8u || pm > mx) mx = pm;
                    hi = dec_sym(&d, base + ((cls * 9u + pm) * 9u + mx) * 17u, 16u);
                    if (hi < 0) return -1;
                    pm = (unsigned)hi & 7u; if (mx == 8u || pm > mx) mx = pm;
                    if (CZ && (lo == 8 || hi == 8)) return -1;
                    w[k >> 1] = (unsigned char)(lo | hi << 4);
                }
            }
            if (d.p != d.end || d.x != RANS_L) return -1;
            at += len; g++;
        }
    }
    memcpy(out + s->body - s->extra, rec + at, s->extra);
    return ceil_inverse(s, out, mask);
}

static int cmp_double(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    int runs, i, ntrain = 0, ntest = 0, mode = 0;
    static Slab train[MAXF], test[MAXF];
    Model M;
    uint64_t sum_raw = 0, sum_total = 0, sum_used = 0, sum_sp = 0, sum_wp = 0, sum_extra = 0, sum_lens = 0;
    double sum_ms = 0;
    int bad = 0;
    if (argc < 4) die("usage: nvwnr RUNS [options] --train FILES --test FILES");
    runs = atoi(argv[1]);
    for (i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--wctx") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "srel7")) { WMODE = 0; WK = 3; } else if (!strcmp(v, "exp7")) { WMODE = 1; WK = 3; }
            else if (!strcmp(v, "none")) { WMODE = 2; WK = 0; } else if (!strncmp(v, "code", 4)) { WMODE = 0; WK = atoi(v + 4); }
            else die("--wctx");
        } else if (!strcmp(argv[i], "--sctx") && i + 1 < argc) {
            const char *v = argv[++i];
            SMODE = !strcmp(v, "o0") ? 0 : !strcmp(v, "o1b") ? 1 : !strcmp(v, "o1") ? 2 : !strcmp(v, "o1p") ? 3 : -1;
            if (SMODE < 0) die("--sctx");
        } else if (!strcmp(argv[i], "--cz")) CZ = 1;
        else if (!strcmp(argv[i], "--ceil")) CEIL = 1;
        else if (!strcmp(argv[i], "--cw") && i + 1 < argc) CW = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cs") && i + 1 < argc) CS = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--train")) mode = 1;
        else if (!strcmp(argv[i], "--test")) mode = 2;
        else if (mode == 1 && ntrain < MAXF) load(argv[i], &train[ntrain++]);
        else if (mode == 2 && ntest < MAXF) load(argv[i], &test[ntest++]);
        else die("argument");
    }
    if (!ntrain || !ntest) die("no files");
    {
        size_t nc = nclass() * 81u, ns = nsctx();
        M.wcnt = (uint64_t *)calloc(3 * nc * 16, 8); M.scnt = (uint64_t *)calloc(3 * ns * 256, 8);
        M.wcdf = (uint16_t *)calloc(3 * nc * 17, 2); M.scdf = (uint16_t *)calloc(3 * ns * 257, 2);
    }
    for (i = 0; i < ntrain; i++) {
        unsigned char *coded = (unsigned char *)malloc(train[i].body), a[6];
        uint64_t mapped;
        if (train[i].B % 2 || CW % train[i].B) die("chunk must hold whole blocks");
        prepare(&train[i], coded, &mapped);
        anchors(&train[i], coded, a);
        count_slab(&M, &train[i], coded, a);
        free(coded); free(train[i].data);
    }
    build(&M);
    for (i = 0; i < ntest; i++) {
        Slab *s = &test[i];
        unsigned char *coded = (unsigned char *)malloc(s->body), *expect = (unsigned char *)malloc(s->body), a[6];
        size_t cap = s->body * 2 + (1u << 20);
        unsigned char *rec = (unsigned char *)malloc(cap), *out = (unsigned char *)malloc(s->body);
        uint64_t mapped;
        uint32_t mask = prepare(s, coded, &mapped);
        double *t = (double *)malloc(sizeof(double) * (size_t)(runs + 1)), med;
        Info info;
        int r, ok = 1;
        anchors(s, coded, a);
        encode(&M, s, coded, mask, a, rec, cap, &info);
        memcpy(expect, s->data, s->body);
        if (CZ) {   /* Z(source) */
            size_t k;
            for (k = s->woff[0]; k < s->body - s->extra; k++) {
                if ((expect[k] & 15u) == 8u) expect[k] &= 0xF0u;
                if ((expect[k] >> 4) == 8u) expect[k] &= 0x0Fu;
            }
        }
        for (r = 0; r <= runs; r++) {
            double t0;
            memset(out, 0xA5, s->body);
            t0 = now_ms();
            if (decode(&M, s, rec, out)) ok = 0;
            t[r] = now_ms() - t0;
            if (memcmp(out, expect, s->body)) ok = 0;
        }
        qsort(t + 1, (size_t)runs, sizeof(double), cmp_double);
        med = runs ? t[1 + runs / 2] : t[0];
        if (!ok) bad++;
        printf("json {\"file\":\"%s\",\"raw\":%zu,\"scale_payload\":%zu,\"weight_payload\":%zu,\"lengths\":%zu,\"extra\":%u,"
               "\"used\":%zu,\"total\":%zu,\"ratio_used\":%.6f,\"ratio_total\":%.6f,\"ceil_mask\":%u,\"cz_mapped\":%llu,"
               "\"anchors\":[%u,%u,%u,%u,%u,%u],\"round_trip\":%s,\"decode_ms_median\":%.3f,\"decode_mb_s\":%.1f}\n",
               s->name, s->body, info.sp, info.wp, info.lens, s->extra, info.used, info.total,
               (double)info.used / (double)s->body, (double)info.total / (double)s->body, info.mask,
               (unsigned long long)mapped, a[0], a[1], a[2], a[3], a[4], a[5], ok ? "true" : "false", med,
               (double)s->body / med / 1000.0);
        fflush(stdout);
        sum_raw += s->body; sum_total += info.total; sum_used += info.used; sum_sp += info.sp; sum_wp += info.wp;
        sum_extra += s->extra; sum_lens += info.lens + 32; sum_ms += med;
        free(coded); free(expect); free(rec); free(out); free(t); free(s->data);
    }
    printf("summary {\"train\":%d,\"test\":%d,\"runs\":%d,\"raw\":%llu,\"used\":%llu,\"total\":%llu,\"scale_payload\":%llu,"
           "\"weight_payload\":%llu,\"headers_and_lengths\":%llu,\"extra\":%llu,\"ratio_used\":%.6f,\"ratio_total\":%.6f,"
           "\"round_trip_failures\":%d,\"decode_mb_s\":%.1f,\"model_tables\":%zu}\n",
           ntrain, ntest, runs, (unsigned long long)sum_raw, (unsigned long long)sum_used, (unsigned long long)sum_total,
           (unsigned long long)sum_sp, (unsigned long long)sum_wp, (unsigned long long)sum_lens, (unsigned long long)sum_extra,
           (double)sum_used / (double)sum_raw, (double)sum_total / (double)sum_raw, bad, (double)sum_raw / sum_ms / 1000.0,
           (size_t)(3u * nclass() * 81u + 3u * nsctx()));
    return bad ? 1 : 0;
}
