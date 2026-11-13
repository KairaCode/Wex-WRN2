/* team6 / cpu-simd: fast CPU decoder for WNR2 records and for the ceil-ref lineage's f+g0+e records
   (= WNR3 v1 profile CEIL|HASH without LENC; ceil-ref 48-B header, record-level F flag, FNV-1a/lowbias32 h23 in x0).
   Format and context rule follow team 4 ceil-ref wnr_common.py (kit copy: src/wnr_common.py) (decode_record, :812-934).
   Build: cl /O2 /arch:AVX2 /openmp /W3 wnrdec.c
   Usage: wnrdec LIST MODE [args]   (see main) */
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <immintrin.h>
#include <intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#if defined(__GNUC__)
#define FINL inline __attribute__((always_inline))
#define ALIGNED(n) __attribute__((aligned(n)))
#define POPCNT(x) ((uint32_t)__builtin_popcount(x))
#define BSWAP32(x) __builtin_bswap32(x)
#else
#define FINL __forceinline
#define ALIGNED(n) __declspec(align(n))
#define POPCNT(x) __popcnt(x)
#define BSWAP32(x) _byteswap_ulong(x)
#endif

#define PB 14
#define PP (1u << PB)
#define PMASK (PP - 1)
#define RL (1u << 23)
#define NCTX 567
#define WCH 2048
#define SCH 512
#define BPS 64
#define FNV_OFF 2166136261u
#define FNV_PRIME 16777619u
#ifndef NBSH
#define NBSH 7                 /* LUT variant: buckets of 2^NBSH slots */
#endif
#define NBK (PP >> NBSH)
#define FLAG_E 8
#define FLAG_F 16
#define WNR2_MAGIC 0x32524E57u
#define WNR3_MAGIC 0x33524E57u
#define PAD 16384

typedef struct {
    int R;
    uint16_t *cx16;   /* [R*16] cum_excl, 32-B aligned rows */
    uint32_t *cx32;   /* [R*16] cum_excl */
    uint32_t *fc32;   /* [R*16] f | c << 16 */
    uint8_t *lut;     /* [R*NBK + 4] symbol at bucket start */
    uint16_t *wcdf;   /* [R*17] inclusive cdf (naive) */
    uint32_t sfc[3 * 256];
    uint32_t scdf[3 * 257];
    uint8_t *slut;    /* [3*16384 + 4] */
} Model;

typedef struct {
    const Model *M;
    uint8_t *rec;      /* record bytes (+PAD zero bytes) */
    size_t rec_bytes;
    uint32_t magic, model_id, flags;
    int hash_in_rec, ceil_in_rec;
    uint32_t layer, expert;
    int anchor[3];
    int R[3], C[3], nss[3], nws[3], SS, WS;
    uint32_t *soff, *slen, *woff, *wlen;
    int *smat, *sidx, *wmat, *widx;
    uint8_t *slab;
    size_t raw_bytes, sc_base[3], w_base[3];
} Dec;

static void *amalloc(size_t n) { void *p = _aligned_malloc(n, 64); if (!p) { fprintf(stderr, "oom\n"); exit(2); } memset(p, 0, n); return p; }
static inline uint32_t ld_be32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return BSWAP32(v); }
static inline uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint32_t lowbias32(uint32_t h) { h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16; return h; }

static inline uint32_t seed_of(const Dec *d, uint32_t kind, uint32_t m, uint32_t j)
{
    uint32_t v[6] = {d->layer, d->expert, d->model_id, kind, m, j};
    uint32_t h = FNV_OFF;
    for (int i = 0; i < 6; i++)
        for (int b = 0; b < 4; b++) { h ^= (v[i] >> (8 * b)) & 255u; h *= FNV_PRIME; }
    return h;
}

static uint8_t *read_file(const char *path, size_t *n, size_t pad)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    _fseeki64(f, 0, SEEK_END);
    size_t sz = (size_t)_ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)amalloc(sz + pad);
    if (fread(b, 1, sz, f) != sz) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    *n = sz;
    return b;
}

static void load_model(Model *M, const char *prefix)
{
    char p[1024];
    size_t n;
    sprintf(p, "%s.wf.u16", prefix);
    uint16_t *wf = (uint16_t *)read_file(p, &n, 0);
    int R = (int)(n / 32);
    sprintf(p, "%s.scf.u16", prefix);
    uint16_t *scf = (uint16_t *)read_file(p, &n, 0);
    if (n != 3 * 256 * 2) { fprintf(stderr, "bad scf\n"); exit(2); }
    M->R = R;
    M->cx16 = (uint16_t *)amalloc((size_t)R * 32);
    M->cx32 = (uint32_t *)amalloc((size_t)R * 64);
    M->fc32 = (uint32_t *)amalloc((size_t)R * 64);
    M->lut = (uint8_t *)amalloc((size_t)R * NBK + 64);
    M->wcdf = (uint16_t *)amalloc((size_t)R * 34);
    for (int r = 0; r < R; r++) {
        uint32_t c = 0;
        for (int s = 0; s < 16; s++) {
            uint32_t f = wf[r * 16 + s];
            M->cx16[r * 16 + s] = (uint16_t)c;
            M->cx32[r * 16 + s] = c;
            M->fc32[r * 16 + s] = f | (c << 16);
            M->wcdf[r * 17 + s] = (uint16_t)c;
            c += f;
        }
        if (c != PP) { fprintf(stderr, "row %d sums %u\n", r, c); exit(2); }
        M->wcdf[r * 17 + 16] = (uint16_t)PP;
        int s = 0;
        for (int i = 0; i < NBK; i++) {
            uint32_t slot = (uint32_t)i << NBSH;
            while (s < 15 && M->cx32[r * 16 + s + 1] <= slot) s++;
            M->lut[(size_t)r * NBK + i] = (uint8_t)s;
        }
    }
    M->slut = (uint8_t *)amalloc(3 * PP + 64);
    for (int m = 0; m < 3; m++) {
        uint32_t c = 0;
        for (int s = 0; s < 256; s++) {
            uint32_t f = scf[m * 256 + s];
            M->sfc[m * 256 + s] = f | (c << 16);
            M->scdf[m * 257 + s] = c;
            for (uint32_t k = 0; k < f; k++) M->slut[m * PP + c + k] = (uint8_t)s;
            c += f;
        }
        if (c != PP) { fprintf(stderr, "scale row %d sums %u\n", m, c); exit(2); }
        M->scdf[m * 257 + 256] = PP;
    }
    _aligned_free(wf); _aligned_free(scf);
}

/* Header parse + length tables + prefix sums. Returns 0 or an error code. Part of the timed decode. */
static int parse(Dec *d)
{
    const uint8_t *r = d->rec;
    if (d->rec_bytes % 4096) return 1;
    d->magic = rd32(r);
    int hlen;
    uint32_t spb = rd32(r + 8), wpb = rd32(r + 12), wch = rd32(r + 16), sch = rd32(r + 20);
    if (d->magic == WNR2_MAGIC) { hlen = 32; d->flags = 0; if (r[27] || rd32(r + 28)) return 2; }
    else if (d->magic == WNR3_MAGIC) { hlen = 48; d->flags = r[27]; if (d->flags & ~(1 | 2 | 4 | 8 | 16 | 32)) return 3; if (d->flags & (1 | 2 | 4 | 32)) return 4; /* A, C, D, T not implemented */ }
    else return 5;
    if (rd32(r + 4) != d->model_id) return 6;
    if (wch != WCH || sch != SCH) return 7;
    d->hash_in_rec = (d->flags & FLAG_E) != 0;
    d->ceil_in_rec = (d->flags & FLAG_F) != 0;
    for (int m = 0; m < 3; m++) d->anchor[m] = r[24 + m];
    const uint8_t *lt = r + hlen;
    int SS = d->SS, WS = d->WS;
    uint32_t p = (uint32_t)(hlen + 2 * (SS + WS));
    uint32_t sbeg = p, tot = 0;
    for (int g = 0; g < SS; g++) {
        uint32_t l = (uint32_t)lt[2 * g] | ((uint32_t)lt[2 * g + 1] << 8);
        if (l < 4) return 8;
        d->slen[g] = l; d->soff[g] = p; p += l; tot += l;
    }
    if (tot != spb) return 9;
    tot = 0;
    for (int g = 0; g < WS; g++) {
        uint32_t l = (uint32_t)lt[2 * (SS + g)] | ((uint32_t)lt[2 * (SS + g) + 1] << 8);
        if (l < 4) return 10;
        d->wlen[g] = l; d->woff[g] = p; p += l; tot += l;
    }
    if (tot != wpb || sbeg + spb + wpb != p) return 11;
    if (((size_t)p + 4095) / 4096 * 4096 != d->rec_bytes) return 12;
    for (size_t i = p; i < d->rec_bytes; i++) if (r[i]) return 13;
    return 0;
}

static void setup_dec(Dec *d, const Model *M, uint8_t *rec, size_t rec_bytes, uint32_t layer, uint32_t expert,
                      uint32_t model_id, const int *R, const int *C)
{
    memset(d, 0, sizeof(*d));
    d->M = M; d->rec = rec; d->rec_bytes = rec_bytes; d->layer = layer; d->expert = expert; d->model_id = model_id;
    size_t sb = 0, wb = 0;
    for (int m = 0; m < 3; m++) {
        d->R[m] = R[m]; d->C[m] = C[m];
        size_t nsym = (size_t)R[m] * C[m];
        d->nws[m] = (int)(nsym / WCH);
        d->nss[m] = (int)(nsym / 32 / SCH);
        d->SS += d->nss[m]; d->WS += d->nws[m];
        sb += nsym / 32; wb += nsym / 2;
    }
    d->raw_bytes = sb + wb;
    size_t a = 0, w = sb;
    for (int m = 0; m < 3; m++) {
        size_t nsym = (size_t)R[m] * C[m];
        d->sc_base[m] = a; a += nsym / 32;
        d->w_base[m] = w; w += nsym / 2;
    }
    d->soff = (uint32_t *)amalloc(4 * (size_t)d->SS); d->slen = (uint32_t *)amalloc(4 * (size_t)d->SS);
    d->woff = (uint32_t *)amalloc(4 * (size_t)d->WS); d->wlen = (uint32_t *)amalloc(4 * (size_t)d->WS);
    d->smat = (int *)amalloc(4 * (size_t)d->SS); d->sidx = (int *)amalloc(4 * (size_t)d->SS);
    d->wmat = (int *)amalloc(4 * (size_t)d->WS); d->widx = (int *)amalloc(4 * (size_t)d->WS);
    int gs = 0, gw = 0;
    for (int m = 0; m < 3; m++) {
        for (int j = 0; j < d->nss[m]; j++) { d->smat[gs] = m; d->sidx[gs] = j; gs++; }
        for (int j = 0; j < d->nws[m]; j++) { d->wmat[gw] = m; d->widx[gw] = j; gw++; }
    }
    d->slab = (uint8_t *)amalloc(d->raw_bytes + 64);
}

/* ---------------------------------------------------------------- ceil inverse (per weight stream, 64 blocks) */
static __m128i TDBL;
static void init_tdbl(void)
{
    static const uint8_t D[8] = {0, 2, 4, 5, 6, 7, 6, 7};
    ALIGNED(16) uint8_t t[16];
    for (int n = 0; n < 16; n++) t[n] = (uint8_t)((n & 8) | D[n & 7]);
    TDBL = _mm_load_si128((const __m128i *)t);
}

static FINL int ceil_inv(uint8_t *out, uint8_t *sc, const uint8_t *bm)
{
    int bad = 0;
    const __m128i m0f = _mm_set1_epi8(0x0F);
    for (int b = 0; b < BPS; b++) {
        uint32_t Mx = bm[b];
        uint32_t odd = (Mx == 5);
        bad |= (Mx < 5) | (odd & (sc[b] == 0));
        sc[b] = (uint8_t)(sc[b] - odd);
        __m128i v = _mm_loadu_si128((const __m128i *)(out + b * 16));
        __m128i lo = _mm_and_si128(v, m0f), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m0f);
        __m128i dv = _mm_or_si128(_mm_shuffle_epi8(TDBL, lo), _mm_slli_epi16(_mm_shuffle_epi8(TDBL, hi), 4));
        _mm_storeu_si128((__m128i *)(out + b * 16), _mm_blendv_epi8(v, dv, _mm_set1_epi8((char)(0 - odd))));
    }
    return bad;
}

/* ---------------------------------------------------------------- naive scalar baseline */
static FINL int sdec_naive(const Dec *d, int g, const int HASH)
{
    const Model *M = d->M;
    int m = d->smat[g], j = d->sidx[g];
    const uint8_t *p = d->rec + d->soff[g], *pend = p + d->slen[g];
    uint32_t x = ld_be32(p); p += 4;
    int bad = (x < RL) | (x >= 0x80000000u);
    if (bad) x = RL;
    uint32_t h = HASH ? seed_of(d, 0, m, j) : 0;
    uint8_t *o = d->slab + d->sc_base[m] + (size_t)j * SCH;
    const uint32_t *cdf = M->scdf + m * 257;
    for (int t = 0; t < SCH; t++) {
        uint32_t slot = x & PMASK;
        int lo = 0, hi = 256;                       /* binary search: largest s with cdf[s] <= slot */
        while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (cdf[mid] <= slot) lo = mid; else hi = mid; }
        uint32_t f = cdf[lo + 1] - cdf[lo];
        x = f * (x >> PB) + slot - cdf[lo];
        while (x < RL) { x = (x << 8) | *p++; if (p > pend + 2 * SCH) break; }
        o[t] = (uint8_t)lo;
        if (HASH) h = (h ^ (uint32_t)lo) * FNV_PRIME;
    }
    uint32_t want = RL + (HASH ? (lowbias32(h) & 0x7FFFFFu) : 0);
    if (HASH || !d->hash_in_rec) bad += (p != pend) | (x != want);
    else bad += (p != pend) | (x < RL) | (x >= RL + 0x800000u);
    return bad;
}

static FINL int wdec_naive(const Dec *d, int g, const int HASH, const int CEIL)
{
    const Model *M = d->M;
    int m = d->wmat[g], j = d->widx[g];
    const uint8_t *p = d->rec + d->woff[g], *pend = p + d->wlen[g];
    uint32_t x = ld_be32(p); p += 4;
    int bad = (x < RL) | (x >= 0x80000000u);
    if (bad) x = RL;
    uint32_t h = HASH ? seed_of(d, 1, m, j) : 0;
    uint8_t *o = d->slab + d->w_base[m] + (size_t)j * (WCH / 2);
    uint8_t *sc = d->slab + d->sc_base[m] + (size_t)j * BPS;
    uint8_t bm[BPS];
    for (int b = 0; b < BPS; b++) {
        int dd = (int)sc[b] - d->anchor[m];
        dd = dd < -3 ? -3 : (dd > 3 ? 3 : dd);
        uint32_t rb = (uint32_t)(m * NCTX + (dd + 3) * 81);
        uint32_t pmx = 80, mx = 0;
        for (int t = 0; t < 32; t++) {
            const uint16_t *cdf = M->wcdf + (size_t)(rb + pmx) * 17;
            uint32_t slot = x & PMASK;
            uint32_t s = 0;
            while (cdf[s + 1] <= slot) s++;
            x = (uint32_t)(cdf[s + 1] - cdf[s]) * (x >> PB) + slot - cdf[s];
            while (x < RL) { x = (x << 8) | *p++; if (p > pend + 2 * WCH) break; }
            if (t & 1) o[b * 16 + (t >> 1)] |= (uint8_t)(s << 4); else o[b * 16 + (t >> 1)] = (uint8_t)s;
            if (HASH) h = (h ^ s) * FNV_PRIME;
            uint32_t mag = s & 7;
            if (mag > mx) mx = mag;
            pmx = mag * 9 + mx;
        }
        bm[b] = (uint8_t)mx;
    }
    uint32_t want = RL + (HASH ? (lowbias32(h) & 0x7FFFFFu) : 0);
    if (HASH || !d->hash_in_rec) bad += (p != pend) | (x != want);
    else bad += (p != pend) | (x < RL) | (x >= RL + 0x800000u);
    if (CEIL) bad += ceil_inv(o, sc, bm);
    return bad;
}

/* ---------------------------------------------------------------- interleaved scalar scales (8 streams) */
static FINL int sdec_il8(const Dec *d, int g0, const int HASH)
{
    const Model *M = d->M;
    uint32_t x[8], h[8];
    const uint8_t *p[8], *pend[8], *lut[8];
    const uint32_t *sfc[8];
    uint8_t *o[8];
    int bad = 0;
#define S_LANES(Mc) Mc(0) Mc(1) Mc(2) Mc(3) Mc(4) Mc(5) Mc(6) Mc(7)
#define S_INIT(l) { int g = g0 + l, m = d->smat[g], j = d->sidx[g]; p[l] = d->rec + d->soff[g]; pend[l] = p[l] + d->slen[g]; \
        x[l] = ld_be32(p[l]); p[l] += 4; bad += (x[l] < RL) | (x[l] >= 0x80000000u); if (x[l] < RL || x[l] >= 0x80000000u) x[l] = RL; \
        h[l] = HASH ? seed_of(d, 0, m, j) : 0; o[l] = d->slab + d->sc_base[m] + (size_t)j * SCH; \
        lut[l] = M->slut + m * PP; sfc[l] = M->sfc + m * 256; }
    S_LANES(S_INIT)
    for (int t = 0; t < SCH; t++) {
#define S_STEP(l) { uint32_t slot = x[l] & PMASK; uint32_t s = lut[l][slot]; uint32_t fc = sfc[l][s]; \
        uint32_t xx = (fc & 0xFFFF) * (x[l] >> PB) + slot - (fc >> 16); \
        uint32_t v = ld_be32(p[l]); uint32_t k = (xx < RL) + (xx < (RL >> 8)); \
        x[l] = (uint32_t)(((uint64_t)xx << (k * 8)) | ((uint64_t)v >> (32 - k * 8))); p[l] += k; \
        if (HASH) h[l] = (h[l] ^ s) * FNV_PRIME; o[l][t] = (uint8_t)s; }
        S_LANES(S_STEP)
    }
#define S_FIN(l) { uint32_t want = RL + (HASH ? (lowbias32(h[l]) & 0x7FFFFFu) : 0); \
        if (HASH || !d->hash_in_rec) bad += (p[l] != pend[l]) | (x[l] != want); \
        else bad += (p[l] != pend[l]) | (x[l] < RL) | (x[l] >= RL + 0x800000u); }
    S_LANES(S_FIN)
    return bad;
}

/* ---------------------------------------------------------------- interleaved scalar weights */
#define ILK 4
#define ILNAME wdec_il4
#define IL_LANES(Mc) Mc(0) Mc(1) Mc(2) Mc(3)
#include "wdec_il.h"
#undef ILK
#undef ILNAME
#undef IL_LANES
#define ILK 8
#define ILNAME wdec_il8
#define IL_LANES(Mc) Mc(0) Mc(1) Mc(2) Mc(3) Mc(4) Mc(5) Mc(6) Mc(7)
#include "wdec_il.h"
#undef ILK
#undef ILNAME
#undef IL_LANES

/* ---------------------------------------------------------------- AVX2 8-lane gather weights */
#define VG 1
#define VLUT 0
#define VNAME wdec_v8bs1
#include "wdec_v8.h"
#undef VG
#undef VLUT
#undef VNAME
#define VG 2
#define VLUT 0
#define VNAME wdec_v8bs2
#include "wdec_v8.h"
#undef VG
#undef VLUT
#undef VNAME
#define VG 1
#define VLUT 1
#define VNAME wdec_v8lut1
#include "wdec_v8.h"
#undef VG
#undef VLUT
#undef VNAME
#define VG 2
#define VLUT 1
#define VNAME wdec_v8lut2
#include "wdec_v8.h"
#undef VG
#undef VLUT
#undef VNAME

/* ---------------------------------------------------------------- record decode */
enum { V_NAIVE, V_IL4, V_IL8, V_BS1, V_BS2, V_LUT1, V_LUT2, V_COUNT };
static const char *VNAMES[V_COUNT] = {"scalar_naive", "il4_scalar_avx2cmp", "il8_scalar_avx2cmp", "avx2x8_gather_bs_g1",
                                      "avx2x8_gather_bs_g2", "avx2x8_gather_lut_g1", "avx2x8_gather_lut_g2"};
static const int VUNIT[V_COUNT] = {1, 4, 8, 8, 16, 8, 16};

#define SPEC(HASH, CEIL) \
    static int wunit_##HASH##CEIL(const Dec *d, int v, int g, uint64_t *nf) { \
        switch (v) { \
        case V_NAIVE: return wdec_naive(d, g, HASH, CEIL); \
        case V_IL4: return wdec_il4(d, g, HASH, CEIL); \
        case V_IL8: return wdec_il8(d, g, HASH, CEIL); \
        case V_BS1: return wdec_v8bs1(d, g, HASH, CEIL, NULL); \
        case V_BS2: return wdec_v8bs2(d, g, HASH, CEIL, NULL); \
        case V_LUT1: return wdec_v8lut1(d, g, HASH, CEIL, nf); \
        default: return wdec_v8lut2(d, g, HASH, CEIL, nf); } }
SPEC(0, 0)
SPEC(0, 1)
SPEC(1, 0)
SPEC(1, 1)
static int sunit_0(const Dec *d, int v, int g) { return v == V_NAIVE ? sdec_naive(d, g, 0) : sdec_il8(d, g, 0); }
static int sunit_1(const Dec *d, int v, int g) { return v == V_NAIVE ? sdec_naive(d, g, 1) : sdec_il8(d, g, 1); }

typedef struct { double t_parse, t_scales, t_weights; uint64_t nfix; } Tm;
static double qpf;
static inline double now(void) { LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart / qpf; }

/* hash/ceil: 1 = do it (only if the record has it), 0 = skip. Returns number of failed streams (or -code). */
static int decode_record(Dec *d, int v, int nth, int hash, int ceil, Tm *tm)
{
    double t0 = now();
    int e = parse(d);
    if (e) return -e;
    hash = hash && d->hash_in_rec;
    ceil = ceil && d->ceil_in_rec;
    double t1 = now();
    int bad = 0;
    int su = (v == V_NAIVE) ? 1 : 8;
    int ns = d->SS / su;
    int (*sfn)(const Dec *, int, int) = hash ? sunit_1 : sunit_0;
#pragma omp parallel for num_threads(nth) schedule(dynamic, 4) reduction(+ : bad)
    for (int u = 0; u < ns; u++) bad += sfn(d, v, u * su);
    double t2 = now();
    int wu = VUNIT[v];
    int nw = d->WS / wu;
    int (*wfn)(const Dec *, int, int, uint64_t *) = hash ? (ceil ? wunit_11 : wunit_10) : (ceil ? wunit_01 : wunit_00);
    uint64_t nfix = 0;
#pragma omp parallel num_threads(nth) reduction(+ : bad, nfix)
    {
        uint64_t nf = 0;
#pragma omp for schedule(dynamic, 2)
        for (int u = 0; u < nw; u++) bad += wfn(d, v, u * wu, &nf);
        nfix += nf;
    }
    double t3 = now();
    if (tm) { tm->t_parse = t1 - t0; tm->t_scales = t2 - t1; tm->t_weights = t3 - t2; tm->nfix = nfix; }
    return bad;
}

/* ---------------------------------------------------------------- MurmurHash3_x86_32 (spec H32) */
static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
static inline uint32_t fmix32(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16; return h; }
static inline uint32_t mm3_block(uint32_t h, uint32_t k)
{
    k *= 0xcc9e2d51; k = rotl32(k, 15); k *= 0x1b873593;
    h ^= k; h = rotl32(h, 13); return h * 5 + 0xe6546b64;
}
static uint32_t murmur3_32(const uint8_t *data, int len, uint32_t seed)
{
    uint32_t h = seed;
    int nb = len / 4;
    for (int i = 0; i < nb; i++) h = mm3_block(h, rd32(data + 4 * i));
    const uint8_t *tail = data + nb * 4;
    uint32_t k = 0;
    switch (len & 3) {
    case 3: k ^= (uint32_t)tail[2] << 16;
    case 2: k ^= (uint32_t)tail[1] << 8;
    case 1: k ^= tail[0]; k *= 0xcc9e2d51; k = rotl32(k, 15); k *= 0x1b873593; h ^= k;
    }
    h ^= (uint32_t)len;
    return fmix32(h);
}
/* H32(ident[4] || bytes) with n a multiple of 4 */
static inline uint32_t h32_stream(const uint32_t *ident, const uint8_t *b, int n, uint32_t seed)
{
    uint32_t h = seed;
    for (int i = 0; i < 4; i++) h = mm3_block(h, ident[i]);
    for (int i = 0; i < n / 4; i++) h = mm3_block(h, rd32(b + 4 * i));
    h ^= (uint32_t)(16 + n);
    return fmix32(h);
}
static uint32_t smhasher_verification(void)
{
    uint8_t key[256], hashes[1024];
    for (int i = 0; i < 256; i++) {
        key[i] = (uint8_t)i;
        uint32_t hv = murmur3_32(key, i, 256 - i);
        memcpy(hashes + 4 * i, &hv, 4);
    }
    return murmur3_32(hashes, 1024, 0);
}

/* ---------------------------------------------------------------- load sampling */
static inline uint64_t ft64(FILETIME f) { return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime; }
typedef struct { uint64_t idle, kern, user, pk, pu; double wall; } Ld;
static void ld_sample(Ld *s)
{
    FILETIME i, k, u, c, e, pk, pu;
    GetSystemTimes(&i, &k, &u);
    GetProcessTimes(GetCurrentProcess(), &c, &e, &pk, &pu);
    s->idle = ft64(i); s->kern = ft64(k); s->user = ft64(u); s->pk = ft64(pk); s->pu = ft64(pu); s->wall = now();
}
/* returns total busy %, own %, other % of all logical CPUs over the interval */
static void ld_diff(const Ld *a, const Ld *b, double *tot, double *own, double *oth)
{
    double all = (double)((b->kern - a->kern) + (b->user - a->user));
    double busy = all - (double)(b->idle - a->idle);
    double mine = (double)((b->pk - a->pk) + (b->pu - a->pu));
    *tot = all > 0 ? 100.0 * busy / all : 0;
    *own = all > 0 ? 100.0 * mine / all : 0;
    *oth = *tot - *own;
}

/* ---------------------------------------------------------------- list file */
typedef struct {
    char rec[512], src[512], cfg[32], model[512];
    uint32_t layer, expert, model_id;
    int R[3], C[3];
} Item;

static int read_list(const char *path, Item *it, int maxn)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open list\n"); exit(2); }
    int n = 0;
    while (n < maxn && fscanf(f, "%511s %511s %u %u %u %31s %d %d %d %d %d %d %511s", it[n].rec, it[n].src, &it[n].layer,
                              &it[n].expert, &it[n].model_id, it[n].cfg, &it[n].R[0], &it[n].C[0], &it[n].R[1], &it[n].C[1],
                              &it[n].R[2], &it[n].C[2], it[n].model) == 13)
        n++;
    fclose(f);
    return n;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* per-stream comparison vs the source slab: returns the number of mismatching streams (scale + weight) */
static int compare_streams(const Dec *d, const uint8_t *src, int *bad_s, int *bad_w, size_t *bytes_diff)
{
    int bs = 0, bw = 0;
    size_t bd = 0;
    for (int g = 0; g < d->SS; g++) {
        size_t o = d->sc_base[d->smat[g]] + (size_t)d->sidx[g] * SCH;
        if (memcmp(d->slab + o, src + o, SCH)) bs++;
    }
    for (int g = 0; g < d->WS; g++) {
        size_t o = d->w_base[d->wmat[g]] + (size_t)d->widx[g] * (WCH / 2);
        if (memcmp(d->slab + o, src + o, WCH / 2)) bw++;
    }
    for (size_t i = 0; i < d->raw_bytes; i++) bd += d->slab[i] != src[i];
    *bad_s = bs; *bad_w = bw; *bytes_diff = bd;
    return bs + bw;
}

int main(int argc, char **argv)
{
    LARGE_INTEGER fq;
    QueryPerformanceFrequency(&fq);
    qpf = (double)fq.QuadPart;
    init_tdbl();
    if (argc < 3) { fprintf(stderr, "usage: wnrdec LIST verify|bench|corrupt|h32|gather [...]\n"); return 1; }
    static Item items[64];
    const char *mode = argv[2];
    if (!strcmp(mode, "h32")) {
        printf("{\"mode\":\"h32\",\"smhasher_verification\":\"0x%08X\",\"expected\":\"0xB0F57EE3\"}\n", smhasher_verification());
    }
    if (!strcmp(mode, "gather")) {
        /* gather throughput / latency microbench on an L1-resident table */
        uint32_t *tab = (uint32_t *)amalloc(4096 * 4);
        for (int i = 0; i < 4096; i++) tab[i] = (uint32_t)((i * 2654435761u) >> 20) & 4095;
        int N = 20000000;
        __m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), b = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        __m256i c4 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23), e4 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
        double t0 = now();
        for (int i = 0; i < N; i++) a = _mm256_i32gather_epi32((const int *)tab, a, 4);  /* dependent chain */
        double t1 = now();
        for (int i = 0; i < N; i++) {                                                       /* 4 independent chains */
            a = _mm256_i32gather_epi32((const int *)tab, a, 4); b = _mm256_i32gather_epi32((const int *)tab, b, 4);
            c4 = _mm256_i32gather_epi32((const int *)tab, c4, 4); e4 = _mm256_i32gather_epi32((const int *)tab, e4, 4);
        }
        double t2 = now();
        uint32_t ia = 5, ib = 6, ic = 7, ie = 8;
        for (int i = 0; i < N; i++) { ia = tab[ia]; ib = tab[ib]; ic = tab[ic]; ie = tab[ie]; }
        double t3 = now();
        __m256i z = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c4, e4));
        printf("{\"mode\":\"gather\",\"ns_per_gather_dependent\":%.3f,\"ns_per_gather_4indep\":%.3f,\"ns_per_scalar_load_4indep\":%.3f,\"sink\":%u}\n",
               (t1 - t0) / N * 1e9, (t2 - t1) / (4.0 * N) * 1e9, (t3 - t2) / (4.0 * N) * 1e9,
               (uint32_t)_mm256_extract_epi32(z, 0) ^ ia ^ ib ^ ic ^ ie);
        return 0;
    }
    int n = read_list(argv[1], items, 64);
    if (!strcmp(mode, "h32")) {
        /* time H32 over every stream's coded bytes of a slab (the bytes are the decoded slab; ident synthetic) */
        int reps = argc > 3 ? atoi(argv[3]) : 5;
        Model M; load_model(&M, items[0].model);
        size_t rb, sb;
        uint8_t *rec = read_file(items[0].rec, &rb, PAD);
        uint8_t *src = read_file(items[0].src, &sb, 64);
        Dec d; setup_dec(&d, &M, rec, rb, items[0].layer, items[0].expert, items[0].model_id, items[0].R, items[0].C);
        int ths[4] = {1, 2, 4, 8};
        for (int ti = 0; ti < 4; ti++) {
            double best = 1e9; uint32_t acc = 0;
            for (int r = 0; r < reps; r++) {
                double t0 = now();
                uint32_t a2 = 0;
#pragma omp parallel for num_threads(ths[ti]) schedule(static) reduction(^ : a2)
                for (int g = 0; g < d.SS + d.WS; g++) {
                    uint32_t ident[4] = {d.model_id, d.layer << 16 | d.expert, (uint32_t)g, 0x1C};
                    if (g < d.SS) a2 ^= h32_stream(ident, src + d.sc_base[d.smat[g]] + (size_t)d.sidx[g] * SCH, SCH, 0x33524E57u);
                    else { int w = g - d.SS; a2 ^= h32_stream(ident, src + d.w_base[d.wmat[w]] + (size_t)d.widx[w] * (WCH / 2), WCH / 2, 0x33524E57u); }
                }
                double t1 = now();
                if (t1 - t0 < best) best = t1 - t0;
                acc ^= a2;
            }
            printf("{\"mode\":\"h32_time\",\"threads\":%d,\"ms_per_record_best\":%.3f,\"bytes_hashed\":%zu,\"xor\":\"0x%08X\"}\n",
                   ths[ti], best * 1e3, d.raw_bytes + (size_t)16 * (d.SS + d.WS), acc);
        }
        return 0;
    }
    if (!strcmp(mode, "verify") || !strcmp(mode, "bench")) {
        /* bench args: variants(csv idx) threads(csv) reps flagsets(csv of hc: 11,10,01,00) */
        int bench = !strcmp(mode, "bench");
        char vlist[256] = "0,1,2,3,4,5,6", tlist[64] = "8", flist[64] = "11,10,01,00";
        int reps = 1;
        if (argc > 3) strncpy(vlist, argv[3], 255);
        if (argc > 4) strncpy(tlist, argv[4], 63);
        if (argc > 5) reps = atoi(argv[5]);
        if (argc > 6) strncpy(flist, argv[6], 63);
        int vs[16], nv = 0, ts[16], nt = 0, fs[8], nf = 0;
        for (char *t = strtok(vlist, ","); t; t = strtok(NULL, ",")) vs[nv++] = atoi(t);
        for (char *t = strtok(tlist, ","); t; t = strtok(NULL, ",")) ts[nt++] = atoi(t);
        for (char *t = strtok(flist, ","); t; t = strtok(NULL, ",")) fs[nf++] = (t[0] - '0') * 2 + (t[1] - '0');
        static Model models[4]; static char mnames[4][512]; int nm = 0;
        for (int i = 0; i < n; i++) {
            int mi = -1;
            for (int k = 0; k < nm; k++) if (!strcmp(mnames[k], items[i].model)) mi = k;
            if (mi < 0) { mi = nm++; strcpy(mnames[mi], items[i].model); load_model(&models[mi], items[i].model); }
            size_t rb, sb;
            uint8_t *rec = read_file(items[i].rec, &rb, PAD);
            uint8_t *src = read_file(items[i].src, &sb, 64);
            Dec d;
            setup_dec(&d, &models[mi], rec, rb, items[i].layer, items[i].expert, items[i].model_id, items[i].R, items[i].C);
            if (sb != d.raw_bytes) { fprintf(stderr, "size mismatch\n"); return 2; }
            int has_e = 0, has_f = 0;
            { int e = parse(&d); if (e) { printf("{\"rec\":\"%s\",\"parse_error\":%d}\n", items[i].rec, e); continue; } has_e = d.hash_in_rec; has_f = d.ceil_in_rec; }
            for (int fi = 0; fi < nf; fi++) {
                int hash = fs[fi] >> 1, ceil = fs[fi] & 1;
                if ((hash && !has_e) || (ceil && !has_f)) continue;
                if (!has_e && !has_f && fs[fi] != 0) continue;
                for (int vi = 0; vi < nv; vi++) {
                    for (int ti = 0; ti < nt; ti++) {
                        int v = vs[vi], th = ts[ti];
                        double tms[64]; Tm tm, tbest = {0}; double best = 1e30; int bad = 0;
                        Ld la, lb;
                        memset(d.slab, 0xA5, d.raw_bytes);
                        ld_sample(&la);
                        int R = bench ? reps : 1;
                        for (int r = 0; r < R && r < 64; r++) {
                            double t0 = now();
                            bad = decode_record(&d, v, th, hash, ceil, &tm);
                            double t1 = now();
                            tms[r] = t1 - t0;
                            if (tms[r] < best) { best = tms[r]; tbest = tm; }
                        }
                        ld_sample(&lb);
                        double ltot, lown, loth;
                        ld_diff(&la, &lb, &ltot, &lown, &loth);
                        qsort(tms, R, sizeof(double), cmpd);
                        int bs = -1, bw = -1; size_t bd = 0;
                        int expect_exact = (has_f == ceil);   /* skipping the ceil inverse leaves coded-domain output */
                        compare_streams(&d, src, &bs, &bw, &bd);
                        printf("{\"rec\":\"%s\",\"cfg\":\"%s\",\"variant\":\"%s\",\"threads\":%d,\"hash\":%d,\"ceil\":%d,\"reps\":%d,"
                               "\"ms_best\":%.3f,\"ms_median\":%.3f,\"GBps_best\":%.4f,\"ms_parse\":%.3f,\"ms_scales\":%.3f,\"ms_weights\":%.3f,"
                               "\"failed_streams\":%d,\"mismatch_scale_streams\":%d,\"mismatch_weight_streams\":%d,\"bytes_differing\":%zu,"
                               "\"expect_byte_exact\":%d,\"byte_exact\":%d,\"streams\":%d,\"lut_fixups\":%llu,"
                               "\"cpu_total_pct\":%.1f,\"cpu_own_pct\":%.1f,\"cpu_other_pct\":%.1f,\"raw_bytes\":%zu,\"rec_bytes\":%zu}\n",
                               items[i].rec, items[i].cfg, VNAMES[v], th, hash, ceil, R, best * 1e3, tms[R / 2] * 1e3,
                               d.raw_bytes / best / 1e9, tbest.t_parse * 1e3, tbest.t_scales * 1e3, tbest.t_weights * 1e3, bad, bs, bw, bd,
                               expect_exact, (int)(bs == 0 && bw == 0 && bd == 0), d.SS + d.WS, (unsigned long long)tbest.nfix,
                               ltot, lown, loth, d.raw_bytes, rb);
                        fflush(stdout);
                    }
                }
            }
            _aligned_free(rec); _aligned_free(src); _aligned_free(d.slab);
        }
        return 0;
    }
    if (!strcmp(mode, "corrupt")) {
        /* flip one byte at several positions of the weight and scale payloads; count rejections (variant il8, 8 threads) */
        int trials = argc > 3 ? atoi(argv[3]) : 20;
        for (int i = 0; i < n; i++) {
            Model M; load_model(&M, items[i].model);
            size_t rb, sb;
            uint8_t *rec = read_file(items[i].rec, &rb, PAD);
            uint8_t *src = read_file(items[i].src, &sb, 64);
            Dec d;
            setup_dec(&d, &M, rec, rb, items[i].layer, items[i].expert, items[i].model_id, items[i].R, items[i].C);
            parse(&d);
            uint32_t sbeg = d.soff[0], wbeg = d.woff[0], wend = d.woff[d.WS - 1] + d.wlen[d.WS - 1];
            int rej = 0, acc_exact = 0, acc_wrong = 0;
            uint32_t rs = 12345u + i;
            for (int t = 0; t < trials; t++) {
                rs = rs * 1103515245u + 12345u;
                uint32_t r2 = rs >> 1;
                uint32_t pos = (t % 4 == 0) ? sbeg + r2 % (wbeg - sbeg) : wbeg + r2 % (wend - wbeg);
                uint8_t old = rec[pos];
                rec[pos] ^= (uint8_t)(1u << (t % 8));
                int bad = decode_record(&d, V_IL8, 8, 1, 1, NULL);
                int bs, bw; size_t bd;
                compare_streams(&d, src, &bs, &bw, &bd);
                if (bad != 0) rej++; else if (bd == 0) acc_exact++; else acc_wrong++;
                rec[pos] = old;
            }
            printf("{\"mode\":\"corrupt\",\"rec\":\"%s\",\"cfg\":\"%s\",\"trials\":%d,\"rejected\":%d,\"accepted_exact\":%d,\"accepted_wrong\":%d}\n",
                   items[i].rec, items[i].cfg, trials, rej, acc_exact, acc_wrong);
            fflush(stdout);
            _aligned_free(rec); _aligned_free(src);
        }
        return 0;
    }
    return 0;
}
