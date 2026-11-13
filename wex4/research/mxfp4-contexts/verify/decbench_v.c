/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* CPU decode-cost classes of the context choices, on one real coded slab (single thread, no GPU).
 * Build: gcc -O2 -mavx2 decbench.c -o out/decbench.exe
 * Run:   decbench SLAB ENTRY_HEX ANCHOR0 ANCHOR1 ANCHOR2      (values from out/manifest.json)
 *
 * Model A = today's WNR3 weight model: 16 symbols, context (srel*9 + prev)*9 + running max, 14-bit CDFs,
 *           32-bit rANS, L = 2^23, byte renormalisation. Decoders: binary search (the reference algorithm,
 *           without the stream hash), AVX2 compare search, and the same with 4 streams interleaved.
 * Model F = the reduced model this study found to cost about +0.02% of raw: 8 magnitudes, context = srel of
 *           the block only (constant for 32 symbols, known before the block), 12-bit CDFs with a slot table,
 *           the sign of every non-zero weight as a raw bit. Byte renormalisation (L = 2^23) or 16-bit
 *           renormalisation (L = 2^16, at most one step per symbol).
 * Models are trained on the slab itself (speed test; the held-out sizes come from ctxeval.py). Every decoder
 * output is compared with the slab. Streams are 2048 nibbles, as in production. */
#include <immintrin.h>
#include "out/stage/wex_neural_rans.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STREAM 2048u
#define L23 (1u << 23)
#define PA 14u
#define PF 12u
#define MAXS 6000u   /* streams per matrix (V4.1: 5760) */

typedef struct { const uint8_t *S, *W; uint32_t R, B, n, streams; int anchor; } Mat;
typedef struct { uint8_t *bytes; uint32_t off[MAXS + 1]; uint8_t *sign; uint32_t soff[MAXS + 1]; } Coded;

static inline uint32_t nib(const uint8_t *W, size_t i) { return (W[i >> 1] >> ((i & 1u) * 4u)) & 15u; }
static inline uint32_t srel_of(int s, int a) { int d = s - a; return (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3); }
static int HASH = 0, FAIL = 0;   /* 0 none, 1 = 23-bit hash in every x0, 2 = as 1 but 16-bit hash for the wide form */
static uint32_t h23(const uint8_t *data, uint32_t n) { return wex_nr3_h32_bytes(0x33524e57u, data, n, 16u) & 0x7fffffu; }
static uint32_t x0_of(const uint8_t *coded_bytes, int wide) {
    uint32_t h = HASH ? h23(coded_bytes, STREAM / 2u) : 0u;
    return wide ? (1u << 16) + (HASH == 2 ? (h & 0xffffu) : h) : L23 + h;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* the wex_nr2_quantize rule for any total: every symbol >= 1, remainders largest first, ties lower symbol */
static void quantize(const uint64_t *counts, uint32_t symbols, uint32_t total_f, uint16_t *cdf) {
    uint64_t total = 0, rem[16];
    uint32_t f[16], spare = total_f - symbols, used = 0, s, extra;
    uint8_t taken[16] = {0};
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

/* rANS encoder of one stream: symbol j uses cdf[j]. wide = 16-bit renormalisation with L = 2^16.
 * Returns the stream length; the decoder reads it forwards. */
static uint32_t encode(const uint16_t *const *cdf, const uint8_t *sym, uint32_t n, uint32_t P, int wide, uint8_t *out, uint32_t x0) {
    static uint8_t tmp[4u * STREAM + 16u];
    uint8_t *end = tmp + sizeof(tmp), *p = end;
    uint32_t x = x0, j;
    for (j = n; j-- > 0;) {
        uint32_t start = cdf[j][sym[j]], f = cdf[j][sym[j] + 1u] - start;
        uint64_t max = wide ? ((uint64_t)((1u << 16) >> P) << 16) * f : (uint64_t)((L23 >> P) << 8) * f;
        while (x >= max) {
            if (wide) { p -= 2; p[0] = (uint8_t)x; p[1] = (uint8_t)(x >> 8); x >>= 16; }
            else { *--p = (uint8_t)x; x >>= 8; }
        }
        x = ((x / f) << P) + x % f + start;
    }
    p -= 4;
    if (wide) memcpy(p, &x, 4);
    else { p[0] = (uint8_t)(x >> 24); p[1] = (uint8_t)(x >> 16); p[2] = (uint8_t)(x >> 8); p[3] = (uint8_t)x; }
    memcpy(out, p, (size_t)(end - p));
    return (uint32_t)(end - p);
}

static void encode_A(const Mat *M, const uint16_t *cdf, Coded *c) {
    static const uint16_t *cp[STREAM];
    static uint8_t sym[STREAM];
    uint32_t s, j, at = 0;
    for (s = 0; s < M->streams; s++) {
        uint32_t sr = 0, prev = 8, mx = 8;
        for (j = 0; j < STREAM; j++) {
            size_t i = (size_t)s * STREAM + j;
            uint32_t x = nib(M->W, i);
            if (!(j & 31u)) { sr = srel_of(M->S[i / 32u], M->anchor); prev = 8; mx = 8; }
            cp[j] = cdf + ((sr * 9u + prev) * 9u + mx) * 17u; sym[j] = (uint8_t)x;
            prev = x & 7u; if (mx == 8u || prev > mx) mx = prev;
        }
        c->off[s] = at; at += encode(cp, sym, STREAM, PA, 0, c->bytes + at, x0_of(M->W + (size_t)s * (STREAM / 2u), 0));
    }
    c->off[s] = at;
}

static void encode_F(const Mat *M, const uint16_t *cdf, int wide, Coded *c) {
    static const uint16_t *cp[STREAM];
    static uint8_t sym[STREAM];
    uint32_t s, j, at = 0, sat = 0;
    for (s = 0; s < M->streams; s++) {
        uint32_t sr = 0, bit = 0;
        memset(c->sign + sat, 0, STREAM / 8u + 8u);
        for (j = 0; j < STREAM; j++) {
            size_t i = (size_t)s * STREAM + j;
            uint32_t x = nib(M->W, i);
            if (!(j & 31u)) sr = srel_of(M->S[i / 32u], M->anchor);
            cp[j] = cdf + sr * 9u; sym[j] = (uint8_t)(x & 7u);
            if (x & 7u) { c->sign[sat + bit / 8u] |= (uint8_t)((x >> 3) << (bit & 7u)); bit++; }
        }
        c->off[s] = at; at += encode(cp, sym, STREAM, PF, wide, c->bytes + at, x0_of(M->W + (size_t)s * (STREAM / 2u), wide));
        c->soff[s] = sat; sat += (bit + 7u) / 8u;   /* sign bits, byte aligned per stream */
    }
    c->off[s] = at; c->soff[s] = sat;
}

/* Model A decoder, NW streams interleaved (the streams are independent, so this needs no format change). */
static inline __attribute__((always_inline)) void decode_A(const Mat *M, const uint16_t *cdf, const Coded *c,
                                                           uint8_t *out, const uint32_t NW, const int simd) {
    uint32_t s, j, k;
    for (s = 0; s < M->streams; s += NW) {
        uint32_t x[4], prev[4], mx[4], base[4], lo[4];
        const uint8_t *ptr[4];
        for (k = 0; k < NW; k++) {
            const uint8_t *p = c->bytes + c->off[s + k];
            x[k] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; ptr[k] = p + 4;
        }
        for (j = 0; j < STREAM; j++) for (k = 0; k < NW; k++) {
            size_t i = (size_t)(s + k) * STREAM + j;
            const uint16_t *t;
            uint32_t slot = x[k] & ((1u << PA) - 1u), sym, start, f;
            if (!(j & 31u)) { base[k] = srel_of(M->S[i / 32u], M->anchor) * 81u; prev[k] = 8; mx[k] = 8; }
            t = cdf + (base[k] + prev[k] * 9u + mx[k]) * 17u;
            if (simd) {   /* symbol = number of the 16 upper bounds t[1..16] that are <= slot */
                __m256i gt = _mm256_cmpgt_epi16(_mm256_loadu_si256((const __m256i *)(t + 1)), _mm256_set1_epi16((short)slot));
                sym = 16u - (uint32_t)__builtin_popcount((uint32_t)_mm256_movemask_epi8(gt)) / 2u;
            } else {
                uint32_t a = 0, b = 16;
                while (a + 1u < b) { uint32_t mid = (a + b) / 2u; if (t[mid] <= slot) a = mid; else b = mid; }
                sym = a;
            }
            start = t[sym]; f = t[sym + 1u] - start;
            x[k] = f * (x[k] >> PA) + slot - start;
            while (x[k] < L23) x[k] = x[k] << 8 | *ptr[k]++;
            if (!(j & 1u)) lo[k] = sym; else out[i / 2u] = (uint8_t)(lo[k] | sym << 4);
            prev[k] = sym & 7u; if (mx[k] == 8u || prev[k] > mx[k]) mx[k] = prev[k];
        }
    }
}

/* Model F decoder: slot table entry = freq << 16 | (slot - start) << 3 | magnitude. */
static inline __attribute__((always_inline)) void decode_F(const Mat *M, const uint32_t *lut, const Coded *c,
                                                           uint8_t *out, const uint32_t NW, const int wide) {
    uint32_t s, j, k;
    for (s = 0; s < M->streams; s += NW) {
        uint32_t x[4], bit[4], lo[4];
        const uint8_t *ptr[4], *sg[4];
        const uint32_t *t[4];
        for (k = 0; k < NW; k++) {
            const uint8_t *p = c->bytes + c->off[s + k];
            if (wide) memcpy(&x[k], p, 4);
            else x[k] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
            ptr[k] = p + 4; sg[k] = c->sign + c->soff[s + k]; bit[k] = 0;
        }
        for (j = 0; j < STREAM; j++) for (k = 0; k < NW; k++) {
            size_t i = (size_t)(s + k) * STREAM + j;
            uint32_t e, mag, nz, sym;
            if (!(j & 31u)) t[k] = lut + ((size_t)srel_of(M->S[i / 32u], M->anchor) << PF);
            e = t[k][x[k] & ((1u << PF) - 1u)];
            x[k] = (e >> 16) * (x[k] >> PF) + ((e >> 3) & 0x1fffu);
            if (wide) {
                uint32_t w, need = x[k] < (1u << 16);
                uint16_t v; memcpy(&v, ptr[k], 2); w = x[k] << 16 | v;
                x[k] = need ? w : x[k]; ptr[k] += 2u * need;
            } else while (x[k] < L23) x[k] = x[k] << 8 | *ptr[k]++;
            mag = e & 7u; nz = mag != 0u;
            sym = mag | ((sg[k][bit[k] >> 3] >> (bit[k] & 7u)) & nz) << 3; bit[k] += nz;
            if (!(j & 1u)) lo[k] = sym; else out[i / 2u] = (uint8_t)(lo[k] | sym << 4);
        }
    }
}
/* Reference-style decoders: what wex_nr2_decode checks per stream (open, bounds on every renormalisation byte,
 * no nibble 8, pointer at the end, final state == x0 with the hash of the decoded bytes). One stream at a time. */
static int decode_A_chk(const Mat *M, const uint16_t *cdf, const Coded *c, uint8_t *out, const int simd) {
    uint32_t s, j;
    for (s = 0; s < M->streams; s++) {
        const uint8_t *p = c->bytes + c->off[s], *end = c->bytes + c->off[s + 1u];
        uint32_t x, prev = 8, mx = 8, base = 0, lo = 0;
        if (end - p < 4) return -1;
        x = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; p += 4;
        if (x < L23 || x >= (L23 << 8)) return -1;
        for (j = 0; j < STREAM; j++) {
            size_t i = (size_t)s * STREAM + j;
            const uint16_t *t;
            uint32_t slot = x & ((1u << PA) - 1u), sym, start, f;
            if (!(j & 31u)) { base = srel_of(M->S[i / 32u], M->anchor) * 81u; prev = 8; mx = 8; }
            t = cdf + (base + prev * 9u + mx) * 17u;
            if (simd) {
                __m256i gt = _mm256_cmpgt_epi16(_mm256_loadu_si256((const __m256i *)(t + 1)), _mm256_set1_epi16((short)slot));
                sym = 16u - (uint32_t)__builtin_popcount((uint32_t)_mm256_movemask_epi8(gt)) / 2u;
            } else {
                uint32_t a = 0, b = 16;
                while (a + 1u < b) { uint32_t mid = (a + b) / 2u; if (t[mid] <= slot) a = mid; else b = mid; }
                sym = a;
            }
            start = t[sym]; f = t[sym + 1u] - start;
            x = f * (x >> PA) + slot - start;
            while (x < L23) { if (p == end) return -1; x = x << 8 | *p++; }
            if (sym == 8u) return -1;
            if (!(j & 1u)) lo = sym; else out[i / 2u] = (uint8_t)(lo | sym << 4);
            prev = sym & 7u; if (mx == 8u || prev > mx) mx = prev;
        }
        if (p != end || x != x0_of(out + (size_t)s * (STREAM / 2u), 0)) return -1;
    }
    return 0;
}
static int decode_F_chk(const Mat *M, const uint32_t *lut, const Coded *c, uint8_t *out, const int wide) {
    uint32_t s, j;
    for (s = 0; s < M->streams; s++) {
        const uint8_t *p = c->bytes + c->off[s], *end = c->bytes + c->off[s + 1u], *sg = c->sign + c->soff[s];
        const uint32_t *t = lut;
        uint32_t x, bit = 0, lo = 0;
        if (end - p < 4) return -1;
        if (wide) memcpy(&x, p, 4); else x = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
        p += 4;
        for (j = 0; j < STREAM; j++) {
            size_t i = (size_t)s * STREAM + j;
            uint32_t e, mag, nz, sym;
            if (!(j & 31u)) t = lut + ((size_t)srel_of(M->S[i / 32u], M->anchor) << PF);
            e = t[x & ((1u << PF) - 1u)];
            x = (e >> 16) * (x >> PF) + ((e >> 3) & 0x1fffu);
            if (wide) {
                uint32_t w, need = x < (1u << 16);
                uint16_t v; memcpy(&v, p, 2); w = x << 16 | v;
                x = need ? w : x; p += 2u * need;
            } else while (x < L23) { if (p == end) return -1; x = x << 8 | *p++; }
            mag = e & 7u; nz = mag != 0u;
            sym = mag | ((sg[bit >> 3] >> (bit & 7u)) & nz) << 3; bit += nz;
            if (!(j & 1u)) lo = sym; else out[i / 2u] = (uint8_t)(lo | sym << 4);
        }
        if (p != end || (bit + 7u) / 8u != c->soff[s + 1u] - c->soff[s] ||
            x != x0_of(out + (size_t)s * (STREAM / 2u), wide)) return -1;
    }
    return 0;
}
static void dA_ref(const Mat *M, const uint16_t *cdf, const Coded *c, uint8_t *o) { decode_A(M, cdf, c, o, 1, 0); }
static void dA_avx(const Mat *M, const uint16_t *cdf, const Coded *c, uint8_t *o) { decode_A(M, cdf, c, o, 1, 1); }
static void dA_avx4(const Mat *M, const uint16_t *cdf, const Coded *c, uint8_t *o) { decode_A(M, cdf, c, o, 4, 1); }
static void dA_ref4(const Mat *M, const uint16_t *cdf, const Coded *c, uint8_t *o) { decode_A(M, cdf, c, o, 4, 0); }
static void dF_1(const Mat *M, const uint32_t *l, const Coded *c, uint8_t *o) { decode_F(M, l, c, o, 1, 0); }
static void dF_4(const Mat *M, const uint32_t *l, const Coded *c, uint8_t *o) { decode_F(M, l, c, o, 4, 0); }
static void dFw_1(const Mat *M, const uint32_t *l, const Coded *c, uint8_t *o) { decode_F(M, l, c, o, 1, 1); }
static void dFw_4(const Mat *M, const uint32_t *l, const Coded *c, uint8_t *o) { decode_F(M, l, c, o, 4, 1); }

int main(int argc, char **argv) {
    static Mat M[3];
    static Coded A[3], F[3], Fw[3];
    static uint16_t cdfA[3][567 * 17], cdfF[3][7 * 9];
    static uint64_t cntA[3][567][16], cntF[3][7][8];
    uint32_t *lut[3];
    uint8_t entry[72], *slab, *out[3];
    uint64_t weights = 0, bytesA = 0, bytesF = 0, bytesFw = 0, signs = 0;
    uint32_t m, i;
    long size;
    FILE *f;
    if (argc > 6) HASH = atoi(argv[6]);
    if (argc < 6 || strlen(argv[2]) != 144u || !(f = fopen(argv[1], "rb"))) return puts("usage: decbench SLAB ENTRY_HEX A0 A1 A2"), 1;
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    slab = (uint8_t *)malloc((size_t)size);
    if (!slab || fread(slab, 1, (size_t)size, f) != (size_t)size) return 1;
    fclose(f);
    for (i = 0; i < 72u; i++) { unsigned v; sscanf(argv[2] + 2u * i, "%2x", &v); entry[i] = (uint8_t)v; }
    for (m = 0; m < 3u; m++) {   /* WexEntry: shapes u16 at 20 (scales) and 32 (weights), offsets u32 at 44 / 56 */
        uint32_t so, wo;
        memcpy(&so, entry + 44u + 4u * m, 4); memcpy(&wo, entry + 56u + 4u * m, 4);
        M[m].R = entry[20u + 4u * m] | entry[21u + 4u * m] << 8; M[m].B = entry[22u + 4u * m] | entry[23u + 4u * m] << 8;
        M[m].S = slab + so; M[m].W = slab + wo; M[m].n = M[m].R * M[m].B * 32u; M[m].streams = M[m].n / STREAM;
        M[m].anchor = atoi(argv[3 + m]);
        if (M[m].n % STREAM || M[m].streams > MAXS || M[m].streams % 4u || (size_t)wo + M[m].n / 2u > (size_t)size) return puts("geometry"), 1;
        weights += M[m].n;
        {   /* counts and models from this slab */
            uint32_t sr = 0, prev = 8, mx = 8; size_t k;
            for (k = 0; k < M[m].n; k++) {
                uint32_t x = nib(M[m].W, k);
                if (!(k & 31u)) { sr = srel_of(M[m].S[k / 32u], M[m].anchor); prev = 8; mx = 8; }
                cntA[m][(sr * 9u + prev) * 9u + mx][x]++; cntF[m][sr][x & 7u]++;
                prev = x & 7u; if (mx == 8u || prev > mx) mx = prev;
            }
        }
        for (i = 0; i < 567u; i++) quantize(cntA[m][i], 16, 1u << PA, cdfA[m] + i * 17u);
        lut[m] = (uint32_t *)malloc(sizeof(uint32_t) * 7u << PF);
        for (i = 0; i < 7u; i++) {
            uint32_t s, slot;
            quantize(cntF[m][i], 8, 1u << PF, cdfF[m] + i * 9u);
            for (s = 0; s < 8u; s++)
                for (slot = cdfF[m][i * 9u + s]; slot < cdfF[m][i * 9u + s + 1u]; slot++)
                    lut[m][(i << PF) + slot] = (uint32_t)(cdfF[m][i * 9u + s + 1u] - cdfF[m][i * 9u + s]) << 16 |
                                               (slot - cdfF[m][i * 9u + s]) << 3 | s;
        }
        A[m].bytes = (uint8_t *)malloc(M[m].n); F[m].bytes = (uint8_t *)malloc(M[m].n); Fw[m].bytes = (uint8_t *)malloc(M[m].n);
        F[m].sign = (uint8_t *)malloc(M[m].n / 8u + 64u * MAXS); Fw[m].sign = F[m].sign;
        out[m] = (uint8_t *)malloc(M[m].n / 2u);
        encode_A(&M[m], cdfA[m], &A[m]); encode_F(&M[m], cdfF[m], 0, &F[m]);
        memcpy(Fw[m].soff, F[m].soff, sizeof(F[m].soff));
        {   /* the wide stream shares the sign bits; encode_F rewrites the same bytes */
            encode_F(&M[m], cdfF[m], 1, &Fw[m]);
        }
        bytesA += A[m].off[M[m].streams]; bytesF += F[m].off[M[m].streams]; bytesFw += Fw[m].off[M[m].streams];
        signs += F[m].soff[M[m].streams];
    }
    printf("weights %llu streams %u\n", (unsigned long long)weights, M[0].streams + M[1].streams + M[2].streams);
    printf("size A (567 ctx, 16 sym, 14 bit)        %llu B = %.5f bit/w\n", (unsigned long long)bytesA, bytesA * 8.0 / weights);
    printf("size F (7 ctx, 8 mag, 12 bit, raw sign) %llu + %llu sign = %llu B = %.5f bit/w\n", (unsigned long long)bytesF,
           (unsigned long long)signs, (unsigned long long)(bytesF + signs), (bytesF + signs) * 8.0 / weights);
    printf("size Fw (same, 16-bit renorm, L 2^16)   %llu + %llu sign = %llu B = %.5f bit/w\n", (unsigned long long)bytesFw,
           (unsigned long long)signs, (unsigned long long)(bytesFw + signs), (bytesFw + signs) * 8.0 / weights);
#define RUN(name, call) do { double best = 1e9; int r, bad = 0; \
        for (r = 0; r < 5; r++) { double t = now(); for (m = 0; m < 3u; m++) call; t = now() - t; if (t < best) best = t; } \
        for (m = 0; m < 3u; m++) bad |= memcmp(out[m], M[m].W, M[m].n / 2u) != 0; \
        printf("%-34s %7.2f ns/weight  %7.1f ms/record  %s\n", name, best * 1e9 / weights, best * 1e3, bad ? "MISMATCH" : "exact"); \
        if (bad) return 1; } while (0)
    RUN("A reference search, 1 stream", dA_ref(&M[m], cdfA[m], &A[m], out[m]));
    RUN("A reference search, 4 interleaved", dA_ref4(&M[m], cdfA[m], &A[m], out[m]));
    RUN("A AVX2 search, 1 stream", dA_avx(&M[m], cdfA[m], &A[m], out[m]));
    RUN("A AVX2 search, 4 interleaved", dA_avx4(&M[m], cdfA[m], &A[m], out[m]));
    RUN("F slot table, 1 stream", dF_1(&M[m], lut[m], &F[m], out[m]));
    RUN("F slot table, 4 interleaved", dF_4(&M[m], lut[m], &F[m], out[m]));
    RUN("Fw slot table, 1 stream", dFw_1(&M[m], lut[m], &Fw[m], out[m]));
    RUN("Fw slot table, 4 interleaved", dFw_4(&M[m], lut[m], &Fw[m], out[m]));
    RUN("A reference search + checks", FAIL |= decode_A_chk(&M[m], cdfA[m], &A[m], out[m], 0));
    RUN("A AVX2 search + checks", FAIL |= decode_A_chk(&M[m], cdfA[m], &A[m], out[m], 1));
    RUN("F slot table + checks", FAIL |= decode_F_chk(&M[m], lut[m], &F[m], out[m], 0));
    RUN("Fw slot table + checks", FAIL |= decode_F_chk(&M[m], lut[m], &Fw[m], out[m], 1));
    printf("checked decoders status %d (0 = every stream passed), hash mode %d\n", FAIL, HASH);
    {   /* work every full decoder still has to do after the streams */
        WexNr2Layout lay;
        unsigned char *sc[3], *wt[3];
        double best = 1e9, bh = 1e9;
        int r;
        uint32_t acc = 0, s;
        for (m = 0; m < 3u; m++) {
            lay.scale_count[m] = M[m].R * M[m].B; lay.symbol_count[m] = M[m].n;
            sc[m] = (unsigned char *)malloc(lay.scale_count[m]); wt[m] = (unsigned char *)malloc(M[m].n / 2u);
        }
        for (r = 0; r < 5; r++) {
            double t;
            for (m = 0; m < 3u; m++) { memcpy(sc[m], M[m].S, lay.scale_count[m]); memcpy(wt[m], M[m].W, M[m].n / 2u); }
            t = now(); if (wex_nr3_ceil_inverse(&lay, 7u, sc, wt)) return puts("ceil inverse failed"), 1; t = now() - t;
            if (t < best) best = t;
            t = now();
            for (m = 0; m < 3u; m++) for (s = 0; s < M[m].streams; s++) acc += h23(M[m].W + (size_t)s * (STREAM / 2u), STREAM / 2u);
            t = now() - t;
            if (t < bh) bh = t;
        }
        printf("ceil inverse of the record            %7.1f ms\n", best * 1e3);
        printf("stream hashes alone (17280 x 1024 B)  %7.1f ms  (%u)\n", bh * 1e3, acc);
    }
    return 0;
}
