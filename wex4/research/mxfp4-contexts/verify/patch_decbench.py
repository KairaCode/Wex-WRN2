# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Builds decbench_v.c from ../decbench.c: optional stream hash folded into the initial state (as WNR3 does),
reference-style checked decoders (bounds, CZ, end of stream, final state), hash-only and ceil-inverse timings."""
import sys
src = open('../decbench.c').read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)

rep('#include <immintrin.h>', '#include <immintrin.h>\n#include "out/stage/wex_neural_rans.h"')
rep('static double now(void)', '''static int HASH = 0, FAIL = 0;   /* 0 none, 1 = 23-bit hash in every x0, 2 = as 1 but 16-bit hash for the wide form */
static uint32_t h23(const uint8_t *data, uint32_t n) { return wex_nr3_h32_bytes(0x33524e57u, data, n, 16u) & 0x7fffffu; }
static uint32_t x0_of(const uint8_t *coded_bytes, int wide) {
    uint32_t h = HASH ? h23(coded_bytes, STREAM / 2u) : 0u;
    return wide ? (1u << 16) + (HASH == 2 ? (h & 0xffffu) : h) : L23 + h;
}
static double now(void)''')
rep('static uint32_t encode(const uint16_t *const *cdf, const uint8_t *sym, uint32_t n, uint32_t P, int wide, uint8_t *out) {',
    'static uint32_t encode(const uint16_t *const *cdf, const uint8_t *sym, uint32_t n, uint32_t P, int wide, uint8_t *out, uint32_t x0) {')
rep('uint32_t x = wide ? 1u << 16 : L23, j;', 'uint32_t x = x0, j;')
rep('at += encode(cp, sym, STREAM, PA, 0, c->bytes + at);',
    'at += encode(cp, sym, STREAM, PA, 0, c->bytes + at, x0_of(M->W + (size_t)s * (STREAM / 2u), 0));')
rep('at += encode(cp, sym, STREAM, PF, wide, c->bytes + at);',
    'at += encode(cp, sym, STREAM, PF, wide, c->bytes + at, x0_of(M->W + (size_t)s * (STREAM / 2u), wide));')
rep('static void dA_ref(', r'''/* Reference-style decoders: what wex_nr2_decode checks per stream (open, bounds on every renormalisation byte,
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
static void dA_ref(''')
rep('if (argc != 6 ||', 'if (argc > 6) HASH = atoi(argv[6]);\n    if (argc < 6 ||')
rep('dFw_4(&M[m], lut[m], &Fw[m], out[m]));\n    return 0;\n}\n', r'''dFw_4(&M[m], lut[m], &Fw[m], out[m]));
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
''')
open('decbench_v.c', 'w').write(src)
