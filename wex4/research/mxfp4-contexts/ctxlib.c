/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Context-model counting library for the MXFP4 weight streams of a WNR3 coded slab
 * (canonical zero + ceil form). CPU only, research code: it never writes records.
 * Build: gcc -O2 -shared ctxlib.c -o out/ctxlib.dll   (thread-safe: no globals are written)
 *
 * A context is the mixed-radix number of a list of features ("spec"). Every feature is a
 * function of data a decoder already has when it reaches the symbol:
 *   - the scales of the whole matrix (all scale streams precede the weight streams),
 *   - earlier nibbles of the SAME stream (stream = 2048 nibbles = 64 blocks of 32),
 * except the ones marked CROSS (row above: lives in an earlier stream, forces a stream order)
 * and SIDE (needs a per-block side symbol coded before the block's nibbles). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __declspec(dllexport)
#define STREAM 2048u

enum {
    F_SREL,   /* 7  clamp(scale - anchor, +-3), today's context                         */
    F_PM1,    /* 9  previous magnitude in the block, 8 at block start, today's context  */
    F_RMAX,   /* 9  running max magnitude in the block, 8 at block start, today's       */
    F_PM2,    /* 9  magnitude two back in the block, 8 if none                          */
    F_POS,    /* 32 position in the block                                               */
    F_POSC,   /* 3  position class 0 / 1-7 / 8-31                                       */
    F_PSIGN,  /* 3  sign of the previous nibble: none / + / -                           */
    F_ABOVE,  /* 9  CROSS magnitude of the nibble above (previous row), 8 on row 0      */
    F_ABOVE3, /* 3  CROSS class of it: <=1 / 2-4 or none / >=5                          */
    F_ASIGN,  /* 3  CROSS sign of the nibble above: none / + / -                        */
    F_DNEXT,  /* 5  clamp(scale(next block in row) - scale, +-2), 0 at row end          */
    F_DLEFT,  /* 5  same with the previous block in the row                             */
    F_DUP,    /* 5  same with the block above                                           */
    F_DDOWN,  /* 5  same with the block below                                           */
    F_LOCMAX, /* 5  clamp(scale - max(4 neighbour scales), +-2)                         */
    F_LOCAVG, /* 5  clamp(round(scale - mean(4 neighbour scales)), +-2)                 */
    F_SROW,   /* 7  clamp(scale - rule-g anchor of the row, +-3)                        */
    F_SCOL,   /* 7  clamp(scale - rule-g anchor of the block column, +-3)               */
    F_ZC,     /* 8  zeros so far in the block, capped at 7                              */
    F_ZC4,    /* 4  zeros so far, capped at 3                                           */
    F_BIG,    /* 4  magnitudes >= 4 so far, capped at 3                                 */
    F_BMAX3,  /* 3  SIDE block max class: <=5 / 6 / 7                                   */
    F_ZBLK,   /* 6  SIDE zero count of the block: <=1 / 2 / 3 / 4 / 5-6 / >=7           */
    F_PBMAX,  /* 4  previous block of the same stream: none / max <=5 / 6 / 7           */
    F_PM1X,   /* 9  previous magnitude NOT reset at block starts (only at stream start) */
    F_PAR,    /* 2  nibble parity                                                       */
    F_REMZ,   /* 6  SIDE zeros still to come in the block (exact count known), cap 5    */
    F_REMP,   /* 6  positions left: 1 / 2 / 3-4 / 5-8 / 9-16 / 17-32                    */
    F_RMAXC,  /* 4  running max class: none / <5 / 5 / >=6                              */
    F_PM1C,   /* 4  previous magnitude class: none / 0 / 1-3 / 4-7                      */
    F_SREL5,  /* 11 clamp(scale - anchor, +-5)                                          */
    F_SUM3,   /* 8  sum of the last three magnitudes in the block (missing = 0) / 3     */
    F_ROWLVL, /* 5  clamp(row anchor - matrix anchor, +-2): the level of the whole row  */
    F_COLLVL, /* 5  clamp(column anchor - matrix anchor, +-2)                           */
    NF
};
static const uint32_t radix[NF] = {7, 9, 9, 9, 32, 3, 3, 9, 3, 3, 5, 5, 5, 5, 5, 5, 7, 7, 8, 4, 4, 3, 6, 4, 9, 2,
                                   6, 6, 4, 4, 11, 8, 5, 5};

EXPORT uint32_t ctx_features(void) { return NF; }
EXPORT uint32_t ctx_radix(uint32_t f) { return f < NF ? radix[f] : 0; }

static inline int clampi(int v, int lim) { return v < -lim ? -lim : v > lim ? lim : v; }

/* rule g of WNR3: lower value of the densest adjacent pair, first maximum wins */
static uint8_t anchor_of(const uint8_t *s, uint32_t n, uint32_t stride) {
    uint32_t h[256] = {0}, i, best = 0, top = 0;
    for (i = 0; i < n; i++) h[s[(size_t)i * stride]]++;
    for (i = 0; i < 255u; i++)
        if (h[i] + h[i + 1u] > top) { top = h[i] + h[i + 1u]; best = i; }
    return (uint8_t)best;
}

/* geom[4m..4m+3] = scale offset, weight offset, rows, blocks per row of matrix m; anchor[m] = record anchors.
 * mode 0: counts[3][nctx][16] += 1 per symbol
 * mode 1: scost[stream][K] += cost[K][3][nctx][16] (bits) per symbol, streams numbered over w1, w2, w3
 * mode 2: counts[K][3][nctx][16] += 1 with the table set assign[stream]
 * mode 3: side[3][7][64] += 1 per block at [srel][v], v = side_kind 0: BMAX3, 1: exact zero count, 2: ZBLK
 * Returns the context count of the spec (0 on a bad spec). */
EXPORT uint32_t ctx_walk(const uint8_t *slab, const uint32_t *geom, const uint8_t *anchor, const int32_t *spec,
                         int nspec, int mode, uint32_t *counts, const float *cost, int K, const uint8_t *assign,
                         double *scost, uint32_t *side, int side_kind) {
    uint32_t nctx = 1, m, sbase = 0;
    int j;
    for (j = 0; j < nspec; j++) {
        if (spec[j] < 0 || spec[j] >= NF) return 0;
        nctx *= radix[spec[j]];
    }
    for (m = 0; m < 3u; m++) {
        const uint8_t *S = slab + geom[4u * m], *W = slab + geom[4u * m + 1u];
        uint32_t R = geom[4u * m + 2u], B = geom[4u * m + 3u], r, b, rowlen = B * 32u;
        uint8_t *rowa = (uint8_t *)malloc(R), *cola = (uint8_t *)malloc(B);
        uint32_t f[NF] = {0}, pbmax = 0, pm1x = 8;
        if (!rowa || !cola) { free(rowa); free(cola); return 0; }
        for (r = 0; r < R; r++) rowa[r] = anchor_of(S + (size_t)r * B, B, 1u);
        for (b = 0; b < B; b++) cola[b] = anchor_of(S + b, R, B);
        for (r = 0; r < R; r++) for (b = 0; b < B; b++) {
            size_t bi = (size_t)r * B + b, i0 = bi * 32u;
            uint32_t stream = sbase + (uint32_t)(i0 / STREAM), p, zc = 0, big = 0, pm1 = 8, pm2 = 8, pm3 = 8, rmax = 8;
            uint32_t bmax = 0, zblk = 0, psign = 0;
            int s = S[bi], nmax = -1, nsum = 0, ncnt = 0, d;
            const uint8_t *q = W + bi * 16u;
            for (p = 0; p < 32u; p++) {
                uint32_t a = (q[p / 2u] >> ((p & 1u) * 4u)) & 7u;
                if (a > bmax) bmax = a;
                zblk += !a;
            }
            f[F_SREL] = (uint32_t)(clampi(s - (int)anchor[m], 3) + 3);
            f[F_SREL5] = (uint32_t)(clampi(s - (int)anchor[m], 5) + 5);
            f[F_SROW] = (uint32_t)(clampi(s - (int)rowa[r], 3) + 3);
            f[F_SCOL] = (uint32_t)(clampi(s - (int)cola[b], 3) + 3);
            f[F_ROWLVL] = (uint32_t)(clampi((int)rowa[r] - (int)anchor[m], 2) + 2);
            f[F_COLLVL] = (uint32_t)(clampi((int)cola[b] - (int)anchor[m], 2) + 2);
            f[F_DNEXT] = f[F_DLEFT] = f[F_DUP] = f[F_DDOWN] = 2;
            if (b + 1u < B) { d = S[bi + 1u]; f[F_DNEXT] = (uint32_t)(clampi(d - s, 2) + 2); nsum += d; ncnt++; if (d > nmax) nmax = d; }
            if (b) { d = S[bi - 1u]; f[F_DLEFT] = (uint32_t)(clampi(d - s, 2) + 2); nsum += d; ncnt++; if (d > nmax) nmax = d; }
            if (r) { d = S[bi - B]; f[F_DUP] = (uint32_t)(clampi(d - s, 2) + 2); nsum += d; ncnt++; if (d > nmax) nmax = d; }
            if (r + 1u < R) { d = S[bi + B]; f[F_DDOWN] = (uint32_t)(clampi(d - s, 2) + 2); nsum += d; ncnt++; if (d > nmax) nmax = d; }
            f[F_LOCMAX] = (uint32_t)(clampi(s - nmax, 2) + 2);
            /* round(s - mean) with integers: floor((2*(s*n - sum) + n) / (2n)) */
            if (!ncnt) ncnt = 1, nsum = s;   /* 1x1 matrix: no neighbour */
            d = 2 * (s * ncnt - nsum) + ncnt;
            f[F_LOCAVG] = (uint32_t)(clampi(d >= 0 ? d / (2 * ncnt) : -((-d + 2 * ncnt - 1) / (2 * ncnt)), 2) + 2);
            f[F_BMAX3] = bmax <= 5u ? 0u : bmax - 5u;
            f[F_ZBLK] = zblk <= 1u ? 0u : zblk <= 4u ? zblk - 1u : zblk <= 6u ? 4u : 5u;
            if (!(i0 % STREAM)) { pbmax = 0; pm1x = 8; }
            f[F_PBMAX] = pbmax;
            if (mode == 3) {
                uint32_t v = side_kind == 0 ? f[F_BMAX3] : side_kind == 1 ? zblk : f[F_ZBLK];
                side[(m * 7u + f[F_SREL]) * 64u + v]++;
                continue;
            }
            for (p = 0; p < 32u; p++) {
                uint32_t x = (q[p / 2u] >> ((p & 1u) * 4u)) & 15u, a = x & 7u, ctx = 0, left = 32u - p;
                uint32_t up = 8, upsign = 0, z = zblk - zc;
                if (r) {
                    uint32_t y = ((q - rowlen / 2u)[p / 2u] >> ((p & 1u) * 4u)) & 15u;
                    up = y & 7u; upsign = 1u + (y >> 3);
                }
                f[F_PM1] = pm1; f[F_RMAX] = rmax; f[F_PM2] = pm2; f[F_POS] = p;
                f[F_POSC] = !p ? 0u : p < 8u ? 1u : 2u;
                f[F_PSIGN] = psign;
                f[F_ABOVE] = up; f[F_ABOVE3] = up == 8u ? 1u : up <= 1u ? 0u : up <= 4u ? 1u : 2u; f[F_ASIGN] = upsign;
                f[F_ZC] = zc > 7u ? 7u : zc; f[F_ZC4] = zc > 3u ? 3u : zc; f[F_BIG] = big > 3u ? 3u : big;
                f[F_PM1X] = pm1x; f[F_PAR] = p & 1u;
                f[F_REMZ] = z > 5u ? 5u : z;
                f[F_REMP] = left <= 2u ? left - 1u : left <= 4u ? 2u : left <= 8u ? 3u : left <= 16u ? 4u : 5u;
                f[F_RMAXC] = rmax == 8u ? 0u : rmax < 5u ? 1u : rmax == 5u ? 2u : 3u;
                f[F_PM1C] = pm1 == 8u ? 0u : !pm1 ? 1u : pm1 < 4u ? 2u : 3u;
                f[F_SUM3] = ((pm1 & 7u) * (pm1 != 8u) + (pm2 & 7u) * (pm2 != 8u) + (pm3 & 7u) * (pm3 != 8u)) / 3u;
                for (j = 0; j < nspec; j++) ctx = ctx * radix[spec[j]] + f[spec[j]];
                if (mode == 0) counts[((size_t)m * nctx + ctx) * 16u + x]++;
                else if (mode == 2) counts[(((size_t)assign[stream] * 3u + m) * nctx + ctx) * 16u + x]++;
                else {
                    int k;
                    for (k = 0; k < K; k++)
                        scost[(size_t)stream * (size_t)K + k] += cost[(((size_t)k * 3u + m) * nctx + ctx) * 16u + x];
                }
                pm3 = pm2; pm2 = pm1; pm1 = a; pm1x = a; psign = 1u + (x >> 3);
                if (rmax == 8u || a > rmax) rmax = a;
                zc += !a; big += a >= 4u;
            }
            pbmax = bmax <= 5u ? 1u : bmax - 4u;
        }
        sbase += (uint32_t)(((size_t)R * rowlen + STREAM - 1u) / STREAM);
        free(rowa); free(cola);
    }
    return nctx;
}
