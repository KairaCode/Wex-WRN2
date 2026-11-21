/* SPDX-License-Identifier: Apache-2.0 */
/* WNR4 research: context statistics of coded expert slabs (MXFP4 or NVFP4).
 * One call per (record, set): adds the record's counts to the train (set 0)
 * or test (set 1) accumulator. Cross-entropy of test under quantized train
 * tables is computed at the end (st_report). CPU only, no codec here. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define EXPORT __declspec(dllexport)

/* weight models (16-symbol alphabet unless noted), per matrix */
enum { W_WNR3, W_CLS, W_RUN8, W_RUN8MX, W_CLSM, W_MAG_CLS, W_MAG_RUN8, W_MAG_CLSM, W_N };
static const char *WNAME[W_N] = {"wnr3 (cls,pm,mx) per block", "cls only", "run8 (cls,pm,mx)", "run8 (cls,mx)",
                                 "cls x blockmax M (nibbles)", "mag8|cls (+sign)", "mag8|run8 (+sign)", "mag8|cls,M (+sign)"};
#define NCLS 13           /* class = clamp(s - anchor, -R, R) + R, R = 3 (MX) or 6 (NV) */
#define WCTX (NCLS * 81)  /* largest context count of a weight model */
/* scale models, per matrix, 256 symbols */
enum { S_O0E, S_O0O, S_LE, S_LO, S_U, S_N };
#define SCTX 257

typedef struct {
    uint64_t w[2][3][W_N][WCTX][16];
    uint64_t s[2][3][S_N][SCTX][256];
    uint64_t m[2][3][NCLS][8];       /* block max M given class (for W_CLSM: cost of M) */
    uint64_t sign_bits[2][3];        /* nonzero-magnitude nibbles (raw sign bits) */
    uint64_t nibbles[2][3], scales[2][3], records[2];
} Stats;

EXPORT void *st_new(void) { return calloc(1, sizeof(Stats)); }
EXPORT void st_free(void *p) { free(p); }

/* geometry per matrix: scale offset, rows, cols (scales per row), weight offset.
 * nv = 0: MXFP4 (32 nibbles per scale), 1: NVFP4 (16). anchor[m] precomputed. */
EXPORT int st_add(void *p, int set, int nv, const unsigned char *slab, const uint32_t *geo, const unsigned char *anchor) {
    Stats *S = (Stats *)p;
    const int B = nv ? 16 : 32, R = nv ? 6 : 3;
    for (int m = 0; m < 3; m++) {
        const unsigned char *sc = slab + geo[4 * m], *wt = slab + geo[4 * m + 3];
        uint32_t rows = geo[4 * m + 1], cols = geo[4 * m + 2];
        uint64_t n = (uint64_t)rows * cols;
        for (uint64_t k = 0; k < n; k++) {
            uint32_t col = (uint32_t)(k % cols);
            int s = sc[k];
            int left = col ? 1 + sc[k - 1] : 0;
            int up = k >= cols ? 1 + sc[k - cols] : 0;
            S->s[set][m][(k & 1) ? S_O0O : S_O0E][0][s]++;
            S->s[set][m][(k & 1) ? S_LO : S_LE][left][s]++;
            S->s[set][m][S_U][up][s]++;
            int d = s - (int)anchor[m];
            int cls = (d < -R ? -R : d > R ? R : d) + R;
            /* block nibbles */
            const unsigned char *bw = wt + k * (B / 2);
            int q[32], M = 0;
            for (int i = 0; i < B; i++) {
                q[i] = (bw[i >> 1] >> ((i & 1) * 4)) & 15;
                if ((q[i] & 7) > M) M = q[i] & 7;
            }
            S->m[set][m][cls][M]++;
            int pm = 8, mx = 8, pm8 = 8, mx8 = 8;
            for (int i = 0; i < B; i++) {
                int x = q[i], a = x & 7;
                if ((i & 7) == 0) pm8 = mx8 = 8;
                S->w[set][m][W_WNR3][(cls * 9 + pm) * 9 + mx][x]++;
                S->w[set][m][W_CLS][cls][x]++;
                S->w[set][m][W_RUN8][(cls * 9 + pm8) * 9 + mx8][x]++;
                S->w[set][m][W_RUN8MX][cls * 9 + mx8][x]++;
                S->w[set][m][W_CLSM][cls * 8 + M][x]++;
                S->w[set][m][W_MAG_CLS][cls][a]++;
                S->w[set][m][W_MAG_RUN8][(cls * 9 + pm8) * 9 + mx8][a]++;
                S->w[set][m][W_MAG_CLSM][cls * 8 + M][a]++;
                if (a) S->sign_bits[set][m]++;
                pm = a; mx = mx == 8 ? a : (a > mx ? a : mx);
                pm8 = a; mx8 = mx8 == 8 ? a : (a > mx8 ? a : mx8);
            }
        }
        S->nibbles[set][m] += n * B;
        S->scales[set][m] += n;
    }
    S->records[set]++;
    return 0;
}

/* cost in bits of test counts under the quantized (precision bits) table of train counts */
static double xent(const uint64_t *train, const uint64_t *test, int nsym, int bits) {
    uint64_t total = 0, tt = 0;
    for (int i = 0; i < nsym; i++) { total += train[i]; tt += test[i]; }
    if (!tt) return 0.0;
    double T = (double)(1u << bits), cost = 0.0;
    double f[256];
    if (!total) {
        for (int i = 0; i < nsym; i++) f[i] = T / nsym;
    } else {   /* 1 + proportional share of the rest (rANS tables need f >= 1) */
        double sum = 0.0;
        for (int i = 0; i < nsym; i++) { f[i] = 1.0 + floor((double)train[i] * (T - nsym) / (double)total); sum += f[i]; }
        int best = 0;
        for (int i = 1; i < nsym; i++) if (f[i] > f[best]) best = i;
        f[best] += T - sum;
    }
    for (int i = 0; i < nsym; i++) if (test[i]) cost += (double)test[i] * (bits - log2(f[i]));
    return cost;
}

/* out (doubles): per matrix m: [W_N weight model bits] [S_N scale bits] [M cost bits] [sign bits] [nibbles] [scales]
 * = W_N + S_N + 4 values; test set only. wbits / sbits = table precisions. */
EXPORT int st_report(void *p, int wbits, int sbits, double *out) {
    Stats *S = (Stats *)p;
    const int stride = W_N + S_N + 4;
    for (int m = 0; m < 3; m++) {
        double *o = out + m * stride;
        for (int v = 0; v < W_N; v++) {
            int nsym = v >= W_MAG_CLS ? 8 : 16;
            double c = 0;
            for (int x = 0; x < WCTX; x++) c += xent(S->w[0][m][v][x], S->w[1][m][v][x], nsym, wbits);
            o[v] = c;
        }
        for (int v = 0; v < S_N; v++) {
            double c = 0;
            for (int x = 0; x < SCTX; x++) c += xent(S->s[0][m][v][x], S->s[1][m][v][x], 256, sbits);
            o[W_N + v] = c;
        }
        double mc = 0;
        for (int x = 0; x < NCLS; x++) mc += xent(S->m[0][m][x], S->m[1][m][x], 8, sbits);
        o[W_N + S_N] = mc;
        o[W_N + S_N + 1] = (double)S->sign_bits[1][m];
        o[W_N + S_N + 2] = (double)S->nibbles[1][m];
        o[W_N + S_N + 3] = (double)S->scales[1][m];
    }
    return stride;
}

EXPORT const char *st_wname(int v) { return v >= 0 && v < W_N ? WNAME[v] : ""; }
EXPORT int st_wn(void) { return W_N; }
EXPORT int st_sn(void) { return S_N; }
