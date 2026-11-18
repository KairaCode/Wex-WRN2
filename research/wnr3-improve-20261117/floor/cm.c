/* cm.c - adaptive context-mixing code length of one coded WNR3 record (the floor experiment).
 *
 * Input: a coded slab (out/coded/rec_LL_EEE.coded = what WNR3-CZ's rANS streams carry: coded scales and coded
 * nibbles after canonical zero + ceil form) and its geometry. Nothing is trained offline unless -p is given:
 * every probability is learned while coding, so the sum of -log2 p is the ideal code length of a real adaptive
 * arithmetic coder (plus <= ~8 B of coder flush per record). With -p PRIME.coded the model first codes another
 * record (a TRAIN record) and only the second record's cost is reported ("pre-trained adaptive").
 *
 * Coding order: all scales (w1, w2, w3), then nibbles of w1, w3, w2 (w3 sees w1 = its sibling at the same
 * [i,j]; w2 sees the neuron statistics of w1 and w3). A nibble = magnitude (3 binary decisions, tree) + sign
 * (one binary decision when magnitude > 0). A scale = 8 binary decisions (tree, MSB first).
 * Every binary decision: N direct-indexed adaptive context models -> 2 gated logistic mixers -> APM.
 *
 * Feature switches (-f mask) allow ablations: bit 0 sibling w1 for w3, bit 1 column statistics,
 * bit 2 neighbours above / two back / zero count, bit 3 w2 neuron statistics, bit 4 position in block,
 * bit 5 exact scale values, bit 6 scale contexts beyond order 0 (left/up/sibling/column), bit 7 sign contexts.
 *
 * Build: gcc -O2 -march=native -o cm cm.c -lm
 * Run:   ./cm [-f MASK] [-p PRIME.coded] FILE soff0 soff1 soff2 woff0 woff1 woff2 R0 NB0 R1 NB1 R2 NB2 a0 a1 a2
 *        (prime uses the same geometry; its anchors follow FILE's as pa0 pa1 pa2 after -P a0,a1,a2)
 * Output: one JSON line with bits per part. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static unsigned FM = 0xffffffffu;
#define F(b) (FM >> (b) & 1u)

/* ---------- primitives ---------- */
static inline float stretchf(float p) { return logf(p / (1.0f - p)); }
static inline float squashf(float x) { if (x > 30) x = 30; if (x < -30) x = -30; return 1.0f / (1.0f + expf(-x)); }

typedef struct { float p; uint16_t n; } Slot;
typedef struct { Slot *t; size_t size; int limit; } CM;
static void cm_init(CM *c, size_t size, int limit) {
    c->size = size; c->limit = limit; c->t = malloc(size * sizeof(Slot));
    for (size_t i = 0; i < size; i++) { c->t[i].p = 0.5f; c->t[i].n = 0; }
}
static inline void cm_upd(Slot *s, int y, int limit) {
    s->p += ((float)y - s->p) / ((float)s->n + 1.5f);
    if (s->n < limit) s->n++;
    if (s->p < 1e-5f) s->p = 1e-5f; if (s->p > 1 - 1e-5f) s->p = 1 - 1e-5f;
}

#define MAXIN 20
typedef struct { float *w; int n, nsel; float lr; } Mixer;
static void mx_init(Mixer *m, int n, int nsel, float lr) {
    m->n = n; m->nsel = nsel; m->lr = lr; m->w = malloc(sizeof(float) * n * nsel);
    for (int i = 0; i < n * nsel; i++) m->w[i] = 0.3f;
}
static inline float mx_dot(Mixer *m, int sel, const float *x) {
    float *w = m->w + (size_t)sel * m->n, s = 0; for (int i = 0; i < m->n; i++) s += w[i] * x[i]; return s;
}
static inline void mx_upd(Mixer *m, int sel, const float *x, float err) {
    float *w = m->w + (size_t)sel * m->n; for (int i = 0; i < m->n; i++) w[i] += m->lr * err * x[i];
}

typedef struct { float *t; uint16_t *n; int nctx; } APM;
static void apm_init(APM *a, int nctx) {
    a->nctx = nctx; a->t = malloc(sizeof(float) * nctx * 33); a->n = calloc((size_t)nctx * 33, 2);
    for (int c = 0; c < nctx; c++) for (int j = 0; j < 33; j++) a->t[c * 33 + j] = squashf((j - 16) * 0.5f);
}
static int apm_idx; static float apm_wt;
static inline float apm_p(APM *a, int ctx, float st) {
    float f = (st + 8.0f) * 2.0f; if (f < 0) f = 0; if (f > 31.999f) f = 31.999f;
    int j = (int)f; apm_wt = f - j; apm_idx = ctx * 33 + j;
    return a->t[apm_idx] * (1 - apm_wt) + a->t[apm_idx + 1] * apm_wt;
}
static inline void apm_upd(APM *a, int y) {
    for (int k = 0; k < 2; k++) {
        int i = apm_idx + k; float w = k ? apm_wt : 1 - apm_wt;
        float r = 1.0f / ((float)a->n[i] + 1.5f); if (r < 0.002f) r = 0.002f;
        a->t[i] += w * r * ((float)y - a->t[i]); if (a->n[i] < 65000) a->n[i]++;
    }
}

/* ---------- a predictor = set of context models + mixers + APM ---------- */
typedef struct {
    int nm; CM cm[MAXIN]; Mixer m1, m2; APM a1;
    Slot *cur[MAXIN]; float x[MAXIN + 1];
} Pred;
static double code_bit(Pred *P, const size_t *idx, int sel1, int sel2, int actx, int y) {
    int n = P->nm;
    for (int i = 0; i < n; i++) { P->cur[i] = &P->cm[i].t[idx[i]]; float st = stretchf(P->cur[i]->p);
        if (st > 8) st = 8; if (st < -8) st = -8; P->x[i] = st; }
    P->x[n] = 0.25f;
    float d1 = mx_dot(&P->m1, sel1, P->x), d2 = mx_dot(&P->m2, sel2, P->x);
    float p1 = squashf(d1), p2 = squashf(d2);
    float st = (d1 + d2) * 0.5f; if (st > 12) st = 12; if (st < -12) st = -12;
    float pm = squashf(st);
    float pa = apm_p(&P->a1, actx, st);
    float p = 0.4f * pm + 0.6f * pa;
    if (p < 1.0f / 65536) p = 1.0f / 65536; if (p > 1 - 1.0f / 65536) p = 1 - 1.0f / 65536;
    double bits = y ? -log2((double)p) : -log2(1.0 - (double)p);
    mx_upd(&P->m1, sel1, P->x, (float)y - p1); mx_upd(&P->m2, sel2, P->x, (float)y - p2);
    apm_upd(&P->a1, y);
    for (int i = 0; i < n; i++) cm_upd(P->cur[i], y, P->cm[i].limit);
    return bits;
}

/* ---------- record geometry ---------- */
typedef struct { const unsigned char *s[3]; const unsigned char *w[3]; int R[3], NB[3], C[3]; int anc[3]; } Rec;
static inline int nib(const Rec *r, int m, int i, int j) {
    unsigned char b = r->w[m][(size_t)i * (r->C[m] / 2) + (j >> 1)]; return j & 1 ? b >> 4 : b & 15;
}
static inline int scl(const Rec *r, int m, int i, int b) { return r->s[m][(size_t)i * r->NB[m] + b]; }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* ---------- models ---------- */
#define NW 13
#define NS 10
static Pred PW, PS, PSG;
static size_t wsz[NW], ssz[NS], gsz[4];

static void init_models(void) {
    /* weight magnitude contexts: every size includes x3 matrices and x8 tree nodes */
    size_t base = 3 * 8;
    wsz[0] = base * 7;                 /* srel */
    wsz[1] = base * 567;               /* srel pm mx (WNR3) */
    wsz[2] = base * 7 * 32 * 9;        /* srel pos mx */
    wsz[3] = base * 7 * 16 * 9;        /* srel colq mx */
    wsz[4] = base * 7 * 9 * 9 * 9;     /* srel up pm mx */
    wsz[5] = base * 7 * 9 * 7 * 9;     /* srel sib_a sib_srel mx  | m1: srel neuronq mx */
    wsz[6] = base * 7 * 9 * 9 * 9;     /* srel sib_a sib_l sib_r */
    wsz[7] = base * 15 * 15 * 9;       /* exact s, exact sib s, sib_a */
    wsz[8] = base * 7 * 9 * 9 * 9;     /* srel sib block max, sib_a, mx */
    wsz[9] = base * 7 * 16 * 16 * 9;   /* srel colq sibcolq/neuronq pm */
    wsz[10] = base * 7 * 9 * 9 * 9;    /* srel pm2 pm mx */
    wsz[11] = base * 7 * 9 * 9;        /* srel zero-count bin mx */
    wsz[12] = base * 7 * 9 * 9 * 9;    /* srel up-left up-right up */
    PW.nm = NW;
    for (int i = 0; i < NW; i++) cm_init(&PW.cm[i], wsz[i], 1023);
    mx_init(&PW.m1, NW + 1, 3 * 8 * 7 * 9, 0.002f);
    mx_init(&PW.m2, NW + 1, 3 * 8 * 9 * 16, 0.002f);
    apm_init(&PW.a1, 3 * 8 * 7 * 9 * 9);
    /* scale contexts: x3 matrices x 256 tree nodes; compact scale value v' = clamp(v - anchor + 16, 0, 31) + 1, 0 = none */
    size_t sb = 3 * 256;
    ssz[0] = sb;                       /* order 0 */
    ssz[1] = sb * 33;                  /* left */
    ssz[2] = sb * 33;                  /* up */
    ssz[3] = sb * 33 * 33;             /* left up */
    ssz[4] = sb * 33 * 33;             /* sib scale, left (m2: w1 scale; m0: w3 not known -> none) */
    ssz[5] = sb * 33 * 33;             /* left left2 */
    ssz[6] = sb * 33 * 33;             /* column mean bin (33), left */
    ssz[7] = sb * 33 * 33;             /* row max so far, left */
    ssz[8] = sb * 33 * 33 * 33;        /* up-left, up, up-right */
    ssz[9] = sb * 33 * 33;             /* row mean so far bin, up */
    PS.nm = NS;
    for (int i = 0; i < NS; i++) cm_init(&PS.cm[i], ssz[i], 1023);
    mx_init(&PS.m1, NS + 1, 3 * 8, 0.002f);
    mx_init(&PS.m2, NS + 1, 3 * 8 * 33, 0.002f);
    apm_init(&PSG.a1, 1); /* unused placeholder */
    apm_init(&PS.a1, 3 * 256);
    /* sign: contexts */
    gsz[0] = 3; gsz[1] = 3 * 3 * 8 * 8; gsz[2] = 3 * 3 * 3 * 8; gsz[3] = 3 * 32 * 8;
    PSG.nm = 4;
    for (int i = 0; i < 4; i++) cm_init(&PSG.cm[i], gsz[i], 1023);
    mx_init(&PSG.m1, 5, 3 * 8, 0.002f);
    mx_init(&PSG.m2, 5, 3 * 3, 0.002f);
    apm_init(&PSG.a1, 3 * 8);
}

typedef struct { double sc[3], mag[3], sgn[3]; } Cost;

static int cv(const Rec *r, int m, int v) { return clampi(v - r->anc[m] + 16, 0, 31) + 1; }

static void code_scales(const Rec *r, Cost *c) {
    int order[3] = {0, 1, 2};
    for (int oi = 0; oi < 3; oi++) {
        int m = order[oi], R = r->R[m], NB = r->NB[m];
        double *colsum = calloc(NB, sizeof(double));
        for (int i = 0; i < R; i++) {
            int rowmax = 0; double rowsum = 0;
            for (int b = 0; b < NB; b++) {
                int v = scl(r, m, i, b);
                int L = b ? cv(r, m, scl(r, m, i, b - 1)) : 0, L2 = b > 1 ? cv(r, m, scl(r, m, i, b - 2)) : 0;
                int U = i ? cv(r, m, scl(r, m, i - 1, b)) : 0;
                int UL = i && b ? cv(r, m, scl(r, m, i - 1, b - 1)) : 0, UR = i && b + 1 < NB ? cv(r, m, scl(r, m, i - 1, b + 1)) : 0;
                int SIB = (m == 2) ? cv(r, 0, scl(r, 0, i, b)) : 0;
                int COL = i ? clampi((int)lround((colsum[b] / i - r->anc[m]) * 4.0) + 16, 0, 31) + 1 : 0;
                int RM = b ? cv(r, m, rowmax) : 0;
                int RS = b ? clampi((int)lround((rowsum / b - r->anc[m]) * 4.0) + 16, 0, 31) + 1 : 0;
                if (!F(6)) L = L2 = U = UL = UR = SIB = COL = RM = RS = 0;
                int node = 1;
                for (int k = 7; k >= 0; k--) {
                    int y = v >> k & 1;
                    size_t mb = (size_t)m * 256 + node, idx[NS];
                    idx[0] = mb;
                    idx[1] = mb * 33 + L;
                    idx[2] = mb * 33 + U;
                    idx[3] = (mb * 33 + L) * 33 + U;
                    idx[4] = (mb * 33 + SIB) * 33 + L;
                    idx[5] = (mb * 33 + L) * 33 + L2;
                    idx[6] = (mb * 33 + COL) * 33 + L;
                    idx[7] = (mb * 33 + RM) * 33 + L;
                    idx[8] = ((mb * 33 + UL) * 33 + U) * 33 + UR;
                    idx[9] = (mb * 33 + RS) * 33 + U;
                    c->sc[m] += code_bit(&PS, idx, m * 8 + (7 - k), (m * 8 + (7 - k)) * 33 + L, m * 256 + node, y);
                    node = node * 2 + y;
                }
                if (v > rowmax) rowmax = v;
                rowsum += v; colsum[b] += v;
            }
        }
        free(colsum);
    }
}

static void code_weights(const Rec *r, Cost *c) {
    int order[3] = {0, 2, 1};
    /* neuron statistics for w2: mean magnitude of row n of w1 and w3 -> 16 bins (filled after w1, w3) */
    int *neuq = calloc(r->R[0], sizeof(int));
    for (int oi = 0; oi < 3; oi++) {
        int m = order[oi], R = r->R[m], C = r->C[m], NB = r->NB[m];
        double *colsum = calloc(C, sizeof(double)); double tot = 0; long totn = 0;
        int *sibcolq = calloc(C, sizeof(int));
        if (m == 2) { /* full column means of w1 (decoded) */
            double *cs = calloc(C, sizeof(double)), g = 0;
            for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) cs[j] += nib(r, 0, i, j) & 7;
            for (int j = 0; j < C; j++) g += cs[j]; g /= (double)R * C;
            for (int j = 0; j < C; j++) sibcolq[j] = clampi((int)lround((cs[j] / R - g) * 20.0) + 8, 0, 15);
            free(cs);
        }
        if (m == 1) {
            double g = 0, *ns = calloc(r->R[0], sizeof(double));
            for (int i = 0; i < r->R[0]; i++) { for (int j = 0; j < r->C[0]; j++) ns[i] += (nib(r, 0, i, j) & 7) + (nib(r, 2, i, j) & 7); g += ns[i]; }
            g /= r->R[0];
            for (int i = 0; i < r->R[0]; i++) neuq[i] = clampi((int)lround((ns[i] - g) / g * 40.0) + 8, 0, 15);
            free(ns);
        }
        for (int i = 0; i < R; i++) {
            for (int b = 0; b < NB; b++) {
                int s = scl(r, m, i, b), srel = clampi(s - r->anc[m], -3, 3) + 3;
                int sx = clampi(s - r->anc[m], -7, 7) + 7;
                int sib_s = 0, sib_srel = 3, sib_bm = 8;
                if (m == 0 || m == 2) { int o = 2 - m; sib_s = scl(r, o, i, b); sib_srel = clampi(sib_s - r->anc[o], -3, 3) + 3; }
                int sibx = (m != 1) ? clampi(sib_s - r->anc[2 - m], -7, 7) + 7 : 7;
                if (m == 2 && F(0)) { sib_bm = 0; for (int k = 0; k < 32; k++) { int a = nib(r, 0, i, b * 32 + k) & 7; if (a > sib_bm) sib_bm = a; } }
                if (!F(5)) { sx = srel + 4; sibx = 7; }
                if (!F(0)) { sib_srel = 3; sibx = 7; }
                int pm = 8, mx = 8, pm2 = 8, zc = 0;
                for (int k = 0; k < 32; k++) {
                    int j = b * 32 + k, q = nib(r, m, i, j), a = q & 7;
                    int up = i ? nib(r, m, i - 1, j) & 7 : 8;
                    int ul = i && j ? nib(r, m, i - 1, j - 1) & 7 : 8, ur = i && j + 1 < C ? nib(r, m, i - 1, j + 1) & 7 : 8;
                    int sa = 8, sl = 8, sr_ = 8, sq = 0;
                    if (m == 2 && F(0)) { sq = nib(r, 0, i, j); sa = sq & 7; sl = j ? nib(r, 0, i, j - 1) & 7 : 8; sr_ = j + 1 < C ? nib(r, 0, i, j + 1) & 7 : 8; }
                    int colq = 8;
                    if (F(1) && i) colq = clampi((int)lround((colsum[j] / i - tot / totn) * 20.0) + 8, 0, 15);
                    int scq = (m == 2 && F(1)) ? sibcolq[j] : (m == 1 && F(3)) ? neuq[j] : 8;
                    int nq = (m == 1 && F(3)) ? clampi(neuq[j] / 2, 0, 8) : sib_srel;
                    int pos = F(4) ? k : 0;
                    if (!F(2)) { up = ul = ur = 8; pm2 = 8; }
                    int zcb = F(2) ? (zc > 7 ? 7 + (zc > 11) : zc) : 0;
                    int node = 1;
                    for (int t = 2; t >= 0; t--) {
                        int y = a >> t & 1;
                        size_t mb = (size_t)m * 8 + node, idx[NW];
                        idx[0] = mb * 7 + srel;
                        idx[1] = mb * 567 + (srel * 9 + pm) * 9 + mx;
                        idx[2] = ((mb * 7 + srel) * 32 + pos) * 9 + mx;
                        idx[3] = ((mb * 7 + srel) * 16 + colq) * 9 + mx;
                        idx[4] = (((mb * 7 + srel) * 9 + up) * 9 + pm) * 9 + mx;
                        idx[5] = (((mb * 7 + srel) * 9 + sa) * 7 + (m == 1 ? clampi(nq, 0, 6) : sib_srel)) * 9 + mx;
                        idx[6] = (((mb * 7 + srel) * 9 + sa) * 9 + sl) * 9 + sr_;
                        idx[7] = ((mb * 15 + sx) * 15 + sibx) * 9 + sa;
                        idx[8] = (((mb * 7 + srel) * 9 + sib_bm) * 9 + sa) * 9 + mx;
                        idx[9] = (((mb * 7 + srel) * 16 + colq) * 16 + scq) * 9 + pm;
                        idx[10] = (((mb * 7 + srel) * 9 + pm2) * 9 + pm) * 9 + mx;
                        idx[11] = ((mb * 7 + srel) * 9 + zcb) * 9 + mx;
                        idx[12] = (((mb * 7 + srel) * 9 + ul) * 9 + ur) * 9 + up;
                        int sel1 = ((m * 8 + node) * 7 + srel) * 9 + mx;
                        int sel2 = ((m * 8 + node) * 9 + sa) * 16 + (m == 1 ? scq : colq);
                        int actx = (((m * 8 + node) * 7 + srel) * 9 + mx) * 9 + sa;
                        c->mag[m] += code_bit(&PW, idx, sel1, sel2, actx, y);
                        node = node * 2 + y;
                    }
                    if (a) {
                        int y = q >> 3, us = i && (nib(r, m, i - 1, j) & 7) ? 1 + (nib(r, m, i - 1, j) >> 3) : 0;
                        int ss = (m == 2 && F(0) && sa && sa != 8) ? 1 + (sq >> 3) : 0;
                        if (!F(7)) us = ss = 0;
                        size_t idx[4];
                        idx[0] = m;
                        idx[1] = ((m * 3 + ss) * 8 + (sa & 7)) * 8 + a;
                        idx[2] = ((m * 3 + us) * 3 + 0) * 8 + a;
                        idx[3] = ((size_t)m * 32 + (F(7) ? k : 0)) * 8 + a;
                        c->sgn[m] += code_bit(&PSG, idx, m * 8 + a, m * 3 + ss, m * 8 + a, y);
                    }
                    pm2 = pm; pm = a; mx = (mx == 8 || a > mx) ? a : mx; zc += a == 0;
                    colsum[j] += a; tot += a; totn++;
                }
            }
        }
        free(colsum); free(sibcolq);
    }
    free(neuq);
}

static unsigned char *readall(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b;
}

static void mkrec(Rec *r, const unsigned char *b, char **a, const int *anc) {
    for (int m = 0; m < 3; m++) {
        r->s[m] = b + atol(a[m]); r->w[m] = b + atol(a[3 + m]);
        r->R[m] = atoi(a[6 + 2 * m]); r->NB[m] = atoi(a[7 + 2 * m]); r->C[m] = r->NB[m] * 32; r->anc[m] = anc[m];
    }
}

int main(int argc, char **argv) {
    const char *prime = NULL; int panc[3] = {0, 0, 0};
    int ai = 1;
    while (ai < argc && argv[ai][0] == '-') {
        if (!strcmp(argv[ai], "-f")) FM = strtoul(argv[ai + 1], 0, 0);
        else if (!strcmp(argv[ai], "-p")) prime = argv[ai + 1];
        else if (!strcmp(argv[ai], "-P")) sscanf(argv[ai + 1], "%d,%d,%d", &panc[0], &panc[1], &panc[2]);
        ai += 2;
    }
    if (argc - ai != 16) { fprintf(stderr, "usage, see source\n"); return 2; }
    size_t n; unsigned char *b = readall(argv[ai], &n);
    int anc[3] = {atoi(argv[ai + 13]), atoi(argv[ai + 14]), atoi(argv[ai + 15])};
    Rec r; mkrec(&r, b, argv + ai + 1, anc);
    init_models();
    if (prime) {
        size_t pn; unsigned char *pb = readall(prime, &pn); Rec pr; mkrec(&pr, pb, argv + ai + 1, panc);
        Cost pc; memset(&pc, 0, sizeof pc); code_scales(&pr, &pc); code_weights(&pr, &pc); free(pb);
    }
    Cost c; memset(&c, 0, sizeof c);
    code_scales(&r, &c); code_weights(&r, &c);
    printf("{\"fmask\": %u, \"primed\": %d", FM, prime != NULL);
    for (int m = 0; m < 3; m++) printf(", \"scale_bits_m%d\": %.1f, \"mag_bits_m%d\": %.1f, \"sign_bits_m%d\": %.1f, \"n_m%d\": %d, \"ns_m%d\": %d",
        m, c.sc[m], m, c.mag[m], m, c.sgn[m], m, r.R[m] * r.C[m], m, r.R[m] * r.NB[m]);
    printf("}\n");
    return 0;
}
