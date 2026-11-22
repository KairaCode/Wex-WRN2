/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-I4 on real int4-g64 expert records (wnr4/wnr4_i4.h): trains a model on the records before "--", codes the
 * records after it at G, decodes each on 1 and on THREADS threads and compares, flips one payload bit (must be
 * refused or decode to other bytes) and prints the ratio (records padded to 4096 B over raw) and the speeds.
 *   gcc -O2 -march=haswell -fopenmp research/wnr4-i4-20261122/bench_i4.c -o bench_i4
 *   bench_i4 GEOMETRY planar|pairs G THREADS train_*.rec -- test_*.rec
 * GEOMETRY: ROWSxCOLS of each matrix joined by commas, e.g. 640x2560,640x2560,2560x640. */
#include "../../wnr4/wnr4_i4.h"
#include <stdio.h>
#include <omp.h>

static unsigned char *load(const char *p, size_t want) {
    FILE *f = fopen(p, "rb");
    unsigned char *b;
    size_t n;
    if (!f) return NULL;
    b = malloc(want + 1); n = fread(b, 1, want + 1, f); fclose(f);
    if (n != want) { free(b); return NULL; }
    return b;
}
int main(int argc, char **argv) {
    uint32_t rows[3], cols[3], nmat = 0, G, i;
    const char *p;
    Wnr4I4Geo g;
    Wnr4I4Model md;
    int th, a = 5, ntr = 0, n = 0, fails = 0;
    double S = 0, R = 0, te = 0, td1 = 0, tdn = 0, t0;
    long long esc = 0;
    uint64_t *cnt;
    unsigned char *blob, *rec, *out;
    size_t cap;
    if (argc < 7) { fprintf(stderr, "usage: bench_i4 GEOMETRY planar|pairs G THREADS train... -- test...\n"); return 2; }
    for (p = argv[1]; *p && nmat < 3u;) {
        char *e;
        rows[nmat] = (uint32_t)strtoul(p, &e, 10);
        if (*e != 'x') return 2;
        cols[nmat++] = (uint32_t)strtoul(e + 1, &e, 10);
        p = *e == ',' ? e + 1 : e;
    }
    if (wnr4_i4_geo(&g, nmat, rows, cols, !strcmp(argv[2], "pairs"))) { fprintf(stderr, "bad geometry\n"); return 2; }
    G = (uint32_t)atoi(argv[3]); th = atoi(argv[4]);
    cnt = calloc(wnr4_i4_counts_words(), 8);
    t0 = omp_get_wtime();
    for (; a < argc && strcmp(argv[a], "--"); a++) {
        unsigned char *r = load(argv[a], g.raw);
        if (!r || wnr4_i4_count(&g, r, cnt)) { fprintf(stderr, "bad %s\n", argv[a]); return 1; }
        free(r); ntr++;
    }
    blob = malloc(WNR4_I4_MODEL_BYTES);
    wnr4_i4_model_build(cnt, blob);
    if (wnr4_i4_model_open(blob, WNR4_I4_MODEL_BYTES, &md)) { fprintf(stderr, "model refused\n"); return 1; }
    printf("trained on %d records in %.1f s, model %08x\n", ntr, omp_get_wtime() - t0, md.id);
    cap = (size_t)g.raw * 4u + 65536u; rec = malloc(cap); out = malloc(g.raw);
    for (a++; a < argc; a++) {
        unsigned char *src = load(argv[a], g.raw);
        Wnr4I4Side sd;
        double t1;
        size_t nb;
        uint32_t e1, e2;
        int bad, det;
        if (!src) { fprintf(stderr, "bad %s\n", argv[a]); return 1; }
        t1 = omp_get_wtime(); nb = wnr4_i4_encode(&md, &g, src, 0x00010002u, G, rec, cap); te += omp_get_wtime() - t1;
        if (!nb) { fprintf(stderr, "encode failed %s\n", argv[a]); return 1; }
        memset(out, 0xA5, g.raw);
        t1 = omp_get_wtime(); e1 = wnr4_i4_decode(&md, &g, rec, nb, 0x00010002u, out, 1); td1 += omp_get_wtime() - t1;
        bad = e1 || memcmp(out, src, g.raw);
        memset(out, 0x5A, g.raw);
        t1 = omp_get_wtime(); e2 = wnr4_i4_decode(&md, &g, rec, nb, 0x00010002u, out, th); tdn += omp_get_wtime() - t1;
        bad |= e2 || memcmp(out, src, g.raw);
        rec[nb / 2] ^= 0x10;
        det = wnr4_i4_decode(&md, &g, rec, nb, 0x00010002u, out, th) != 0 || memcmp(out, src, g.raw) != 0;
        if (!wnr4_i4_side(&g, src, &sd)) {
            for (i = 0; i < g.cells[0] + g.cells[1] + g.cells[2]; i++) esc += sd.k[i] == WNR4_I4_ESC;
            wnr4_i4_side_free(&sd);
        }
        if (bad || !det) { fails++; printf("FAIL %s (err %u %u, flip noticed %d)\n", argv[a], e1, e2, det); }
        S += (double)nb; R += (double)g.raw; n++; free(src);
    }
    printf("SUMMARY %s %s G %u records %d fails %d stored/raw %.5f escaped scales/record %.1f encode1 %.1f decode1 %.1f"
           " decode%d %.1f MB/s\n", argv[1], argv[2], G, n, fails, S / R, (double)esc / (n ? n : 1), R / 1e6 / te,
           R / 1e6 / td1, th, R / 1e6 / tdn);
    return fails != 0;
}
