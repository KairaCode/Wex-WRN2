/* SPDX-License-Identifier: Apache-2.0 */
/* CPU decode speed on a pack (measure.py --pack): WNR4 reference decoder
 * (tiles in parallel, OpenMP) at 1..T threads, and WNR3's CPU reference
 * decoder (wex_decode_nr2_record, one thread) for scale.
 *   bench_cpu PACK [records] [max_threads] */
#include "../../../wnr4/wnr4.h"
#include "../../../c/wex_decode.c"
#include <stdio.h>
#include <omp.h>

#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    char magic[4]; uint32_t ver, n, kind, G, geo[13], l4, l2, i;
    REQ(f && fread(magic, 4, 1, f) == 1 && !memcmp(magic, "WP4K", 4));
    REQ(fread(&ver, 4, 1, f) == 1 && fread(&n, 4, 1, f) == 1 && fread(&kind, 4, 1, f) == 1 && fread(&G, 4, 1, f) == 1);
    REQ(fread(geo, 4, 13, f) == 13 && fread(&l4, 4, 1, f) == 1 && fread(&l2, 4, 1, f) == 1);
    WexEntry e0; REQ(fread(&e0, 72, 1, f) == 1);
    unsigned char *wnm4 = malloc(l4), *wnm2 = malloc(l2 + 1);
    REQ(fread(wnm4, 1, l4, f) == l4); if (l2) REQ(fread(wnm2, 1, l2, f) == l2);
    uint32_t want_n = argc > 2 ? (uint32_t)atoi(argv[2]) : 8, tmax = argc > 3 ? (uint32_t)atoi(argv[3]) : 16;
    if (want_n > n) want_n = n;
    unsigned char **w4 = calloc(want_n, sizeof(void *)), **w3 = calloc(want_n, sizeof(void *)), **want = calloc(want_n, sizeof(void *));
    uint32_t *len4 = calloc(want_n, 4), *len3 = calloc(want_n, 4), *own = calloc(want_n, 4), rawlen = 0;
    for (i = 0; i < want_n; i++) {
        uint32_t h[4]; REQ(fread(h, 4, 4, f) == 4);
        own[i] = h[0]; len4[i] = h[1]; len3[i] = h[2]; rawlen = h[3];
        w4[i] = malloc(h[1]); w3[i] = malloc(h[2] + 1); want[i] = malloc(h[3]);
        REQ(fread(w4[i], 1, h[1], f) == h[1]); if (h[2]) REQ(fread(w3[i], 1, h[2], f) == h[2]); REQ(fread(want[i], 1, h[3], f) == h[3]);
    }
    fclose(f);
    Wnr4Geo g; REQ(!wnr4_geo(&g, geo[0], geo + 1, geo + 4, geo + 7, geo + 10));
    Wnr4Model m; REQ(!wnr4_model_open(wnm4, l4, &m));
    unsigned char *out = malloc(rawlen);
    printf("pack %s: %u records of %u B, kind %s, G %u; %d CPUs\n", argv[1], want_n, rawlen, kind ? "NV" : "MX", G, omp_get_num_procs());
    for (uint32_t t = 1; t <= tmax; t *= 2) {
        double best = 1e30;
        for (int it = 0; it < 3; it++) {
            double t0 = omp_get_wtime();
            for (i = 0; i < want_n; i++) REQ(wnr4_decode(&m, &g, w4[i], len4[i], own[i], out, (int)t) == 0);
            double dt = omp_get_wtime() - t0; if (dt < best) best = dt;
        }
        REQ(!memcmp(out, want[want_n - 1], rawlen));
        printf("WNR4 CPU %2u threads: %7.1f ms/record  %7.3f GB/s\n", t, best * 1e3 / want_n, (double)rawlen * want_n / best / 1e9);
    }
    {   /* batch: every tile of every loaded record in one parallel loop */
        unsigned char **outs = calloc(want_n, sizeof(void *)); size_t *tot = calloc(want_n, sizeof(size_t));
        uint32_t *st = calloc(want_n, 4);
        for (i = 0; i < want_n; i++) { outs[i] = malloc(rawlen); tot[i] = len4[i]; }
        for (uint32_t t = 1; t <= 2 * tmax; t *= 2) {
            double best = 1e30;
            for (int it = 0; it < 3; it++) {
                double t0 = omp_get_wtime();
                REQ(wnr4_decode_batch(&m, &g, (int)want_n, (const unsigned char *const *)w4, tot, own, outs, st, (int)t) == 0);
                double dt = omp_get_wtime() - t0; if (dt < best) best = dt;
            }
            for (i = 0; i < want_n; i++) REQ(!memcmp(outs[i], want[i], rawlen));
            printf("WNR4 CPU batch of %u, %2u threads: %7.3f GB/s\n", want_n, t, (double)rawlen * want_n / best / 1e9);
        }
    }
    if (l2) {
        WexNr2Model model; REQ(!wex_nr2_model_open(wnm2, l2, &model));
        WexEntry e = e0; uint64_t raw, aligned;
        e.flags = WEX_ENTRY_NEURAL_RANS3;
        REQ(!wex_entry_raw_sizes(&e, &raw, &aligned));
        unsigned char *slab = malloc((size_t)aligned);
        uint32_t k = want_n < 4 ? want_n : 4;
        double t0 = omp_get_wtime();
        for (i = 0; i < k; i++) {
            e.total_bytes = len3[i];
            memcpy(slab + wex_record_read_offset(&e), w3[i], len3[i]);
            REQ(wex_decode_nr2_record(&e, slab, (size_t)aligned, &model) == 0);
            REQ(!memcmp(slab, want[i], rawlen));
        }
        double dt = omp_get_wtime() - t0;
        printf("WNR3 CPU reference, 1 thread: %7.1f ms/record  %7.3f GB/s\n", dt * 1e3 / k, (double)rawlen * k / dt / 1e9);
    }
    return 0;
}
