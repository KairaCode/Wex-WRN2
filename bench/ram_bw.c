// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// RAM bandwidth: read (sum), write (fill) and copy over a buffer far larger than the caches, on 1 thread
// and on all threads (OpenMP). Prints one JSON object. Build: gcc -O2 -fopenmp ram_bw.c -o ram_bw
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return omp_get_wtime(); }

static double run(int threads, int kind, uint64_t *a, uint64_t *b, size_t n, int reps) {
    double best = 1e30;
    volatile uint64_t sink = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        if (kind == 0) {            /* read */
            uint64_t s = 0;
#pragma omp parallel for num_threads(threads) reduction(+:s) schedule(static)
            for (size_t i = 0; i < n; i++) s += a[i];
            sink += s;
        } else if (kind == 1) {     /* write */
#pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i = 0; i < n; i++) a[i] = i;
        } else {                    /* copy */
#pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i = 0; i < n; i++) b[i] = a[i];
        }
        double t = now() - t0;
        if (t < best) best = t;
    }
    (void)sink;
    double bytes = (double)n * sizeof(uint64_t) * (kind == 2 ? 2 : 1);
    return bytes / best / 1e9;      /* GB/s */
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 1024;   /* per buffer */
    size_t n = mb * 1024 * 1024 / sizeof(uint64_t);
    uint64_t *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b);
    if (!a || !b) { fprintf(stderr, "ram_bw: cannot allocate 2 x %zu MiB\n", mb); return 1; }
    memset(a, 1, n * sizeof *a); memset(b, 0, n * sizeof *b);   /* fault the pages in */
    int all = omp_get_max_threads();
    const char *names[3] = {"read", "write", "copy"};
    printf("{\"buffer_mib\": %zu, \"threads_all\": %d", mb, all);
    for (int k = 0; k < 3; k++)
        printf(", \"%s_gbs_1t\": %.2f, \"%s_gbs_all\": %.2f", names[k], run(1, k, a, b, n, 3),
               names[k], run(all, k, a, b, n, 5));
    printf("}\n");
    free(a); free(b);
    return 0;
}
