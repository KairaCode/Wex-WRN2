// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// Host <-> device copy bandwidth with pinned and pageable host memory, for transfer sizes that match
// expert records (1, 8, 32 MiB) and a large one (256 MiB). Prints one JSON object.
// Build: nvcc -O2 h2d.cu -o h2d
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "h2d: %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

static float timed(void *dst, const void *src, size_t n, cudaMemcpyKind kind, int reps) {
    cudaEvent_t a, b; float best = 1e30f, ms;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); cudaMemcpy(dst, src, n, kind); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (float)(n / (best / 1e3) / 1e9);   /* GB/s */
}

int main(void) {
    int dev = 0; cudaDeviceProp p;
    CK(cudaSetDevice(dev)); CK(cudaGetDeviceProperties(&p, dev));
    const size_t sizes[4] = {1u << 20, 8u << 20, 32u << 20, 256u << 20};
    size_t max = sizes[3];
    void *d, *pinned, *pageable = malloc(max);
    if (!pageable) return 1;
    CK(cudaMalloc(&d, max)); CK(cudaMallocHost(&pinned, max));
    printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"integrated\": %d, \"rows\": [", p.name, p.major, p.minor,
           p.integrated);
    for (int i = 0; i < 4; i++) {
        size_t n = sizes[i];
        printf("%s{\"mib\": %zu, \"h2d_pinned_gbs\": %.2f, \"d2h_pinned_gbs\": %.2f, "
               "\"h2d_pageable_gbs\": %.2f, \"d2h_pageable_gbs\": %.2f}", i ? ", " : "", n >> 20,
               timed(d, pinned, n, cudaMemcpyHostToDevice, 10), timed(pinned, d, n, cudaMemcpyDeviceToHost, 10),
               timed(d, pageable, n, cudaMemcpyHostToDevice, 5), timed(pageable, d, n, cudaMemcpyDeviceToHost, 5));
    }
    printf("]}\n");
    cudaFree(d); cudaFreeHost(pinned); free(pageable);
    return 0;
}
