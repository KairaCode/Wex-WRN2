/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Pinned H2D bandwidth for one 11 MiB expert record: one copy + sync each
 * (the helper's pattern) against 32 copies back to back + one sync. */
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
int main() {
    const size_t n = 11u << 20; const int reps = 64;
    unsigned char *h, *d; cudaMallocHost(&h, n); cudaMalloc(&d, n);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    for (int i = 0; i < 8; i++) cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) { cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s); cudaStreamSynchronize(s); }
    double a = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    double b = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("{\"bytes\":%zu,\"sync_each_gib_s\":%.2f,\"sync_each_ms\":%.3f,\"batched_gib_s\":%.2f,\"err\":\"%s\"}\n",
           n, reps * (double)n / a / 1073741824.0, 1e3 * a / reps, reps * (double)n / b / 1073741824.0,
           cudaGetErrorString(cudaGetLastError()));
}
