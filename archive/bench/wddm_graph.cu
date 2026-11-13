/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* RTX 5090 under WDDM: copy engines, and a 40-kernel chain + sync launched as
 * plain stream launches vs one CUDA graph launch (p50/p90 over 2000 runs). */
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
__global__ void busy(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f;
    for (int i = 0; i < iters; i++) a = a * b + 1e-7f;
    if (a == 12345.f) out[blockIdx.x] = a;
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    float *out; cudaMalloc(&out, 4096 * 4);
    const int K = 40, G = p.multiProcessorCount * 2;
    auto pct = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[(size_t)(q * (v.size() - 1))]; };
    for (int iters : {50, 2000}) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < K; k++) busy<<<G, 256, 0, s>>>(out, iters);
        cudaStreamEndCapture(s, &g);
        cudaGraphInstantiate(&ge, g, 0);
        std::vector<double> a, b;
        for (int r = 0; r < 2200; r++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < K; k++) busy<<<G, 256, 0, s>>>(out, iters);
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::steady_clock::now();
            cudaGraphLaunch(ge, s);
            cudaStreamSynchronize(s);
            auto t2 = std::chrono::steady_clock::now();
            if (r >= 200) {
                a.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                b.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
            }
        }
        printf("{\"kernel_iters\":%d,\"stream_us\":[%.1f,%.1f],\"graph_us\":[%.1f,%.1f]}\n", iters, pct(a, .5), pct(a, .9), pct(b, .5), pct(b, .9));
    }
    printf("{\"gpu\":\"%s\",\"asyncEngineCount\":%d,\"concurrentKernels\":%d,\"tccDriver\":%d,\"driver\":%d,\"runtime\":%d,\"err\":\"%s\"}\n",
           p.name, p.asyncEngineCount, p.concurrentKernels, p.tccDriver, drv, rt, cudaGetErrorString(cudaGetLastError()));
}
