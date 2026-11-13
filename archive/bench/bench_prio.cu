/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Contention micro-benchmark for DSV4_CUDA_STREAM_PRIORITY.
 *
 * A background thread decodes real WNR2 records straight into VRAM through the
 * WEX helper (wex_cuda_decode_to_device, as COLI_WEX_GPU_DIRECT=2 prefetch does),
 * while the main thread runs a sync-bound "inference" chain: KERNELS short
 * kernels on one stream, then cudaStreamSynchronize (the shape of decode
 * attention/mHC). Reports chain latency without background, with background at
 * equal priority, and with the chain stream at the device's highest priority.
 *
 * usage: bench_prio.exe CONTAINER.wex HELPER.dll [records=16] [reps=400] [bg=decode|copy|kernel]
 *   copy:   the background only uploads each record (cudaMemcpyAsync + sync)
 *   kernel: the background only runs busy kernels (+ sync), no copies
 * Reads only: header page, table, WNM2 model and a few records. */
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "wex_format.h"

typedef int (*RegFn)(const unsigned char *, size_t);
typedef int (*DevFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void busy(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f;
    for (int i = 0; i < iters; i++) a = a * b + 1e-7f;
    if (a == 12345.f) out[blockIdx.x] = a;   /* never true: keeps the loop */
}

static bool read_at(FILE *f, uint64_t at, void *dst, size_t n) {
    return _fseeki64(f, (long long)at, SEEK_SET) == 0 && fread(dst, 1, n, f) == n;
}

static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t)(p * (v.size() - 1))];
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s CONTAINER HELPER [records] [reps]\n", argv[0]); return 2; }
    int nrec = argc > 3 ? atoi(argv[3]) : 16, reps = argc > 4 ? atoi(argv[4]) : 400;
    const char *bgmode = argc > 5 ? argv[5] : "decode";
    HMODULE h = LoadLibraryA(argv[2]);
    if (!h) { fprintf(stderr, "helper load failed\n"); return 1; }
    RegFn reg = (RegFn)GetProcAddress(h, "wex_cuda_register_model");
    DevFn dev = (DevFn)GetProcAddress(h, "wex_cuda_decode_to_device");
    if (!reg || !dev) { fprintf(stderr, "helper lacks WNR2 device decode\n"); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open failed\n"); return 1; }
    _fseeki64(f, 0, SEEK_END);
    uint64_t file_bytes = (uint64_t)_ftelli64(f);
    static unsigned char page[WEX_ALIGN];
    if (!read_at(f, 0, page, WEX_ALIGN)) return 1;
    const WexHeader *hdr = (const WexHeader *)page;
    size_t table_bytes = 0, aligned_table = 0;
    if (wex_validate_header(hdr, file_bytes, hdr->num_layers, hdr->num_experts, &table_bytes, &aligned_table)) {
        fprintf(stderr, "bad header\n"); return 1;
    }
    std::vector<WexEntry> table((size_t)hdr->num_layers * hdr->num_experts);
    if (!read_at(f, hdr->table_offset, table.data(), table.size() * sizeof(WexEntry))) return 1;
    uint64_t model_at; uint32_t model_bytes, model_id;
    if (wex_nr2_model_location(hdr, page, &model_at, &model_bytes, &model_id)) { fprintf(stderr, "no WNM2\n"); return 1; }
    std::vector<unsigned char> model(model_bytes);
    if (!read_at(f, model_at, model.data(), model_bytes) || reg(model.data(), model_bytes)) {
        fprintf(stderr, "model register failed\n"); return 1;
    }

    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::vector<const WexEntry *> ent;
    std::vector<unsigned char *> rec;
    size_t step = table.size() / (size_t)nrec;
    for (size_t i = 0; i < table.size() && (int)ent.size() < nrec; i += step) {
        const WexEntry *e = &table[i];
        if (e->flags != WEX_ENTRY_NEURAL_RANS2) continue;
        unsigned char *p; CK(cudaMallocHost((void **)&p, e->total_bytes));
        if (!read_at(f, e->file_offset, p, e->total_bytes)) return 1;
        ent.push_back(e); rec.push_back(p);
    }
    fclose(f);
    if (ent.empty()) { fprintf(stderr, "no WNR2 records\n"); return 1; }
    unsigned char *ds[3], *dw[3];
    for (int m = 0; m < 3; m++) {
        size_t sb = 0, wb = 0;
        for (auto e : ent) {
            sb = std::max(sb, (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1]);
            wb = std::max(wb, (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1]);
        }
        CK(cudaMalloc(&ds[m], sb)); CK(cudaMalloc(&dw[m], wb));
    }
    /* warm the helper (workspace, model tables) */
    for (size_t k = 0; k < ent.size(); k++)
        if (dev(ent[k], rec[k], 0, ds, dw)) { fprintf(stderr, "decode failed\n"); return 1; }

    int least = 0, greatest = 0;
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t s_def, s_hi;
    CK(cudaStreamCreateWithPriority(&s_def, cudaStreamNonBlocking, least));
    CK(cudaStreamCreateWithPriority(&s_hi, cudaStreamNonBlocking, greatest));
    float *out; CK(cudaMalloc(&out, 4096 * sizeof(float)));
    const int KERNELS = 40, GRID = prop.multiProcessorCount * 2, ITERS = 2000;

    auto chain = [&](cudaStream_t s) {
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < KERNELS; k++) busy<<<GRID, 256, 0, s>>>(out, ITERS);
        CK(cudaStreamSynchronize(s));
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    };
    for (int i = 0; i < 50; i++) chain(s_def);

    std::atomic<bool> run{false}, stop{false};
    std::atomic<long long> decodes{0};
    std::thread bg([&] {
        CK(cudaSetDevice(0));
        size_t k = 0;
        while (!stop.load()) {
            if (!run.load()) { std::this_thread::yield(); continue; }
            if (dev(ent[k % ent.size()], rec[k % ent.size()], 0, ds, dw)) { fprintf(stderr, "bg decode failed\n"); exit(1); }
            decodes++; k++;
        }
    });

    std::vector<double> idle, eq, hi;
    double eq_s = 0, hi_s = 0; long long eq_d = 0, hi_d = 0;
    for (int i = 0; i < reps; i++) idle.push_back(chain(s_def));
    run = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (int round = 0; round < 8; round++)          /* alternate to cancel drift */
        for (int which = 0; which < 2; which++) {
            cudaStream_t s = which ? s_hi : s_def;
            long long d0 = decodes.load();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps / 8; i++) (which ? hi : eq).push_back(chain(s));
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            (which ? hi_s : eq_s) += sec; (which ? hi_d : eq_d) += decodes.load() - d0;
        }
    run = false; stop = true; bg.join();

    printf("{\"bg\":\"%s\",\"gpu\":\"%s\",\"records\":%zu,\"record_mib\":%.2f,\"chain_kernels\":%d,\"grid\":%d,"
           "\"priority_range\":[%d,%d],\n", bgmode, prop.name, ent.size(), ent[0]->total_bytes / 1048576.0,
           KERNELS, GRID, least, greatest);
    printf(" \"idle_us\":{\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f},\n", pct(idle, .5), pct(idle, .9), pct(idle, .99));
    printf(" \"bg_equal_us\":{\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"decodes_per_s\":%.1f},\n",
           pct(eq, .5), pct(eq, .9), pct(eq, .99), eq_d / eq_s);
    printf(" \"bg_high_us\":{\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"decodes_per_s\":%.1f}}\n",
           pct(hi, .5), pct(hi, .9), pct(hi, .99), hi_d / hi_s);
    return 0;
}
