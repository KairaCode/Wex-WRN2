/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Prototype of a batched WNR2 device decode: N records in pinned RAM (one
 * layer's misses) go to VRAM through K = N lanes (own copy stream, work stream,
 * device workspace), H2D + decode pipelined, ONE host sync per batch, instead
 * of today's helper call per record (H2D, kernels, commit, two syncs).
 * Reports wall ms per record (host clock, sync included) by batch size, and
 * checks every decoded slab against a reference decode.
 *
 * usage: bench_nr2_batch.exe CONTAINER.wex [records=32] [rounds=40] [contend_rate=0]
 * contend_rate > 0: also run a latency-critical chain (40 short kernels + sync)
 * while a background thread decodes contend_rate records/s, either one record
 * per batch or 6 per batch, and report the chain's p50/p90 for each. */
#include <thread>
#include <atomic>
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "wex_cuda.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static bool read_at(FILE *f, uint64_t at, void *dst, size_t n) {
    return _fseeki64(f, (long long)at, SEEK_SET) == 0 && fread(dst, 1, n, f) == n;
}

struct Rec { const WexEntry *e; unsigned char *host; uint32_t *host_off; WexCudaNr2Plan plan; uint64_t raw; };
struct Lane {
    cudaStream_t copy, work; cudaEvent_t landed;
    unsigned char *record, *raw; uint32_t *offsets, *error;
};

__global__ void busy(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f;
    for (int i = 0; i < iters; i++) a = a * b + 1e-7f;
    if (a == 12345.f) out[blockIdx.x] = a;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int nrec = argc > 2 ? atoi(argv[2]) : 32, rounds = argc > 3 ? atoi(argv[3]) : 40;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    _fseeki64(f, 0, SEEK_END);
    uint64_t file_bytes = (uint64_t)_ftelli64(f);
    static unsigned char page[WEX_ALIGN];
    if (!read_at(f, 0, page, WEX_ALIGN)) return 1;
    const WexHeader *hdr = (const WexHeader *)page;
    size_t tb, atb;
    if (wex_validate_header(hdr, file_bytes, hdr->num_layers, hdr->num_experts, &tb, &atb)) return 1;
    std::vector<WexEntry> table((size_t)hdr->num_layers * hdr->num_experts);
    if (!read_at(f, hdr->table_offset, table.data(), table.size() * sizeof(WexEntry))) return 1;
    uint64_t mat; uint32_t mbytes, mid;
    std::vector<unsigned char> blob;
    if (wex_nr2_model_location(hdr, page, &mat, &mbytes, &mid)) return 1;
    blob.resize(mbytes);
    if (!read_at(f, mat, blob.data(), mbytes)) return 1;

    CK(cudaSetDevice(0));
    const int buckets = WEX_CUDA_NR2_LOOKUP == 0;
    unsigned char *base; CK(cudaMalloc(&base, wex_cuda_nr2_table_bytes(buckets)));
    CK(cudaMemcpy(base, blob.data(), WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice));
    WexCudaNr2Tables tables;
    CK(wex_cuda_build_nr2_tables(base, buckets, 0, &tables));
    CK(cudaDeviceSynchronize());

    std::vector<Rec> recs;
    size_t step = table.size() / (size_t)nrec, max_rec = 0, max_raw = 0, max_streams = 0;
    for (size_t i = 0; i < table.size() && (int)recs.size() < nrec; i += step) {
        const WexEntry *e = &table[i];
        if (e->flags != WEX_ENTRY_NEURAL_RANS2) continue;
        Rec r; r.e = e;
        uint64_t aligned;
        if (wex_entry_raw_sizes(e, &r.raw, &aligned)) return 1;
        CK(cudaMallocHost((void **)&r.host, e->total_bytes));
        if (!read_at(f, e->file_offset, r.host, e->total_bytes)) return 1;
        WexNr2Record view;
        if (wex_cuda_plan_nr2(e, r.host, e->total_bytes, &view, &r.plan)) return 1;
        CK(cudaMallocHost((void **)&r.host_off, (r.plan.streams + 1u) * 4u));
        wex_cuda_nr2_offsets(&view, &r.plan, r.host_off);
        max_rec = std::max(max_rec, (size_t)e->total_bytes);
        max_raw = std::max(max_raw, (size_t)aligned);
        max_streams = std::max(max_streams, (size_t)r.plan.streams + 1u);
        recs.push_back(r);
    }
    fclose(f);

    /* reference: each record decoded alone, synchronously */
    std::vector<std::vector<unsigned char>> ref(recs.size());
    {
        unsigned char *dr, *draw; uint32_t *doff, *derr;
        CK(cudaMalloc(&dr, max_rec)); CK(cudaMalloc(&draw, max_raw));
        CK(cudaMalloc(&doff, max_streams * 4u)); CK(cudaMalloc(&derr, 4));
        for (size_t k = 0; k < recs.size(); k++) {
            Rec &r = recs[k];
            CK(cudaMemcpy(dr, r.host, r.e->total_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(doff, r.host_off, (r.plan.streams + 1u) * 4u, cudaMemcpyHostToDevice));
            CK(wex_cuda_launch_nr2(&r.plan, &tables, dr, doff, draw, derr, 0));
            uint32_t err; CK(cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost));
            if (err) { fprintf(stderr, "reference decode error\n"); return 1; }
            ref[k].resize(r.raw);
            CK(cudaMemcpy(ref[k].data(), draw, r.raw, cudaMemcpyDeviceToHost));
        }
        cudaFree(dr); cudaFree(draw); cudaFree(doff); cudaFree(derr);
    }

    const int MAXK = 8;
    std::vector<Lane> lanes(MAXK);
    for (auto &l : lanes) {
        CK(cudaStreamCreateWithFlags(&l.copy, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&l.work, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&l.landed, cudaEventDisableTiming));
        CK(cudaMalloc(&l.record, max_rec)); CK(cudaMalloc(&l.raw, max_raw));
        CK(cudaMalloc(&l.offsets, max_streams * 4u)); CK(cudaMalloc(&l.error, 4));
    }
    std::vector<uint32_t> host_err(MAXK);
    uint32_t *pinned_err; CK(cudaMallocHost((void **)&pinned_err, MAXK * 4u));

    printf("{\"records\":%zu,\"record_mib\":%.2f,\"rounds\":%d,\"batches\":[", recs.size(), recs[0].e->total_bytes / 1048576.0, rounds);
    int sizes[] = {1, 2, 4, 6, 8};
    for (int si = 0; si < 5; si++) {
        int N = sizes[si];
        std::vector<double> ms;
        size_t next = 0; long checked = 0;
        for (int round = 0; round < rounds; round++) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<size_t> which(N);
            for (int k = 0; k < N; k++) {
                Lane &l = lanes[k];
                Rec &r = recs[which[k] = next++ % recs.size()];
                CK(cudaMemcpyAsync(l.offsets, r.host_off, (r.plan.streams + 1u) * 4u, cudaMemcpyHostToDevice, l.copy));
                CK(cudaMemcpyAsync(l.record, r.host, r.e->total_bytes, cudaMemcpyHostToDevice, l.copy));
                CK(cudaEventRecord(l.landed, l.copy));
                CK(cudaStreamWaitEvent(l.work, l.landed, 0));
                CK(wex_cuda_launch_nr2(&r.plan, &tables, l.record, l.offsets, l.raw, l.error, l.work));
                CK(cudaMemcpyAsync(pinned_err + k, l.error, 4, cudaMemcpyDeviceToHost, l.work));
            }
            for (int k = 0; k < N; k++) CK(cudaStreamSynchronize(lanes[k].work));
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            /* verify outside the timed region, every 8th round */
            if (round % 8 == 0)
                for (int k = 0; k < N; k++) {
                    if (pinned_err[k]) { fprintf(stderr, "batch decode error\n"); return 1; }
                    std::vector<unsigned char> got(recs[which[k]].raw);
                    CK(cudaMemcpy(got.data(), lanes[k].raw, got.size(), cudaMemcpyDeviceToHost));
                    if (memcmp(got.data(), ref[which[k]].data(), got.size())) { fprintf(stderr, "MISMATCH\n"); return 1; }
                    checked++;
                }
        }
        std::sort(ms.begin(), ms.end());
        double med = ms[ms.size() / 2];
        printf("%s{\"batch\":%d,\"ms_per_batch_p50\":%.3f,\"ms_per_record_p50\":%.3f,\"checked_identical\":%ld}",
               si ? "," : "", N, med, med / N, checked);
    }
    printf("]");
    double rate = argc > 4 ? atof(argv[4]) : 0;
    if (rate > 0) {
        cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
        cudaStream_t fg; CK(cudaStreamCreateWithFlags(&fg, cudaStreamNonBlocking));
        float *out; CK(cudaMalloc(&out, 4096 * 4));
        auto chain = [&] {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 40; k++) busy<<<prop.multiProcessorCount * 2, 256, 0, fg>>>(out, 2000);
            CK(cudaStreamSynchronize(fg));
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        };
        auto pct = [](std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; };
        std::vector<double> idle;
        for (int i = 0; i < 300; i++) idle.push_back(chain());
        printf(",\"contention\":{\"rate_per_s\":%.0f,\"idle_us\":[%.1f,%.1f]", rate, pct(idle, .5), pct(idle, .9));
        int modes[] = {1, 6};
        for (int mi = 0; mi < 2; mi++) {
            int N = modes[mi];
            std::atomic<bool> stop{false}; std::atomic<long> done{0};
            std::thread bg([&] {
                CK(cudaSetDevice(0));
                size_t next = 0;
                auto start = std::chrono::steady_clock::now();
                while (!stop.load()) {
                    for (int k = 0; k < N; k++) {
                        Lane &l = lanes[k]; Rec &r = recs[next++ % recs.size()];
                        CK(cudaMemcpyAsync(l.offsets, r.host_off, (r.plan.streams + 1u) * 4u, cudaMemcpyHostToDevice, l.copy));
                        CK(cudaMemcpyAsync(l.record, r.host, r.e->total_bytes, cudaMemcpyHostToDevice, l.copy));
                        CK(cudaEventRecord(l.landed, l.copy));
                        CK(cudaStreamWaitEvent(l.work, l.landed, 0));
                        CK(wex_cuda_launch_nr2(&r.plan, &tables, l.record, l.offsets, l.raw, l.error, l.work));
                    }
                    for (int k = 0; k < N; k++) CK(cudaStreamSynchronize(lanes[k].work));
                    long d = done += N;
                    auto due = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(d / rate));
                    while (!stop.load() && std::chrono::steady_clock::now() < due) std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            });
            std::vector<double> lat;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 1500; i++) lat.push_back(chain());
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            stop = true; bg.join();
            printf(",\"batch%d_us\":[%.1f,%.1f],\"batch%d_achieved_per_s\":%.0f", N, pct(lat, .5), pct(lat, .9), N, done / sec);
        }
        printf("}");
    }
    printf("}\n");
    return 0;
}
