/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR2 kernel latency/throughput by stream length: loads records_<chunk>.bin
 * written by rechunk.py (same 16 real experts re-encoded at each chunk), puts
 * records in VRAM, checks every GPU decode against the stored raw slab, then
 * times kernel A+B per record with 1 and 8 concurrent streams.
 * usage: bench_nr2_rechunk.exe records_A.bin [records_B.bin ...] */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "wex_cuda.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

struct Rec { WexEntry e; std::vector<unsigned char> raw, rec; WexCudaNr2Plan plan;
             unsigned char *d_rec, *d_raw; uint32_t *d_off, *d_err; };

static double run(std::vector<Rec> &recs, const WexCudaNr2Tables &tables, int S, int iters) {
    std::vector<cudaStream_t> st(S);
    for (auto &s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a, 0));
    for (int it = 0; it < iters; it++)
        for (size_t k = 0; k < recs.size(); k++) {
            Rec &r = recs[k];
            CK(wex_cuda_launch_nr2(&r.plan, &tables, r.d_rec, r.d_off, r.d_raw, r.d_err, st[k % S]));
        }
    for (auto &s : st) { cudaEvent_t d; CK(cudaEventCreate(&d)); CK(cudaEventRecord(d, s)); CK(cudaStreamWaitEvent(0, d, 0)); }
    CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    for (auto &s : st) cudaStreamDestroy(s);
    return ms / (iters * (double)recs.size());
}

int main(int argc, char **argv) {
    CK(cudaSetDevice(0));
    const int buckets = WEX_CUDA_NR2_LOOKUP == 0;
    printf("[");
    for (int fi = 1; fi < argc; fi++) {
        FILE *f = fopen(argv[fi], "rb");
        if (!f) return 1;
        uint32_t mb, n;
        fread(&mb, 4, 1, f);
        std::vector<unsigned char> blob(mb);
        fread(blob.data(), 1, mb, f);
        fread(&n, 4, 1, f);
        unsigned char *base; CK(cudaMalloc(&base, wex_cuda_nr2_table_bytes(buckets)));
        CK(cudaMemcpy(base, blob.data(), WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice));
        WexCudaNr2Tables tables; CK(wex_cuda_build_nr2_tables(base, buckets, 0, &tables));
        std::vector<Rec> recs(n);
        uint64_t comp = 0, rawsum = 0;
        for (auto &r : recs) {
            uint32_t rb;
            fread(&r.e, sizeof(WexEntry), 1, f); fread(&rb, 4, 1, f);
            r.raw.resize(rb); fread(r.raw.data(), 1, rb, f);
            r.rec.resize(r.e.total_bytes); fread(r.rec.data(), 1, r.e.total_bytes, f);
            WexNr2Record view;
            if (wex_cuda_plan_nr2(&r.e, r.rec.data(), r.rec.size(), &view, &r.plan)) { fprintf(stderr, "plan %s\n", argv[fi]); return 1; }
            std::vector<uint32_t> off(r.plan.streams + 1u);
            wex_cuda_nr2_offsets(&view, &r.plan, off.data());
            uint64_t raw, aligned; wex_entry_raw_sizes(&r.e, &raw, &aligned);
            CK(cudaMalloc(&r.d_rec, r.rec.size())); CK(cudaMalloc(&r.d_raw, aligned));
            CK(cudaMalloc(&r.d_off, off.size() * 4)); CK(cudaMalloc(&r.d_err, 4));
            CK(cudaMemcpy(r.d_rec, r.rec.data(), r.rec.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(r.d_off, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
            CK(wex_cuda_launch_nr2(&r.plan, &tables, r.d_rec, r.d_off, r.d_raw, r.d_err, 0));
            uint32_t err; CK(cudaMemcpy(&err, r.d_err, 4, cudaMemcpyDeviceToHost));
            std::vector<unsigned char> got(raw);
            CK(cudaMemcpy(got.data(), r.d_raw, raw, cudaMemcpyDeviceToHost));
            for (int m = 0; m < 3; m++) {
                size_t sb = (size_t)r.e.scale_shapes[m][0] * r.e.scale_shapes[m][1];
                size_t wb = (size_t)r.e.weight_shapes[m][0] * r.e.weight_shapes[m][1];
                if (err || memcmp(got.data() + r.plan.scale_offset[m], r.raw.data() + r.e.scale_offsets[m], sb) ||
                    memcmp(got.data() + r.plan.weight_offset[m], r.raw.data() + r.e.weight_offsets[m], wb)) {
                    fprintf(stderr, "MISMATCH %s\n", argv[fi]); return 1;
                }
            }
            comp += r.e.total_bytes; rawsum += raw;
        }
        fclose(f);
        double one = run(recs, tables, 1, 20), eight = run(recs, tables, 8, 20);
        printf("%s{\"file\":\"%s\",\"weight_chunk\":%u,\"weight_blocks\":%u,\"compressed_mib\":%.2f,"
               "\"ms_1\":%.4f,\"ms_8\":%.4f,\"decoded_gib_s_8\":%.1f,\"all_identical\":true}\n",
               fi > 1 ? "," : "", argv[fi], recs[0].plan.chunk[1], recs[0].plan.block_base[1][3],
               comp / (double)n / 1048576.0, one, eight, rawsum / (double)n / (eight / 1e3) / 1073741824.0);
        for (auto &r : recs) { cudaFree(r.d_rec); cudaFree(r.d_raw); cudaFree(r.d_off); cudaFree(r.d_err); }
        cudaFree(base);
    }
    printf("]\n");
    return 0;
}
