/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Pure GPU throughput of the WNR2 decoder (kernel A scales + kernel B weights)
 * with the compressed record ALREADY in VRAM: no PCIe, no helper, no syncs per
 * record. Answers: if experts stayed compressed in VRAM and were decoded on a
 * VRAM miss, what would a decode cost?
 *
 * usage: bench_nr2_kernel.exe CONTAINER.wex [records=16] [iters=20] [streams=1,2,4,8]
 * Checks 2 records byte-for-byte against the CPU decoder first. */
#include <windows.h>
#include <cuda_runtime.h>
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

struct Rec {
    const WexEntry *e;
    std::vector<unsigned char> host;
    WexCudaNr2Plan plan;
    unsigned char *d_record, *d_raw;
    uint32_t *d_offsets, *d_error;
    uint64_t raw;
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s CONTAINER [records] [iters]\n", argv[0]); return 2; }
    int nrec = argc > 2 ? atoi(argv[2]) : 16, iters = argc > 3 ? atoi(argv[3]) : 20;
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
    if (wex_nr2_model_location(hdr, page, &mat, &mbytes, &mid)) return 1;
    std::vector<unsigned char> blob(mbytes);
    WexNr2Model model;
    if (!read_at(f, mat, blob.data(), mbytes) || wex_nr2_model_open(blob.data(), mbytes, &model)) return 1;

    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int buckets = WEX_CUDA_NR2_LOOKUP == 0;
    unsigned char *base; CK(cudaMalloc(&base, wex_cuda_nr2_table_bytes(buckets)));
    CK(cudaMemcpy(base, blob.data(), WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice));
    WexCudaNr2Tables tables;
    CK(wex_cuda_build_nr2_tables(base, buckets, 0, &tables));
    CK(cudaDeviceSynchronize());

    std::vector<Rec> recs;
    size_t step = table.size() / (size_t)nrec;
    for (size_t i = 0; i < table.size() && (int)recs.size() < nrec; i += step) {
        const WexEntry *e = &table[i];
        if (e->flags != WEX_ENTRY_NEURAL_RANS2) continue;
        Rec r; r.e = e;
        uint64_t aligned;
        if (wex_entry_raw_sizes(e, &r.raw, &aligned)) return 1;
        r.host.resize(e->total_bytes);
        if (!read_at(f, e->file_offset, r.host.data(), e->total_bytes)) return 1;
        WexNr2Record view;
        if (wex_cuda_plan_nr2(e, r.host.data(), e->total_bytes, &view, &r.plan)) { fprintf(stderr, "plan\n"); return 1; }
        std::vector<uint32_t> off(r.plan.streams + 1u);
        wex_cuda_nr2_offsets(&view, &r.plan, off.data());
        CK(cudaMalloc(&r.d_record, e->total_bytes));
        CK(cudaMalloc(&r.d_raw, aligned));
        CK(cudaMalloc(&r.d_offsets, off.size() * 4u));
        CK(cudaMalloc(&r.d_error, 4));
        CK(cudaMemcpy(r.d_record, r.host.data(), e->total_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(r.d_offsets, off.data(), off.size() * 4u, cudaMemcpyHostToDevice));
        recs.push_back(std::move(r));
    }
    fclose(f);

    /* correctness: GPU vs CPU decoder on two records */
    for (int k = 0; k < 2 && k < (int)recs.size(); k++) {
        Rec &r = recs[k];
        CK(wex_cuda_launch_nr2(&r.plan, &tables, r.d_record, r.d_offsets, r.d_raw, r.d_error, 0));
        uint32_t err; CK(cudaMemcpy(&err, r.d_error, 4, cudaMemcpyDeviceToHost));
        std::vector<unsigned char> gpu(r.raw + 4096 * 4);
        CK(cudaMemcpy(gpu.data(), r.d_raw, r.raw, cudaMemcpyDeviceToHost));
        uint64_t aligned; wex_entry_raw_sizes(r.e, &r.raw, &aligned);
        std::vector<unsigned char> slab(aligned);
        size_t at = wex_record_read_offset(r.e);
        memcpy(slab.data() + at, r.host.data(), r.e->total_bytes);
        if (err || wex_decode_nr2_record(r.e, slab.data(), slab.size(), &model)) { fprintf(stderr, "decode error\n"); return 1; }
        for (int m = 0; m < 3; m++) {
            size_t sb = (size_t)r.e->scale_shapes[m][0] * r.e->scale_shapes[m][1];
            size_t wb = (size_t)r.e->weight_shapes[m][0] * r.e->weight_shapes[m][1];
            if (memcmp(gpu.data() + r.plan.scale_offset[m], slab.data() + r.e->scale_offsets[m], sb) ||
                memcmp(gpu.data() + r.plan.weight_offset[m], slab.data() + r.e->weight_offsets[m], wb)) {
                fprintf(stderr, "MISMATCH record %d matrix %d\n", k, m); return 1;
            }
        }
    }

    printf("{\"gpu\":\"%s\",\"sms\":%d,\"records\":%zu,\"record_mib\":%.2f,\"raw_mib\":%.2f,\"cpu_check\":\"2 records identical\",\n",
           prop.name, prop.multiProcessorCount, recs.size(), recs[0].e->total_bytes / 1048576.0, recs[0].raw / 1048576.0);
    printf(" \"grid\":{\"scale_blocks\":%u,\"weight_blocks\":%u,\"threads\":%u},\n \"concurrency\":[",
           recs[0].plan.block_base[0][3], recs[0].plan.block_base[1][3], WEX_CUDA_NR2_THREADS);
    int lanes[] = {1, 2, 4, 8};
    for (int li = 0; li < 4; li++) {
        int S = lanes[li];
        std::vector<cudaStream_t> st(S);
        for (auto &s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        double comp = 0, raw = 0;
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a, 0));
        for (int it = 0; it < iters; it++)
            for (size_t k = 0; k < recs.size(); k++) {
                Rec &r = recs[k];
                cudaStream_t s = st[k % S];
                CK(wex_cuda_launch_nr2(&r.plan, &tables, r.d_record, r.d_offsets, r.d_raw, r.d_error, s));
                comp += r.e->total_bytes; raw += (double)r.raw;
            }
        for (auto &s : st) { cudaEvent_t done; CK(cudaEventCreate(&done)); CK(cudaEventRecord(done, s)); CK(cudaStreamWaitEvent(0, done, 0)); }
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double n = (double)iters * recs.size();
        printf("%s{\"streams\":%d,\"ms_per_record\":%.4f,\"decoded_gib_s\":%.2f,\"compressed_gib_s\":%.2f}",
               li ? "," : "", S, ms / n, raw / (ms / 1e3) / 1073741824.0, comp / (ms / 1e3) / 1073741824.0);
        for (auto &s : st) cudaStreamDestroy(s);
    }
    printf("]}\n");
    return 0;
}
