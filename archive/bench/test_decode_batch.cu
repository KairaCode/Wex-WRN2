/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* wex_cuda_decode_batch_to_device (helper feature 8) against real WNR2 records.
 * Input: a records_*.bin sample (rechunk.py / reencode_bin.py: WNM2 model,
 * entries, raw slabs, records). Checks:
 *   1. batches of 1..16 records: status 0 and every destination byte equal to
 *      the stored raw slab (all 64 records covered);
 *   2. a batch with one corrupted payload and one invalid entry: those report
 *      -1 with their destination untouched (sentinel), the others decode;
 *   3. wall ms per record: batch API at each size vs one single call per record.
 * usage: test_decode_batch.exe HELPER.dll RECORDS.bin */
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "wex_format.h"

typedef int (*RegFn)(const unsigned char *, size_t);
typedef int (*DevFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*BatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                       unsigned char *const *, unsigned char *const *, int *);
typedef unsigned int (*FeatFn)(void);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

struct Rec { WexEntry e; std::vector<unsigned char> raw; unsigned char *host; };

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    HMODULE h = LoadLibraryA(argv[1]);
    CHECK(h);
    FeatFn feat = (FeatFn)GetProcAddress(h, "wex_cuda_helper_features");
    RegFn reg = (RegFn)GetProcAddress(h, "wex_cuda_register_model");
    DevFn dev = (DevFn)GetProcAddress(h, "wex_cuda_decode_to_device");
    BatchFn batch = (BatchFn)GetProcAddress(h, "wex_cuda_decode_batch_to_device");
    CHECK(feat && (feat() & 8u) && reg && dev && batch);

    FILE *f = fopen(argv[2], "rb");
    CHECK(f);
    uint32_t mb, n;
    CHECK(fread(&mb, 4, 1, f) == 1);
    std::vector<unsigned char> blob(mb);
    CHECK(fread(blob.data(), 1, mb, f) == mb && fread(&n, 4, 1, f) == 1);
    CHECK(reg(blob.data(), mb) == 0);
    CK(cudaSetDevice(0));
    std::vector<Rec> recs(n);
    size_t max_s[3] = {0, 0, 0}, max_w[3] = {0, 0, 0};
    for (auto &r : recs) {
        uint32_t rb;
        CHECK(fread(&r.e, sizeof(WexEntry), 1, f) == 1 && fread(&rb, 4, 1, f) == 1);
        r.raw.resize(rb);
        CHECK(fread(r.raw.data(), 1, rb, f) == rb);
        CK(cudaMallocHost((void **)&r.host, r.e.total_bytes));
        CHECK(fread(r.host, 1, r.e.total_bytes, f) == r.e.total_bytes);
        for (int m = 0; m < 3; m++) {
            max_s[m] = std::max(max_s[m], (size_t)r.e.scale_shapes[m][0] * r.e.scale_shapes[m][1]);
            max_w[m] = std::max(max_w[m], (size_t)r.e.weight_shapes[m][0] * r.e.weight_shapes[m][1]);
        }
    }
    fclose(f);
    const int MAXB = 16;
    size_t per = 0;
    for (int m = 0; m < 3; m++) per += max_s[m] + max_w[m];
    unsigned char *arena; CK(cudaMalloc(&arena, per * MAXB));
    std::vector<unsigned char *> ds(3 * MAXB), dw(3 * MAXB);
    for (int b = 0; b < MAXB; b++) {
        unsigned char *p = arena + per * b;
        for (int m = 0; m < 3; m++) { ds[3 * b + m] = p; p += max_s[m]; }
        for (int m = 0; m < 3; m++) { dw[3 * b + m] = p; p += max_w[m]; }
    }
    std::vector<unsigned char> got(per);
    auto verify = [&](int slot, const Rec &r) {
        for (int m = 0; m < 3; m++) {
            size_t sb = (size_t)r.e.scale_shapes[m][0] * r.e.scale_shapes[m][1];
            size_t wb = (size_t)r.e.weight_shapes[m][0] * r.e.weight_shapes[m][1];
            CK(cudaMemcpy(got.data(), ds[3 * slot + m], sb, cudaMemcpyDeviceToHost));
            if (memcmp(got.data(), r.raw.data() + r.e.scale_offsets[m], sb)) return false;
            CK(cudaMemcpy(got.data(), dw[3 * slot + m], wb, cudaMemcpyDeviceToHost));
            if (memcmp(got.data(), r.raw.data() + r.e.weight_offsets[m], wb)) return false;
        }
        return true;
    };
    auto untouched = [&](int slot) {   /* sentinel 0xA5 still everywhere */
        CK(cudaMemcpy(got.data(), arena + per * slot, per, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < per; i++) if (got[i] != 0xA5) return false;
        return true;
    };

    /* 1. correctness over every record, batch sizes 1..16 */
    int sizes[] = {1, 2, 3, 4, 6, 8, 10, 16};
    long verified = 0;
    for (int B : sizes) {
        for (size_t start = 0; start < recs.size(); start += B) {
            int cnt = (int)std::min<size_t>(B, recs.size() - start);
            std::vector<const WexEntry *> es(cnt); std::vector<const unsigned char *> rs(cnt); std::vector<int> st(cnt, 7);
            for (int i = 0; i < cnt; i++) { es[i] = &recs[start + i].e; rs[i] = recs[start + i].host; }
            CK(cudaMemset(arena, 0xA5, per * MAXB));
            CHECK(batch(cnt, es.data(), rs.data(), 0, ds.data(), dw.data(), st.data()) == 0);
            for (int i = 0; i < cnt; i++) { CHECK(st[i] == 0); CHECK(verify(i, recs[start + i])); verified++; }
        }
    }
    /* 2. corrupted payload (record 2) and invalid entry (record 4) inside a batch of 6 */
    {
        std::vector<unsigned char> bad(recs[2].host, recs[2].host + recs[2].e.total_bytes);
        unsigned char *badp; CK(cudaMallocHost((void **)&badp, bad.size()));
        memcpy(badp, bad.data(), bad.size());
        for (size_t i = bad.size() / 2; i < bad.size() / 2 + 4096; i++) badp[i] ^= 0x5Au;  /* middle of the weight payload */
        WexEntry invalid = recs[4].e; invalid.flags = 0;
        std::vector<const WexEntry *> es(6); std::vector<const unsigned char *> rs(6); std::vector<int> st(6, 7);
        for (int i = 0; i < 6; i++) { es[i] = &recs[i].e; rs[i] = recs[i].host; }
        rs[2] = badp; es[4] = &invalid;
        CK(cudaMemset(arena, 0xA5, per * MAXB));
        int failed = batch(6, es.data(), rs.data(), 0, ds.data(), dw.data(), st.data());
        printf("{\"corruption_statuses\":[%d,%d,%d,%d,%d,%d],\"failed\":%d}\n", st[0], st[1], st[2], st[3], st[4], st[5], failed);
        CHECK(failed == 2 && st[2] == -1 && st[4] == -1);
        CHECK(untouched(2) && untouched(4));
        for (int i : {0, 1, 3, 5}) { CHECK(st[i] == 0); CHECK(verify(i, recs[i])); }
        cudaFreeHost(badp);
    }
    /* 3. timing: batch API vs one single call per record */
    auto ms_now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    printf("{\"records\":%u,\"verified\":%ld,\"record_mib\":%.2f,\"timing\":[", n, verified, recs[0].e.total_bytes / 1048576.0);
    std::vector<double> single;
    for (int round = 0; round < 6; round++) {
        double t0 = ms_now();
        for (int i = 0; i < 16; i++) {
            unsigned char *s3[3] = {ds[3 * (i % MAXB)], ds[3 * (i % MAXB) + 1], ds[3 * (i % MAXB) + 2]};
            unsigned char *w3[3] = {dw[3 * (i % MAXB)], dw[3 * (i % MAXB) + 1], dw[3 * (i % MAXB) + 2]};
            CHECK(dev(&recs[i].e, recs[i].host, 0, s3, w3) == 0);
        }
        single.push_back((ms_now() - t0) / 16);
    }
    std::sort(single.begin(), single.end());
    printf("{\"api\":\"single\",\"ms_per_record_p50\":%.3f}", single[single.size() / 2]);
    for (int B : {1, 2, 4, 6, 8, 10, 16}) {
        std::vector<double> t;
        for (int round = 0; round < 12; round++) {
            std::vector<const WexEntry *> es(B); std::vector<const unsigned char *> rs(B); std::vector<int> st(B);
            for (int i = 0; i < B; i++) { size_t k = (round * B + i) % recs.size(); es[i] = &recs[k].e; rs[i] = recs[k].host; }
            double t0 = ms_now();
            CHECK(batch(B, es.data(), rs.data(), 0, ds.data(), dw.data(), st.data()) == 0);
            t.push_back((ms_now() - t0) / B);
        }
        std::sort(t.begin(), t.end());
        printf(",{\"api\":\"batch\",\"batch\":%d,\"ms_per_record_p50\":%.3f}", B, t[t.size() / 2]);
    }
    printf("]}\nALL OK\n");
    return 0;
}
