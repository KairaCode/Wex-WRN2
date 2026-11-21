// SPDX-License-Identifier: Apache-2.0
/* WNR4 vs WNR3 GPU decode on the records of a pack (measure.py --pack).
 *   bench_gpu PACK [iterations]
 * WNR4: every record decoded on the GPU and compared with its expected bytes;
 * kernel-only GB/s (records resident in VRAM) by batch size; end-to-end GB/s from
 * page-locked host memory (plan + H2D + kernel, 2-stream pipelined: batch i+1 uploads while batch i
 * decodes; error words read once at the end; no commit copy).
 * WNR3 (MX packs): the production helper batch (wex_cuda_decode_batch_to_device,
 * H2D included) and its kernels alone (records resident, one stream per record
 * slot as the helper's lanes). GB/s = decoded (raw slab) bytes / time. */
#define NOMINMAX
#include "../../../c/tools/wex_cuda_bridge.cu"
#include "../../../wnr4/wnr4_cuda.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

struct Rec { uint32_t owner; std::vector<unsigned char> w4, w3, want; };

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    REQ(argc >= 2);
    const int iters = argc > 2 ? atoi(argv[2]) : 5;
    FILE *f = fopen(argv[1], "rb"); REQ(f);
    char magic[4]; uint32_t ver, n, kind, G, geo[13], l4, l2;
    REQ(fread(magic, 4, 1, f) == 1 && !memcmp(magic, "WP4K", 4));
    REQ(fread(&ver, 4, 1, f) == 1 && fread(&n, 4, 1, f) == 1 && fread(&kind, 4, 1, f) == 1 && fread(&G, 4, 1, f) == 1);
    REQ(fread(geo, 4, 13, f) == 13 && fread(&l4, 4, 1, f) == 1 && fread(&l2, 4, 1, f) == 1);
    WexEntry entry0; REQ(fread(&entry0, 72, 1, f) == 1);
    std::vector<unsigned char> wnm4(l4), wnm2(l2);
    REQ(fread(wnm4.data(), 1, l4, f) == l4); if (l2) REQ(fread(wnm2.data(), 1, l2, f) == l2);
    std::vector<Rec> recs(n);
    for (auto &r : recs) {
        uint32_t h[4]; REQ(fread(h, 4, 4, f) == 4);
        r.owner = h[0]; r.w4.resize(h[1]); r.w3.resize(h[2]); r.want.resize(h[3]);
        REQ(fread(r.w4.data(), 1, h[1], f) == h[1]);
        if (h[2]) REQ(fread(r.w3.data(), 1, h[2], f) == h[2]);
        REQ(fread(r.want.data(), 1, h[3], f) == h[3]);
        REQ(h[3] == recs[0].want.size());
    }
    fclose(f);
    Wnr4Geo g; REQ(!wnr4_geo(&g, geo[0], geo + 1, geo + 4, geo + 7, geo + 10));
    Wnr4Model model; REQ(!wnr4_model_open(wnm4.data(), wnm4.size(), &model));
    const size_t raw = (size_t)g.raw;
    REQ(recs[0].want.size() == raw);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, %d SMs; pack %s: %u records, kind %s, G %u, raw %zu B\n", prop.name, prop.multiProcessorCount,
           argv[1], n, kind ? "NV" : "MX", G, raw);
    double r4 = 0, r3 = 0;
    for (auto &r : recs) { r4 += r.w4.size(); r3 += r.w3.size(); }
    printf("stored/raw: WNR4 %.5f", r4 / ((double)raw * n));
    if (l2) printf(", WNR3 %.5f", r3 / ((double)raw * n));
    printf("\n");

    /* device model tables */
    std::vector<unsigned char> img(wnr4_cuda_table_bytes(&model));
    wnr4_cuda_table_image(&model, img.data());
    unsigned char *dtab; CK(cudaMalloc(&dtab, img.size())); CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
    Wnr4CudaTables tab; wnr4_cuda_tables_view(&model, dtab, &tab);

    /* records: pinned host copies + resident device copies; one destination slab per record */
    size_t tot4 = 0; std::vector<size_t> at4(n);
    for (uint32_t i = 0; i < n; i++) { at4[i] = tot4; tot4 += recs[i].w4.size(); }
    unsigned char *h4, *d4, *dst; CK(cudaMallocHost(&h4, tot4)); CK(cudaMalloc(&d4, tot4));
    for (uint32_t i = 0; i < n; i++) memcpy(h4 + at4[i], recs[i].w4.data(), recs[i].w4.size());
    CK(cudaMemcpy(d4, h4, tot4, cudaMemcpyHostToDevice));
    const size_t slab = (raw + 4096u + 255u) & ~(size_t)255u;   /* WNR3 kernels write up to 15 B past raw */
    CK(cudaMalloc(&dst, slab * n));
    uint32_t *derr; CK(cudaMalloc(&derr, 4u * n));
    Wnr4CudaRec *drec; CK(cudaMalloc(&drec, sizeof(Wnr4CudaRec) * (2u * n + 2u)));   /* two slots of the pipelined loop */
    Wnr4CudaJob *djob; CK(cudaMalloc(&djob, sizeof(Wnr4CudaJob) * (n * 2048u + 64u)));
    cudaStream_t s0, s1; CK(cudaStreamCreate(&s0)); CK(cudaStreamCreate(&s1));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    /* plan a batch [b0, b0 + cnt) from device copy base (records laid out as in h4) */
    auto plan = [&](uint32_t b0, uint32_t cnt, const unsigned char *base, std::vector<Wnr4CudaJob> &jobs,
                    std::vector<Wnr4CudaRec> &infos) {
        std::vector<Wnr4CudaJob> by_m[3];
        infos.resize(cnt);
        for (uint32_t i = 0; i < cnt; i++) {
            uint32_t k = b0 + i;
            unsigned char *ds[3], *dw[3];
            for (int m = 0; m < 3; m++) { ds[m] = dst + slab * k + g.soff[m]; dw[m] = dst + slab * k + g.woff[m]; }
            REQ(!wnr4_cuda_plan(&g, &model, h4 + at4[k], recs[k].w4.size(), recs[k].owner, base + at4[k], i, ds, dw,
                                &infos[i], by_m));
        }
        wnr4_cuda_jobs(by_m, jobs);
        REQ(jobs.size() <= n * 1024u + 32u);   /* each half of the device job array (the pipelined loop uses two) */
    };

    /* 1. correctness: every record, batch = all */
    {
        std::vector<Wnr4CudaJob> jobs; std::vector<Wnr4CudaRec> infos;
        plan(0, n, d4, jobs, infos);
        CK(cudaMemset(dst, 0xA5, slab * n)); CK(cudaMemset(derr, 0, 4u * n));
        CK(cudaMemcpy(drec, infos.data(), sizeof(Wnr4CudaRec) * n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), drec, derr, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> errs(n); CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        std::vector<unsigned char> back(raw);
        uint32_t good = 0;
        for (uint32_t i = 0; i < n; i++) {
            CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost));
            bool same = !memcmp(back.data(), recs[i].want.data(), raw);
            good += same && !errs[i];
            if (!same || errs[i]) printf("  record %u: error 0x%x, %s\n", i, errs[i], same ? "bytes equal" : "BYTES DIFFER");
        }
        printf("WNR4 GPU bit-exact: %u of %u records (%zu jobs)\n", good, n, jobs.size());
        /* negative: flip one bit in every record's payload copy on the device, all must be rejected */
        uint32_t rejected = 0;
        for (uint32_t i = 0; i < n; i++) {
            size_t len = recs[i].w4.size(), used = len;
            while (used > 0 && !recs[i].w4[used - 1]) used--;
            size_t p0 = 64u + 4u * ((size_t)wnr4_u32(recs[i].w4.data() + 20) + 1u);   /* payload start */
            size_t pos = p0 + (size_t)((i * 2654435761u) % (uint32_t)(used - p0));
            unsigned char b = recs[i].w4[pos] ^ (unsigned char)(1u << (i % 8));
            CK(cudaMemcpy(d4 + at4[i] + pos, &b, 1, cudaMemcpyHostToDevice));
        }
        CK(cudaMemset(derr, 0, 4u * n));
        CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), drec, derr, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) rejected += errs[i] != 0;
        printf("WNR4 GPU single-bit flips (one per record, payload bytes; header and offsets are checked on the host): %u of %u rejected\n", rejected, n);
        CK(cudaMemcpy(d4, h4, tot4, cudaMemcpyHostToDevice));
    }

    /* 2. kernel-only GB/s by batch size (records resident): word fetch variant x warps per block */
    static const int CFG[][3] = {{0, 4, 0}, {1, 4, 0}, {1, 8, 0}, {1, 16, 0}, {1, 16, 1}, {1, 0, -1}};   /* LD, warps (0 auto), rows global (-1 auto) */
    for (int cfg = 0; cfg < 6; cfg++) {
    wnr4_cuda_lds = CFG[cfg][0]; wnr4_cuda_warps = (uint32_t)CFG[cfg][1]; wnr4_cuda_rows_global = CFG[cfg][2];
    printf("WNR4 kernel only (records in VRAM), %s, warps/block %s, rows %s, median of %d:\n",
           CFG[cfg][0] ? "LD lookup" : "row_find", CFG[cfg][1] ? std::to_string(CFG[cfg][1]).c_str() : "auto",
           CFG[cfg][2] < 0 ? "auto" : CFG[cfg][2] ? "L1" : "shared", iters);
    for (uint32_t b = 1; b <= n; b *= 2) {
        std::vector<Wnr4CudaJob> jobs; std::vector<Wnr4CudaRec> infos;
        plan(0, b, d4, jobs, infos);
        CK(cudaMemcpy(drec, infos.data(), sizeof(Wnr4CudaRec) * b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        std::vector<float> ms;
        for (int it = 0; it < iters + 1; it++) {
            CK(cudaMemsetAsync(derr, 0, 4u * b, s0));
            CK(cudaEventRecord(e0, s0));
            CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), drec, derr, s0));
            CK(cudaEventRecord(e1, s0)); CK(cudaEventSynchronize(e1));
            float t; CK(cudaEventElapsedTime(&t, e0, e1)); if (it) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        float t = ms[ms.size() / 2];
        printf("  batch %4u: %8.3f ms  %7.1f GB/s  (%zu warps)\n", b, t, (double)raw * b / (t * 1e-3) / 1e9, jobs.size());
        if (b < n && b * 2 > n) b = n / 2;   /* last row = all records */
    }
    }

    wnr4_cuda_lds = 1; wnr4_cuda_warps = 0u; wnr4_cuda_rows_global = -1;   /* defaults: LD, auto */
    {   /* re-check correctness with the defaults after the sweep */
        std::vector<Wnr4CudaJob> jobs; std::vector<Wnr4CudaRec> infos;
        plan(0, n, d4, jobs, infos);
        CK(cudaMemset(dst, 0xA5, slab * n)); CK(cudaMemset(derr, 0, 4u * n));
        CK(cudaMemcpy(drec, infos.data(), sizeof(Wnr4CudaRec) * n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), drec, derr, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> errs(n); CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        std::vector<unsigned char> back(raw); uint32_t good = 0;
        for (uint32_t i = 0; i < n; i++) { CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost)); good += !errs[i] && !memcmp(back.data(), recs[i].want.data(), raw); }
        printf("WNR4 GPU bit-exact (defaults: LD, auto block size): %u of %u records\n", good, n);
    }
    /* 3. end-to-end from pinned host: plan + H2D + kernel; whole pack in batches of b, 2 streams pipelined */
    printf("WNR4 end to end from page-locked host (plan + upload + decode), whole pack:\n");
    for (uint32_t b : {1u, 4u, 8u, 16u, 32u}) {
        if (b > n) break;
        double best = 1e30;
        for (int it = 0; it < iters; it++) {
            CK(cudaDeviceSynchronize());
            double t0 = now();
            std::vector<Wnr4CudaJob> jobs[2]; std::vector<Wnr4CudaRec> infos[2];
            Wnr4CudaRec *drs[2] = {drec, drec + n + 1u};
            Wnr4CudaJob *djs[2] = {djob, djob + n * 1024u + 32u};
            cudaEvent_t done[2]; CK(cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming));
            int k = 0;
            for (uint32_t b0 = 0; b0 < n; b0 += b, k ^= 1) {
                uint32_t cnt = std::min(b, n - b0);
                cudaStream_t s = k ? s1 : s0;
                CK(cudaEventSynchronize(done[k]));   /* host vectors of this slot are free again */
                plan(b0, cnt, d4, jobs[k], infos[k]);
                size_t bytes = at4[b0 + cnt - 1] + recs[b0 + cnt - 1].w4.size() - at4[b0];
                CK(cudaMemcpyAsync(d4 + at4[b0], h4 + at4[b0], bytes, cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(drs[k], infos[k].data(), sizeof(Wnr4CudaRec) * cnt, cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(djs[k], jobs[k].data(), sizeof(Wnr4CudaJob) * jobs[k].size(), cudaMemcpyHostToDevice, s));
                CK(cudaMemsetAsync(derr + b0, 0, 4u * cnt, s));
                CK(wnr4_cuda_launch(tab, djs[k], (uint32_t)jobs[k].size(), drs[k], derr + b0, s));
                CK(cudaEventRecord(done[k], s));
            }
            CK(cudaDeviceSynchronize());
            best = std::min(best, now() - t0);
            cudaEventDestroy(done[0]); cudaEventDestroy(done[1]);
        }
        std::vector<uint32_t> errs(n); CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        uint32_t bad = 0; for (auto e : errs) bad += e != 0;
        printf("  batch %3u: %8.2f ms for %u records  %7.1f GB/s decoded (%.1f GB/s read)%s\n", b, best * 1e3, n,
               (double)raw * n / best / 1e9, (double)tot4 / best / 1e9, bad ? "  ERRORS" : "");
    }

    /* 4. WNR3 baseline (MX packs) */
    if (l2) {
        REQ(wex_cuda_register_model(wnm2.data(), wnm2.size()) == 0);
        size_t tot3 = 0; std::vector<size_t> at3(n);
        for (uint32_t i = 0; i < n; i++) { at3[i] = tot3; tot3 += recs[i].w3.size(); }
        unsigned char *h3, *d3; CK(cudaMallocHost(&h3, tot3)); CK(cudaMalloc(&d3, tot3));
        for (uint32_t i = 0; i < n; i++) memcpy(h3 + at3[i], recs[i].w3.data(), recs[i].w3.size());
        CK(cudaMemcpy(d3, h3, tot3, cudaMemcpyHostToDevice));
        std::vector<WexEntry> ents(n, entry0);
        std::vector<const WexEntry *> ep(n); std::vector<const unsigned char *> rp(n);
        std::vector<unsigned char *> ds(3 * n), dw(3 * n);
        for (uint32_t i = 0; i < n; i++) {
            ents[i].flags = WEX_ENTRY_NEURAL_RANS3; ents[i].total_bytes = (uint32_t)recs[i].w3.size();
            ep[i] = &ents[i]; rp[i] = h3 + at3[i];
            for (int m = 0; m < 3; m++) { ds[3 * i + m] = dst + slab * i + entry0.scale_offsets[m]; dw[3 * i + m] = dst + slab * i + entry0.weight_offsets[m]; }
        }
        std::vector<int> status(n);
        /* correctness through the production batch */
        CK(cudaMemset(dst, 0xA5, slab * n));
        int bad = wex_cuda_decode_batch_to_device((int)n, ep.data(), rp.data(), 0, ds.data(), dw.data(), status.data());
        std::vector<unsigned char> back(raw); uint32_t good = 0;
        for (uint32_t i = 0; i < n; i++) {
            CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost));
            good += !status[i] && !memcmp(back.data(), recs[i].want.data(), raw);
        }
        printf("WNR3 production batch: %u of %u records bit-exact (failures reported %d)\n", good, n, bad);
        printf("WNR3 production helper batch (upload included), whole pack:\n");
        for (uint32_t b : {1u, 4u, 8u, 16u, 32u}) {
            if (b > n) break;
            double best = 1e30;
            for (int it = 0; it < iters; it++) {
                double t0 = now();
                for (uint32_t b0 = 0; b0 < n; b0 += b) {
                    int cnt = (int)std::min(b, n - b0);
                    REQ(wex_cuda_decode_batch_to_device(cnt, ep.data() + b0, rp.data() + b0, 0, ds.data() + 3 * b0,
                                                        dw.data() + 3 * b0, status.data() + b0) == 0);
                }
                best = std::min(best, now() - t0);
            }
            printf("  batch %3u: %8.2f ms  %7.1f GB/s decoded\n", b, best * 1e3, (double)raw * n / best / 1e9);
        }
        /* WNR3 kernels alone: records resident, decode into a scratch raw per stream slot, as the helper's lanes */
        printf("WNR3 kernels only (records in VRAM; scales + weights kernels per record, 8 streams), median of %d:\n", iters);
        const int S = 8;
        cudaStream_t ss[S]; for (int i = 0; i < S; i++) CK(cudaStreamCreate(&ss[i]));
        std::vector<WexNr2Record> views(n); std::vector<WexCudaNr2Plan> plans(n);
        uint32_t *doff; size_t offs = 0; std::vector<size_t> oat(n);
        for (uint32_t i = 0; i < n; i++) {
            REQ(wex_cuda_plan_nr2(&ents[i], recs[i].w3.data(), recs[i].w3.size(), &views[i], &plans[i]) == 0);
            oat[i] = offs; offs += plans[i].streams + 1u;
        }
        std::vector<uint32_t> hoff(offs);
        for (uint32_t i = 0; i < n; i++) wex_cuda_nr2_offsets(&views[i], &plans[i], hoff.data() + oat[i]);
        CK(cudaMalloc(&doff, 4u * offs)); CK(cudaMemcpy(doff, hoff.data(), 4u * offs, cudaMemcpyHostToDevice));
        WexCudaNr2Tables t3; REQ(wex_cuda_nr2_tables(plans[0].model_id, 0, ss[0], &t3) == 0);
        CK(cudaDeviceSynchronize());
        uint32_t *derr3; CK(cudaMalloc(&derr3, 4u * n));
        for (uint32_t b = 1; b <= n; b *= 2) {
            std::vector<double> ts;
            for (int it = 0; it < iters + 1; it++) {
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0, 0));
                for (uint32_t i = 0; i < b; i++) {
                    cudaStream_t s = ss[i % S];
                    CK(cudaStreamWaitEvent(s, e0, 0));
                    CK(wex_cuda_launch_nr2(&plans[i], &t3, d3 + at3[i], doff + oat[i], dst + slab * i, derr3 + i, s));
                }
                for (int i = 0; i < S; i++) { cudaEvent_t ev; CK(cudaEventCreate(&ev)); CK(cudaEventRecord(ev, ss[i])); CK(cudaStreamWaitEvent(0, ev, 0)); cudaEventDestroy(ev); }
                CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
                float t; CK(cudaEventElapsedTime(&t, e0, e1)); if (it) ts.push_back(t);
            }
            std::sort(ts.begin(), ts.end());
            double t = ts[ts.size() / 2];
            printf("  batch %4u: %8.3f ms  %7.1f GB/s\n", b, t, (double)raw * b / (t * 1e-3) / 1e9);
            if (b < n && b * 2 > n) b = n / 2;
        }
    }
    printf("done\n");
    return 0;
}
