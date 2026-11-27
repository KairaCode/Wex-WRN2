// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
/* WNR4 MX/NV CUDA correctness test and L2-flushed benchmark (wnr4/wnr4_cuda.cuh +
 * wnr4/tools/wnr4_cuda_bridge.cu). Needs nvcc and an NVIDIA GPU:
 *   research\wnr4-20261121\nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler=/O2,/EHsc wnr4/tests/test_wnr4_gpu.cu -o out/test_wnr4_gpu_v2.exe
 *   out\test_wnr4_gpu_v2.exe [--bench-only]
 * Verifies:
 *   - ABI 6 & full correctness matrix: MX & NV x pb in {1, 5} x W in {0, 1, 2, 4, 8, 16},
 *     partial tiles (n % 32 != 0), NV scols < 33 (ctx-63 fallback), G in {1, 3, 7, 32, 128},
 *     inline (<= 32) vs ptr (> 32) direct launch, mixed-G batch (G=32 + G=64 in one submit),
 *     job-list API, fault isolation, and negative cases (ceil underflow, CZ nibble 8, unaligned
 *     dst_w, mismatched model kind, wrong owner, unknown model, 4-slot ring overflow, bad state,
 *     unused tile word).
 *   - L2-flushed kernel-only GB/s (direct-launch + job-list) and end-to-end ABI-6 bridge
 *     throughput/latency (sync and async2) at Qwen3.8 and DeepSeek V4.1 production shapes. */
#include "../tools/wnr4_cuda_bridge.cu"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static inline uint32_t rnd32_s(uint64_t &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint32_t)(s >> 16);
}

struct Wnm2Tables {
    bool loaded = false;
    uint16_t wcdf[3][567][17];
    uint16_t scdf[3][257];
    unsigned char anchor[3] = {120, 120, 120};
};

static Wnm2Tables load_v41_wnm2(void) {
    Wnm2Tables t;
    const char *paths[] = {
        "research/wnr3-5090-20261107/ceil-kaira/model_CFG_rule_g.wnm2",
        "../../research/wnr3-5090-20261107/ceil-kaira/model_CFG_rule_g.wnm2"
    };
    FILE *f = nullptr;
    for (const char *p : paths) { f = fopen(p, "rb"); if (f) break; }
    if (!f) return t;
    std::vector<unsigned char> b(59408);
    if (fread(b.data(), 1, b.size(), f) != b.size()) { fclose(f); return t; }
    fclose(f);
    if (wnr4_u32(b.data()) != 0x324D4E57u) return t;
    size_t off = 32;
    for (int m = 0; m < 3; m++)
        for (int c = 0; c < 567; c++)
            for (int k = 0; k < 17; k++, off += 2)
                t.wcdf[m][c][k] = (uint16_t)(b[off] | (b[off + 1] << 8));
    for (int m = 0; m < 3; m++)
        for (int k = 0; k < 257; k++, off += 2)
            t.scdf[m][k] = (uint16_t)(b[off] | (b[off + 1] << 8));
    t.loaded = true;
    return t;
}

/* Sample a realistic MXFP4 raw slab from V4.1 ceil-form WNM2 tables (inverse-transformed back to raw
 * so wnr4_prepare triggers authentic ~28% ceil blocks and flags = 0x0F). */
static void sample_mx_v41(const Wnr4Geo &g, const Wnm2Tables &wnm2, unsigned char *raw, int cz, uint64_t seed) {
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ull;
    memset(raw, 0, (size_t)g.raw);
    if (!wnm2.loaded) {
        for (uint32_t m = 0; m < 3u; m++) {
            for (uint32_t i = 0; i < g.scount[m]; i++)
                raw[g.soff[m] + i] = (unsigned char)(120u + (rnd32_s(rng) % 100u < 35u ? 1u : 0u));
            for (uint32_t c = 0; c < g.cells[m]; c++) {
                uint32_t even = (c % 3u == 0u);
                for (uint32_t i = 0; i < 32u; i++) {
                    uint32_t r = rnd32_s(rng) % 100u;
                    uint32_t a = r < 12 ? 0 : r < 30 ? 1 : r < 48 ? 2 : r < 63 ? 3 : r < 78 ? 4 : r < 89 ? 5 : r < 96 ? 6 : 7;
                    static const uint32_t ev[6] = {0, 2, 4, 5, 6, 7};
                    if (even) a = ev[rnd32_s(rng) % 6u];
                    if (i == 31u) a = even ? 7u : (rnd32_s(rng) & 1u ? 7u : 6u);
                    uint32_t x = a | (rnd32_s(rng) & 8u);
                    if (cz && x == 8u) x = 0u;
                    raw[g.woff[m] + (size_t)c * 16u + i / 2u] |= (unsigned char)(x << ((i & 1u) * 4u));
                }
            }
        }
        return;
    }
    uint32_t s_tot[3] = {0, 0, 0};
    uint32_t s_cdf[3][257];
    for (int m = 0; m < 3; m++) {
        s_cdf[m][0] = 0;
        for (int k = 0; k < 256; k++) {
            uint32_t f = (uint32_t)wnm2.scdf[m][k + 1] - wnm2.scdf[m][k];
            uint32_t adj = f > 1u ? f - 1u : 0u;
            s_tot[m] += adj;
            s_cdf[m][k + 1] = s_tot[m];
        }
    }
    for (uint32_t m = 0; m < 3u; m++) {
        unsigned char *so = raw + g.soff[m];
        unsigned char *wo = raw + g.woff[m];
        for (uint32_t c = 0; c < g.cells[m]; c++) {
            uint32_t u = rnd32_s(rng) % s_tot[m], sc = 0;
            uint32_t lo = 0, hi = 256;
            while (hi - lo > 1u) { uint32_t mid = (lo + hi) >> 1; if (s_cdf[m][mid] <= u) lo = mid; else hi = mid; }
            sc = lo;
            uint32_t cls = wnr4_cls(sc, wnm2.anchor[m], 3);
            uint32_t pm = 8u, mx = 8u;
            uint32_t w[4] = {0, 0, 0, 0};
            for (uint32_t k = 0; k < 32u; k++) {
                const uint16_t *cdf = wnm2.wcdf[m][wnr4_wctx(cls, pm, mx)];
                uint32_t slot = rnd32_s(rng) & 16383u;
                uint32_t y = wnr4_find16(cdf, slot);
                if (cz && y == 8u) y = 0u;
                w[k / 8u] |= y << ((k & 7u) * 4u);
                pm = y & 7u;
                mx = wnr4_mx(mx, pm);
            }
            if (mx == 5u && sc > 0u) {
                sc--;
                for (int q = 0; q < 4; q++) w[q] = wnr4_ceil_dbl8(w[q]);
            } else if (mx < 6u) {
                w[3] = (w[3] & 0x0fffffffu) | ((6u | (rnd32_s(rng) & 8u)) << 28);
            }
            so[c] = (unsigned char)sc;
            memcpy(wo + (size_t)c * 16u, w, 16u);
        }
    }
}

/* Sample realistic NVFP4 slab (correlated E4M3 scales along rows/columns + skewed FP4 nibbles). */
static void sample_nv_q38(const Wnr4Geo &g, unsigned char *raw, int cz, uint64_t seed) {
    uint64_t rng = seed ? seed : 0x517CC1B727220A95ull;
    memset(raw, 0, (size_t)g.raw);
    for (uint32_t m = 0; m < 3u; m++) {
        unsigned char *so = raw + g.soff[m];
        unsigned char *wo = raw + g.woff[m];
        const uint32_t cols = g.scols[m];
        for (uint32_t i = 0; i < g.scount[m]; i++) {
            int base = (i >= cols) ? (int)so[i - cols] : (56 + (int)((i % cols) % 7u));
            int delta = (int)(rnd32_s(rng) % 5u) - 2;
            if ((rnd32_s(rng) & 7u) == 0u) delta += (int)(rnd32_s(rng) % 7u) - 3;
            int v = base + delta;
            if (v < 24) v = 24;
            if (v > 96) v = 96;
            so[i] = (unsigned char)v;
        }
        for (uint32_t c = 0; c < g.cells[m]; c++) {
            for (uint32_t i = 0; i < 32u; i++) {
                uint32_t r = rnd32_s(rng) % 100u;
                uint32_t a = r < 14 ? 0 : r < 33 ? 1 : r < 52 ? 2 : r < 68 ? 3 : r < 82 ? 4 : r < 91 ? 5 : r < 97 ? 6 : 7;
                if ((i & 15u) == 15u && (rnd32_s(rng) & 1u)) a = 7u;
                uint32_t x = a | (rnd32_s(rng) & 8u);
                if (cz && x == 8u) x = 0u;
                wo[(size_t)c * 16u + i / 2u] |= (unsigned char)(x << ((i & 1u) * 4u));
            }
        }
    }
}

struct Pack {
    const char *name = nullptr;
    Wnr4Geo g = {};
    uint32_t geo[13] = {};
    uint32_t pb = 0, G = 0;
    int cz = 0;
    std::vector<unsigned char> blob;
    Wnr4Model model = {};
    std::vector<std::vector<unsigned char>> expected;
    std::vector<std::vector<unsigned char>> enc;
    std::vector<uint32_t> owners;
    ~Pack() { wnr4_model_free(&model); }
};

static void build_pack(Pack &p, const char *name, uint32_t kind, const uint32_t sc[3], const uint32_t cols[3],
                       uint32_t pb, uint32_t G, int cz, uint32_t ntrain, uint32_t nrec, const Wnm2Tables &wnm2,
                       bool quiet = false) {
    p.name = name; p.pb = pb; p.G = G; p.cz = cz;
    uint32_t so[3] = {0, sc[0], sc[0] + sc[1]};
    const uint32_t wmul = kind ? 8u : 16u;
    uint32_t wo[3] = {so[2] + sc[2], so[2] + sc[2] + sc[0] * wmul, so[2] + sc[2] + (sc[0] + sc[1]) * wmul};
    REQ(!wnr4_geo(&p.g, kind, sc, cols, so, wo));
    p.geo[0] = kind;
    for (int m = 0; m < 3; m++) { p.geo[1 + m] = sc[m]; p.geo[4 + m] = cols[m]; p.geo[7 + m] = so[m]; p.geo[10 + m] = wo[m]; }
    std::vector<uint64_t> counts(wnr4_counts_words(kind, pb), 0);
    std::vector<unsigned char> raw((size_t)p.g.raw), coded((size_t)p.g.raw);
    for (uint32_t i = 0; i < ntrain; i++) {
        uint64_t sd = 0x123456789ABCDEF0ull ^ ((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)kind << 48);
        if (kind == 0) sample_mx_v41(p.g, wnm2, raw.data(), cz, sd);
        else sample_nv_q38(p.g, raw.data(), cz, sd);
        (void)wnr4_prepare(&p.g, raw.data(), coded.data(), cz, nullptr);
        unsigned char a[3];
        for (int m = 0; m < 3; m++) a[m] = wnr4_anchor(coded.data() + p.g.soff[m], p.g.scount[m]);
        wnr4_count(&p.g, coded.data(), a, pb, counts.data());
    }
    p.blob.resize(wnr4_model_bytes(kind, pb));
    wnr4_model_build(kind, pb, counts.data(), p.blob.data());
    REQ(!wnr4_model_open(p.blob.data(), p.blob.size(), &p.model));
    p.expected.resize(nrec);
    p.enc.resize(nrec);
    p.owners.resize(nrec);
    const size_t cap = (size_t)p.g.raw * 2u + 65536u;
    const uint32_t nthr = std::min<uint32_t>(nrec, std::max<uint32_t>(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (uint32_t tid = 0; tid < nthr; tid++) {
        pool.emplace_back([&, tid]() {
            std::vector<unsigned char> tcoded((size_t)p.g.raw), encbuf(cap);
            for (uint32_t i = tid; i < nrec; i += nthr) {
                p.expected[i].resize((size_t)p.g.raw);
                uint64_t sd = 0xFEDCBA9876543210ull ^ ((uint64_t)(i + 1) * 0xBF58476D1CE4E5B9ull) ^ ((uint64_t)kind << 48);
                if (kind == 0) sample_mx_v41(p.g, wnm2, p.expected[i].data(), cz, sd);
                else sample_nv_q38(p.g, p.expected[i].data(), cz, sd);
                uint32_t flags = wnr4_prepare(&p.g, p.expected[i].data(), tcoded.data(), cz, nullptr);
                if (cz) {
                    for (int m = 0; m < 3; m++) {
                        unsigned char *w = p.expected[i].data() + p.g.woff[m];
                        for (size_t b = 0; b < (size_t)p.g.cells[m] * 16u; b++) {
                            uint32_t lo = w[b] & 15u, hi = w[b] >> 4;
                            if (lo == 8u) lo = 0;
                            if (hi == 8u) hi = 0;
                            w[b] = (unsigned char)(lo | (hi << 4));
                        }
                    }
                }
                p.owners[i] = (i << 16) | (i & 0xffu);
                size_t n = wnr4_encode(&p.model, &p.g, tcoded.data(), flags, p.owners[i], G, encbuf.data(), cap);
                REQ(n > 0 && !(n & 4095u));
                p.enc[i].assign(encbuf.data(), encbuf.data() + n);
            }
        });
    }
    for (auto &th : pool) th.join();
    double sum_enc = 0;
    for (uint32_t i = 0; i < nrec; i++) sum_enc += (double)p.enc[i].size();
    std::vector<unsigned char> chk((size_t)p.g.raw);
    REQ(wnr4_decode(&p.model, &p.g, p.enc[0].data(), p.enc[0].size(), p.owners[0], chk.data(), 4) == 0);
    REQ(chk == p.expected[0]);
    if (!quiet) {
        printf("pack %-18s: kind=%u pb=%u G=%3u n=%2u raw=%8zu B stored/raw=%.5f\n",
               name, kind, pb, G, nrec, (size_t)p.g.raw, sum_enc / ((double)p.g.raw * nrec));
    }
}

/* Verify every record of p through BOTH bridge (direct-launch) and job-list API at all W in {0, 1, 2, 4, 8, 16}. */
static void verify_pack_all_W(const Pack &p) {
    const uint32_t n = (uint32_t)p.enc.size();
    const size_t slab_s = ((size_t)p.g.scount[0] + p.g.scount[1] + p.g.scount[2] + 255u) & ~(size_t)255u;
    const size_t w0b = (size_t)p.g.cells[0] * 16u, w1b = (size_t)p.g.cells[1] * 16u, w2b = (size_t)p.g.cells[2] * 16u;
    const size_t slab_w = (w0b + w1b + w2b + 255u) & ~(size_t)255u;
    size_t all = 0;
    std::vector<size_t> at(n), tot(n);
    for (uint32_t i = 0; i < n; i++) {
        at[i] = all; tot[i] = p.enc[i].size(); all += (tot[i] + 255u) & ~(size_t)255u;
    }
    unsigned char *hrec, *drec, *ds_buf, *dw_buf, *dtab;
    uint32_t *derr;
    CK(cudaMallocHost(&hrec, all));
    CK(cudaMalloc(&drec, all));
    CK(cudaMalloc(&ds_buf, slab_s * n));
    CK(cudaMalloc(&dw_buf, slab_w * n));
    CK(cudaMalloc(&derr, 4u * n));
    std::vector<const unsigned char *> hrp(n);
    std::vector<unsigned char *> dsc(3 * n), dwt(3 * n);
    for (uint32_t i = 0; i < n; i++) {
        memcpy(hrec + at[i], p.enc[i].data(), tot[i]);
        hrp[i] = hrec + at[i];
        dsc[3 * i + 0] = ds_buf + slab_s * i;
        dsc[3 * i + 1] = dsc[3 * i + 0] + p.g.scount[0];
        dsc[3 * i + 2] = dsc[3 * i + 1] + p.g.scount[1];
        dwt[3 * i + 0] = dw_buf + slab_w * i;
        dwt[3 * i + 1] = dwt[3 * i + 0] + w0b;
        dwt[3 * i + 2] = dwt[3 * i + 1] + w1b;
    }
    CK(cudaMemcpy(drec, hrec, all, cudaMemcpyHostToDevice));
    std::vector<unsigned char> img(wnr4_cuda_table_bytes(&p.model));
    wnr4_cuda_table_image(&p.model, img.data());
    CK(cudaMalloc(&dtab, img.size()));
    CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
    Wnr4CudaTables tab;
    wnr4_cuda_tables_view(&p.model, dtab, &tab);
    REQ(wnr4_cuda_register_model(p.blob.data(), p.blob.size(), 0) == 0);

    std::vector<unsigned char> sb(p.g.scount[0] + p.g.scount[1] + p.g.scount[2]);
    std::vector<unsigned char> wb(w0b + w1b + w2b);
    auto check_exact = [&](uint32_t b) {
        for (uint32_t i = 0; i < b; i++) {
            CK(cudaMemcpy(sb.data(), dsc[3 * i], sb.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(wb.data(), dwt[3 * i], wb.size(), cudaMemcpyDeviceToHost));
            REQ(!memcmp(sb.data(), p.expected[i].data(), sb.size()));
            REQ(!memcmp(wb.data(), p.expected[i].data() + p.g.woff[0], wb.size()));
        }
    };

    std::vector<int> st(n);
    for (uint32_t W : {0u, 1u, 2u, 4u, 8u, 16u}) {
        wnr4_cuda_warps = W;
        /* Bridge path (exercises inline direct-launch for n <= 32, ptr direct-launch for n > 32) */
        CK(cudaMemset(ds_buf, 0x5A, slab_s * n));
        CK(cudaMemset(dw_buf, 0x5A, slab_w * n));
        CK(cudaDeviceSynchronize());
        std::fill(st.begin(), st.end(), 7);
        REQ(wnr4_cuda_decode_batch((int)n, p.geo, hrp.data(), tot.data(), p.owners.data(),
                                   p.model.id, 0, dsc.data(), dwt.data(), st.data()) == 0);
        for (uint32_t i = 0; i < n; i++) REQ(st[i] == 0);
        check_exact(n);

        /* Also test b = 1 (single-record fast path inside direct-launch kernel) */
        CK(cudaMemset(ds_buf, 0x5A, slab_s));
        CK(cudaMemset(dw_buf, 0x5A, slab_w));
        CK(cudaDeviceSynchronize());
        st[0] = 7;
        REQ(wnr4_cuda_decode_batch(1, p.geo, hrp.data(), tot.data(), p.owners.data(),
                                   p.model.id, 0, dsc.data(), dwt.data(), st.data()) == 0);
        REQ(st[0] == 0);
        check_exact(1);

        /* Job-list API path */
        std::vector<Wnr4CudaRec> hinfo(n);
        std::vector<Wnr4CudaJob> by_m[3], jobs;
        for (uint32_t i = 0; i < n; i++) {
            unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
            unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
            REQ(!wnr4_cuda_plan(&p.g, &p.model, p.enc[i].data(), p.enc[i].size(), p.owners[i],
                                drec + at[i], i, ds3, dw3, &hinfo[i], by_m));
        }
        wnr4_cuda_jobs(by_m, jobs);
        Wnr4CudaRec *dinfo; Wnr4CudaJob *djob;
        CK(cudaMalloc(&dinfo, sizeof(Wnr4CudaRec) * n));
        CK(cudaMalloc(&djob, sizeof(Wnr4CudaJob) * jobs.size()));
        CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4CudaRec) * n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(derr, 0, 4u * n));
        CK(cudaMemset(ds_buf, 0xA5, slab_s * n));
        CK(cudaMemset(dw_buf, 0xA5, slab_w * n));
        CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), dinfo, derr, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> err(n);
        CK(cudaMemcpy(err.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) REQ(err[i] == 0);
        check_exact(n);
        CK(cudaFree(dinfo)); CK(cudaFree(djob));
    }
    wnr4_cuda_warps = 0u;
    CK(cudaFreeHost(hrec)); CK(cudaFree(drec)); CK(cudaFree(ds_buf)); CK(cudaFree(dw_buf));
    CK(cudaFree(derr)); CK(cudaFree(dtab));
}

/* Comprehensive correctness matrix & negative test suite. */
static void run_correctness_matrix(const Wnm2Tables &wnm2) {
    printf("=== Correctness Matrix ===\n");
    REQ(wnr4_cuda_abi() == 6u);

    /* 1. MX & NV x pb in {1, 5} x W in {0, 1, 2, 4, 8, 16} x partial tiles x G in {1, 3, 7, 32, 128} */
    const uint32_t sc_mx_part[3] = {100, 37, 1}, cols_mx_part[3] = {20, 37, 1};
    const uint32_t sc_nv_part[3] = {200, 148, 70}, cols_nv_part[3] = {40, 37, 35};
    const uint32_t sc_nv_lt33[3] = {130, 64, 18}, cols_nv_lt33[3] = {26, 32, 6};

    for (uint32_t pb : {1u, 5u}) {
        for (uint32_t G : {1u, 3u, 7u, 32u, 128u}) {
            Pack pmx, pnv, pnv_lt33;
            build_pack(pmx, "mx-part", 0, sc_mx_part, cols_mx_part, pb, G, 1, 4, 7, wnm2, true);
            verify_pack_all_W(pmx);
            build_pack(pnv, "nv-part", 1, sc_nv_part, cols_nv_part, pb, G, 1, 4, 7, wnm2, true);
            verify_pack_all_W(pnv);
            build_pack(pnv_lt33, "nv-lt33", 1, sc_nv_lt33, cols_nv_lt33, pb, G, 0, 4, 7, wnm2, true);
            verify_pack_all_W(pnv_lt33);
        }
        printf("  pb=%u: MX partial, NV partial (scols>=33), NV scols<33 (ctx-63) x G={1,3,7,32,128} x W={0,1,2,4,8,16}: OK\n", pb);
    }

    /* 2. Inline (32 recs) vs Ptr (33 & 64 recs) direct-launch boundary check for MX & NV */
    {
        const uint32_t sc_mx[3] = {256, 128, 64}, cols_mx[3] = {32, 32, 32};
        const uint32_t sc_nv[3] = {512, 256, 128}, cols_nv[3] = {64, 64, 64};
        for (uint32_t nrec : {32u, 33u, 64u}) {
            Pack pmx, pnv;
            build_pack(pmx, "mx-bnd", 0, sc_mx, cols_mx, 5, 4, 1, 2, nrec, wnm2, true);
            verify_pack_all_W(pmx);
            build_pack(pnv, "nv-bnd", 1, sc_nv, cols_nv, 5, 4, 0, 2, nrec, wnm2, true);
            verify_pack_all_W(pnv);
        }
        printf("  direct-launch boundaries (n=32 inline, n=33/64 ptr) x MX/NV x W={0..16}: OK\n");
    }

    /* 3. Mixed-G batch (G=32 + G=64 in one submit, including fault isolation) for MX and NV */
    for (uint32_t kind : {0u, 1u}) {
        const uint32_t sc[3] = {kind ? 4096u : 2048u, kind ? 4096u : 2048u, kind ? 4096u : 2048u};
        const uint32_t cols[3] = {64u, 64u, 64u};
        Pack p32;
        build_pack(p32, kind ? "nv-mix" : "mx-mix", kind, sc, cols, 5, 32, kind ? 0 : 1, 4, 16, wnm2, true);
        REQ(wnr4_cuda_register_model(p32.blob.data(), p32.blob.size(), 0) == 0);
        /* Re-encode odd records at G = 64 using the SAME model */
        const size_t cap = (size_t)p32.g.raw * 2u + 65536u;
        std::vector<unsigned char> tcoded((size_t)p32.g.raw), encbuf(cap);
        for (uint32_t i = 1; i < 16u; i += 2u) {
            uint32_t flags = wnr4_prepare(&p32.g, p32.expected[i].data(), tcoded.data(), p32.cz, nullptr);
            size_t nb = wnr4_encode(&p32.model, &p32.g, tcoded.data(), flags, p32.owners[i], 64u, encbuf.data(), cap);
            REQ(nb > 0 && !(nb & 4095u));
            p32.enc[i].assign(encbuf.data(), encbuf.data() + nb);
        }
        const uint32_t n = 16u;
        const size_t slab_s = ((size_t)p32.g.scount[0] + p32.g.scount[1] + p32.g.scount[2] + 255u) & ~(size_t)255u;
        const size_t w0b = (size_t)p32.g.cells[0] * 16u, w1b = (size_t)p32.g.cells[1] * 16u, w2b = (size_t)p32.g.cells[2] * 16u;
        const size_t slab_w = (w0b + w1b + w2b + 255u) & ~(size_t)255u;
        size_t all = 0;
        std::vector<size_t> at(n), tot(n);
        for (uint32_t i = 0; i < n; i++) { at[i] = all; tot[i] = p32.enc[i].size(); all += (tot[i] + 255u) & ~(size_t)255u; }
        unsigned char *hrec, *ds_buf, *dw_buf;
        CK(cudaMallocHost(&hrec, all));
        CK(cudaMalloc(&ds_buf, slab_s * n));
        CK(cudaMalloc(&dw_buf, slab_w * n));
        std::vector<const unsigned char *> hrp(n);
        std::vector<unsigned char *> dsc(3 * n), dwt(3 * n);
        for (uint32_t i = 0; i < n; i++) {
            memcpy(hrec + at[i], p32.enc[i].data(), tot[i]);
            hrp[i] = hrec + at[i];
            dsc[3 * i + 0] = ds_buf + slab_s * i;
            dsc[3 * i + 1] = dsc[3 * i + 0] + p32.g.scount[0];
            dsc[3 * i + 2] = dsc[3 * i + 1] + p32.g.scount[1];
            dwt[3 * i + 0] = dw_buf + slab_w * i;
            dwt[3 * i + 1] = dwt[3 * i + 0] + w0b;
            dwt[3 * i + 2] = dwt[3 * i + 1] + w1b;
        }
        std::vector<int> st(n, 7);
        CK(cudaMemset(ds_buf, 0x5A, slab_s * n));
        CK(cudaMemset(dw_buf, 0x5A, slab_w * n));
        CK(cudaDeviceSynchronize());
        REQ(wnr4_cuda_decode_batch((int)n, p32.geo, hrp.data(), tot.data(), p32.owners.data(),
                                   p32.model.id, 0, dsc.data(), dwt.data(), st.data()) == 0);
        std::vector<unsigned char> sb(p32.g.scount[0] + p32.g.scount[1] + p32.g.scount[2]);
        std::vector<unsigned char> wb(w0b + w1b + w2b);
        for (uint32_t i = 0; i < n; i++) {
            REQ(st[i] == 0);
            CK(cudaMemcpy(sb.data(), dsc[3 * i], sb.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(wb.data(), dwt[3 * i], wb.size(), cudaMemcpyDeviceToHost));
            REQ(!memcmp(sb.data(), p32.expected[i].data(), sb.size()));
            REQ(!memcmp(wb.data(), p32.expected[i].data() + p32.g.woff[0], wb.size()));
        }
        /* Mixed-G fault isolation: flip 1 payload bit in record 3 (G=64) and record 4 (G=32) */
        for (uint32_t idx : {3u, 4u}) {
            Wnr4Record r;
            REQ(!wnr4_parse(hrp[idx], tot[idx], &p32.g, &r));
            hrec[at[idx] + (size_t)(r.data - hrp[idx]) + 130u] ^= 0x10u;
        }
        REQ(wnr4_cuda_decode_batch((int)n, p32.geo, hrp.data(), tot.data(), p32.owners.data(),
                                   p32.model.id, 0, dsc.data(), dwt.data(), st.data()) == 2);
        for (uint32_t i = 0; i < n; i++) REQ(st[i] == ((i == 3u || i == 4u) ? -1 : 0));
        CK(cudaFreeHost(hrec)); CK(cudaFree(ds_buf)); CK(cudaFree(dw_buf));
    }
    printf("  mixed-G batch (G=32 + G=64 in one submit) + fault isolation (MX & NV): OK\n");

    /* 4. Negative & Error cases */
    {
        /* Use small MX (with partial + full tiles) & NV packs for negative cases */
        const uint32_t sc_mx[3] = {96, 64, 37}, cols_mx[3] = {32, 32, 37};
        const uint32_t sc_nv[3] = {192, 128, 74}, cols_nv[3] = {64, 64, 37};
        Pack pmx, pnv;
        build_pack(pmx, "mx-neg", 0, sc_mx, cols_mx, 5, 4, 1, 4, 6, wnm2, true);
        build_pack(pnv, "nv-neg", 1, sc_nv, cols_nv, 5, 4, 1, 4, 2, wnm2, true);
        REQ(wnr4_cuda_register_model(pmx.blob.data(), pmx.blob.size(), 0) == 0);
        REQ(wnr4_cuda_register_model(pnv.blob.data(), pnv.blob.size(), 0) == 0);

        const size_t slab_s = ((size_t)pmx.g.scount[0] + pmx.g.scount[1] + pmx.g.scount[2] + 255u) & ~(size_t)255u;
        const size_t w0b = (size_t)pmx.g.cells[0] * 16u, w1b = (size_t)pmx.g.cells[1] * 16u, w2b = (size_t)pmx.g.cells[2] * 16u;
        const size_t slab_w = (w0b + w1b + w2b + 255u) & ~(size_t)255u;
        const size_t cap = (size_t)pmx.g.raw * 2u + 65536u;
        unsigned char *hbuf, *ds_buf, *dw_buf;
        CK(cudaMallocHost(&hbuf, cap * 6u));
        CK(cudaMalloc(&ds_buf, slab_s * 6u));
        CK(cudaMalloc(&dw_buf, slab_w * 6u));
        unsigned char *ds3[3] = {ds_buf, ds_buf + pmx.g.scount[0], ds_buf + pmx.g.scount[0] + pmx.g.scount[1]};
        unsigned char *dw3[3] = {dw_buf, dw_buf + w0b, dw_buf + w0b + w1b};

        /* 4a. Ceil underflow (s == 0 with max magnitude 5): test fast round (cell 5, mat 0),
         * fast tail round (cell 70, mat 0, lane 6 tail!), and partial tile (cell 35, mat 2) */
        for (int which = 0; which < 3; which++) {
            std::vector<unsigned char> coded((size_t)pmx.g.raw);
            uint32_t flags = wnr4_prepare(&pmx.g, pmx.expected[0].data(), coded.data(), 1, nullptr);
            uint32_t m = (which == 2) ? 2u : 0u;
            uint32_t cell = (which == 0) ? 5u : (which == 1 ? 70u : 35u);
            coded[pmx.g.soff[m] + cell] = 0u;
            /* Set every nibble of cell to magnitude 5 (nibble = 5) so mx == 5 (and tail nibbles also 5) */
            memset(coded.data() + pmx.g.woff[m] + (size_t)cell * 16u, 0x55, 16u);
            flags |= (1u << m);
            size_t nb = wnr4_encode(&pmx.model, &pmx.g, coded.data(), flags, 42u, 4u, hbuf, cap);
            REQ(nb > 0);
            std::vector<unsigned char> cpu_out((size_t)pmx.g.raw);
            REQ((wnr4_decode(&pmx.model, &pmx.g, hbuf, nb, 42u, cpu_out.data(), 1) & WNR4_ERR_CEIL) != 0);
            const unsigned char *rp = hbuf;
            uint32_t own = 42u;
            int st = 7;
            REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id, 0, ds3, dw3, &st) == 1 && st == -1);
        }

        /* 4b. CZ nibble 8 (FP4 -0) with WNR4_FLAG_CZ set: test fast round, fast tail (inside tail nibble!), and partial tile */
        for (int which = 0; which < 3; which++) {
            std::vector<unsigned char> coded((size_t)pmx.g.raw);
            uint32_t flags = wnr4_prepare(&pmx.g, pmx.expected[0].data(), coded.data(), 1, nullptr) | WNR4_FLAG_CZ;
            uint32_t m = (which == 2) ? 2u : 0u;
            uint32_t cell = (which == 0) ? 5u : (which == 1 ? 70u : 35u);
            /* Byte 15 high nibble is nibble 31 (inside tail for lane >= 2!) */
            coded[pmx.g.woff[m] + (size_t)cell * 16u + 15u] = (coded[pmx.g.woff[m] + (size_t)cell * 16u + 15u] & 0x0fu) | 0x80u;
            size_t nb = wnr4_encode(&pmx.model, &pmx.g, coded.data(), flags, 43u, 4u, hbuf, cap);
            REQ(nb > 0);
            std::vector<unsigned char> cpu_out((size_t)pmx.g.raw);
            REQ((wnr4_decode(&pmx.model, &pmx.g, hbuf, nb, 43u, cpu_out.data(), 1) & WNR4_ERR_CZ) != 0);
            const unsigned char *rp = hbuf;
            uint32_t own = 43u;
            int st = 7;
            REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id, 0, ds3, dw3, &st) == 1 && st == -1);
        }

        /* 4c. Valid record checks + unaligned dst_w, mismatched model kind, wrong owner, unknown model */
        memcpy(hbuf, pmx.enc[0].data(), pmx.enc[0].size());
        const unsigned char *rp = hbuf;
        size_t nb = pmx.enc[0].size();
        uint32_t own = pmx.owners[0];
        int st = 7;
        REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id, 0, ds3, dw3, &st) == 0 && st == 0);

        /* Unaligned dst_w -> -1 */
        unsigned char *dw_bad[3] = {dw3[0] + 4u, dw3[1], dw3[2]};
        st = 7;
        REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id, 0, ds3, dw_bad, &st) == 1 && st == -1);

        /* Mismatched model kind (MX geo with NV model ID) -> -1 */
        st = 7;
        REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pnv.model.id, 0, ds3, dw3, &st) == 1 && st == -1);

        /* Wrong owner -> -1 */
        uint32_t wrong_own = own ^ 1u;
        st = 7;
        REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &wrong_own, pmx.model.id, 0, ds3, dw3, &st) == 1 && st == -1);

        /* Unknown model -> 1 */
        st = 7;
        REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id ^ 1u, 0, ds3, dw3, &st) == 1 && st == 1);

        /* Bad initial state (x < WNR4_L) -> -1 */
        {
            std::vector<unsigned char> badst = pmx.enc[0];
            Wnr4Record r;
            REQ(!wnr4_parse(badst.data(), badst.size(), &pmx.g, &r));
            size_t data_at = (size_t)(r.data - badst.data());
            wnr4_put32(badst.data() + data_at + 12u, 0x1234u); /* lane 3 initial state < 65536 */
            memcpy(hbuf, badst.data(), badst.size());
            st = 7;
            REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &nb, &own, pmx.model.id, 0, ds3, dw3, &st) == 1 && st == -1);
        }

        /* Unused extra 2-byte word in tile 0 (ptr != nwords) -> -1 */
        {
            Wnr4Record r;
            REQ(!wnr4_parse(pmx.enc[0].data(), pmx.enc[0].size(), &pmx.g, &r));
            REQ(r.ntiles >= 2u);
            const size_t data_at = (size_t)(r.data - pmx.enc[0].data());
            const size_t cut = data_at + wnr4_u32(r.offsets + 4u);
            const size_t end = data_at + r.payload;
            std::vector<unsigned char> wide(pmx.enc[0].begin(), pmx.enc[0].begin() + cut);
            wide.push_back(0); wide.push_back(0);
            wide.insert(wide.end(), pmx.enc[0].begin() + cut, pmx.enc[0].begin() + end);
            wide.resize((end + 2u + 4095u) & ~(size_t)4095u, 0);
            wnr4_put32(wide.data() + 24u, r.payload + 2u);
            for (uint32_t k = 1u; k <= r.ntiles; k++) {
                unsigned char *pp = wide.data() + WNR4_HEADER_BYTES + 4u * k;
                wnr4_put32(pp, wnr4_u32(pp) + 2u);
            }
            REQ(!wnr4_parse(wide.data(), wide.size(), &pmx.g, &r));
            memcpy(hbuf, wide.data(), wide.size());
            size_t wnb = wide.size();
            st = 7;
            REQ(wnr4_cuda_decode_batch(1, pmx.geo, &rp, &wnb, &own, pmx.model.id, 0, ds3, dw3, &st) == 1 && st == -1);
        }

        /* 4d. 4-slot ring overflow -> -4 (WNR4_E_BUSY) */
        {
            size_t off = 0;
            const unsigned char *rps[5];
            size_t nbs[5];
            uint32_t owns[5];
            unsigned char *dss[15], *dws[15];
            for (int i = 0; i < 5; i++) {
                memcpy(hbuf + off, pmx.enc[i].data(), pmx.enc[i].size());
                rps[i] = hbuf + off;
                nbs[i] = pmx.enc[i].size();
                owns[i] = pmx.owners[i];
                off += (nbs[i] + 255u) & ~(size_t)255u;
                dss[3 * i + 0] = ds_buf + slab_s * i;
                dss[3 * i + 1] = dss[3 * i + 0] + pmx.g.scount[0];
                dss[3 * i + 2] = dss[3 * i + 1] + pmx.g.scount[1];
                dws[3 * i + 0] = dw_buf + slab_w * i;
                dws[3 * i + 1] = dws[3 * i + 0] + w0b;
                dws[3 * i + 2] = dws[3 * i + 1] + w1b;
            }
            int tk[5];
            for (int i = 0; i < 4; i++) {
                tk[i] = wnr4_cuda_batch_submit(1, pmx.geo, &rps[i], &nbs[i], &owns[i], pmx.model.id, 0, &dss[3 * i], &dws[3 * i], nullptr);
                REQ(tk[i] >= 0);
            }
            tk[4] = wnr4_cuda_batch_submit(1, pmx.geo, &rps[4], &nbs[4], &owns[4], pmx.model.id, 0, &dss[12], &dws[12], nullptr);
            REQ(tk[4] == -4);
            for (int i = 3; i >= 0; i--) {
                int s1 = 7;
                REQ(wnr4_cuda_batch_wait(tk[i], &s1) == 0 && s1 == 0);
            }
            REQ(wnr4_cuda_batch_wait(tk[0], &st) == -1);
        }

        CK(cudaFreeHost(hbuf)); CK(cudaFree(ds_buf)); CK(cudaFree(dw_buf));
        printf("  negative checks (ceil underflow, CZ -0, unaligned dst_w, wrong kind/owner, unknown model, bad state, unused word, 4-slot overflow -4): OK\n");
    }
    printf("=== Correctness Matrix PASSED ===\n\n");
}

static void bench_pack(const Pack &p, const std::vector<uint32_t> &batches) {
    const uint32_t n = (uint32_t)p.enc.size();
    const size_t raw = (size_t)p.g.raw;
    const size_t slab_s = ((size_t)p.g.scount[0] + p.g.scount[1] + p.g.scount[2] + 255u) & ~(size_t)255u;
    const size_t w0b = (size_t)p.g.cells[0] * 16u, w1b = (size_t)p.g.cells[1] * 16u, w2b = (size_t)p.g.cells[2] * 16u;
    const size_t slab_w = (w0b + w1b + w2b + 255u) & ~(size_t)255u;
    size_t all = 0;
    std::vector<size_t> at(n), tot(n);
    for (uint32_t i = 0; i < n; i++) {
        at[i] = all; tot[i] = p.enc[i].size(); all += (tot[i] + 255u) & ~(size_t)255u;
    }
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t flush_bytes = 2u * (size_t)l2 + (1u << 20);
    unsigned char *hrec, *drec, *ds_buf, *dw_buf, *dtab, *flush;
    uint32_t *derr;
    CK(cudaMallocHost(&hrec, all));
    CK(cudaMalloc(&drec, all));
    CK(cudaMalloc(&ds_buf, slab_s * n * 2u));
    CK(cudaMalloc(&dw_buf, slab_w * n * 2u));
    CK(cudaMalloc(&derr, 4u * n));
    CK(cudaMalloc(&flush, flush_bytes));
    std::vector<const unsigned char *> hrp(n);
    std::vector<unsigned char *> dsc(3 * n), dwt(3 * n), dsc2(3 * n), dwt2(3 * n);
    for (uint32_t i = 0; i < n; i++) {
        memcpy(hrec + at[i], p.enc[i].data(), tot[i]);
        hrp[i] = hrec + at[i];
        dsc[3 * i + 0] = ds_buf + slab_s * i;
        dsc[3 * i + 1] = dsc[3 * i + 0] + p.g.scount[0];
        dsc[3 * i + 2] = dsc[3 * i + 1] + p.g.scount[1];
        dwt[3 * i + 0] = dw_buf + slab_w * i;
        dwt[3 * i + 1] = dwt[3 * i + 0] + w0b;
        dwt[3 * i + 2] = dwt[3 * i + 1] + w1b;
        dsc2[3 * i + 0] = ds_buf + slab_s * (n + i);
        dsc2[3 * i + 1] = dsc2[3 * i + 0] + p.g.scount[0];
        dsc2[3 * i + 2] = dsc2[3 * i + 1] + p.g.scount[1];
        dwt2[3 * i + 0] = dw_buf + slab_w * (n + i);
        dwt2[3 * i + 1] = dwt2[3 * i + 0] + w0b;
        dwt2[3 * i + 2] = dwt2[3 * i + 1] + w1b;
    }
    CK(cudaMemcpy(drec, hrec, all, cudaMemcpyHostToDevice));
    std::vector<unsigned char> img(wnr4_cuda_table_bytes(&p.model));
    wnr4_cuda_table_image(&p.model, img.data());
    CK(cudaMalloc(&dtab, img.size()));
    CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
    Wnr4CudaTables tab;
    wnr4_cuda_tables_view(&p.model, dtab, &tab);
    REQ(wnr4_cuda_register_model(p.blob.data(), p.blob.size(), 0) == 0);

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    auto verify_range = [&](uint32_t b) {
        std::vector<unsigned char> sb(p.g.scount[0] + p.g.scount[1] + p.g.scount[2]);
        std::vector<unsigned char> wb(w0b + w1b + w2b);
        for (uint32_t i = 0; i < b; i++) {
            CK(cudaMemcpy(sb.data(), dsc[3 * i], sb.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(wb.data(), dwt[3 * i], wb.size(), cudaMemcpyDeviceToHost));
            REQ(!memcmp(sb.data(), p.expected[i].data(), sb.size()));
            REQ(!memcmp(wb.data(), p.expected[i].data() + p.g.woff[0], wb.size()));
        }
    };

    printf("  [kernel-only, L2 %d MiB flushed, median of 20]\n", l2 >> 20);
    for (uint32_t b : batches) {
        if (b > n) continue;

        /* Direct-launch (inline <= 32, ptr > 32) */
        {
            std::vector<Wnr4CudaDirectRec> mrecs(b);
            Wnr4Record r0 = {};
            for (uint32_t i = 0; i < b; i++) {
                Wnr4Record r;
                REQ(!wnr4_parse(p.enc[i].data(), p.enc[i].size(), &p.g, &r));
                if (i == 0) r0 = r;
                Wnr4CudaDirectRec rec = {};
                rec.rec = drec + at[i];
                for (uint32_t m = 0; m < 3u; m++) { rec.dst_s[m] = dsc[3 * i + m]; rec.dst_w[m] = dwt[3 * i + m]; }
                rec.owner = r.owner;
                rec.flags = r.flags;
                rec.anchors = (uint32_t)r.anchor[0] | (uint32_t)r.anchor[1] << 8 | (uint32_t)r.anchor[2] << 16;
                rec.rec_idx = i;
                rec.data_at = (uint32_t)(r.data - p.enc[i].data());
                rec.payload = r.payload;
                mrecs[i] = rec;
            }
            const size_t tiles = (size_t)r0.ntiles * b;
            const uint32_t W = wnr4_cuda_choose_warps(tiles);
            Wnr4CudaBatchGeo bg = {};
            bg.kind = p.g.kind; bg.G = p.G; bg.nrecs = b;
            for (uint32_t m = 0; m < 3u; m++) { bg.cells[m] = p.g.cells[m]; bg.scols[m] = p.g.scols[m]; }
            for (uint32_t m = 0; m < 4u; m++) bg.tile_base[m] = r0.tile_base[m];
            uint32_t blk = 0u;
            for (uint32_t m = 0; m < 3u; m++) {
                bg.mat_blocks[m] = blk;
                const uint32_t mt = (bg.tile_base[m + 1u] - bg.tile_base[m]) * b;
                blk += (mt + W - 1u) / W;
            }
            bg.mat_blocks[3] = blk;
            Wnr4InlineRecs irecs = {};
            Wnr4CudaDirectRec *dmrec = nullptr;
            if (b <= WNR4_CUDA_INLINE_RECS) {
                memcpy(irecs.r, mrecs.data(), sizeof(Wnr4CudaDirectRec) * b);
            } else {
                CK(cudaMalloc(&dmrec, sizeof(Wnr4CudaDirectRec) * b));
                CK(cudaMemcpy(dmrec, mrecs.data(), sizeof(Wnr4CudaDirectRec) * b, cudaMemcpyHostToDevice));
            }
            CK(cudaMemset(derr, 0, 4u * b));
            CK(cudaMemset(ds_buf, 0x5A, slab_s * b));
            CK(cudaMemset(dw_buf, 0x5A, slab_w * b));
            std::vector<float> td;
            for (int it = 0; it < 21; it++) {
                CK(cudaMemsetAsync(flush, it, flush_bytes, 0));
                CK(cudaEventRecord(e0, 0));
                if (b <= WNR4_CUDA_INLINE_RECS) CK(wnr4_cuda_launch_recs(tab, bg, irecs, W, derr, 0));
                else CK(wnr4_cuda_launch_recs_ptr(tab, bg, dmrec, W, derr, 0));
                CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                if (it) td.push_back(ms);
            }
            std::sort(td.begin(), td.end());
            std::vector<uint32_t> err(b);
            CK(cudaMemcpy(err.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < b; i++) REQ(err[i] == 0);
            verify_range(b);
            double ms = td[td.size() / 2];
            printf("    direct   b=%3u (tiles=%5zu W=%2u): %6.3f ms = %6.1f GB/s\n",
                   b, tiles, W, ms, (double)raw * b / ms / 1e6);
            if (dmrec) CK(cudaFree(dmrec));
        }

        /* Job-list API */
        {
            std::vector<Wnr4CudaRec> hinfo(b);
            std::vector<Wnr4CudaJob> by_m[3], jobs;
            for (uint32_t i = 0; i < b; i++) {
                unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
                unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
                REQ(!wnr4_cuda_plan(&p.g, &p.model, p.enc[i].data(), p.enc[i].size(), p.owners[i],
                                    drec + at[i], i, ds3, dw3, &hinfo[i], by_m));
            }
            const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
            const uint32_t W = wnr4_cuda_jobs(by_m, jobs);
            Wnr4CudaRec *dinfo; Wnr4CudaJob *djob;
            CK(cudaMalloc(&dinfo, sizeof(Wnr4CudaRec) * b));
            CK(cudaMalloc(&djob, sizeof(Wnr4CudaJob) * jobs.size()));
            CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4CudaRec) * b, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
            CK(cudaMemset(derr, 0, 4u * b));
            CK(cudaMemset(ds_buf, 0x5A, slab_s * b));
            CK(cudaMemset(dw_buf, 0x5A, slab_w * b));
            std::vector<float> td;
            for (int it = 0; it < 21; it++) {
                CK(cudaMemsetAsync(flush, it, flush_bytes, 0));
                CK(cudaEventRecord(e0, 0));
                CK(wnr4_cuda_launch(tab, djob, (uint32_t)jobs.size(), dinfo, derr, 0));
                CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                if (it) td.push_back(ms);
            }
            std::sort(td.begin(), td.end());
            std::vector<uint32_t> err(b);
            CK(cudaMemcpy(err.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < b; i++) REQ(err[i] == 0);
            verify_range(b);
            double ms = td[td.size() / 2];
            printf("    job-list b=%3u (tiles=%5zu W=%2u): %6.3f ms = %6.1f GB/s\n",
                   b, tiles, W, ms, (double)raw * b / ms / 1e6);
            CK(cudaFree(dinfo)); CK(cudaFree(djob));
        }
    }

    printf("  [bridge e2e from pinned host, median of 15]\n");
    std::vector<int> st(n);
    for (uint32_t b : batches) {
        if (b > n) continue;
        CK(cudaMemset(ds_buf, 0x5A, slab_s * b));
        CK(cudaMemset(dw_buf, 0x5A, slab_w * b));
        CK(cudaDeviceSynchronize());
        std::vector<double> ts;
        for (int it = 0; it < 16; it++) {
            auto t0 = std::chrono::steady_clock::now();
            int bad = wnr4_cuda_decode_batch((int)b, p.geo, hrp.data(), tot.data(), p.owners.data(),
                                             p.model.id, 0, dsc.data(), dwt.data(), st.data());
            auto t1 = std::chrono::steady_clock::now();
            REQ(bad == 0);
            if (it) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        verify_range(b);
        double ms_sync = ts[ts.size() / 2];
        double ms_a2 = 0.0;
        if (b <= n) {
            const int reps = 8;
            std::vector<double> ta;
            for (int it = 0; it < 7; it++) {
                auto t0 = std::chrono::steady_clock::now();
                int prev = -1;
                for (int r = 0; r < reps; r++) {
                    unsigned char *const *ds_cur = (r & 1) ? dsc2.data() : dsc.data();
                    unsigned char *const *dw_cur = (r & 1) ? dwt2.data() : dwt.data();
                    int tk = wnr4_cuda_batch_submit((int)b, p.geo, hrp.data(), tot.data(), p.owners.data(),
                                                    p.model.id, 0, ds_cur, dw_cur, nullptr);
                    REQ(tk >= 0);
                    if (prev >= 0) REQ(wnr4_cuda_batch_wait(prev, st.data()) == 0);
                    prev = tk;
                }
                REQ(wnr4_cuda_batch_wait(prev, st.data()) == 0);
                auto t1 = std::chrono::steady_clock::now();
                if (it) ta.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
            }
            std::sort(ta.begin(), ta.end());
            ms_a2 = ta[ta.size() / 2];
        }
        printf("    bridge   b=%3u: sync %6.3f ms (%5.1f GB/s) | async2 %6.3f ms (%5.1f GB/s)\n",
               b, ms_sync, (double)raw * b / ms_sync / 1e6, ms_a2, (double)raw * b / ms_a2 / 1e6);
    }

    /* Fault isolation check: flip 1 payload bit in odd records, verify even ones bit-exact */
    if (n >= 4) {
        std::vector<size_t> pos(n);
        for (uint32_t i = 1; i < n; i += 2) {
            Wnr4Record r;
            REQ(!wnr4_parse(hrp[i], tot[i], &p.g, &r));
            pos[i] = at[i] + (size_t)(r.data - hrp[i]) + (size_t)((i * 2654435761u + 12345u) % r.payload);
            hrec[pos[i]] ^= (unsigned char)(1u << (i % 8u));
        }
        CK(cudaMemset(ds_buf, 0x5A, slab_s * n));
        CK(cudaMemset(dw_buf, 0x5A, slab_w * n));
        CK(cudaDeviceSynchronize());
        int bad = wnr4_cuda_decode_batch((int)n, p.geo, hrp.data(), tot.data(), p.owners.data(),
                                         p.model.id, 0, dsc.data(), dwt.data(), st.data());
        REQ(bad == (int)(n / 2));
        std::vector<unsigned char> sb(p.g.scount[0] + p.g.scount[1] + p.g.scount[2]);
        std::vector<unsigned char> wb(w0b + w1b + w2b);
        for (uint32_t i = 0; i < n; i++) {
            if (i & 1u) {
                REQ(st[i] == -1);
                hrec[pos[i]] ^= (unsigned char)(1u << (i % 8u));
            } else {
                REQ(st[i] == 0);
                CK(cudaMemcpy(sb.data(), dsc[3 * i], sb.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(wb.data(), dwt[3 * i], wb.size(), cudaMemcpyDeviceToHost));
                REQ(!memcmp(sb.data(), p.expected[i].data(), sb.size()));
                REQ(!memcmp(wb.data(), p.expected[i].data() + p.g.woff[0], wb.size()));
            }
        }
        printf("  [fault isolation] %u/%u corrupted refused, %u/%u clean bit-exact OK\n", n / 2, n / 2, n - n / 2, n - n / 2);
    }

    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    CK(cudaFreeHost(hrec)); CK(cudaFree(drec)); CK(cudaFree(ds_buf)); CK(cudaFree(dw_buf));
    CK(cudaFree(derr)); CK(cudaFree(dtab)); CK(cudaFree(flush));
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool bench_only = (argc > 1 && !strcmp(argv[1], "--bench-only"));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, L2 %d MiB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20);
    Wnm2Tables wnm2 = load_v41_wnm2();
    printf("V4.1 ceil WNM2 loaded: %s\n\n", wnm2.loaded ? "yes" : "no (fallback synth)");

    if (!bench_only) run_correctness_matrix(wnm2);

    /* Qwen3.8 MXFP4 (raw = 2,611,200 B) */
    {
        const uint32_t sc[3] = {51200, 51200, 51200}, cols[3] = {80, 20, 80};
        for (uint32_t G : {32u, 128u}) {
            Pack p;
            build_pack(p, G == 32 ? "q38-mx-G32" : "q38-mx-G128", 0, sc, cols, 5, G, 1, 8, 64, wnm2);
            bench_pack(p, {1u, 8u, 16u, 32u, 64u});
        }
    }
    /* Qwen3.8 NVFP4 (raw = 2,764,800 B) */
    {
        const uint32_t sc[3] = {102400, 102400, 102400}, cols[3] = {160, 40, 160};
        for (uint32_t G : {32u, 128u}) {
            Pack p;
            build_pack(p, G == 32 ? "q38-nv-G32" : "q38-nv-G128", 1, sc, cols, 5, G, 0, 8, 64, wnm2);
            bench_pack(p, {1u, 8u, 16u, 32u, 64u});
        }
    }
    /* DeepSeek V4.1 REAP-192 MXFP4 (raw = 18,800,640 B) */
    {
        const uint32_t sc[3] = {368640, 368640, 368640}, cols[3] = {160, 72, 160};
        for (uint32_t G : {32u, 128u}) {
            Pack p;
            build_pack(p, G == 32 ? "v41-mx-G32" : "v41-mx-G128", 0, sc, cols, 5, G, 1, 4, 16, wnm2);
            bench_pack(p, {1u, 4u, 8u, 16u});
        }
    }
    wnr4_cuda_release_thread();
    printf("ALL OK\n");
    return 0;
}
