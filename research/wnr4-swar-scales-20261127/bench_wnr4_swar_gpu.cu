// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
/* Head-to-head GPU verification & L2-flushed benchmark:
 *   - Standard WNR4-MX (pb=1 & pb=5, absolute order-0 scales)
 *   - Prototype 3a: WNR4-MX + s_rec16 (pb=1 & pb=5, per-record 16-symbol scales)
 *   - Prototype 3b: WNR4-SWAR-8MAG (63-ctx 8-mag carry-free SWAR + aligned raw
 *     signs + MAGT 480-bit tail + s_rec16 + 4-slot cp.async ring + 32-bit hash)
 *
 * Datasets:
 *   1. DeepSeek V4.1 MXFP4 (typical, sampled from model_CFG_rule_g.wnm2, raw=18,800,640 B, G=32,64,128)
 *   2. DeepSeek V4.1 L39 E120 anchor-shifted (121,122,121) outliers (w2 ~65% 123, 27% 122, 8% 121/124)
 *   3. Escaped-scale stress (all 256 byte values 0..255 verified bit-exact on GPU)
 *   4. Qwen3.8 MXFP4 (raw=2,611,200 B, G=32,128)
 *
 * Build:
 *   research\wnr4-20261121\nvcc.cmd -O3 -std=c++17 -arch=sm_120 \
 *       -Xcompiler=/O2,/EHsc --ptxas-options=-v \
 *       research\wnr4-swar-scales-20261127\bench_wnr4_swar_gpu.cu \
 *       -o out\bench_wnr4_swar_gpu.exe
 * Run:
 *   out\bench_wnr4_swar_gpu.exe [--verify-only]
 */
#define NOMINMAX
#include "wnr4_swar_cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do {                                                                                  \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
        fprintf(stderr, "CUDA %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_));             \
        exit(1);                                                                                    \
    }                                                                                               \
} while (0)
#define REQ(x) do {                                                                                 \
    if (!(x)) {                                                                                     \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                                        \
        exit(1);                                                                                    \
    }                                                                                               \
} while (0)

static uint64_t g_rng = UINT64_C(0x9E3779B97F4A7C15);
static inline uint32_t rnd32(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 16);
}

struct Wnm2Tables {
    bool loaded = false;
    uint16_t wcdf[3][567][17];
    uint16_t scdf[3][257];
    uint32_t s_tot[3] = {0, 0, 0};
    uint32_t s_cdf[3][257];
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
    for (int m = 0; m < 3; m++) {
        t.s_cdf[m][0] = 0;
        t.s_tot[m] = 0;
        for (int k = 0; k < 257; k++, off += 2)
            t.scdf[m][k] = (uint16_t)(b[off] | (b[off + 1] << 8));
        for (int k = 0; k < 256; k++) {
            uint32_t freq = (uint32_t)t.scdf[m][k + 1] - t.scdf[m][k];
            uint32_t adj = freq > 1u ? freq - 1u : 0u;
            t.s_tot[m] += adj;
            t.s_cdf[m][k + 1] = t.s_tot[m];
        }
    }
    t.loaded = true;
    return t;
}

/* Sample a realistic MXFP4 raw slab from V4.1 ceil-form WNM2 tables.
 * mode = 0: typical V4.1 (anchors ~120,120,120)
 * mode = 1: L39 E120 anchor-shifted (121,122,121) outlier:
 *           w1/w3 shifted +1 (anchor 121), w2 ~65% 123, ~27% 122, ~4% 121, ~4% 124 (anchor 122)
 * mode = 2: escape stress (all 256 byte values 0..255 present, anchors ~120)
 */
static void sample_mx_slab(const Wnr4Geo &g, const Wnm2Tables &wnm2, unsigned char *raw, int mode) {
    REQ(wnm2.loaded);
    memset(raw, 0, (size_t)g.raw);
    for (uint32_t m = 0; m < 3u; m++) {
        unsigned char *so = raw + g.soff[m];
        unsigned char *wo = raw + g.woff[m];
        const unsigned char eff_anchor = (mode == 1) ? (m == 1u ? 122u : 121u) : wnm2.anchor[m];
        for (uint32_t c = 0; c < g.cells[m]; c++) {
            uint32_t sc = 0;
            if (mode == 1) {
                if (m == 1u) {
                    /* w2 realistic L39 E120 coded histogram: ~65% 123, ~27% 122, ~4% 121, ~4% 124 */
                    uint32_t r = rnd32() % 1000u;
                    sc = r < 650u ? 123u : r < 920u ? 122u : r < 960u ? 121u : 124u;
                } else {
                    /* w1 / w3 shifted +1 from typical V4.1 so coded anchor = 121 */
                    uint32_t u = rnd32() % wnm2.s_tot[m], lo = 0, hi = 256;
                    while (hi - lo > 1u) { uint32_t mid = (lo + hi) >> 1; if (wnm2.s_cdf[m][mid] <= u) lo = mid; else hi = mid; }
                    sc = lo + 1u;
                    if (sc > 254u) sc = 254u;
                }
            } else if (mode == 2 && c < 256u) {
                sc = c;
            } else {
                uint32_t u = rnd32() % wnm2.s_tot[m], lo = 0, hi = 256;
                while (hi - lo > 1u) { uint32_t mid = (lo + hi) >> 1; if (wnm2.s_cdf[m][mid] <= u) lo = mid; else hi = mid; }
                sc = lo;
            }
            uint32_t cls = wnr4_cls(sc, eff_anchor, 3);
            uint32_t pm = 8u, mx = 8u;
            uint32_t w[4] = {0, 0, 0, 0};
            for (uint32_t k = 0; k < 32u; k++) {
                const uint16_t *cdf = wnm2.wcdf[m][wnr4_wctx(cls, pm, mx)];
                uint32_t slot = rnd32() & 16383u;
                uint32_t y = wnr4_find16(cdf, slot);
                if (y == 8u) y = 0u; /* CZ */
                w[k / 8u] |= y << ((k & 7u) * 4u);
                pm = y & 7u;
                mx = wnr4_mx(mx, pm);
            }
            if (mode != 2) {
                if (mx == 5u && sc > 0u) {
                    sc--;
                    for (int q = 0; q < 4; q++) w[q] = wnr4_ceil_dbl8(w[q]);
                } else if (mx < 6u) {
                    w[3] = (w[3] & 0x0fffffffu) | ((6u | (rnd32() & 8u)) << 28);
                }
            }
            so[c] = (unsigned char)sc;
            memcpy(wo + (size_t)c * 16u, w, 16u);
        }
    }
}

/* Measure exact steady-state Phase-1 scale payload bytes of standard WNR4-MX (static order-0 256-sym) */
static double measure_wnr4_mx_scale_bytes(const Wnr4Model *model, const Wnr4Geo *g, const unsigned char *coded) {
    double bits = 0.0;
    for (uint32_t m = 0; m < 3u; m++) {
        const unsigned char *sc = coded + g->soff[m];
        const uint16_t *cdf = model->scdf + (size_t)m * 257u;
        uint32_t hist[256] = {0};
        for (uint32_t i = 0; i < g->scount[m]; i++) hist[sc[i]]++;
        for (uint32_t s = 0; s < 256u; s++) {
            if (!hist[s]) continue;
            double p = (double)((uint32_t)cdf[s + 1u] - cdf[s]) / (double)WNR4_TOTAL;
            bits += (double)hist[s] * (-std::log2(p));
        }
    }
    return bits / 8.0;
}

struct TrainedModels {
    std::vector<unsigned char> blob_pb1, blob_pb5, blob_swar;
    Wnr4Model m_pb1, m_pb5;
    Wnr4SwarModel m_swar;
    ~TrainedModels() { wnr4_model_free(&m_pb1); wnr4_model_free(&m_pb5); }
};

static void train_models(const Wnr4Geo &g, const Wnm2Tables &wnm2, uint32_t ntrain, TrainedModels &tm) {
    std::vector<uint64_t> c1(wnr4_counts_words(WNR4_KIND_MX, 1), 0);
    std::vector<uint64_t> c5(wnr4_counts_words(WNR4_KIND_MX, 5), 0);
    std::vector<uint64_t> cs(wnr4_swar_counts_words(), 0);
    std::vector<unsigned char> raw((size_t)g.raw), coded((size_t)g.raw);
    for (uint32_t i = 0; i < ntrain; i++) {
        sample_mx_slab(g, wnm2, raw.data(), 0);
        (void)wnr4_prepare(&g, raw.data(), coded.data(), 1, nullptr);
        unsigned char a[3];
        for (int m = 0; m < 3; m++) a[m] = wnr4_anchor(coded.data() + g.soff[m], g.scount[m]);
        wnr4_count(&g, coded.data(), a, 1, c1.data());
        wnr4_count(&g, coded.data(), a, 5, c5.data());
        wnr4_swar_count(&g, coded.data(), a, cs.data());
    }
    tm.blob_pb1.resize(wnr4_model_bytes(WNR4_KIND_MX, 1));
    tm.blob_pb5.resize(wnr4_model_bytes(WNR4_KIND_MX, 5));
    tm.blob_swar.resize(wnr4_swar_model_bytes());
    wnr4_model_build(WNR4_KIND_MX, 1, c1.data(), tm.blob_pb1.data());
    wnr4_model_build(WNR4_KIND_MX, 5, c5.data(), tm.blob_pb5.data());
    wnr4_swar_model_build(cs.data(), tm.blob_swar.data());
    REQ(!wnr4_model_open(tm.blob_pb1.data(), tm.blob_pb1.size(), &tm.m_pb1));
    REQ(!wnr4_model_open(tm.blob_pb5.data(), tm.blob_pb5.size(), &tm.m_pb5));
    REQ(!wnr4_swar_model_open(tm.blob_swar.data(), tm.blob_swar.size(), &tm.m_swar));
}

struct EncodedSuite {
    std::string name;
    Wnr4Geo g;
    uint32_t G;
    std::vector<std::vector<unsigned char>> expected;
    std::vector<std::vector<unsigned char>> enc_mx_pb1, enc_mx_pb5;
    std::vector<std::vector<unsigned char>> enc_3a_pb1, enc_3a_pb5;
    std::vector<std::vector<unsigned char>> enc_swar;
    std::vector<uint32_t> owners;
    double mean_mx_pb1 = 0, mean_mx_pb5 = 0, mean_3a_pb1 = 0, mean_3a_pb5 = 0, mean_swar = 0;
    size_t max_mx_pb1 = 0, max_mx_pb5 = 0, max_3a_pb1 = 0, max_3a_pb5 = 0, max_swar = 0;
    double mean_scale_mx = 0, mean_scale_3a = 0, mean_scale_swar = 0;
    Wnr4SwarStats sum_st = {};
};

static void build_suite(EncodedSuite &es, const char *name, const Wnr4Geo &g, const Wnm2Tables &wnm2,
                        const TrainedModels &tm, uint32_t G, uint32_t nrec, int mode) {
    es.name = name; es.g = g; es.G = G;
    es.expected.resize(nrec);
    es.enc_mx_pb1.resize(nrec); es.enc_mx_pb5.resize(nrec);
    es.enc_3a_pb1.resize(nrec); es.enc_3a_pb5.resize(nrec);
    es.enc_swar.resize(nrec);
    es.owners.resize(nrec);
    const size_t cap = (size_t)g.raw * 2u + 65536u;
    std::vector<unsigned char> coded((size_t)g.raw), buf(cap), chk((size_t)g.raw);
    memset(&es.sum_st, 0, sizeof(es.sum_st));
    for (uint32_t i = 0; i < nrec; i++) {
        es.expected[i].resize((size_t)g.raw);
        sample_mx_slab(g, wnm2, es.expected[i].data(), mode);
        uint32_t flags = wnr4_prepare(&g, es.expected[i].data(), coded.data(), 1, nullptr);
        es.owners[i] = (i << 16) | (i & 0xffu);

        size_t n1 = wnr4_encode(&tm.m_pb1, &g, coded.data(), flags, es.owners[i], G, buf.data(), cap);
        REQ(n1 > 0 && !(n1 & 4095u));
        es.enc_mx_pb1[i].assign(buf.data(), buf.data() + n1);

        size_t n5 = wnr4_encode(&tm.m_pb5, &g, coded.data(), flags, es.owners[i], G, buf.data(), cap);
        REQ(n5 > 0 && !(n5 & 4095u));
        es.enc_mx_pb5[i].assign(buf.data(), buf.data() + n5);

        size_t sc3a_1 = 0, sc3a_5 = 0;
        size_t na1 = wnr4_srec16_encode(&tm.m_pb1, &g, coded.data(), flags, es.owners[i], G, buf.data(), cap, &sc3a_1);
        REQ(na1 > 0 && !(na1 & 4095u));
        es.enc_3a_pb1[i].assign(buf.data(), buf.data() + na1);

        size_t na5 = wnr4_srec16_encode(&tm.m_pb5, &g, coded.data(), flags, es.owners[i], G, buf.data(), cap, &sc3a_5);
        REQ(na5 > 0 && !(na5 & 4095u));
        es.enc_3a_pb5[i].assign(buf.data(), buf.data() + na5);

        Wnr4SwarStats st;
        size_t nsw = wnr4_swar_encode(&tm.m_swar, &g, coded.data(), flags, es.owners[i], G, buf.data(), cap, &st);
        REQ(nsw > 0 && !(nsw & 4095u));
        es.enc_swar[i].assign(buf.data(), buf.data() + nsw);

        if (i == 0) {
            REQ(wnr4_decode(&tm.m_pb5, &g, es.enc_mx_pb5[0].data(), n5, es.owners[0], chk.data(), 4) == 0);
            REQ(chk == es.expected[0]);
            REQ(wnr4_srec16_decode(&tm.m_pb5, &g, es.enc_3a_pb5[0].data(), na5, es.owners[0], chk.data(), 4) == 0);
            REQ(chk == es.expected[0]);
            REQ(wnr4_swar_decode(&tm.m_swar, &g, es.enc_swar[0].data(), nsw, es.owners[0], chk.data(), 4) == 0);
            REQ(chk == es.expected[0]);
        }

        es.mean_mx_pb1 += (double)n1; es.max_mx_pb1 = std::max(es.max_mx_pb1, n1);
        es.mean_mx_pb5 += (double)n5; es.max_mx_pb5 = std::max(es.max_mx_pb5, n5);
        es.mean_3a_pb1 += (double)na1; es.max_3a_pb1 = std::max(es.max_3a_pb1, na1);
        es.mean_3a_pb5 += (double)na5; es.max_3a_pb5 = std::max(es.max_3a_pb5, na5);
        es.mean_swar += (double)nsw; es.max_swar = std::max(es.max_swar, nsw);

        es.mean_scale_mx += measure_wnr4_mx_scale_bytes(&tm.m_pb5, &g, coded.data());
        es.mean_scale_3a += (double)sc3a_5;
        es.mean_scale_swar += (double)(st.srec_header_bytes + st.scale_rans_bytes);
        es.sum_st.srec_header_bytes += st.srec_header_bytes;
        es.sum_st.scale_rans_bytes += st.scale_rans_bytes;
        es.sum_st.mag_rans_bytes += st.mag_rans_bytes;
        es.sum_st.sign_stream_bytes += st.sign_stream_bytes;
        es.sum_st.magt_saved_bits += st.magt_saved_bits;
        es.sum_st.tile_pad2_bytes += st.tile_pad2_bytes;
    }
    es.mean_mx_pb1 /= nrec; es.mean_mx_pb5 /= nrec;
    es.mean_3a_pb1 /= nrec; es.mean_3a_pb5 /= nrec;
    es.mean_swar /= nrec;
    es.mean_scale_mx /= nrec; es.mean_scale_3a /= nrec; es.mean_scale_swar /= nrec;
    const double raw = (double)g.raw;
    printf("\n=== Suite %-22s (G=%3u, n=%2u, raw=%zu B) ===\n", name, G, nrec, (size_t)g.raw);
    printf("  WNR4-MX pb=1 (abs scales): mean=%10.0f B (%.5f)  max=%8zu B\n",
           es.mean_mx_pb1, es.mean_mx_pb1 / raw, es.max_mx_pb1);
    printf("  WNR4-MX pb=5 (abs scales): mean=%10.0f B (%.5f)  max=%8zu B  scale_payload=%.0f B\n",
           es.mean_mx_pb5, es.mean_mx_pb5 / raw, es.max_mx_pb5, es.mean_scale_mx);
    printf("  3a WNR4-MX+s_rec16 pb=1  : mean=%10.0f B (%.5f)  max=%8zu B  delta_vs_pb1=%+.4f%%\n",
           es.mean_3a_pb1, es.mean_3a_pb1 / raw, es.max_3a_pb1, 100.0 * (es.mean_3a_pb1 - es.mean_mx_pb1) / raw);
    printf("  3a WNR4-MX+s_rec16 pb=5  : mean=%10.0f B (%.5f)  max=%8zu B  delta_vs_pb5=%+.4f%%  scale_payload=%.0f B (srec_hdr=%.0f B)\n",
           es.mean_3a_pb5, es.mean_3a_pb5 / raw, es.max_3a_pb5, 100.0 * (es.mean_3a_pb5 - es.mean_mx_pb5) / raw,
           es.mean_scale_3a, (double)es.sum_st.srec_header_bytes / nrec);
    printf("  3b WNR4-SWAR-8MAG (63ctx): mean=%10.0f B (%.5f)  max=%8zu B  delta_vs_pb5=%+.4f%%  delta_vs_3a_pb5=%+.4f%%\n",
           es.mean_swar, es.mean_swar / raw, es.max_swar,
           100.0 * (es.mean_swar - es.mean_mx_pb5) / raw, 100.0 * (es.mean_swar - es.mean_3a_pb5) / raw);
    printf("     3b breakdown/rec: srec_hdr=%.0f B, scale_rans=%.0f B, mag_rans=%.0f B, signs=%.0f B, pad2=%.0f B, MAGT_saved=%.0f bits (%.0f B)\n",
           (double)es.sum_st.srec_header_bytes / nrec, (double)es.sum_st.scale_rans_bytes / nrec,
           (double)es.sum_st.mag_rans_bytes / nrec, (double)es.sum_st.sign_stream_bytes / nrec,
           (double)es.sum_st.tile_pad2_bytes / nrec, (double)es.sum_st.magt_saved_bits / nrec,
           (double)es.sum_st.magt_saved_bits / (8.0 * nrec));
}

static void print_occupancy(void) {
    printf("\n=== Kernel Occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor) ===\n");
    int blks = 0;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    auto show_swar = [&](const char *lbl, auto fn, uint32_t W) {
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blks, fn, (int)(W * 32u), 0));
        printf("  %-26s W=%2u (%3u thr): %2d blocks/SM = %2d/%d warps/SM (%.1f%% occ)\n",
               lbl, W, W * 32u, blks, blks * (int)W, max_warps, 100.0 * (blks * (int)W) / max_warps);
    };
    show_swar("wnr4_swar_cuda_kernel", wnr4_swar_cuda_kernel<1u>, 1u);
    show_swar("wnr4_swar_cuda_kernel", wnr4_swar_cuda_kernel<2u>, 2u);
    show_swar("wnr4_swar_cuda_kernel", wnr4_swar_cuda_kernel<4u>, 4u);
    show_swar("wnr4_swar_cuda_kernel", wnr4_swar_cuda_kernel<8u>, 8u);
    show_swar("wnr4_swar_cuda_kernel", wnr4_swar_cuda_kernel<16u>, 16u);
    auto show_mx = [&](const char *lbl, auto fn, uint32_t W) {
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blks, fn, (int)(W * 32u), 0));
        printf("  %-26s W=%2u (%3u thr): %2d blocks/SM = %2d/%d warps/SM (%.1f%% occ)\n",
               lbl, W, W * 32u, blks, blks * (int)W, max_warps, 100.0 * (blks * (int)W) / max_warps);
    };
    show_mx("wnr4_mx_v2_pb1", wnr4_cuda_kernel<0u, false, 1u>, 1u);
    show_mx("wnr4_mx_v2_pb1", wnr4_cuda_kernel<0u, false, 2u>, 2u);
    show_mx("wnr4_mx_v2_pb1", wnr4_cuda_kernel<0u, false, 4u>, 4u);
    show_mx("wnr4_mx_v2_pb1", wnr4_cuda_kernel<0u, false, 8u>, 8u);
    show_mx("wnr4_mx_v2_pb1", wnr4_cuda_kernel<0u, false, 16u>, 16u);
    show_mx("wnr4_mx_v2_pb5", wnr4_cuda_kernel<0u, true, 1u>, 1u);
    show_mx("wnr4_mx_v2_pb5", wnr4_cuda_kernel<0u, true, 2u>, 2u);
    show_mx("wnr4_mx_v2_pb5", wnr4_cuda_kernel<0u, true, 4u>, 4u);
    show_mx("wnr4_mx_v2_pb5", wnr4_cuda_kernel<0u, true, 8u>, 8u);
    show_mx("wnr4_mx_v2_pb5", wnr4_cuda_kernel<0u, true, 16u>, 16u);
}

static void verify_and_bench_gpu(const EncodedSuite &es, const TrainedModels &tm,
                                 const std::vector<uint32_t> &batches, bool verify_only) {
    const uint32_t n = (uint32_t)es.expected.size();
    const size_t raw = (size_t)es.g.raw;
    const size_t s_tot = (size_t)es.g.scount[0] + es.g.scount[1] + es.g.scount[2];
    const size_t slab_s = (s_tot + 255u) & ~(size_t)255u;
    const size_t w0b = (size_t)es.g.cells[0] * 16u, w1b = (size_t)es.g.cells[1] * 16u, w2b = (size_t)es.g.cells[2] * 16u;
    const size_t slab_w = (w0b + w1b + w2b + 255u) & ~(size_t)255u;

    int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t flush_bytes = 2u * (size_t)l2 + (1u << 20);
    unsigned char *ds_buf, *dw_buf, *flush;
    uint32_t *derr;
    CK(cudaMalloc(&ds_buf, slab_s * n));
    CK(cudaMalloc(&dw_buf, slab_w * n));
    CK(cudaMalloc(&derr, 4u * n));
    CK(cudaMalloc(&flush, flush_bytes));

    std::vector<unsigned char *> dsc(3 * n), dwt(3 * n);
    for (uint32_t i = 0; i < n; i++) {
        dsc[3 * i + 0] = ds_buf + slab_s * i;
        dsc[3 * i + 1] = dsc[3 * i + 0] + es.g.scount[0];
        dsc[3 * i + 2] = dsc[3 * i + 1] + es.g.scount[1];
        dwt[3 * i + 0] = dw_buf + slab_w * i;
        dwt[3 * i + 1] = dwt[3 * i + 0] + w0b;
        dwt[3 * i + 2] = dwt[3 * i + 1] + w1b;
    }

    auto verify_range = [&](uint32_t cnt) {
        std::vector<unsigned char> sb(s_tot), wb(w0b + w1b + w2b);
        for (uint32_t i = 0; i < cnt; i++) {
            CK(cudaMemcpy(sb.data(), dsc[3 * i], sb.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(wb.data(), dwt[3 * i], wb.size(), cudaMemcpyDeviceToHost));
            REQ(!memcmp(sb.data(), es.expected[i].data(), sb.size()));
            REQ(!memcmp(wb.data(), es.expected[i].data() + es.g.woff[0], wb.size()));
        }
    };

    /* Upload WNR4-SWAR tables & records */
    std::vector<unsigned char> sw_img(wnr4_swar_cuda_table_bytes());
    wnr4_swar_cuda_table_image(&tm.m_swar, sw_img.data());
    unsigned char *dtab_sw; CK(cudaMalloc(&dtab_sw, sw_img.size()));
    CK(cudaMemcpy(dtab_sw, sw_img.data(), sw_img.size(), cudaMemcpyHostToDevice));
    Wnr4SwarCudaTables tab_sw; wnr4_swar_cuda_tables_view(&tm.m_swar, dtab_sw, &tab_sw);

    size_t tot_sw = 0; std::vector<size_t> at_sw(n);
    for (uint32_t i = 0; i < n; i++) { at_sw[i] = tot_sw; tot_sw += (es.enc_swar[i].size() + 255u) & ~(size_t)255u; }
    unsigned char *drec_sw; CK(cudaMalloc(&drec_sw, tot_sw));
    for (uint32_t i = 0; i < n; i++)
        CK(cudaMemcpy(drec_sw + at_sw[i], es.enc_swar[i].data(), es.enc_swar[i].size(), cudaMemcpyHostToDevice));

    /* 1. Verify WNR4-SWAR bit-exactness across EVERY W in {1, 2, 4, 8, 16} */
    for (uint32_t W : {1u, 2u, 4u, 8u, 16u}) {
        std::vector<Wnr4SwarCudaRec> hinfo(n);
        std::vector<Wnr4SwarCudaJob> by_m[3], jobs;
        for (uint32_t i = 0; i < n; i++) {
            unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
            unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
            REQ(!wnr4_swar_cuda_plan(&es.g, &tm.m_swar, es.enc_swar[i].data(), es.enc_swar[i].size(), es.owners[i],
                                     drec_sw + at_sw[i], i, ds3, dw3, &hinfo[i], by_m));
        }
        wnr4_swar_cuda_jobs(by_m, jobs, W);
        Wnr4SwarCudaRec *dinfo; Wnr4SwarCudaJob *djob;
        CK(cudaMalloc(&dinfo, sizeof(Wnr4SwarCudaRec) * n));
        CK(cudaMalloc(&djob, sizeof(Wnr4SwarCudaJob) * jobs.size()));
        CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4SwarCudaRec) * n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4SwarCudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(derr, 0, 4u * n));
        CK(cudaMemset(ds_buf, 0x5A, slab_s * n));
        CK(cudaMemset(dw_buf, 0x5A, slab_w * n));
        CK(wnr4_swar_cuda_launch(tab_sw, djob, (uint32_t)jobs.size(), dinfo, derr, W, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> errs(n); CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) REQ(errs[i] == 0);
        verify_range(n);
        CK(cudaFree(dinfo)); CK(cudaFree(djob));
    }
    printf("  [GPU verify] WNR4-SWAR bit-exact on %u/%u records across all W={1,2,4,8,16}\n", n, n);

    /* 2. GPU single-bit sign-stream flip check (flip 1 bit near tile end of odd records -> 100% rejected, even OK) */
    if (n >= 2) {
        std::vector<Wnr4SwarCudaRec> hinfo(n);
        std::vector<Wnr4SwarCudaJob> by_m[3], jobs;
        std::vector<size_t> flip_at(n, 0);
        std::vector<unsigned char> orig_b(n, 0);
        for (uint32_t i = 0; i < n; i++) {
            unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
            unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
            REQ(!wnr4_swar_cuda_plan(&es.g, &tm.m_swar, es.enc_swar[i].data(), es.enc_swar[i].size(), es.owners[i],
                                     drec_sw + at_sw[i], i, ds3, dw3, &hinfo[i], by_m));
            if (i & 1u) {
                Wnr4SwarRecord r;
                REQ(!wnr4_swar_parse(es.enc_swar[i].data(), es.enc_swar[i].size(), &es.g, &r));
                uint32_t o1 = wnr4_u32(r.offsets + 4);
                /* Tile 0's last 4 bytes are the first 32 raw sign bits! Flip 1 bit there. */
                size_t pos = (size_t)(r.data - es.enc_swar[i].data()) + o1 - 2u;
                flip_at[i] = pos;
                orig_b[i] = es.enc_swar[i][pos];
                unsigned char fb = orig_b[i] ^ (unsigned char)(1u << (i % 8u));
                CK(cudaMemcpy(drec_sw + at_sw[i] + pos, &fb, 1, cudaMemcpyHostToDevice));
            }
        }
        uint32_t W = wnr4_swar_cuda_jobs(by_m, jobs, 4u);
        Wnr4SwarCudaRec *dinfo; Wnr4SwarCudaJob *djob;
        CK(cudaMalloc(&dinfo, sizeof(Wnr4SwarCudaRec) * n));
        CK(cudaMalloc(&djob, sizeof(Wnr4SwarCudaJob) * jobs.size()));
        CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4SwarCudaRec) * n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4SwarCudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(derr, 0, 4u * n));
        CK(wnr4_swar_cuda_launch(tab_sw, djob, (uint32_t)jobs.size(), dinfo, derr, W, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> errs(n); CK(cudaMemcpy(errs.data(), derr, 4u * n, cudaMemcpyDeviceToHost));
        uint32_t caught = 0, clean = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (i & 1u) {
                REQ(errs[i] != 0);
                caught++;
                CK(cudaMemcpy(drec_sw + at_sw[i] + flip_at[i], &orig_b[i], 1, cudaMemcpyHostToDevice));
            } else {
                REQ(errs[i] == 0);
                clean++;
            }
        }
        printf("  [GPU fault isolation + sign-bit flip] %u/%u sign-flipped rejected, %u/%u clean OK\n",
               caught, n / 2, clean, n - n / 2);
        CK(cudaFree(dinfo)); CK(cudaFree(djob));
    }

    if (!verify_only) {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

        /* Upload standard WNR4-MX (pb=5 and pb=1) tables & records for head-to-head L2-flushed benchmark */
        auto bench_mx_model = [&](const char *label, const Wnr4Model &mdl,
                                  const std::vector<std::vector<unsigned char>> &encs) {
            std::vector<unsigned char> img(wnr4_cuda_table_bytes(&mdl));
            wnr4_cuda_table_image(&mdl, img.data());
            unsigned char *dtab; CK(cudaMalloc(&dtab, img.size()));
            CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
            Wnr4CudaTables tab; wnr4_cuda_tables_view(&mdl, dtab, &tab);
            size_t tot = 0; std::vector<size_t> at(n);
            for (uint32_t i = 0; i < n; i++) { at[i] = tot; tot += (encs[i].size() + 255u) & ~(size_t)255u; }
            unsigned char *drec; CK(cudaMalloc(&drec, tot));
            for (uint32_t i = 0; i < n; i++)
                CK(cudaMemcpy(drec + at[i], encs[i].data(), encs[i].size(), cudaMemcpyHostToDevice));

            printf("  [%s v2 kernel-only, L2 %d MiB flushed, median of 20]\n", label, l2 >> 20);
            for (uint32_t b : batches) {
                if (b > n) continue;
                double best_ms = 1e30; uint32_t best_W = 0; size_t tiles_out = 0;
                double w_ms[5] = {0};
                const uint32_t Ws[5] = {1u, 2u, 4u, 8u, 16u};
                for (int wi = 0; wi < 5; wi++) {
                    uint32_t W = Ws[wi];
                    wnr4_cuda_warps = W;
                    std::vector<Wnr4CudaRec> hinfo(b);
                    std::vector<Wnr4CudaJob> by_m[3], jobs;
                    for (uint32_t i = 0; i < b; i++) {
                        unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
                        unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
                        REQ(!wnr4_cuda_plan(&es.g, &mdl, encs[i].data(), encs[i].size(), es.owners[i],
                                            drec + at[i], i, ds3, dw3, &hinfo[i], by_m));
                    }
                    tiles_out = by_m[0].size() + by_m[1].size() + by_m[2].size();
                    wnr4_cuda_jobs(by_m, jobs);
                    Wnr4CudaRec *dinfo; Wnr4CudaJob *djob;
                    CK(cudaMalloc(&dinfo, sizeof(Wnr4CudaRec) * b));
                    CK(cudaMalloc(&djob, sizeof(Wnr4CudaJob) * jobs.size()));
                    CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4CudaRec) * b, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
                    CK(cudaMemset(derr, 0, 4u * b));
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
                    std::vector<uint32_t> errs(b); CK(cudaMemcpy(errs.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
                    for (uint32_t i = 0; i < b; i++) REQ(errs[i] == 0);
                    if (W == 4u) verify_range(b);
                    double med = td[td.size() / 2];
                    w_ms[wi] = med;
                    if (med < best_ms) { best_ms = med; best_W = W; }
                    CK(cudaFree(dinfo)); CK(cudaFree(djob));
                }
                printf("    b=%3u (tiles=%5zu best_W=%2u): %6.3f ms = %6.1f GB/s  [W1=%.3f W2=%.3f W4=%.3f W8=%.3f W16=%.3f ms]\n",
                       b, tiles_out, best_W, best_ms, (double)raw * b / best_ms / 1e6,
                       w_ms[0], w_ms[1], w_ms[2], w_ms[3], w_ms[4]);
            }
            CK(cudaFree(drec)); CK(cudaFree(dtab));
        };

        bench_mx_model("WNR4-MX pb=5", tm.m_pb5, es.enc_mx_pb5);
        bench_mx_model("WNR4-MX pb=1", tm.m_pb1, es.enc_mx_pb1);

        /* Benchmark WNR4-SWAR across W in {1, 2, 4, 8, 16} */
        printf("  [WNR4-SWAR-8MAG kernel-only, L2 %d MiB flushed, median of 20]\n", l2 >> 20);
        for (uint32_t b : batches) {
            if (b > n) continue;
            double best_ms = 1e30; uint32_t best_W = 0; size_t tiles_out = 0;
            double w_ms[5] = {0};
            const uint32_t Ws[5] = {1u, 2u, 4u, 8u, 16u};
            for (int wi = 0; wi < 5; wi++) {
                uint32_t W = Ws[wi];
                std::vector<Wnr4SwarCudaRec> hinfo(b);
                std::vector<Wnr4SwarCudaJob> by_m[3], jobs;
                for (uint32_t i = 0; i < b; i++) {
                    unsigned char *ds3[3] = {dsc[3 * i], dsc[3 * i + 1], dsc[3 * i + 2]};
                    unsigned char *dw3[3] = {dwt[3 * i], dwt[3 * i + 1], dwt[3 * i + 2]};
                    REQ(!wnr4_swar_cuda_plan(&es.g, &tm.m_swar, es.enc_swar[i].data(), es.enc_swar[i].size(),
                                             es.owners[i], drec_sw + at_sw[i], i, ds3, dw3, &hinfo[i], by_m));
                }
                tiles_out = by_m[0].size() + by_m[1].size() + by_m[2].size();
                wnr4_swar_cuda_jobs(by_m, jobs, W);
                Wnr4SwarCudaRec *dinfo; Wnr4SwarCudaJob *djob;
                CK(cudaMalloc(&dinfo, sizeof(Wnr4SwarCudaRec) * b));
                CK(cudaMalloc(&djob, sizeof(Wnr4SwarCudaJob) * jobs.size()));
                CK(cudaMemcpy(dinfo, hinfo.data(), sizeof(Wnr4SwarCudaRec) * b, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4SwarCudaJob) * jobs.size(), cudaMemcpyHostToDevice));
                CK(cudaMemset(derr, 0, 4u * b));
                std::vector<float> td;
                for (int it = 0; it < 21; it++) {
                    CK(cudaMemsetAsync(flush, it, flush_bytes, 0));
                    CK(cudaEventRecord(e0, 0));
                    CK(wnr4_swar_cuda_launch(tab_sw, djob, (uint32_t)jobs.size(), dinfo, derr, W, 0));
                    CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (it) td.push_back(ms);
                }
                std::sort(td.begin(), td.end());
                std::vector<uint32_t> errs(b); CK(cudaMemcpy(errs.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
                for (uint32_t i = 0; i < b; i++) REQ(errs[i] == 0);
                double med = td[td.size() / 2];
                w_ms[wi] = med;
                if (med < best_ms) { best_ms = med; best_W = W; }
                CK(cudaFree(dinfo)); CK(cudaFree(djob));
            }
            printf("    b=%3u (tiles=%5zu best_W=%2u): %6.3f ms = %6.1f GB/s  [W1=%.3f W2=%.3f W4=%.3f W8=%.3f W16=%.3f ms]\n",
                   b, tiles_out, best_W, best_ms, (double)raw * b / best_ms / 1e6,
                   w_ms[0], w_ms[1], w_ms[2], w_ms[3], w_ms[4]);
        }
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }

    CK(cudaFree(drec_sw)); CK(cudaFree(dtab_sw));
    CK(cudaFree(ds_buf)); CK(cudaFree(dw_buf)); CK(cudaFree(derr)); CK(cudaFree(flush));
}

static void print_ram_cache_table(size_t max_mx_pb5, size_t max_3a_pb5, size_t max_swar) {
    printf("\n=== V4WexCrc Fixed-Slot RAM Cache Impact (worst-case L39 E120 outlier slot size) ===\n");
    const uint64_t gib16 = UINT64_C(16) * 1024u * 1024u * 1024u;
    const uint64_t gib32 = UINT64_C(32) * 1024u * 1024u * 1024u;
    auto row = [&](const char *lbl, size_t max_b, size_t base_b) {
        uint64_t s16 = gib16 / max_b, s32 = gib32 / max_b;
        uint64_t b16 = gib16 / base_b, b32 = gib32 / base_b;
        printf("  %-26s max=%8zu B (%.2f MB) | 16 GiB: %4llu slots (%+4lld, %+.2f%%) | 32 GiB: %4llu slots (%+4lld, %+.2f%%)\n",
               lbl, max_b, (double)max_b / 1e6,
               (unsigned long long)s16, (long long)(s16 - b16), 100.0 * ((double)s16 - (double)b16) / (double)b16,
               (unsigned long long)s32, (long long)(s32 - b32), 100.0 * ((double)s32 - (double)b32) / (double)b32);
    };
    row("WNR4-MX pb=5 (baseline)", max_mx_pb5, max_mx_pb5);
    row("3a WNR4-MX + s_rec16 pb=5", max_3a_pb5, max_mx_pb5);
    row("3b WNR4-SWAR-8MAG", max_swar, max_mx_pb5);
}

int main(int argc, char **argv) {
    bool verify_only = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verify-only")) verify_only = true;
    }

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, L2 %d MiB, mode=%s)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20,
           verify_only ? "verify-only" : "verify+bench");
    Wnm2Tables wnm2 = load_v41_wnm2();
    REQ(wnm2.loaded);
    printf("Loaded model_CFG_rule_g.wnm2 (exact V4.1 ceil-form distributions)\n");
    print_occupancy();

    /* 1. DeepSeek V4.1 MXFP4 geometry (raw = 18,800,640 B) */
    {
        Wnr4Geo g;
        const uint32_t sc[3] = {368640, 368640, 368640}, cols[3] = {160, 72, 160};
        const uint32_t so[3] = {0, sc[0], sc[0] + sc[1]};
        const uint32_t wo[3] = {so[2] + sc[2], so[2] + sc[2] + sc[0] * 16u, so[2] + sc[2] + (sc[0] + sc[1]) * 16u};
        REQ(!wnr4_geo(&g, WNR4_KIND_MX, sc, cols, so, wo));
        TrainedModels tm;
        train_models(g, wnm2, 4, tm);

        /* Typical V4.1 across G = 32, 64, 128 */
        for (uint32_t G : {32u, 64u, 128u}) {
            EncodedSuite es;
            char nm[64]; snprintf(nm, sizeof(nm), "v41-typical-G%u", G);
            build_suite(es, nm, g, wnm2, tm, G, verify_only ? 4u : 16u, 0);
            verify_and_bench_gpu(es, tm, {1u, 4u, 8u, 16u}, verify_only);
        }

        /* L39 E120 anchor-shifted (121,122,121) outliers across G = 32, 64, 128 */
        size_t worst_mx = 0, worst_3a = 0, worst_swar = 0;
        for (uint32_t G : {32u, 64u, 128u}) {
            EncodedSuite es;
            char nm[64]; snprintf(nm, sizeof(nm), "v41-L39E120-G%u", G);
            build_suite(es, nm, g, wnm2, tm, G, verify_only ? 4u : 8u, 1);
            if (G == 128u) { worst_mx = es.max_mx_pb5; worst_3a = es.max_3a_pb5; worst_swar = es.max_swar; }
            verify_and_bench_gpu(es, tm, {1u, 4u, 8u}, verify_only || G != 128u);
        }
        print_ram_cache_table(worst_mx, worst_3a, worst_swar);

        /* Escaped scales stress (all 256 byte values 0..255) verified bit-exact on GPU */
        {
            EncodedSuite es;
            build_suite(es, "v41-escapes-0..255", g, wnm2, tm, 128u, 2u, 2);
            verify_and_bench_gpu(es, tm, {1u}, true);
        }
    }

    /* 2. Qwen3.8 MXFP4 geometry (raw = 2,611,200 B) */
    {
        Wnr4Geo g;
        const uint32_t sc[3] = {51200, 51200, 51200}, cols[3] = {80, 20, 80};
        const uint32_t so[3] = {0, sc[0], sc[0] + sc[1]};
        const uint32_t wo[3] = {so[2] + sc[2], so[2] + sc[2] + sc[0] * 16u, so[2] + sc[2] + (sc[0] + sc[1]) * 16u};
        REQ(!wnr4_geo(&g, WNR4_KIND_MX, sc, cols, so, wo));
        TrainedModels tm;
        train_models(g, wnm2, 8, tm);

        for (uint32_t G : {32u, 128u}) {
            EncodedSuite es;
            char nm[64]; snprintf(nm, sizeof(nm), "q38-mx-G%u", G);
            build_suite(es, nm, g, wnm2, tm, G, verify_only ? 8u : 64u, 0);
            verify_and_bench_gpu(es, tm, {1u, 8u, 16u, 32u, 64u}, verify_only);
        }
    }

    printf("\nALL OK\n");
    return 0;
}
