/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Live GPU verification suite for libwex's unified CUDA device backend
 * (libwex_cuda.dll / libwex_cuda.so).
 * Dynamically loads the backend shared library, resolves wex_dev_get_ops(),
 * and verifies all 4 GPU codecs against their CPU reference decoders:
 *   1. WNR3 (plain & CZ, host strict/scratch, direct-to-device, E8M0->f32, corrupt untouched)
 *   2. WNR4-MX & WNR4-NV (cz=0/1, pb=1/5, waves {1,2,4,6}, host strict/scratch,
 *      direct-to-device + E8M0->f32 bit-exact vs ldexpf(1.0f, e-127), mixed corrupt 2-stream)
 *   3. WNR4-I4 (int4-g64 G=4 & G=32, planar/pairs & mixed-G, host strict/scratch,
 *      direct-to-device, corrupt / G>64 / unaligned checks)
 *   4. WNR4-GG (Q2_0 + IQ4_NL + verbatim slices, heterogeneous per-record geos,
 *      host strict/scratch, direct-to-device, corrupt checks)
 *   5. Thread churn (32 rounds x 4 fresh threads across all 4 codecs, bounded VRAM) */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../src/wex_dev_ops.h"
#include "../../wnr4/wnr4.h"
#include "../../wnr4/wnr4_i4.h"
#include "../../wnr4/wnr4_gg.h"

#ifndef WEX_ENTRY_WNR4
#define WEX_ENTRY_WNR4 UINT32_C(8)
#endif
#ifndef WEX_ENTRY_WNR4_NV
#define WEX_ENTRY_WNR4_NV UINT32_C(9)
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    std::exit(1); } } while (0)
#define REQ(c) do { if (!(c)) { \
    std::fprintf(stderr, "CHECK FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    std::exit(1); } } while (0)

static const wex_dev_ops_t *g_ops = nullptr;

static uint32_t xorshift32(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

static bool all_byte(const std::vector<uint8_t> &v, uint8_t b) {
    for (uint8_t c : v) if (c != b) return false;
    return true;
}

static uint32_t float_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

static void verify_e8m0_f32_exact(const uint8_t *expected_e8m0, const uint8_t *got_f32_bytes, size_t nscales) {
    for (size_t i = 0; i < nscales; i++) {
        const int e = (int)expected_e8m0[i];
        const float ref_f = std::ldexp(1.0f, e - 127);
        const uint32_t ref_u = float_bits(ref_f);
        uint32_t got_u = 0;
        std::memcpy(&got_u, got_f32_bytes + i * 4u, 4);
        if (got_u != ref_u) {
            std::fprintf(stderr, "E8M0 mismatch at %zu (e=%d): got 0x%08x expected 0x%08x\n",
                         i, e, got_u, ref_u);
            std::exit(1);
        }
    }
}

/* ---- 1. WNR3 fixtures & tests ------------------------------------------- */

struct Nr3Rec {
    WexEntry e;
    std::vector<uint8_t> expected;
    std::vector<uint8_t> record;
};

static WexEntry make_mx_entry(uint32_t rows0, uint32_t scols0, uint32_t rows1, uint32_t scols1, uint32_t flags) {
    WexEntry e;
    std::memset(&e, 0, sizeof(e));
    const uint32_t r[3] = {rows0, rows1, rows0};
    const uint32_t sc[3] = {scols0, scols1, scols0};
    uint32_t soff = 0;
    for (int m = 0; m < 3; m++) {
        e.scale_shapes[m][0] = (uint16_t)r[m];
        e.scale_shapes[m][1] = (uint16_t)sc[m];
        e.scale_offsets[m] = soff;
        soff += r[m] * sc[m];
    }
    e.scale_bytes = soff;
    uint32_t woff = soff;
    for (int m = 0; m < 3; m++) {
        e.weight_shapes[m][0] = (uint16_t)r[m];
        e.weight_shapes[m][1] = (uint16_t)(sc[m] * 16u);
        e.weight_offsets[m] = woff;
        woff += r[m] * sc[m] * 16u;
    }
    e.weight_bytes = woff - soff;
    e.flags = flags;
    return e;
}

static WexEntry make_nv_entry(uint32_t rows0, uint32_t scols0, uint32_t rows1, uint32_t scols1, uint32_t flags) {
    WexEntry e;
    std::memset(&e, 0, sizeof(e));
    const uint32_t r[3] = {rows0, rows1, rows0};
    const uint32_t sc[3] = {scols0, scols1, scols0};
    uint32_t soff = 0;
    for (int m = 0; m < 3; m++) {
        e.scale_shapes[m][0] = (uint16_t)r[m];
        e.scale_shapes[m][1] = (uint16_t)sc[m];
        e.scale_offsets[m] = soff;
        soff += r[m] * sc[m];
    }
    e.scale_bytes = soff;
    uint32_t woff = soff;
    for (int m = 0; m < 3; m++) {
        e.weight_shapes[m][0] = (uint16_t)r[m];
        e.weight_shapes[m][1] = (uint16_t)(sc[m] * 8u);
        e.weight_offsets[m] = woff;
        woff += r[m] * sc[m] * 8u;
    }
    e.weight_bytes = woff - soff;
    e.flags = flags;
    return e;
}

static void fill_mx_slab(const WexEntry &e, uint8_t *raw, uint32_t seed, int inject_edge_scales) {
    static const uint8_t general[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    static const uint8_t even[8] = {0, 0, 2, 2, 4, 5, 6, 7};
    uint32_t s = seed;
    for (int m = 0; m < 3; m++) {
        const uint32_t blocks = (uint32_t)e.scale_shapes[m][0] * e.scale_shapes[m][1];
        uint8_t *sc = raw + e.scale_offsets[m];
        uint8_t *w = raw + e.weight_offsets[m];
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t kind = xorshift32(&s) % 8u;
            uint8_t *q = w + (size_t)b * 16u;
            sc[b] = (uint8_t)(118u + xorshift32(&s) % 5u);
            if (kind == 4u && b % 37u == 0u) sc[b] = (uint8_t)(b & 1u ? 255u : 254u);
            if (kind == 6u && b % 41u == 0u) sc[b] = 0u;
            for (uint32_t k = 0; k < 32u; k++) {
                const uint32_t r = xorshift32(&s);
                const uint32_t a = kind < 4u ? general[r & 15u] : kind < 6u ? even[r & 7u] : kind == 6u ? (r & 7u) % 5u : 0u;
                uint32_t x = a | ((r >> 8) & 8u);
                if (k < 2u && kind < 6u) x = (x & 8u) | 7u;
                if (k & 1u) q[k / 2u] = (uint8_t)(q[k / 2u] | (x << 4));
                else q[k / 2u] = (uint8_t)x;
            }
        }
        if (inject_edge_scales && blocks >= 5u) {
            /* Ensure e in {0, 1, 127, 254, 255} all occur on non-ceil-modified blocks (odd mag 1 present). */
            static const uint8_t edges[5] = {0u, 1u, 127u, 254u, 255u};
            for (uint32_t b = 0; b < 5u; b++) {
                sc[b] = edges[b];
                w[(size_t)b * 16u] = 0x71u; /* has mag 1 and mag 7 -> never ceil-halved */
            }
        }
    }
}

static void nr3_views(const WexEntry &e, uint8_t *raw, uint8_t *s[3], uint8_t *w[3]) {
    for (int m = 0; m < 3; m++) {
        s[m] = raw + e.scale_offsets[m];
        w[m] = raw + e.weight_offsets[m];
    }
}

static Nr3Rec build_nr3_fixture(int cz, uint32_t seed, WexNr2Model *model_out, std::vector<uint8_t> *blob_out) {
    WexEntry e = make_mx_entry(256u, 64u, 256u, 64u, WEX_ENTRY_NEURAL_RANS3);
    WexNr2Layout layout;
    wex_nr2_layout(&e, &layout);
    const size_t raw = (size_t)e.scale_bytes + e.weight_bytes;
    uint64_t r64 = 0, aligned = 0;
    REQ(!wex_entry_raw_sizes(&e, &r64, &aligned));

    std::vector<uint8_t> src(raw), coded(raw), out(raw * 3u + 65536u);
    uint8_t *s[3], *w[3];
    std::vector<uint64_t> wc(3u * 567u * 16u, 0), sc(3u * 256u, 0);
    for (uint32_t i = 1; i <= 2u; i++) {
        fill_mx_slab(e, src.data(), seed + i * 1009u, 1);
        coded = src;
        if (cz) wex_nr3_canonical_zero(coded.data() + e.scale_bytes, e.weight_bytes);
        nr3_views(e, coded.data(), s, w);
        wex_nr3_ceil_forward(&layout, s, w, NULL);
        wex_nr3_count(&layout, (const unsigned char *const *)s, (const unsigned char *const *)w, wc.data(), sc.data());
    }
    blob_out->resize(WEX_NR2_MODEL_BYTES);
    wex_nr2_model_build(wc.data(), sc.data(), blob_out->data());
    REQ(!wex_nr2_model_open(blob_out->data(), blob_out->size(), model_out));
    REQ(g_ops->register_model(0, blob_out->data(), blob_out->size()) == 0);
    REQ(g_ops->register_model(0, blob_out->data(), blob_out->size()) == 0); /* idempotent */

    fill_mx_slab(e, src.data(), seed + 4242u, 1);
    std::vector<uint8_t> expected = src;
    if (cz) wex_nr3_canonical_zero(expected.data() + e.scale_bytes, e.weight_bytes);
    coded = expected;
    nr3_views(e, coded.data(), s, w);
    const uint32_t flags = wex_nr3_ceil_forward(&layout, s, w, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
    size_t written = 0;
    std::fill(out.begin(), out.end(), 0);
    REQ(!wex_nr3_encode(model_out, &layout, (const unsigned char *const *)s, (const unsigned char *const *)w,
                        2048u, 512u, flags, 0x00020007u, out.data(), out.size(), &written));
    Nr3Rec rec;
    rec.e = e;
    rec.e.total_bytes = (uint32_t)((written + 4095u) & ~(size_t)4095u);
    REQ(rec.e.total_bytes < aligned);
    rec.record.assign(out.begin(), out.begin() + rec.e.total_bytes);
    rec.expected = expected;

    /* Cross-check CPU reference */
    std::vector<uint8_t> cpu_slab((size_t)aligned, 0x5Au);
    const size_t at = wex_record_read_offset(&rec.e);
    std::memcpy(cpu_slab.data() + at, rec.record.data(), rec.record.size());
    REQ(wex_decode_nr2_record(&rec.e, cpu_slab.data(), cpu_slab.size(), model_out) == 0);
    REQ(std::memcmp(cpu_slab.data(), rec.expected.data(), raw) == 0);
    return rec;
}

static void test_wnr3(Nr3Rec *churn_rec_out) {
    cudaStream_t user_stream = nullptr;
    CK(cudaStreamCreateWithFlags(&user_stream, cudaStreamNonBlocking));

    for (int cz = 0; cz < 2; cz++) {
        WexNr2Model model;
        std::vector<uint8_t> blob;
        Nr3Rec good = build_nr3_fixture(cz, 1000u + (uint32_t)cz * 77u, &model, &blob);
        if (cz == 0 && churn_rec_out) *churn_rec_out = good;

        /* Corrupt copy: flip a weight payload bit */
        Nr3Rec bad = good;
        WexNr2Layout layout;
        WexNr2Record view;
        wex_nr2_layout(&bad.e, &layout);
        REQ(!wex_nr3_parse(bad.record.data(), bad.record.size(), &layout, model.id, &view));
        bad.record[view.metadata_bytes + view.scale_payload_bytes + 12u] ^= 0x01u;

        /* 1) Host strict & host scratch on mixed [good, bad, good] */
        const size_t raw = (size_t)good.e.scale_bytes + good.e.weight_bytes;
        const WexEntry *es[3] = {&good.e, &bad.e, &good.e};
        const uint8_t *rs[3] = {good.record.data(), bad.record.data(), good.record.data()};
        std::vector<uint8_t> h0(raw, 0xA5u), h1(raw, 0xA5u), h2(raw, 0xA5u);
        uint8_t *hd[3] = {h0.data(), h1.data(), h2.data()};
        int st[3] = {7, 7, 7};

        REQ(g_ops->decode_expert_host(3, es, rs, 0, hd, 0, st) == 1);
        REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
        REQ(h0 == good.expected && all_byte(h1, 0xA5u) && h2 == good.expected);

        std::fill(h0.begin(), h0.end(), 0xA5u);
        std::fill(h1.begin(), h1.end(), 0xA5u);
        std::fill(h2.begin(), h2.end(), 0xA5u);
        REQ(g_ops->decode_expert_host(3, es, rs, 0, hd, 1, st) == 1);
        REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
        REQ(h0 == good.expected && h2 == good.expected);

        /* 2) Direct-to-device (expand_e8m0_f32=0 & =1) with NULL and user_stream */
        uint8_t *ds_dev[9] = {}, *dw_dev[9] = {};
        for (int i = 0; i < 3; i++) {
            for (int m = 0; m < 3; m++) {
                const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
                const size_t wb = (size_t)good.e.weight_shapes[m][0] * good.e.weight_shapes[m][1];
                CK(cudaMalloc(&ds_dev[3 * i + m], sb * 4u));
                CK(cudaMalloc(&dw_dev[3 * i + m], wb));
                CK(cudaMemset(ds_dev[3 * i + m], 0xA5, sb * 4u));
                CK(cudaMemset(dw_dev[3 * i + m], 0xA5, wb));
            }
        }
        CK(cudaDeviceSynchronize());

        REQ(g_ops->decode_expert_device(3, es, rs, 0, ds_dev, dw_dev, nullptr, 0, st) == 1);
        REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
        for (int i : {0, 2}) {
            for (int m = 0; m < 3; m++) {
                const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
                const size_t wb = (size_t)good.e.weight_shapes[m][0] * good.e.weight_shapes[m][1];
                std::vector<uint8_t> gs(sb), gw(wb);
                CK(cudaMemcpy(gs.data(), ds_dev[3 * i + m], sb, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], wb, cudaMemcpyDeviceToHost));
                REQ(std::memcmp(gs.data(), good.expected.data() + good.e.scale_offsets[m], sb) == 0);
                REQ(std::memcmp(gw.data(), good.expected.data() + good.e.weight_offsets[m], wb) == 0);
            }
        }
        /* Corrupt slot 1 strictly untouched on device! */
        for (int m = 0; m < 3; m++) {
            const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
            const size_t wb = (size_t)good.e.weight_shapes[m][0] * good.e.weight_shapes[m][1];
            std::vector<uint8_t> gs(sb * 4u), gw(wb);
            CK(cudaMemcpy(gs.data(), ds_dev[3 + m], sb * 4u, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gw.data(), dw_dev[3 + m], wb, cudaMemcpyDeviceToHost));
            REQ(all_byte(gs, 0xA5u) && all_byte(gw, 0xA5u));
        }

        /* Now expand_e8m0_f32=1 on caller stream */
        for (int i = 0; i < 3; i++) {
            for (int m = 0; m < 3; m++) {
                const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
                const size_t wb = (size_t)good.e.weight_shapes[m][0] * good.e.weight_shapes[m][1];
                CK(cudaMemsetAsync(ds_dev[3 * i + m], 0xA5, sb * 4u, user_stream));
                CK(cudaMemsetAsync(dw_dev[3 * i + m], 0xA5, wb, user_stream));
            }
        }
        REQ(g_ops->decode_expert_device(3, es, rs, 0, ds_dev, dw_dev, (void *)user_stream, 1, st) == 1);
        REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
        for (int i : {0, 2}) {
            for (int m = 0; m < 3; m++) {
                const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
                const size_t wb = (size_t)good.e.weight_shapes[m][0] * good.e.weight_shapes[m][1];
                std::vector<uint8_t> gs_f32(sb * 4u), gw(wb);
                CK(cudaMemcpy(gs_f32.data(), ds_dev[3 * i + m], sb * 4u, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], wb, cudaMemcpyDeviceToHost));
                verify_e8m0_f32_exact(good.expected.data() + good.e.scale_offsets[m], gs_f32.data(), sb);
                REQ(std::memcmp(gw.data(), good.expected.data() + good.e.weight_offsets[m], wb) == 0);
            }
        }
        for (int m = 0; m < 3; m++) {
            const size_t sb = (size_t)good.e.scale_shapes[m][0] * good.e.scale_shapes[m][1];
            std::vector<uint8_t> gs_f32(sb * 4u);
            CK(cudaMemcpy(gs_f32.data(), ds_dev[3 + m], sb * 4u, cudaMemcpyDeviceToHost));
            REQ(all_byte(gs_f32, 0xA5u));
        }

        for (int k = 0; k < 9; k++) { CK(cudaFree(ds_dev[k])); CK(cudaFree(dw_dev[k])); }
        std::printf("[PASS] WNR3 cz=%d: host strict/scratch, direct device, E8M0->f32 bit-exact, corrupt untouched\n", cz);
    }
    CK(cudaStreamDestroy(user_stream));
}

/* ---- 2. WNR4-MX & WNR4-NV fixtures & tests ------------------------------ */

struct Wnr4MxRec {
    WexEntry e;
    Wnr4Geo g;
    uint32_t owner;
    std::vector<uint8_t> expected;
    std::vector<uint8_t> record;
};

static uint32_t wnr4_ceil_forward(const Wnr4Geo *g, uint8_t *coded) {
    uint32_t mask = 0;
    if (g->kind) return 0;
    for (uint32_t m = 0; m < 3u; m++) {
        int ok = 1;
        for (uint32_t c = 0; c < g->cells[m] && ok; c++) {
            uint32_t mx = 0;
            for (uint32_t i = 0; i < 32u; i++) {
                const uint32_t a = (coded[g->woff[m] + c * 16u + i / 2u] >> ((i & 1u) * 4u)) & 7u;
                if (a > mx) mx = a;
            }
            if (mx == 5u) ok = 0;
        }
        if (!ok) continue;
        mask |= 1u << m;
        for (uint32_t c = 0; c < g->cells[m]; c++) {
            uint32_t mx = 0, odd = 0;
            uint8_t *w = coded + g->woff[m] + c * 16u;
            uint8_t *s = coded + g->soff[m] + c;
            for (uint32_t i = 0; i < 32u; i++) {
                const uint32_t a = (w[i / 2u] >> ((i & 1u) * 4u)) & 7u;
                if (a > mx) mx = a;
                if (a == 1u || a == 3u) odd = 1;
            }
            if (mx == 7u && !odd && *s <= 254u) {
                static const uint8_t H[8] = {0, 0, 1, 0, 2, 3, 4, 5};
                for (uint32_t i = 0; i < 16u; i++) {
                    const uint32_t lo = w[i] & 15u, hi = w[i] >> 4;
                    w[i] = (uint8_t)(((lo & 8u) | H[lo & 7u]) | (((hi & 8u) | H[hi & 7u]) << 4));
                }
                (*s)++;
            }
        }
    }
    return mask;
}

static void fill_wnr4_slab(const Wnr4Geo *g, uint8_t *slab, int cz, int even_blocks, uint32_t seed) {
    static const uint32_t even_mags[6] = {0, 2, 4, 5, 6, 7};
    const uint32_t B = g->kind ? 16u : 32u;
    uint32_t s = seed;
    std::memset(slab, 0, (size_t)g->raw);
    for (uint32_t m = 0; m < 3u; m++) {
        for (uint32_t i = 0; i < g->scount[m]; i++) {
            slab[g->soff[m] + i] = (uint8_t)(g->kind
                ? (40u + (i / g->scols[m]) % 7u * 3u + xorshift32(&s) % 9u)
                : (118u + xorshift32(&s) % 5u));
        }
        for (uint32_t c = 0; c < g->cells[m]; c++) {
            for (uint32_t i = 0; i < 32u; i++) {
                const uint32_t r = xorshift32(&s) % 100u;
                uint32_t a = r < 30 ? 0u : r < 55 ? 1u : r < 70 ? 2u : r < 80 ? 3u : r < 88 ? 4u : r < 93 ? 5u : r < 97 ? 6u : 7u;
                if (even_blocks && !g->kind && (c % 3u == 0u)) a = even_mags[xorshift32(&s) % 6u];
                if (i % B == B - 1u && (even_blocks || !(c % 2u))) a = 7u;
                uint32_t x = a | (xorshift32(&s) & 8u);
                if (cz && x == 8u) x = 0u;
                slab[g->woff[m] + c * 16u + i / 2u] |= (uint8_t)(x << ((i & 1u) * 4u));
            }
        }
        /* For MX, inject E8M0 edge values {0, 1, 127, 254, 255} on non-even cells. */
        if (!g->kind && g->scount[m] >= 5u) {
            static const uint8_t edges[5] = {0u, 1u, 127u, 254u, 255u};
            for (uint32_t c = 0; c < 5u; c++) {
                slab[g->soff[m] + c] = edges[c];
                slab[g->woff[m] + c * 16u] = 0x71u; /* odd mag 1 + max 7 -> untouched by ceil */
            }
        }
    }
}

static void test_wnr4_mx_and_nv(Wnr4MxRec *churn_mx_out) {
    cudaStream_t user_stream = nullptr;
    CK(cudaStreamCreateWithFlags(&user_stream, cudaStreamNonBlocking));

    /* Test MXFP4 for cz = 0 (pb=1, non-16-aligned scale_bytes!) and cz = 1 (pb=5, ceil form) */
    for (int cz = 0; cz < 2; cz++) {
        const uint32_t pb = cz ? 5u : 1u;
        /* cz=0: 25*40 + 21*17 + 25*40 = 2357 scales (NOT divisible by 16 -> exercises weight_base alignment!) */
        const WexEntry base_e = cz ? make_mx_entry(64u, 32u, 64u, 32u, WEX_ENTRY_WNR4)
                                   : make_mx_entry(25u, 40u, 21u, 17u, WEX_ENTRY_WNR4);
        uint32_t sc[3], cols[3], soff[3], woff[3];
        for (int m = 0; m < 3; m++) {
            sc[m] = (uint32_t)base_e.scale_shapes[m][0] * base_e.scale_shapes[m][1];
            cols[m] = base_e.scale_shapes[m][1];
            soff[m] = base_e.scale_offsets[m];
            woff[m] = base_e.weight_offsets[m];
        }
        Wnr4Geo g;
        REQ(!wnr4_geo(&g, WNR4_KIND_MX, sc, cols, soff, woff));

        std::vector<uint64_t> counts(wnr4_counts_words(WNR4_KIND_MX, pb), 0);
        std::vector<uint8_t> tmp((size_t)g.raw);
        for (uint32_t k = 0; k < 2u; k++) {
            fill_wnr4_slab(&g, tmp.data(), cz, cz, 5000u + (uint32_t)cz * 97u + k * 31u);
            wnr4_ceil_forward(&g, tmp.data());
            uint8_t anc[3];
            for (int m = 0; m < 3; m++) anc[m] = wnr4_anchor(tmp.data() + g.soff[m], g.scount[m]);
            wnr4_count(&g, tmp.data(), anc, pb, counts.data());
        }
        std::vector<uint8_t> blob(wnr4_model_bytes(WNR4_KIND_MX, pb));
        wnr4_model_build(WNR4_KIND_MX, pb, counts.data(), blob.data());
        Wnr4Model model;
        REQ(!wnr4_model_open(blob.data(), blob.size(), &model));
        REQ(g_ops->register_model(1, blob.data(), blob.size()) == 0);
        REQ(g_ops->register_model(1, blob.data(), blob.size()) == 0);

        /* Build 6 distinct valid records for waves {1, 2, 4, 6} */
        std::vector<Wnr4MxRec> recs(6);
        const size_t cap = (size_t)g.raw * 3u + 65536u;
        for (int i = 0; i < 6; i++) {
            Wnr4MxRec &r = recs[(size_t)i];
            r.e = base_e;
            r.g = g;
            r.owner = 0x00030000u + (uint32_t)i;
            r.expected.resize((size_t)g.raw);
            fill_wnr4_slab(&g, r.expected.data(), cz, cz, 8000u + (uint32_t)cz * 500u + (uint32_t)i * 73u);
            std::vector<uint8_t> coded = r.expected;
            const uint32_t flags = wnr4_ceil_forward(&g, coded.data()) | (cz ? WNR4_FLAG_CZ : 0u);
            r.record.resize(cap);
            const size_t nbytes = wnr4_encode(&model, &g, coded.data(), flags, r.owner, 4u, r.record.data(), cap);
            REQ(nbytes > 0 && !(nbytes & 4095u));
            r.record.resize(nbytes);
            r.e.total_bytes = (uint32_t)nbytes;
            /* Verify CPU reference */
            std::vector<uint8_t> cpu((size_t)g.raw, 0x5Au);
            REQ(wnr4_decode(&model, &g, r.record.data(), r.record.size(), r.owner, cpu.data(), 2) == 0);
            REQ(cpu == r.expected);
        }
        if (cz == 1 && churn_mx_out) *churn_mx_out = recs[0];

        /* Allocate device buffers for up to 6 records */
        uint8_t *ds_dev[18] = {}, *dw_dev[18] = {};
        for (int i = 0; i < 6; i++) {
            for (int m = 0; m < 3; m++) {
                CK(cudaMalloc(&ds_dev[3 * i + m], (size_t)sc[m] * 4u));
                CK(cudaMalloc(&dw_dev[3 * i + m], (size_t)sc[m] * 16u));
            }
        }

        const int waves[4] = {1, 2, 4, 6};
        for (int wi = 0; wi < 4; wi++) {
            const int w = waves[wi];
            std::vector<const WexEntry *> es((size_t)w);
            std::vector<const uint8_t *> rs((size_t)w);
            std::vector<std::vector<uint8_t>> host_out((size_t)w);
            std::vector<uint8_t *> hd((size_t)w);
            std::vector<int> st((size_t)w, 7);
            for (int i = 0; i < w; i++) {
                es[(size_t)i] = &recs[(size_t)i].e;
                rs[(size_t)i] = recs[(size_t)i].record.data();
                host_out[(size_t)i].assign((size_t)g.raw, 0xA5u);
                hd[(size_t)i] = host_out[(size_t)i].data();
            }

            /* Host strict */
            REQ(g_ops->decode_expert_host(w, es.data(), rs.data(), 0, hd.data(), 0, st.data()) == 0);
            for (int i = 0; i < w; i++) REQ(st[(size_t)i] == 0 && host_out[(size_t)i] == recs[(size_t)i].expected);

            /* Host scratch */
            for (int i = 0; i < w; i++) std::fill(host_out[(size_t)i].begin(), host_out[(size_t)i].end(), 0xA5u);
            REQ(g_ops->decode_expert_host(w, es.data(), rs.data(), 0, hd.data(), 1, st.data()) == 0);
            for (int i = 0; i < w; i++) REQ(st[(size_t)i] == 0 && host_out[(size_t)i] == recs[(size_t)i].expected);

            /* Direct-to-device (raw uint8 E8M0 scales) */
            for (int i = 0; i < w; i++) {
                for (int m = 0; m < 3; m++) {
                    CK(cudaMemset(ds_dev[3 * i + m], 0xA5, (size_t)sc[m] * 4u));
                    CK(cudaMemset(dw_dev[3 * i + m], 0xA5, (size_t)sc[m] * 16u));
                }
            }
            CK(cudaDeviceSynchronize());
            {
                const int rc = g_ops->decode_expert_device(w, es.data(), rs.data(), 0, ds_dev, dw_dev,
                                                           (wi & 1) ? (void *)user_stream : nullptr, 0, st.data());
                if (rc != 0) {
                    std::fprintf(stderr, "WNR4-MX device failed: cz=%d w=%d rc=%d st[0]=%d\n", cz, w, rc, st[0]);
                }
                REQ(rc == 0);
            }
            for (int i = 0; i < w; i++) {
                REQ(st[(size_t)i] == 0);
                for (int m = 0; m < 3; m++) {
                    std::vector<uint8_t> gs(sc[m]), gw((size_t)sc[m] * 16u);
                    CK(cudaMemcpy(gs.data(), ds_dev[3 * i + m], sc[m], cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], (size_t)sc[m] * 16u, cudaMemcpyDeviceToHost));
                    REQ(std::memcmp(gs.data(), recs[(size_t)i].expected.data() + soff[m], sc[m]) == 0);
                    REQ(std::memcmp(gw.data(), recs[(size_t)i].expected.data() + woff[m], (size_t)sc[m] * 16u) == 0);
                }
            }

            /* Direct-to-device with expand_e8m0_f32=1 (bit-exact vs ldexpf(1.0f, e-127)) */
            for (int i = 0; i < w; i++) {
                for (int m = 0; m < 3; m++) {
                    CK(cudaMemsetAsync(ds_dev[3 * i + m], 0xA5, (size_t)sc[m] * 4u, user_stream));
                    CK(cudaMemsetAsync(dw_dev[3 * i + m], 0xA5, (size_t)sc[m] * 16u, user_stream));
                }
            }
            REQ(g_ops->decode_expert_device(w, es.data(), rs.data(), 0, ds_dev, dw_dev,
                                            (void *)user_stream, 1, st.data()) == 0);
            for (int i = 0; i < w; i++) {
                REQ(st[(size_t)i] == 0);
                for (int m = 0; m < 3; m++) {
                    std::vector<uint8_t> gs_f32((size_t)sc[m] * 4u), gw((size_t)sc[m] * 16u);
                    CK(cudaMemcpy(gs_f32.data(), ds_dev[3 * i + m], (size_t)sc[m] * 4u, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], (size_t)sc[m] * 16u, cudaMemcpyDeviceToHost));
                    verify_e8m0_f32_exact(recs[(size_t)i].expected.data() + soff[m], gs_f32.data(), sc[m]);
                    REQ(std::memcmp(gw.data(), recs[(size_t)i].expected.data() + woff[m], (size_t)sc[m] * 16u) == 0);
                }
            }
        }

        /* Mixed corrupt 2-stream wave (n=6: corrupt index 1 in lane 0 & index 4 in lane 1) */
        {
            std::vector<Wnr4MxRec> mix = recs;
            Wnr4Record parsed;
            REQ(!wnr4_parse(mix[1].record.data(), mix[1].record.size(), &g, &parsed));
            mix[1].record[(size_t)(parsed.data - mix[1].record.data()) + 200u] ^= 0x10u; /* GPU hash fail */
            REQ(!wnr4_parse(mix[4].record.data(), mix[4].record.size(), &g, &parsed));
            mix[4].record[(size_t)(parsed.data - mix[4].record.data()) + 300u] ^= 0x20u; /* GPU hash fail */

            const WexEntry *es[6];
            const uint8_t *rs[6];
            std::vector<std::vector<uint8_t>> hstrict(6), hscratch(6);
            uint8_t *hd_s[6], *hd_c[6];
            for (int i = 0; i < 6; i++) {
                es[i] = &mix[(size_t)i].e;
                rs[i] = mix[(size_t)i].record.data();
                hstrict[(size_t)i].assign((size_t)g.raw, 0xA5u);
                hscratch[(size_t)i].assign((size_t)g.raw, 0xA5u);
                hd_s[i] = hstrict[(size_t)i].data();
                hd_c[i] = hscratch[(size_t)i].data();
            }
            int st_s[6] = {7, 7, 7, 7, 7, 7}, st_c[6] = {7, 7, 7, 7, 7, 7}, st_d[6] = {7, 7, 7, 7, 7, 7};
            REQ(g_ops->decode_expert_host(6, es, rs, 0, hd_s, 0, st_s) == 2);
            REQ(g_ops->decode_expert_host(6, es, rs, 0, hd_c, 1, st_c) == 2);
            for (int i = 0; i < 6; i++) {
                const bool is_bad = (i == 1 || i == 4);
                REQ(st_s[i] == (is_bad ? -1 : 0));
                REQ(st_c[i] == (is_bad ? -1 : 0));
                if (!is_bad) {
                    REQ(hstrict[(size_t)i] == recs[(size_t)i].expected);
                    REQ(hscratch[(size_t)i] == recs[(size_t)i].expected);
                } else {
                    REQ(all_byte(hstrict[(size_t)i], 0xA5u)); /* strict untouched! */
                }
            }
            /* Mixed corrupt 2-stream direct-to-device with expand_e8m0_f32=1 */
            for (int i = 0; i < 6; i++) {
                for (int m = 0; m < 3; m++) {
                    CK(cudaMemset(ds_dev[3 * i + m], 0xA5, (size_t)sc[m] * 4u));
                    CK(cudaMemset(dw_dev[3 * i + m], 0xA5, (size_t)sc[m] * 16u));
                }
            }
            CK(cudaDeviceSynchronize());
            REQ(g_ops->decode_expert_device(6, es, rs, 0, ds_dev, dw_dev, nullptr, 1, st_d) == 2);
            for (int i = 0; i < 6; i++) {
                const bool is_bad = (i == 1 || i == 4);
                REQ(st_d[i] == (is_bad ? -1 : 0));
                if (!is_bad) {
                    for (int m = 0; m < 3; m++) {
                        std::vector<uint8_t> gs_f32((size_t)sc[m] * 4u), gw((size_t)sc[m] * 16u);
                        CK(cudaMemcpy(gs_f32.data(), ds_dev[3 * i + m], (size_t)sc[m] * 4u, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], (size_t)sc[m] * 16u, cudaMemcpyDeviceToHost));
                        verify_e8m0_f32_exact(recs[(size_t)i].expected.data() + soff[m], gs_f32.data(), sc[m]);
                        REQ(std::memcmp(gw.data(), recs[(size_t)i].expected.data() + woff[m], (size_t)sc[m] * 16u) == 0);
                    }
                } else {
                    for (int m = 0; m < 3; m++) {
                        std::vector<uint8_t> gs_f32((size_t)sc[m] * 4u);
                        CK(cudaMemcpy(gs_f32.data(), ds_dev[3 * i + m], (size_t)sc[m] * 4u, cudaMemcpyDeviceToHost));
                        REQ(all_byte(gs_f32, 0xA5u)); /* gated e8m0 kernel did not write corrupt scale slot */
                    }
                }
            }
        }

        for (int k = 0; k < 18; k++) { CK(cudaFree(ds_dev[k])); CK(cudaFree(dw_dev[k])); }
        wnr4_model_free(&model);
        std::printf("[PASS] WNR4-MX cz=%d pb=%u: waves {1,2,4,6}, host strict/scratch, device + E8M0->f32, mixed corrupt 2-stream\n",
                    cz, pb);
    }

    /* Also verify WNR4-NV (kind 1, E4M3 scales per 16 nibbles) through both codec 8 and codec 9 flags */
    {
        WexEntry e = make_nv_entry(30u, 40u, 22u, 12u, WEX_ENTRY_WNR4);
        uint32_t sc[3], cols[3], soff[3], woff[3];
        for (int m = 0; m < 3; m++) {
            sc[m] = (uint32_t)e.scale_shapes[m][0] * e.scale_shapes[m][1];
            cols[m] = e.scale_shapes[m][1];
            soff[m] = e.scale_offsets[m];
            woff[m] = e.weight_offsets[m];
        }
        Wnr4Geo g;
        REQ(!wnr4_geo(&g, WNR4_KIND_NV, sc, cols, soff, woff));
        std::vector<uint8_t> src((size_t)g.raw);
        fill_wnr4_slab(&g, src.data(), 1, 0, 91234u);
        std::vector<uint64_t> counts(wnr4_counts_words(WNR4_KIND_NV, 1u), 0);
        uint8_t anc[3];
        for (int m = 0; m < 3; m++) anc[m] = wnr4_anchor(src.data() + g.soff[m], g.scount[m]);
        wnr4_count(&g, src.data(), anc, 1u, counts.data());
        std::vector<uint8_t> blob(wnr4_model_bytes(WNR4_KIND_NV, 1u));
        wnr4_model_build(WNR4_KIND_NV, 1u, counts.data(), blob.data());
        Wnr4Model model;
        REQ(!wnr4_model_open(blob.data(), blob.size(), &model));
        REQ(g_ops->register_model(1, blob.data(), blob.size()) == 0);

        std::vector<uint8_t> rec((size_t)g.raw * 3u + 65536u);
        const size_t nbytes = wnr4_encode(&model, &g, src.data(), WNR4_FLAG_CZ, 0x00040001u, 4u, rec.data(), rec.size());
        REQ(nbytes > 0);
        rec.resize(nbytes);
        e.total_bytes = (uint32_t)nbytes;
        WexEntry e9 = e;
        e9.flags = WEX_ENTRY_WNR4_NV;

        const WexEntry *es[2] = {&e, &e9};
        const uint8_t *rs[2] = {rec.data(), rec.data()};
        std::vector<uint8_t> h0((size_t)g.raw, 0xA5u), h1((size_t)g.raw, 0xA5u);
        uint8_t *hd[2] = {h0.data(), h1.data()};
        int st[2] = {7, 7};
        REQ(g_ops->decode_expert_host(2, es, rs, 0, hd, 0, st) == 0);
        REQ(st[0] == 0 && st[1] == 0 && h0 == src && h1 == src);

        uint8_t *ds_dev[6] = {}, *dw_dev[6] = {};
        for (int i = 0; i < 2; i++) {
            for (int m = 0; m < 3; m++) {
                CK(cudaMalloc(&ds_dev[3 * i + m], sc[m]));
                CK(cudaMalloc(&dw_dev[3 * i + m], (size_t)sc[m] * 8u));
            }
        }
        REQ(g_ops->decode_expert_device(2, es, rs, 0, ds_dev, dw_dev, nullptr, 0, st) == 0);
        REQ(st[0] == 0 && st[1] == 0);
        for (int i = 0; i < 2; i++) {
            for (int m = 0; m < 3; m++) {
                std::vector<uint8_t> gs(sc[m]), gw((size_t)sc[m] * 8u);
                CK(cudaMemcpy(gs.data(), ds_dev[3 * i + m], sc[m], cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gw.data(), dw_dev[3 * i + m], (size_t)sc[m] * 8u, cudaMemcpyDeviceToHost));
                REQ(std::memcmp(gs.data(), src.data() + soff[m], sc[m]) == 0);
                REQ(std::memcmp(gw.data(), src.data() + woff[m], (size_t)sc[m] * 8u) == 0);
            }
        }
        /* E8M0 expand on NVFP4 must be refused (-1) */
        REQ(g_ops->decode_expert_device(2, es, rs, 0, ds_dev, dw_dev, nullptr, 1, st) == 2 && st[0] == -1 && st[1] == -1);
        for (int k = 0; k < 6; k++) { CK(cudaFree(ds_dev[k])); CK(cudaFree(dw_dev[k])); }
        wnr4_model_free(&model);
        std::printf("[PASS] WNR4-NV: host & device bit-exact (flags 8 & 9), E8M0 expand refused\n");
    }

    CK(cudaStreamDestroy(user_stream));
}

/* ---- 3. WNR4-I4 fixtures & tests ---------------------------------------- */

struct I4Fixture {
    wex_dev_i4_geo_t geo;
    Wnr4I4Geo g;
    std::vector<uint8_t> raw;
    std::vector<uint8_t> enc;
    uint32_t owner;
};

static std::vector<uint8_t> synth_i4_slab(const Wnr4I4Geo &g, uint32_t seed) {
    uint32_t s = seed;
    std::vector<uint8_t> slab((size_t)g.raw);
    std::vector<uint32_t> B(g.nbases + 1u);
    for (uint32_t i = 0; i < g.nbases; i++) B[i] = 0x3a000000u + (xorshift32(&s) % 0x01000000u);
    for (uint32_t m = 0; m < g.nmat; m++) {
        for (uint32_t i = 0; i < g.cells[m]; i++) {
            const uint32_t k = 126u - (xorshift32(&s) % 16u) - (xorshift32(&s) % 16u);
            const uint32_t r = xorshift32(&s) % 100u;
            const uint32_t sc = (r < 3u)
                ? ((xorshift32(&s) << 16) ^ xorshift32(&s))
                : wnr4_i4_lat(k, B[wnr4_i4_block(g.bbase[m], g.gc[m], g.nbc[m], i)]);
            wnr4_put32(slab.data() + g.soff[m] + 4u * (size_t)i, sc);
            uint32_t v[64];
            for (uint32_t e = 0; e < 64u; e++) v[e] = 8u + (xorshift32(&s) % 5u) - 2u;
            for (uint32_t e = 0; e < 32u; e++) {
                slab[g.coff[m] + 32u * (size_t)i + e] = (uint8_t)(g.pairs
                    ? (v[2u * e] | (v[2u * e + 1u] << 4))
                    : (v[e] | (v[e + 32u] << 4)));
            }
        }
    }
    return slab;
}

static void test_wnr4_i4(I4Fixture *churn_i4_out) {
    cudaStream_t user_stream = nullptr;
    CK(cudaStreamCreateWithFlags(&user_stream, cudaStreamNonBlocking));

    for (uint32_t pairs = 0; pairs < 2u; pairs++) {
        wex_dev_i4_geo_t dgeo = {};
        dgeo.nmat = 3u;
        dgeo.pairs = pairs;
        dgeo.rows[0] = 130u; dgeo.cols[0] = 256u;
        dgeo.rows[1] = 130u; dgeo.cols[1] = 256u;
        dgeo.rows[2] = 256u; dgeo.cols[2] = 128u;
        Wnr4I4Geo g;
        REQ(!wnr4_i4_geo(&g, dgeo.nmat, dgeo.rows, dgeo.cols, dgeo.pairs));

        std::vector<uint64_t> counts(wnr4_i4_counts_words(), 0);
        for (uint32_t k = 0; k < 2u; k++) {
            auto s = synth_i4_slab(g, 30000u + pairs * 500u + k * 79u);
            REQ(!wnr4_i4_count(&g, s.data(), counts.data()));
        }
        std::vector<uint8_t> blob(WNR4_I4_MODEL_BYTES);
        wnr4_i4_model_build(counts.data(), blob.data());
        Wnr4I4Model model;
        REQ(!wnr4_i4_model_open(blob.data(), blob.size(), &model));
        REQ(g_ops->register_model(3, blob.data(), blob.size()) == 0);
        REQ(g_ops->register_model(3, blob.data(), blob.size()) == 0);

        const uint32_t Gs[2] = {4u, 32u};
        std::vector<I4Fixture> fix(2);
        for (int gi = 0; gi < 2; gi++) {
            I4Fixture &f = fix[(size_t)gi];
            f.geo = dgeo;
            f.g = g;
            f.owner = 0x00070000u + pairs * 16u + (uint32_t)gi;
            f.raw = synth_i4_slab(g, 40000u + pairs * 900u + (uint32_t)gi * 131u);
            f.enc.resize((size_t)g.raw * 4u + 65536u);
            const size_t nb = wnr4_i4_encode(&model, &g, f.raw.data(), f.owner, Gs[gi], f.enc.data(), f.enc.size());
            REQ(nb > 0);
            f.enc.resize(nb);
            std::vector<uint8_t> cpu((size_t)g.raw, 0x5Au);
            REQ(wnr4_i4_decode(&model, &g, f.enc.data(), f.enc.size(), f.owner, cpu.data(), 2) == 0 && cpu == f.raw);
        }
        if (pairs == 0u && churn_i4_out) *churn_i4_out = fix[0];

        /* Uniform-G (G=4 and G=32 separately) + Mixed-G ({G=4, G=32} together) */
        uint8_t *dev_buf[3] = {};
        for (int i = 0; i < 3; i++) CK(cudaMalloc(&dev_buf[i], (size_t)g.raw + 64u));

        for (int gi = 0; gi < 2; gi++) {
            const I4Fixture &f = fix[(size_t)gi];
            std::vector<uint8_t> corrupt = f.enc;
            Wnr4I4Record pr;
            REQ(!wnr4_i4_parse(corrupt.data(), corrupt.size(), &g, &pr));
            corrupt[(size_t)(pr.data - corrupt.data()) + 150u] ^= 0x04u;

            const uint8_t *rs[3] = {f.enc.data(), corrupt.data(), f.enc.data()};
            const size_t tots[3] = {f.enc.size(), corrupt.size(), f.enc.size()};
            const uint32_t owns[3] = {f.owner, f.owner, f.owner};
            std::vector<uint8_t> h0((size_t)g.raw, 0xA5u), h1((size_t)g.raw, 0xA5u), h2((size_t)g.raw, 0xA5u);
            uint8_t *hd[3] = {h0.data(), h1.data(), h2.data()};
            int st[3] = {7, 7, 7};

            /* Host strict: corrupt slot untouched */
            REQ(g_ops->decode_i4_batch(3, &dgeo, rs, tots, owns, 0, hd, 0, 0, nullptr, st) == 1);
            REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
            REQ(h0 == f.raw && all_byte(h1, 0xA5u) && h2 == f.raw);

            /* Host scratch */
            std::fill(h0.begin(), h0.end(), 0xA5u);
            std::fill(h2.begin(), h2.end(), 0xA5u);
            REQ(g_ops->decode_i4_batch(3, &dgeo, rs, tots, owns, 0, hd, 0, 1, nullptr, st) == 1);
            REQ(st[0] == 0 && st[1] == -1 && st[2] == 0 && h0 == f.raw && h2 == f.raw);

            /* Direct-to-device */
            if (gi) {
                for (int i = 0; i < 3; i++) CK(cudaMemsetAsync(dev_buf[i], 0xA5, (size_t)g.raw, user_stream));
            } else {
                for (int i = 0; i < 3; i++) CK(cudaMemset(dev_buf[i], 0xA5, (size_t)g.raw));
                CK(cudaDeviceSynchronize());
            }
            REQ(g_ops->decode_i4_batch(3, &dgeo, rs, tots, owns, 0, dev_buf, 1, 0,
                                       gi ? (void *)user_stream : nullptr, st) == 1);
            REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
            for (int i : {0, 2}) {
                std::vector<uint8_t> back((size_t)g.raw);
                CK(cudaMemcpy(back.data(), dev_buf[i], (size_t)g.raw, cudaMemcpyDeviceToHost));
                REQ(back == f.raw);
            }
        }

        /* Mixed-G batch {G=4, G=32} */
        {
            const uint8_t *rs[2] = {fix[0].enc.data(), fix[1].enc.data()};
            const size_t tots[2] = {fix[0].enc.size(), fix[1].enc.size()};
            const uint32_t owns[2] = {fix[0].owner, fix[1].owner};
            std::vector<uint8_t> h0((size_t)g.raw, 0xA5u), h1((size_t)g.raw, 0xA5u);
            uint8_t *hd[2] = {h0.data(), h1.data()};
            int st[2] = {7, 7};
            REQ(g_ops->decode_i4_batch(2, &dgeo, rs, tots, owns, 0, hd, 0, 0, nullptr, st) == 0);
            REQ(st[0] == 0 && st[1] == 0 && h0 == fix[0].raw && h1 == fix[1].raw);
            REQ(g_ops->decode_i4_batch(2, &dgeo, rs, tots, owns, 0, dev_buf, 1, 0, nullptr, st) == 0);
            REQ(st[0] == 0 && st[1] == 0);
            for (int i = 0; i < 2; i++) {
                std::vector<uint8_t> back((size_t)g.raw);
                CK(cudaMemcpy(back.data(), dev_buf[i], (size_t)g.raw, cudaMemcpyDeviceToHost));
                REQ(back == fix[(size_t)i].raw);
            }
        }

        /* Refusals: G=128 (> 64), wrong owner, unaligned device pointer */
        {
            std::vector<uint8_t> g128((size_t)g.raw * 4u + 65536u);
            const size_t nb = wnr4_i4_encode(&model, &g, fix[0].raw.data(), fix[0].owner, 128u, g128.data(), g128.size());
            REQ(nb > 0);
            g128.resize(nb);
            const uint8_t *rp = g128.data();
            size_t tot = g128.size();
            int st = 7;
            REQ(g_ops->decode_i4_batch(1, &dgeo, &rp, &tot, &fix[0].owner, 0, &dev_buf[0], 1, 0, nullptr, &st) == 1 && st == -1);

            rp = fix[0].enc.data();
            tot = fix[0].enc.size();
            uint32_t bad_owner = fix[0].owner ^ 1u;
            REQ(g_ops->decode_i4_batch(1, &dgeo, &rp, &tot, &bad_owner, 0, &dev_buf[0], 1, 0, nullptr, &st) == 1 && st == -1);

            uint8_t *unaligned = dev_buf[0] + 2u;
            REQ(g_ops->decode_i4_batch(1, &dgeo, &rp, &tot, &fix[0].owner, 0, &unaligned, 1, 0, nullptr, &st) == 1 && st == -1);
        }

        for (int i = 0; i < 3; i++) CK(cudaFree(dev_buf[i]));
        wnr4_i4_model_free(&model);
        std::printf("[PASS] WNR4-I4 pairs=%u: G=4 & G=32 & mixed-G, host strict/scratch, device, corrupt/refusals\n", pairs);
    }

    CK(cudaStreamDestroy(user_stream));
}

/* ---- 4. WNR4-GG fixtures & tests ---------------------------------------- */

struct GgFixture {
    wex_dev_gg_geo_t geo;
    Wnr4GgGeo g;
    std::vector<uint8_t> raw;
    std::vector<uint8_t> enc;
    uint32_t owner;
};

static std::vector<uint8_t> synth_gg_slab(const Wnr4GgGeo &g, uint32_t seed) {
    uint32_t s = seed;
    std::vector<uint8_t> slab((size_t)g.raw);
    for (uint32_t seg = 0; seg < g.nseg; seg++) {
        uint8_t *p = slab.data() + g.off[seg];
        if (g.sub[seg] >= 0) {
            for (uint32_t c = 0; c < g.cells[seg]; c++) {
                uint8_t *blk = p + (size_t)c * 18u;
                blk[1] = (uint8_t)(0x38u + (xorshift32(&s) % 8u)); /* skewed high byte */
                blk[0] = (uint8_t)(xorshift32(&s) & 0xffu);
                for (uint32_t k = 2; k < 18u; k++) blk[k] = (uint8_t)(xorshift32(&s) & 0xffu);
            }
        } else {
            for (uint32_t k = 0; k < g.bytes[seg]; k++) p[k] = (uint8_t)(xorshift32(&s) & 0xffu);
        }
    }
    return slab;
}

static void test_wnr4_gg(GgFixture *churn_gg_out) {
    cudaStream_t user_stream = nullptr;
    CK(cudaStreamCreateWithFlags(&user_stream, cudaStreamNonBlocking));

    /* Two heterogeneous geometries in the same batch:
     *   Geo A (4 slices): Q2_0 (1000 blocks) + verbatim (4097 B, odd offset!) + IQ4_NL (37 blocks) + Q2_0 (5 blocks)
     *   Geo B (3 slices): IQ4_NL (640 blocks) + verbatim (513 B) + Q2_0 (256 blocks) */
    wex_dev_gg_geo_t dgA = {}, dgB = {};
    dgA.nslice = 4u;
    dgA.ggml_type[0] = WNR4_GG_Q2_0;   dgA.nbytes[0] = 18u * 1000u;
    dgA.ggml_type[1] = 16u;            dgA.nbytes[1] = 4097u;
    dgA.ggml_type[2] = WNR4_GG_IQ4_NL; dgA.nbytes[2] = 18u * 37u;
    dgA.ggml_type[3] = WNR4_GG_Q2_0;   dgA.nbytes[3] = 18u * 5u;

    dgB.nslice = 3u;
    dgB.ggml_type[0] = WNR4_GG_IQ4_NL; dgB.nbytes[0] = 18u * 640u;
    dgB.ggml_type[1] = 22u;            dgB.nbytes[1] = 513u;
    dgB.ggml_type[2] = WNR4_GG_Q2_0;   dgB.nbytes[2] = 18u * 256u;

    auto make_geo = [](wex_dev_gg_geo_t &dg, Wnr4GgGeo &gg) {
        uint32_t b32[4] = {};
        uint64_t off = 0;
        for (uint32_t s = 0; s < dg.nslice; s++) {
            dg.offsets[s] = off;
            b32[s] = (uint32_t)dg.nbytes[s];
            off += dg.nbytes[s];
        }
        REQ(!wnr4_gg_geo(&gg, dg.nslice, dg.ggml_type, b32));
    };
    Wnr4GgGeo gA, gB;
    make_geo(dgA, gA);
    make_geo(dgB, gB);

    std::vector<uint64_t> counts(wnr4_gg_counts_words(), 0);
    {
        auto sA = synth_gg_slab(gA, 70001u);
        auto sB = synth_gg_slab(gB, 70002u);
        wnr4_gg_count(&gA, sA.data(), counts.data());
        wnr4_gg_count(&gB, sB.data(), counts.data());
    }
    std::vector<uint8_t> blob(WNR4_GG_MODEL_BYTES);
    REQ(wnr4_gg_model_build(counts.data(), blob.data()) == 0);
    Wnr4GgModel model;
    REQ(!wnr4_gg_model_open(blob.data(), blob.size(), &model));
    REQ(g_ops->register_model(2, blob.data(), blob.size()) == 0);
    REQ(g_ops->register_model(2, blob.data(), blob.size()) == 0);

    GgFixture fA = {}, fB = {};
    fA.geo = dgA; fA.g = gA; fA.owner = 0x00080001u;
    fA.raw = synth_gg_slab(gA, 71001u);
    fA.enc.resize((size_t)gA.raw * 3u + 65536u);
    size_t nA = wnr4_gg_encode(&model, &gA, fA.raw.data(), fA.owner, 3u, fA.enc.data(), fA.enc.size());
    REQ(nA > 0);
    fA.enc.resize(nA);
    {
        std::vector<uint8_t> cpu((size_t)gA.raw, 0x5Au);
        REQ(wnr4_gg_decode(&model, &gA, fA.enc.data(), fA.enc.size(), fA.owner, cpu.data(), 2) == 0 && cpu == fA.raw);
    }

    fB.geo = dgB; fB.g = gB; fB.owner = 0x00080002u;
    fB.raw = synth_gg_slab(gB, 71002u);
    fB.enc.resize((size_t)gB.raw * 3u + 65536u);
    size_t nB = wnr4_gg_encode(&model, &gB, fB.raw.data(), fB.owner, 8u, fB.enc.data(), fB.enc.size());
    REQ(nB > 0);
    fB.enc.resize(nB);
    {
        std::vector<uint8_t> cpu((size_t)gB.raw, 0x5Au);
        REQ(wnr4_gg_decode(&model, &gB, fB.enc.data(), fB.enc.size(), fB.owner, cpu.data(), 2) == 0 && cpu == fB.raw);
    }
    if (churn_gg_out) *churn_gg_out = fA;

    /* Corrupt copy of fA (flip payload bit) */
    std::vector<uint8_t> corruptA = fA.enc;
    {
        Wnr4GgRecord pr;
        REQ(!wnr4_gg_parse(corruptA.data(), corruptA.size(), &gA, &pr));
        corruptA[(size_t)(pr.data - corruptA.data()) + 140u] ^= 0x08u;
    }

    const wex_dev_gg_geo_t geos[3] = {fA.geo, fA.geo, fB.geo};
    const uint8_t *rs[3] = {fA.enc.data(), corruptA.data(), fB.enc.data()};
    const size_t tots[3] = {fA.enc.size(), corruptA.size(), fB.enc.size()};
    const uint32_t owns[3] = {fA.owner, fA.owner, fB.owner};
    std::vector<uint8_t> h0((size_t)gA.raw, 0xA5u), h1((size_t)gA.raw, 0xA5u), h2((size_t)gB.raw, 0xA5u);
    uint8_t *hd[3] = {h0.data(), h1.data(), h2.data()};
    int st[3] = {7, 7, 7};

    /* Host strict: corrupt slot 1 untouched */
    REQ(g_ops->decode_gg_batch(3, geos, rs, tots, owns, 0, hd, 0, 0, nullptr, st) == 1);
    REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
    REQ(h0 == fA.raw && all_byte(h1, 0xA5u) && h2 == fB.raw);

    /* Host scratch */
    std::fill(h0.begin(), h0.end(), 0xA5u);
    std::fill(h2.begin(), h2.end(), 0xA5u);
    REQ(g_ops->decode_gg_batch(3, geos, rs, tots, owns, 0, hd, 0, 1, nullptr, st) == 1);
    REQ(st[0] == 0 && st[1] == -1 && st[2] == 0 && h0 == fA.raw && h2 == fB.raw);

    /* Direct-to-device on user_stream */
    uint8_t *d0 = nullptr, *d1 = nullptr, *d2 = nullptr;
    CK(cudaMalloc(&d0, (size_t)gA.raw));
    CK(cudaMalloc(&d1, (size_t)gA.raw));
    CK(cudaMalloc(&d2, (size_t)gB.raw));
    CK(cudaMemsetAsync(d0, 0xA5, (size_t)gA.raw, user_stream));
    CK(cudaMemsetAsync(d1, 0xA5, (size_t)gA.raw, user_stream));
    CK(cudaMemsetAsync(d2, 0xA5, (size_t)gB.raw, user_stream));
    uint8_t *dd[3] = {d0, d1, d2};
    REQ(g_ops->decode_gg_batch(3, geos, rs, tots, owns, 0, dd, 1, 0, (void *)user_stream, st) == 1);
    REQ(st[0] == 0 && st[1] == -1 && st[2] == 0);
    {
        std::vector<uint8_t> b0((size_t)gA.raw), b2((size_t)gB.raw);
        CK(cudaMemcpy(b0.data(), d0, (size_t)gA.raw, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b2.data(), d2, (size_t)gB.raw, cudaMemcpyDeviceToHost));
        REQ(b0 == fA.raw && b2 == fB.raw);
    }

    /* Wrong owner & bad geometry refusal */
    {
        uint32_t bad_own = fA.owner ^ 1u;
        int s1 = 7;
        const uint8_t *rp = fA.enc.data();
        size_t t1 = fA.enc.size();
        REQ(g_ops->decode_gg_batch(1, &fA.geo, &rp, &t1, &bad_own, 0, &d0, 1, 0, nullptr, &s1) == 1 && s1 == -1);
        wex_dev_gg_geo_t bad_geo = fA.geo;
        bad_geo.nslice = 0u;
        REQ(g_ops->decode_gg_batch(1, &bad_geo, &rp, &t1, &fA.owner, 0, &d0, 1, 0, nullptr, &s1) == 1 && s1 == -1);
    }

    CK(cudaFree(d0));
    CK(cudaFree(d1));
    CK(cudaFree(d2));
    CK(cudaStreamDestroy(user_stream));
    wnr4_gg_model_free(&model);
    std::printf("[PASS] WNR4-GG: Q2_0 + IQ4_NL + verbatim (heterogeneous geos), host strict/scratch, device, corrupt checks\n");
}

/* ---- 5. Thread churn (32 rounds x 4 fresh threads, bounded VRAM) -------- */

static void test_thread_churn(const Nr3Rec &nr3, const Wnr4MxRec &mx, const I4Fixture &i4, const GgFixture &gg) {
    const int ROUNDS = 32;
    const int THREADS = 4;

    /* Shared device scratch per thread slot (allocated once outside the churn loop). */
    const size_t mx_s_bytes = (size_t)mx.e.scale_bytes;
    const size_t mx_w_bytes = (size_t)mx.e.weight_bytes;
    uint8_t *d_mx_s[THREADS * 3] = {}, *d_mx_w[THREADS * 3] = {};
    uint8_t *d_i4[THREADS] = {}, *d_gg[THREADS] = {};
    for (int t = 0; t < THREADS; t++) {
        for (int m = 0; m < 3; m++) {
            const size_t sb = (size_t)mx.e.scale_shapes[m][0] * mx.e.scale_shapes[m][1];
            const size_t wb = (size_t)mx.e.weight_shapes[m][0] * mx.e.weight_shapes[m][1];
            CK(cudaMalloc(&d_mx_s[3 * t + m], sb * 4u));
            CK(cudaMalloc(&d_mx_w[3 * t + m], wb));
        }
        CK(cudaMalloc(&d_i4[t], (size_t)i4.g.raw));
        CK(cudaMalloc(&d_gg[t], (size_t)gg.g.raw));
    }
    (void)mx_s_bytes;
    (void)mx_w_bytes;

    auto run_round = [&]() {
        std::vector<int> errs(THREADS, 0);
        std::vector<std::thread> team;
        team.reserve(THREADS);
        for (int t = 0; t < THREADS; t++) {
            team.emplace_back([&, t]() {
                int f = 0;
                /* WNR3 host */
                {
                    std::vector<uint8_t> out(nr3.expected.size(), 0xA5u);
                    const WexEntry *e = &nr3.e;
                    const uint8_t *r = nr3.record.data();
                    uint8_t *d = out.data();
                    int st = 7;
                    if (g_ops->decode_expert_host(1, &e, &r, 0, &d, 1, &st) != 0 || st != 0 || out != nr3.expected) f++;
                }
                /* WNR4-MX host + device (E8M0 expand) */
                {
                    std::vector<uint8_t> out(mx.expected.size(), 0xA5u);
                    const WexEntry *e = &mx.e;
                    const uint8_t *r = mx.record.data();
                    uint8_t *d = out.data();
                    int st = 7;
                    if (g_ops->decode_expert_host(1, &e, &r, 0, &d, 0, &st) != 0 || st != 0 || out != mx.expected) f++;
                    if (g_ops->decode_expert_device(1, &e, &r, 0, &d_mx_s[3 * t], &d_mx_w[3 * t], nullptr, 1, &st) != 0 || st != 0) f++;
                }
                /* WNR4-I4 host + device */
                {
                    std::vector<uint8_t> out(i4.raw.size(), 0xA5u);
                    const uint8_t *r = i4.enc.data();
                    size_t tot = i4.enc.size();
                    uint8_t *d = out.data();
                    int st = 7;
                    if (g_ops->decode_i4_batch(1, &i4.geo, &r, &tot, &i4.owner, 0, &d, 0, 1, nullptr, &st) != 0 || st != 0 || out != i4.raw) f++;
                    if (g_ops->decode_i4_batch(1, &i4.geo, &r, &tot, &i4.owner, 0, &d_i4[t], 1, 0, nullptr, &st) != 0 || st != 0) f++;
                }
                /* WNR4-GG host + device */
                {
                    std::vector<uint8_t> out(gg.raw.size(), 0xA5u);
                    const uint8_t *r = gg.enc.data();
                    size_t tot = gg.enc.size();
                    uint8_t *d = out.data();
                    int st = 7;
                    if (g_ops->decode_gg_batch(1, &gg.geo, &r, &tot, &gg.owner, 0, &d, 0, 0, nullptr, &st) != 0 || st != 0 || out != gg.raw) f++;
                    if (g_ops->decode_gg_batch(1, &gg.geo, &r, &tot, &gg.owner, 0, &d_gg[t], 1, 0, nullptr, &st) != 0 || st != 0) f++;
                }
                errs[(size_t)t] = f;
            });
        }
        for (auto &th : team) th.join();
        for (int t = 0; t < THREADS; t++) REQ(errs[(size_t)t] == 0);
    };

    /* Warmup round 0 populates all 4 pooled workspace sets for concurrency=4 */
    run_round();
    size_t free_before = 0, free_after = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));
    for (int r = 1; r < ROUNDS; r++) run_round();
    CK(cudaMemGetInfo(&free_after, &total));

    /* Spot-check last round's device outputs */
    for (int t = 0; t < THREADS; t++) {
        std::vector<uint8_t> bi4(i4.raw.size()), bgg(gg.raw.size());
        CK(cudaMemcpy(bi4.data(), d_i4[t], i4.raw.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bgg.data(), d_gg[t], gg.raw.size(), cudaMemcpyDeviceToHost));
        REQ(bi4 == i4.raw && bgg == gg.raw);
        for (int m = 0; m < 3; m++) {
            const size_t sb = (size_t)mx.e.scale_shapes[m][0] * mx.e.scale_shapes[m][1];
            std::vector<uint8_t> gs_f32(sb * 4u);
            CK(cudaMemcpy(gs_f32.data(), d_mx_s[3 * t + m], sb * 4u, cudaMemcpyDeviceToHost));
            verify_e8m0_f32_exact(mx.expected.data() + mx.e.scale_offsets[m], gs_f32.data(), sb);
        }
    }

    for (int t = 0; t < THREADS; t++) {
        for (int m = 0; m < 3; m++) { CK(cudaFree(d_mx_s[3 * t + m])); CK(cudaFree(d_mx_w[3 * t + m])); }
        CK(cudaFree(d_i4[t]));
        CK(cudaFree(d_gg[t]));
    }

    const double grew_mib = free_before > free_after ? (double)(free_before - free_after) / 1048576.0 : 0.0;
    std::printf("[PASS] Thread churn (%d rounds x %d fresh threads across all 4 codecs): VRAM growth = %.2f MiB\n",
                ROUNDS, THREADS, grew_mib);
    REQ(grew_mib <= 64.0);
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_libwex_gpu <path-to-libwex_cuda.dll|.so>\n");
        return 2;
    }

#ifdef _WIN32
    HMODULE lib = LoadLibraryA(argv[1]);
    REQ(lib != nullptr);
    auto get_ops = (wex_dev_get_ops_fn)GetProcAddress(lib, WEX_DEV_INIT_SYMBOL);
#else
    void *lib = dlopen(argv[1], RTLD_NOW);
    REQ(lib != nullptr);
    auto get_ops = (wex_dev_get_ops_fn)dlsym(lib, WEX_DEV_INIT_SYMBOL);
#endif
    REQ(get_ops != nullptr);
    g_ops = get_ops();
    REQ(g_ops != nullptr);

    const uint32_t expected_caps = WEX_DEV_CAP_PIN_HOST | WEX_DEV_CAP_WNR3 | WEX_DEV_CAP_WNR4_MX |
                                   WEX_DEV_CAP_WNR4_NV | WEX_DEV_CAP_WNR4_I4 | WEX_DEV_CAP_WNR4_GG |
                                   WEX_DEV_CAP_DIRECT_DEV | WEX_DEV_CAP_SCRATCH_HOST | WEX_DEV_CAP_E8M0_EXPAND;
    REQ(g_ops->abi_version == WEX_DEV_ABI_VERSION);
    REQ((g_ops->caps & expected_caps) == expected_caps);
    REQ(g_ops->name != nullptr && std::strcmp(g_ops->name, "cuda") == 0);
    REQ(g_ops->device_count && g_ops->device_count() >= 1);
    REQ(g_ops->host_register && g_ops->host_unregister && g_ops->register_model);
    REQ(g_ops->decode_expert_host && g_ops->decode_expert_device);
    REQ(g_ops->decode_i4_batch && g_ops->decode_gg_batch);

    cudaDeviceProp prop = {};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Loaded %s (backend=%s, abi=%u, caps=0x%x) on GPU 0: %s (sm_%d%d)\n",
                argv[1], g_ops->name, g_ops->abi_version, g_ops->caps, prop.name, prop.major, prop.minor);

    /* Pin / unpin & bad model checks */
    std::vector<uint8_t> pin_buf(65536u, 0x33u);
    REQ(g_ops->host_register(pin_buf.data(), pin_buf.size()) == 0);
    REQ(g_ops->host_unregister(pin_buf.data()) == 0);
    REQ(g_ops->host_register(nullptr, 64u) == -1);
    REQ(g_ops->host_unregister(nullptr) == -1);
    uint8_t junk[64] = {0};
    for (int k = -1; k <= 4; k++) REQ(g_ops->register_model(k, junk, sizeof(junk)) == -1);
    std::printf("[PASS] ABI, caps, host_register/unregister, invalid model rejection\n");

    Nr3Rec churn_nr3;
    Wnr4MxRec churn_mx;
    I4Fixture churn_i4;
    GgFixture churn_gg;

    test_wnr3(&churn_nr3);
    test_wnr4_mx_and_nv(&churn_mx);
    test_wnr4_i4(&churn_i4);
    test_wnr4_gg(&churn_gg);
    test_thread_churn(churn_nr3, churn_mx, churn_i4, churn_gg);

    std::printf("ALL LIBWEX GPU TESTS PASSED\n");
    return 0;
}
