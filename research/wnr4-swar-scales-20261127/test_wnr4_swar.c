/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* CPU test suite for Prototype 3a (WNR4-MX + s_rec16) and Prototype 3b
 * (WNR4-SWAR-8MAG: 63-ctx 8-mag carry-free SWAR + aligned backward raw signs +
 * MAGT 480-bit tail + s_rec16 + ceil-form + 32-bit MurmurHash3).
 *
 * Build & run:
 *   gcc -O3 -march=haswell -fopenmp -Wall -Wextra \
 *       research/wnr4-swar-scales-20261127/test_wnr4_swar.c -o out/test_wnr4_swar.exe
 *   out/test_wnr4_swar.exe [fuzz_iterations]
 */
#include "wnr4_swar.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(c, ...) do {                                                    \
    if (!(c)) {                                                               \
        failures++;                                                           \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);                           \
        printf(__VA_ARGS__);                                                  \
        printf("\n");                                                         \
    }                                                                         \
} while (0)

static uint64_t rs = UINT64_C(0x9E3779B97F4A7C15);
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 16);
}

static void geo_mx(Wnr4Geo *g, uint32_t r0, uint32_t c0, uint32_t r1, uint32_t c1) {
    uint32_t sc[3] = {r0 * c0, r1 * c1, r0 * c0};
    uint32_t cols[3] = {c0, c1, c0};
    uint32_t so[3], wo[3];
    so[0] = 0; so[1] = sc[0]; so[2] = sc[0] + sc[1];
    wo[0] = so[2] + sc[2];
    wo[1] = wo[0] + sc[0] * 16u;
    wo[2] = wo[1] + sc[1] * 16u;
    if (wnr4_geo(g, WNR4_KIND_MX, sc, cols, so, wo)) { printf("bad geo\n"); exit(1); }
}

/* Modes:
 *  0 = normal V4.1-like scales (~120..121) + ceil-friendly blocks
 *  1 = L39 E120 anchor-shifted (121, 122, 121) with w2 ~65% 123, 27% 122, 8% 121/124
 *  2 = escaped scales covering EVERY byte 0..255 (100% lossless stress)
 *  3 = constant single-scale matrices (K = 1 edge case) + sparse/all-zero cells (MAGT nz < 16)
 */
static void make_slab(const Wnr4Geo *g, unsigned char *slab, int mode, int even_blocks) {
    uint32_t m, c, i;
    memset(slab, 0, (size_t)g->raw);
    for (m = 0; m < 3u; m++) {
        for (i = 0; i < g->scount[m]; i++) {
            unsigned char sv;
            if (mode == 0) {
                uint32_t r = rnd() % 100u;
                sv = (unsigned char)(r < 30u ? 120u : r < 98u ? 121u : 122u);
            } else if (mode == 1) {
                uint32_t r = rnd() % 100u;
                if (m == 1u) sv = (unsigned char)(r < 65u ? 123u : r < 92u ? 122u : r < 96u ? 121u : 124u);
                else sv = (unsigned char)(r < 30u ? 121u : r < 98u ? 122u : 123u);
            } else if (mode == 2) {
                if (i < 256u) sv = (unsigned char)i;
                else if ((rnd() % 10u) == 0u) sv = (unsigned char)(rnd() & 255u);
                else sv = (unsigned char)((m == 0u ? 3u : m == 1u ? 252u : 120u) + (rnd() % 4u));
            } else {
                sv = (unsigned char)(m == 0u ? 0u : m == 1u ? 255u : 121u);
            }
            slab[g->soff[m] + i] = sv;
        }
        for (c = 0; c < g->cells[m]; c++) {
            int sparse = (mode == 3) && ((c % 3u) == 0u);
            int allzero = (mode == 3) && ((c % 5u) == 0u);
            for (i = 0; i < 32u; i++) {
                uint32_t a;
                if (allzero) {
                    a = 0u;
                } else if (sparse) {
                    a = (i < 5u) ? (1u + (rnd() % 6u)) : 0u;
                } else {
                    uint32_t r = rnd() % 100u;
                    a = r < 25u ? 0u : r < 50u ? 1u : r < 68u ? 2u : r < 80u ? 3u :
                        r < 88u ? 4u : r < 93u ? 5u : r < 97u ? 6u : 7u;
                    if (even_blocks && (c % 3u == 0u)) {
                        static const uint32_t ev[6] = {0, 2, 4, 5, 6, 7};
                        a = ev[rnd() % 6u];
                    }
                    if (i == 31u && even_blocks) a = 7u;
                }
                uint32_t x = a ? (a | (rnd() & 8u)) : 0u; /* CZ */
                slab[g->woff[m] + (size_t)c * 16u + i / 2u] |= (unsigned char)(x << ((i & 1u) * 4u));
            }
        }
    }
}

static void test_find8_exact(void) {
    uint16_t cdf[9];
    uint64_t cnt[8];
    int trial;
    for (trial = 0; trial < 200; trial++) {
        uint32_t s, slot;
        for (s = 0; s < 8u; s++) {
            uint32_t r = rnd() % 10u;
            cnt[s] = (r == 0u) ? 0u : (r < 4u ? 1u : (uint64_t)(rnd() % 100000u));
        }
        wnr4_quantize(cnt, 8u, cdf);
        for (slot = 0; slot < WNR4_TOTAL; slot++) {
            uint32_t got = wnr4_swar_find8(cdf, slot);
            uint32_t want = 0;
            while (want < 7u && cdf[want + 1u] <= slot) want++;
            if (got != want) {
                CHECK(0, "find8 mismatch trial %d slot %u: got %u want %u", trial, slot, got, want);
                return;
            }
        }
    }
    printf("unit find8 SWAR: 200 random CDFs x 16,384 slots = 3,276,800 exact\n");
}

static void test_srec16_edge_cases(void) {
    Wnr4Geo g;
    geo_mx(&g, 2, 16, 2, 16);
    unsigned char *slab = (unsigned char *)malloc((size_t)g.raw);
    unsigned char buf[WNR4_SWAR_SREC16_MAX_BYTES];
    uint16_t scdf_enc[3][17], scdf_dec[3][17];
    unsigned char anchor[3];
    uint32_t bytes = 0, m, s;

    /* Mode 3: K=1 constant scales at boundaries 0, 255, 121 */
    make_slab(&g, slab, 3, 0);
    for (m = 0; m < 3u; m++) anchor[m] = wnr4_anchor(slab + g.soff[m], g.scount[m]);
    CHECK(!wnr4_srec16_build(&g, slab, anchor, scdf_enc, buf, &bytes), "srec16 K=1 build");
    CHECK(bytes == 8u, "srec16 K=1 bytes expected 8 (3*2 + 2 pad), got %u", bytes);
    if (bytes == 0u || bytes > WNR4_SWAR_SREC16_MAX_BYTES) { free(slab); return; }
    CHECK(!wnr4_srec16_parse(buf, bytes, anchor, scdf_dec), "srec16 K=1 parse");
    CHECK(!memcmp(scdf_enc, scdf_dec, sizeof(scdf_enc)), "srec16 K=1 roundtrip cdf");
    /* Verify wnr4_find16 returns the sole symbol for every slot 0..16383 */
    for (m = 0; m < 3u; m++) {
        uint32_t want = (uint32_t)((int)slab[g.soff[m]] - (int)anchor[m] + 7);
        for (s = 0; s < WNR4_TOTAL; s++) {
            uint32_t got = wnr4_find16(scdf_dec[m], s);
            if (got != want) { CHECK(0, "srec16 K=1 find16 m=%u slot=%u got=%u want=%u", m, s, got, want); break; }
        }
    }

    /* Negative parse checks */
    {
        unsigned char bad[WNR4_SWAR_SREC16_MAX_BYTES];
        /* Zero mask */
        memcpy(bad, buf, bytes); bad[0] = 0; bad[1] = 0;
        CHECK(wnr4_srec16_parse(bad, bytes, anchor, scdf_dec) != 0, "srec16 zero mask accepted");
        /* Out-of-range in-window bit (anchor[0]=0, bit 0 => 0+0-7 = -7 < 0) */
        memcpy(bad, buf, bytes); bad[0] |= 1u;
        CHECK(wnr4_srec16_parse(bad, bytes, anchor, scdf_dec) != 0, "srec16 anchor-underflow bit accepted");
        /* Non-zero alignment pad */
        memcpy(bad, buf, bytes); bad[bytes - 1u] = 1u;
        CHECK(wnr4_srec16_parse(bad, bytes, anchor, scdf_dec) != 0, "srec16 nonzero pad accepted");
        /* Bad length (not multiple of 4) */
        CHECK(wnr4_srec16_parse(buf, bytes - 2u, anchor, scdf_dec) != 0, "srec16 unaligned len accepted");
    }

    free(slab);
    printf("unit s_rec16 edge cases (K=1, boundaries 0/255, negative parse): OK\n");
}

/* Exhaustive bit-flip check of EVERY bit in the backward raw sign stream of tile 0 */
static void test_sign_stream_flips(const Wnr4SwarModel *model, const Wnr4Geo *g, const unsigned char *rec,
                                   size_t rec_len, uint32_t owner, const unsigned char *expected) {
    Wnr4SwarRecord r;
    CHECK(!wnr4_swar_parse(rec, rec_len, g, &r), "sign-flip parse");
    /* First decode tile 0 normally to find exact forward/backward boundary */
    uint32_t o0 = wnr4_u32(r.offsets), o1 = wnr4_u32(r.offsets + 4);
    uint32_t tile_bytes = o1 - o0;
    size_t tile_off = (size_t)(r.data - rec) + o0;
    unsigned char *bad = (unsigned char *)malloc(rec_len);
    unsigned char *out = (unsigned char *)malloc((size_t)g->raw);
    uint32_t m, c0, n;
    wnr4_tile_cells(r.tile_base, g->cells, r.G, 0, &m, &c0, &n);
    /* Count exact number of non-zero nibbles in tile 0 (minus MAGT) to locate sign words */
    uint32_t total_nz = 0, tail_nz[32] = {0}, j, l, k;
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u;
        for (k = 0; k < 32u; k++) {
            for (l = 0; l < act; l++) {
                uint32_t mag = wnr4_nib(expected + g->woff[m], c0 + 32u * j + l, k) & 7u;
                if (mag != 0u) {
                    int is_last = (l >= 2u) && (32u * j + l + 32u >= n);
                    if (is_last && tail_nz[l] < 16u) tail_nz[l]++;
                    else total_nz++;
                }
            }
        }
    }
    uint32_t sign_words = (total_nz + 31u) / 32u;
    uint32_t sign_bytes = sign_words * 4u;
    CHECK(sign_bytes > 0u && sign_bytes < tile_bytes - 128u, "tile 0 sign_bytes=%u tile_bytes=%u", sign_bytes, tile_bytes);
    size_t sign_start = tile_off + (size_t)(tile_bytes - sign_bytes);
    uint32_t caught_hash = 0, caught_struct = 0, missed = 0;
    uint32_t bits_tested = sign_bytes * 8u;
    /* Cap exhaustive loop if huge, else test EVERY single bit */
    uint32_t step = bits_tested > 4096u ? bits_tested / 4096u : 1u;
    uint32_t tested = 0;
    for (uint32_t bit = 0; bit < bits_tested; bit += step) {
        memcpy(bad, rec, rec_len);
        bad[sign_start + (bit >> 3)] ^= (unsigned char)(1u << (bit & 7u));
        uint32_t err = wnr4_swar_decode(model, g, bad, rec_len, owner, out, 1);
        tested++;
        if (err & WNR4_ERR_HASH) caught_hash++;
        else if (err) caught_struct++;
        else missed++;
    }
    CHECK(missed == 0u, "sign-stream single-bit flips: %u of %u MISSED!", missed, tested);
    printf("  sign-stream single-bit flips: %u/%u caught (hash=%u, struct/pad=%u, missed=%u)\n",
           tested - missed, tested, caught_hash, caught_struct, missed);
    free(bad);
    free(out);
}

static void run_profile(const char *name, const Wnr4Geo *g, int mode, int even, const uint32_t *Gs, int nG, int fuzz) {
    unsigned char *src = (unsigned char *)malloc((size_t)g->raw);
    unsigned char *coded = (unsigned char *)malloc((size_t)g->raw);
    unsigned char *out = (unsigned char *)malloc((size_t)g->raw);
    size_t cap = (size_t)g->raw * 3u + 65536u;
    unsigned char *rec = (unsigned char *)malloc(cap);
    unsigned char *bad = (unsigned char *)malloc(cap);
    unsigned char *blob_s = (unsigned char *)malloc(wnr4_swar_model_bytes());
    unsigned char *blob_m = (unsigned char *)malloc(wnr4_model_bytes(WNR4_KIND_MX, 5));
    uint64_t *cnt_s = (uint64_t *)calloc(wnr4_swar_counts_words(), 8u);
    uint64_t *cnt_m = (uint64_t *)calloc(wnr4_counts_words(WNR4_KIND_MX, 5), 8u);
    Wnr4SwarModel ms;
    Wnr4Model mm;
    unsigned char anchor[3];
    uint32_t flags, owner = 0x00270078u, m;
    int gi, it;

    make_slab(g, src, mode, even);
    flags = wnr4_prepare(g, src, coded, 1, NULL);
    for (m = 0; m < 3u; m++) anchor[m] = wnr4_anchor(coded + g->soff[m], g->scount[m]);

    /* Train both 3b (SWAR) and 3a (WNR4-MX pb=5) models */
    wnr4_swar_count(g, coded, anchor, cnt_s);
    wnr4_swar_model_build(cnt_s, blob_s);
    CHECK(!wnr4_swar_model_open(blob_s, wnr4_swar_model_bytes(), &ms), "%s: swar model", name);

    wnr4_count(g, coded, anchor, 5, cnt_m);
    wnr4_model_build(WNR4_KIND_MX, 5, cnt_m, blob_m);
    CHECK(!wnr4_model_open(blob_m, wnr4_model_bytes(WNR4_KIND_MX, 5), &mm), "%s: mx model", name);

    for (gi = 0; gi < nG; gi++) {
        uint32_t G = Gs[gi];
        /* Item 3a: WNR4-MX + s_rec16 round trip */
        size_t sbytes = 0;
        size_t n3a = wnr4_srec16_encode(&mm, g, coded, flags, owner, G, rec, cap, &sbytes);
        CHECK(n3a && !(n3a & 4095u), "%s G=%u: 3a encode", name, G);
        if (n3a) {
            memset(out, 0xA5, (size_t)g->raw);
            uint32_t err = wnr4_srec16_decode(&mm, g, rec, n3a, owner, out, 4);
            CHECK(!err && !memcmp(out, src, (size_t)g->raw), "%s G=%u: 3a roundtrip (err 0x%x)", name, G, err);
            CHECK(wnr4_srec16_decode(&mm, g, rec, n3a, owner ^ 1u, out, 1) != 0, "%s G=%u: 3a wrong owner", name, G);
        }

        /* Item 3b: WNR4-SWAR-8MAG round trip */
        Wnr4SwarStats st;
        size_t n3b = wnr4_swar_encode(&ms, g, coded, flags, owner, G, rec, cap, &st);
        CHECK(n3b && !(n3b & 4095u), "%s G=%u: 3b encode", name, G);
        if (!n3b) continue;
        memset(out, 0x5A, (size_t)g->raw);
        uint32_t err = wnr4_swar_decode(&ms, g, rec, n3b, owner, out, 4);
        CHECK(!err && !memcmp(out, src, (size_t)g->raw), "%s G=%u: 3b roundtrip (err 0x%x)", name, G, err);
        CHECK(wnr4_swar_decode(&ms, g, rec, n3b, owner ^ 1u, out, 1) != 0, "%s G=%u: 3b wrong owner", name, G);

        /* Batch decode with corrupted middle copy */
        if (gi == 0) {
            unsigned char *outs3[3] = {
                (unsigned char *)malloc((size_t)g->raw),
                (unsigned char *)malloc((size_t)g->raw),
                (unsigned char *)malloc((size_t)g->raw)
            };
            const unsigned char *recs3[3] = {rec, bad, rec};
            size_t tot3[3] = {n3b, n3b, n3b};
            uint32_t own3[3] = {owner, owner, owner}, st3[3];
            memcpy(bad, rec, n3b);
            bad[n3b / 2u] ^= 0x20u;
            int nb = wnr4_swar_decode_batch(&ms, g, 3, recs3, tot3, own3, outs3, st3, 4);
            CHECK(nb == 1 && !st3[0] && st3[1] && !st3[2] &&
                  !memcmp(outs3[0], src, (size_t)g->raw) && !memcmp(outs3[2], src, (size_t)g->raw),
                  "%s G=%u: 3b batch decode failed=%d st=%u/%u/%u", name, G, nb, st3[0], st3[1], st3[2]);
            free(outs3[0]); free(outs3[1]); free(outs3[2]);
        }

        /* Header + s_rec16 byte flips */
        if (gi == 0) {
            static const uint32_t hfields[] = {0, 4, 8, 12, 13, 14, 16, 17, 18, 19, 20, 24, 28, 40, 64, 65};
            for (it = 0; it < (int)(sizeof(hfields) / sizeof(hfields[0])); it++) {
                memcpy(bad, rec, n3b);
                bad[hfields[it]] ^= 0x40u;
                CHECK(wnr4_swar_decode(&ms, g, bad, n3b, owner, out, 1) != 0,
                      "%s G=%u: header/srec byte %u flip accepted", name, G, hfields[it]);
            }
            /* Explicit exhaustive sign-stream single-bit flips on tile 0 */
            if (mode != 3) test_sign_stream_flips(&ms, g, rec, n3b, owner, coded);
            /* Mutation fuzz over all used bytes (header, s_rec16, offsets, states, rANS words, pad2, sign stream) */
            if (fuzz > 0) {
                int refused = 0, same = 0;
                size_t used = n3b;
                while (used && !rec[used - 1u]) used--;
                for (it = 0; it < fuzz; it++) {
                    int kf, f = 1 + (int)(rnd() % 3u);
                    memcpy(bad, rec, n3b);
                    for (kf = 0; kf < f; kf++) bad[rnd() % used] ^= (unsigned char)(1u << (rnd() % 8u));
                    if (!memcmp(bad, rec, n3b)) { same++; continue; }
                    err = wnr4_swar_decode(&ms, g, bad, n3b, owner, out, 1);
                    if (err) refused++;
                    else CHECK(!memcmp(out, src, (size_t)g->raw), "%s: fuzz trial %d accepted with wrong bytes!", name, it);
                }
                printf("  %s fuzz: %d trials, %d refused, %d no-op, %d accepted (identical)\n",
                       name, fuzz, refused, same, fuzz - refused - same);
            }
        }
    }

    printf("%s: mode=%d flags=0x%02x anchors=(%u,%u,%u) %d G values OK\n",
           name, mode, flags, anchor[0], anchor[1], anchor[2], nG);
    wnr4_model_free(&mm);
    free(src); free(coded); free(out); free(rec); free(bad);
    free(blob_s); free(blob_m); free(cnt_s); free(cnt_m);
}

static void test_negative_cases(void) {
    Wnr4Geo g;
    geo_mx(&g, 4, 40, 4, 40);
    unsigned char *src = (unsigned char *)malloc((size_t)g.raw);
    unsigned char *out = (unsigned char *)malloc((size_t)g.raw);
    size_t cap = (size_t)g.raw * 3u + 65536u;
    unsigned char *rec = (unsigned char *)malloc(cap);
    unsigned char *blob = (unsigned char *)malloc(wnr4_swar_model_bytes());
    uint64_t *cnt = (uint64_t *)calloc(wnr4_swar_counts_words(), 8u);
    Wnr4SwarModel ms;
    unsigned char a[3] = {120, 120, 120};

    make_slab(&g, src, 0, 0);
    wnr4_swar_count(&g, src, a, cnt);
    wnr4_swar_model_build(cnt, blob);
    CHECK(!wnr4_swar_model_open(blob, wnr4_swar_model_bytes(), &ms), "neg swar model");

    /* 1. Non-CZ flag (missing WNR4_FLAG_CZ) refused by encoder */
    CHECK(wnr4_swar_encode(&ms, &g, src, 0u, 7u, 4u, rec, cap, NULL) == 0, "non-CZ flag accepted by encoder");

    /* 2. Nibble 8 (-0) refused by encoder */
    src[g.woff[0]] = 0x08u;
    CHECK(wnr4_swar_encode(&ms, &g, src, WNR4_FLAG_CZ, 7u, 4u, rec, cap, NULL) == 0, "nibble 8 accepted by encoder");

    /* 3. Ceil underflow (coded max==5 with scale==0 under ceil flag) refused by decoder */
    make_slab(&g, src, 0, 0);
    memset(src + g.woff[0], 0x11, 16u);
    src[g.woff[0]] = 0x15u;
    src[g.soff[0]] = 0u;
    size_t n = wnr4_swar_encode(&ms, &g, src, WNR4_FLAG_CZ | 1u, 7u, 4u, rec, cap, NULL);
    CHECK(n > 0 && (wnr4_swar_decode(&ms, &g, rec, n, 7u, out, 1) & WNR4_ERR_CEIL), "ceil underflow not refused");

    /* 4. Different model refused */
    cnt[3]++;
    wnr4_swar_model_build(cnt, blob);
    Wnr4SwarModel other;
    CHECK(!wnr4_swar_model_open(blob, wnr4_swar_model_bytes(), &other) && other.id != ms.id, "other swar model");
    CHECK(wnr4_swar_decode(&other, &g, rec, n, 7u, out, 1) != 0, "decoded under wrong swar model");

    /* 5. Corrupt model blob refused */
    blob[60] ^= 0xffu; blob[61] ^= 0xffu;
    CHECK(wnr4_swar_model_open(blob, wnr4_swar_model_bytes(), &other) != 0, "corrupt swar model accepted");

    /* 6. Non-zero unused high sign bit in last sign word refused */
    make_slab(&g, src, 3, 0); /* sparse so last sign word almost certainly has unused high bits */
    n = wnr4_swar_encode(&ms, &g, src, WNR4_FLAG_CZ, 7u, 4u, rec, cap, NULL);
    CHECK(n > 0, "sparse encode");
    {
        Wnr4SwarRecord r;
        CHECK(!wnr4_swar_parse(rec, n, &g, &r), "sparse parse");
        uint32_t o0 = wnr4_u32(r.offsets), o1 = wnr4_u32(r.offsets + 4);
        /* Flip bit 31 of the first sign word (at tile_end - 4) -> caught either by hash or unused bit check */
        rec[(size_t)(r.data - rec) + o1 - 1u] ^= 0x80u;
        CHECK(wnr4_swar_decode(&ms, &g, rec, n, 7u, out, 1) != 0, "flipped sign word MSB accepted");
        (void)o0;
    }

    free(src); free(out); free(rec); free(blob); free(cnt);
    printf("negative cases: OK\n");
}

int main(int argc, char **argv) {
    int fuzz = argc > 1 ? atoi(argv[1]) : 2000;
    static const uint32_t Gs[] = {1, 3, 7, 32, 64, 128};
    Wnr4Geo g;

    test_find8_exact();
    test_srec16_edge_cases();

    /* Normal partial-tile geometry (1000 and 357 cells) */
    geo_mx(&g, 25, 40, 21, 17);
    run_profile("mx-normal-ceil", &g, 0, 1, Gs, 6, fuzz);
    run_profile("mx-l39-e120-shifted", &g, 1, 1, Gs, 6, fuzz);
    run_profile("mx-all256-escapes", &g, 2, 0, Gs, 6, fuzz);
    run_profile("mx-constK1-sparseMAGT", &g, 3, 0, Gs, 6, fuzz);

    /* Tiny sub-warp matrices (< 32 cells: 1, 2, 17 cells) to stress MAGT + partial warps */
    {
        uint32_t sc[3] = {1, 2, 17}, cols[3] = {1, 2, 17};
        uint32_t so[3] = {0, 1, 3};
        uint32_t wo[3] = {20, 20 + 16, 20 + 48};
        CHECK(!wnr4_geo(&g, WNR4_KIND_MX, sc, cols, so, wo), "tiny geo");
        run_profile("mx-tiny-subwarp", &g, 2, 0, Gs, 6, fuzz / 4);
    }

    test_negative_cases();

    printf(failures ? "FAILED: %d\n" : "ALL OK\n", failures);
    return failures != 0;
}
