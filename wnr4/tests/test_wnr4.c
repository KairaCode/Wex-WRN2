/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CPU tests: round trips of synthetic MX/NV slabs (partial tiles, G 1..256, ceil form, CZ),
 * negative cases (CZ nibble 8, ceil underflow, header fields, owner, model) and a mutation fuzz:
 * every mutated record must be refused or decode to the exact expected bytes.
 *   gcc -O2 -fopenmp wnr4/tests/test_wnr4.c -o test_wnr4 && ./test_wnr4 [fuzz_iterations] */
#include "../wnr4.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }

/* a slab with FP4-like statistics: per block a scale near 120 (MX) or a wide E4M3-like byte (NV),
 * nibble magnitudes skewed towards small values, the block max often 6/7, signs random */
static void make_slab(const Wnr4Geo *g, unsigned char *slab, int cz, int even_blocks) {
    const uint32_t B = g->kind ? 16u : 32u, spc = WNR4_SPC(g->kind);
    uint32_t m, c, i;
    memset(slab, 0, (size_t)g->raw);
    for (m = 0; m < 3u; m++) {
        for (i = 0; i < g->scount[m]; i++)
            slab[g->soff[m] + i] = (unsigned char)(g->kind ? 40u + (i / g->scols[m]) % 7u * 3u + rnd() % 9u : 118u + rnd() % 5u);
        for (c = 0; c < g->cells[m]; c++)
            for (i = 0; i < 32u; i++) {
                uint32_t r = rnd() % 100u, a = r < 30 ? 0 : r < 55 ? 1 : r < 70 ? 2 : r < 80 ? 3 : r < 88 ? 4 : r < 93 ? 5 : r < 97 ? 6 : 7;
                uint32_t blk = i / B;
                if (even_blocks && !g->kind && (c % 3u == 0u)) a = (uint32_t[]){0, 2, 4, 5, 6, 7}[rnd() % 6u];   /* ceil-form candidates */
                if (i % B == B - 1u && (even_blocks || !(c % 2u))) a = 7;     /* blocks reach the max (always for ceil tests: no max-5 block) */
                uint32_t x = a | (rnd() & 8u);
                if (cz && x == 8u) x = 0;
                (void)blk; (void)spc;
                slab[g->woff[m] + c * 16u + i / 2u] |= (unsigned char)(x << ((i & 1u) * 4u));
            }
    }
}

/* coded domain for MX with the ceil form of WNR3 4.2 applied by this test (gate: no block max 5) */
static uint32_t ceil_forward(const Wnr4Geo *g, unsigned char *coded) {
    uint32_t m, c, i, mask = 0;
    if (g->kind) return 0;
    for (m = 0; m < 3u; m++) {
        int ok = 1;
        for (c = 0; c < g->cells[m] && ok; c++) {
            uint32_t mx = 0;
            for (i = 0; i < 32u; i++) { uint32_t a = (coded[g->woff[m] + c * 16u + i / 2u] >> ((i & 1u) * 4u)) & 7u; if (a > mx) mx = a; }
            if (mx == 5u) ok = 0;
        }
        if (!ok) continue;
        mask |= 1u << m;
        for (c = 0; c < g->cells[m]; c++) {
            uint32_t mx = 0, odd = 0;
            unsigned char *w = coded + g->woff[m] + c * 16u, *s = coded + g->soff[m] + c;
            for (i = 0; i < 32u; i++) { uint32_t a = (w[i / 2u] >> ((i & 1u) * 4u)) & 7u; if (a > mx) mx = a; if (a == 1u || a == 3u) odd = 1; }
            if (mx == 7u && !odd && *s <= 254u) {
                static const unsigned char H[8] = {0, 0, 1, 0, 2, 3, 4, 5};
                for (i = 0; i < 16u; i++) {
                    uint32_t lo = w[i] & 15u, hi = w[i] >> 4;
                    w[i] = (unsigned char)(((lo & 8u) | H[lo & 7u]) | (((hi & 8u) | H[hi & 7u]) << 4));
                }
                (*s)++;
            }
        }
    }
    return mask;
}

static void geo_mx(Wnr4Geo *g, uint32_t r0, uint32_t c0, uint32_t r1, uint32_t c1) {
    uint32_t sc[3] = {r0 * c0, r1 * c1, r0 * c0}, cols[3] = {c0, c1, c0}, so[3], wo[3];
    so[0] = 0; so[1] = sc[0]; so[2] = sc[0] + sc[1];
    wo[0] = so[2] + sc[2]; wo[1] = wo[0] + sc[0] * 16u; wo[2] = wo[1] + sc[1] * 16u;
    if (wnr4_geo(g, 0, sc, cols, so, wo)) { printf("bad geo\n"); exit(1); }
}
static void geo_nv(Wnr4Geo *g, uint32_t r0, uint32_t c0, uint32_t r1, uint32_t c1) {
    uint32_t sc[3] = {r0 * c0, r1 * c1, r0 * c0}, cols[3] = {c0, c1, c0}, so[3], wo[3];
    so[0] = 0; so[1] = sc[0]; so[2] = sc[0] + sc[1];
    wo[0] = so[2] + sc[2]; wo[1] = wo[0] + sc[0] * 8u; wo[2] = wo[1] + sc[1] * 8u;
    if (wnr4_geo(g, 1, sc, cols, so, wo)) { printf("bad geo\n"); exit(1); }
}

/* the expected decode: the source with Z (CZ) - i.e. the slab before the ceil form */
static int roundtrip(const char *name, const Wnr4Geo *g, uint32_t pb, int cz, int even, const uint32_t *Gs, int nG,
                     int fuzz) {
    unsigned char *src = malloc(g->raw), *coded = malloc(g->raw), *out = malloc(g->raw);
    size_t cap = (size_t)g->raw * 3u + 65536u, mb = wnr4_model_bytes(g->kind, pb);
    unsigned char *rec = malloc(cap), *blob = malloc(mb), *bad = malloc(cap);
    uint64_t *counts = calloc(wnr4_counts_words(g->kind, pb), 8);
    Wnr4Model model;
    uint32_t flags, gi, owner = 0x00050011u;
    unsigned char a[3];
    int m, it;
    make_slab(g, src, cz, even);
    memcpy(coded, src, g->raw);
    flags = ceil_forward(g, coded) | (cz ? WNR4_FLAG_CZ : 0u);
    for (m = 0; m < 3; m++) a[m] = wnr4_anchor(coded + g->soff[m], g->scount[m]);
    wnr4_count(g, coded, a, pb, counts);
    wnr4_model_build(g->kind, pb, counts, blob);
    CHECK(!wnr4_model_open(blob, mb, &model), "%s: model", name);
    for (gi = 0; gi < (uint32_t)nG; gi++) {
        size_t n = wnr4_encode(&model, g, coded, flags, owner, Gs[gi], rec, cap);
        uint32_t err;
        CHECK(n && !(n & 4095u), "%s G=%u: encode", name, Gs[gi]);
        if (!n) continue;
        memset(out, 0xA5, g->raw);
        err = wnr4_decode(&model, g, rec, n, owner, out, 4);
        CHECK(!err && !memcmp(out, src, g->raw), "%s G=%u: round trip (err %u)", name, Gs[gi], err);
        CHECK(wnr4_decode(&model, g, rec, n, owner ^ 1u, out, 1), "%s G=%u: wrong owner accepted", name, Gs[gi]);
        /* batch decode: the record twice and a corrupted copy in the middle; only the copy fails */
        {
            unsigned char *outs3[3] = {malloc(g->raw), malloc(g->raw), malloc(g->raw)};
            const unsigned char *recs3[3] = {rec, bad, rec};
            size_t tot3[3] = {n, n, n};
            uint32_t own3[3] = {owner, owner, owner}, st3[3];
            memcpy(bad, rec, n); bad[n / 2] ^= 0x20;
            int nb = wnr4_decode_batch(&model, g, 3, recs3, tot3, own3, outs3, st3, 4);
            CHECK(nb == 1 && !st3[0] && st3[1] && !st3[2] && !memcmp(outs3[0], src, g->raw) && !memcmp(outs3[2], src, g->raw),
                  "%s G=%u: batch decode (failed %d, status %u %u %u)", name, Gs[gi], nb, st3[0], st3[1], st3[2]);
            free(outs3[0]); free(outs3[1]); free(outs3[2]);
        }
        /* header fields */
        static const int fields[] = {0, 4, 8, 12, 13, 14, 16, 19, 20, 24, 40, 64};
        for (it = 0; it < (int)(sizeof(fields) / sizeof(fields[0])); it++) {
            memcpy(bad, rec, n); bad[fields[it]] ^= 0x40;
            CHECK(wnr4_decode(&model, g, bad, n, owner, out, 1), "%s G=%u: header byte %d flip accepted", name, Gs[gi], fields[it]);
        }
        /* fuzz: random flips anywhere in the used bytes, 1..3 per trial */
        if (fuzz && gi == 0) {
            int refused = 0, same = 0;
            size_t used = n;
            while (used && !rec[used - 1]) used--;
            for (it = 0; it < fuzz; it++) {
                int k, f = 1 + (int)(rnd() % 3u);
                memcpy(bad, rec, n);
                for (k = 0; k < f; k++) bad[rnd() % used] ^= (unsigned char)(1u << (rnd() % 8u));
                if (!memcmp(bad, rec, n)) { same++; continue; }
                err = wnr4_decode(&model, g, bad, n, owner, out, 1);
                if (err) refused++;
                else CHECK(!memcmp(out, src, g->raw), "%s: fuzz trial %d accepted with different bytes", name, it);
            }
            printf("  %s fuzz: %d trials, %d refused, %d no-op, %d accepted (all identical)\n", name, fuzz, refused, same,
                   fuzz - refused - same);
        }
    }
    printf("%s: kind %s pb %u cz %d ceil mask %u, %d G values OK\n", name, g->kind ? "NV" : "MX", pb, cz, flags & 7u, nG);
    wnr4_model_free(&model);
    free(src); free(coded); free(out); free(rec); free(blob); free(bad); free(counts);
    return 0;
}

/* test wnr4_prepare() on MX (plain, CZ, ceil gate, in-place alias) and NV (plain, CZ, no ceil) */
static void test_prepare(void) {
    Wnr4Geo g;
    /* MX: even_blocks=1 passes ceil gate on all 3 matrices */
    geo_mx(&g, 8, 20, 6, 16);
    {
        unsigned char *src = malloc(g.raw), *coded = malloc(g.raw), *alias = malloc(g.raw), *ref = malloc(g.raw);
        unsigned char *out = malloc(g.raw);
        size_t cap = (size_t)g.raw * 3u + 65536u, mb = wnr4_model_bytes(0, 1);
        unsigned char *rec = malloc(cap), *blob = malloc(mb);
        uint64_t *counts = calloc(wnr4_counts_words(0, 1), 8), mapped = 999, neg0 = 0;
        Wnr4Model model;
        unsigned char a[3];
        uint32_t m, flags, f_ref, f_alias;
        make_slab(&g, src, 0, 1);
        for (m = 0; m < 3u; m++) {
            size_t nb = (size_t)g.cells[m] * 16u, b;
            for (b = 0; b < nb; b++) {
                unsigned char v = src[g.woff[m] + b];
                if ((v & 15u) == 8u) neg0++;
                if ((v >> 4) == 8u) neg0++;
            }
        }
        /* plain MX: ceil mask = 7, cz_mapped = 0, matches ceil_forward, alias == out-of-place */
        memcpy(ref, src, g.raw);
        f_ref = ceil_forward(&g, ref);
        flags = wnr4_prepare(&g, src, coded, 0, &mapped);
        memcpy(alias, src, g.raw);
        f_alias = wnr4_prepare(&g, alias, alias, 0, NULL);
        CHECK(flags == 7u && flags == f_ref && f_alias == flags && mapped == 0 && !memcmp(coded, ref, g.raw) &&
              !memcmp(alias, coded, g.raw), "prepare MX plain (flags %u ref %u mapped %llu)", flags, f_ref,
              (unsigned long long)mapped);
        for (m = 0; m < 3u; m++) a[m] = wnr4_anchor(coded + g.soff[m], g.scount[m]);
        wnr4_count(&g, coded, a, 1, counts);
        wnr4_model_build(0, 1, counts, blob);
        CHECK(!wnr4_model_open(blob, mb, &model), "prepare MX model");
        {
            size_t n = wnr4_encode(&model, &g, coded, flags, 42u, 4u, rec, cap);
            CHECK(n && !wnr4_decode(&model, &g, rec, n, 42u, out, 2) && !memcmp(out, src, g.raw),
                  "prepare MX plain round-trip");
        }
        /* MX with CZ: maps exact -0 count, round-trips to Z(src) */
        src[g.woff[0]] = (unsigned char)((src[g.woff[0]] & 0xf0u) | 8u);   /* ensure at least one -0 */
        neg0 = 0;
        memcpy(ref, src, g.raw);
        for (m = 0; m < 3u; m++) {
            size_t nb = (size_t)g.cells[m] * 16u, b;
            for (b = 0; b < nb; b++) {
                unsigned char v = ref[g.woff[m] + b];
                if ((v & 15u) == 8u) { v &= 0xf0u; neg0++; }
                if ((v >> 4) == 8u) { v &= 0x0fu; neg0++; }
                ref[g.woff[m] + b] = v;
            }
        }
        flags = wnr4_prepare(&g, src, coded, 1, &mapped);
        CHECK(flags == (7u | WNR4_FLAG_CZ) && mapped == neg0 && neg0 > 0, "prepare MX CZ (flags %u mapped %llu want %llu)",
              flags, (unsigned long long)mapped, (unsigned long long)neg0);
        {
            size_t n = wnr4_encode(&model, &g, coded, flags, 43u, 4u, rec, cap);
            CHECK(n && !wnr4_decode(&model, &g, rec, n, 43u, out, 2) && !memcmp(out, ref, g.raw),
                  "prepare MX CZ round-trip");
        }
        /* MX ceil gate: inject a block of max magnitude 5 in matrix 1 -> bit 1 cleared */
        memset(src + g.woff[1], 0x11, 16u);
        src[g.woff[1]] = 0x15u;
        flags = wnr4_prepare(&g, src, coded, 0, &mapped);
        CHECK(flags == 5u && !memcmp(coded + g.soff[1], src + g.soff[1], g.scount[1]) &&
              !memcmp(coded + g.woff[1], src + g.woff[1], (size_t)g.cells[1] * 16u),
              "prepare MX ceil gate (flags %u)", flags);
        {
            size_t n = wnr4_encode(&model, &g, coded, flags, 44u, 4u, rec, cap);
            CHECK(n && !wnr4_decode(&model, &g, rec, n, 44u, out, 2) && !memcmp(out, src, g.raw),
                  "prepare MX ceil gate round-trip");
        }
        wnr4_model_free(&model);
        free(src); free(coded); free(alias); free(ref); free(out); free(rec); free(blob); free(counts);
    }
    /* NV: never sets ceil bits, plain leaves slab unchanged, CZ maps -0 only */
    geo_nv(&g, 8, 20, 6, 12);
    {
        unsigned char *src = malloc(g.raw), *coded = malloc(g.raw), *ref = malloc(g.raw), *out = malloc(g.raw);
        size_t cap = (size_t)g.raw * 3u + 65536u, mb = wnr4_model_bytes(1, 1);
        unsigned char *rec = malloc(cap), *blob = malloc(mb);
        uint64_t *counts = calloc(wnr4_counts_words(1, 1), 8), mapped = 999, neg0 = 0;
        Wnr4Model model;
        unsigned char a[3];
        uint32_t m, flags;
        make_slab(&g, src, 0, 1);
        src[g.woff[2]] = 0x88u;   /* guarantee -0 nibbles */
        flags = wnr4_prepare(&g, src, coded, 0, &mapped);
        CHECK(flags == 0u && mapped == 0 && !memcmp(coded, src, g.raw), "prepare NV plain (flags %u)", flags);
        memcpy(ref, src, g.raw);
        for (m = 0; m < 3u; m++) {
            size_t nb = (size_t)g.cells[m] * 16u, b;
            for (b = 0; b < nb; b++) {
                unsigned char v = ref[g.woff[m] + b];
                if ((v & 15u) == 8u) { v &= 0xf0u; neg0++; }
                if ((v >> 4) == 8u) { v &= 0x0fu; neg0++; }
                ref[g.woff[m] + b] = v;
            }
        }
        flags = wnr4_prepare(&g, src, coded, 1, &mapped);
        CHECK(flags == WNR4_FLAG_CZ && mapped == neg0 && neg0 >= 2 && !memcmp(coded, ref, g.raw),
              "prepare NV CZ (flags %u mapped %llu want %llu)", flags, (unsigned long long)mapped, (unsigned long long)neg0);
        for (m = 0; m < 3u; m++) a[m] = wnr4_anchor(coded + g.soff[m], g.scount[m]);
        wnr4_count(&g, coded, a, 1, counts);
        wnr4_model_build(1, 1, counts, blob);
        CHECK(!wnr4_model_open(blob, mb, &model), "prepare NV model");
        {
            size_t n = wnr4_encode(&model, &g, coded, flags, 45u, 4u, rec, cap);
            CHECK(n && !wnr4_decode(&model, &g, rec, n, 45u, out, 2) && !memcmp(out, ref, g.raw),
                  "prepare NV CZ round-trip");
        }
        wnr4_model_free(&model);
        free(src); free(coded); free(ref); free(out); free(rec); free(blob); free(counts);
    }
    printf("prepare tests done\n");
}

int main(int argc, char **argv) {
    int fuzz = argc > 1 ? atoi(argv[1]) : 2000;
    static const uint32_t Gs[] = {1, 2, 3, 7, 32, 128, 256};
    Wnr4Geo g;
    test_prepare();
    /* MX: matrices of 1000 cells (31.25 lanes-rounds at G=1), 357 cells (odd) */
    geo_mx(&g, 25, 40, 21, 17);
    roundtrip("mx-plain", &g, 1, 0, 0, Gs, 7, fuzz);
    roundtrip("mx-cz-ceil", &g, 5, 1, 1, Gs, 7, fuzz);
    /* NV: rows of 40 scales (row-above context) and of 12 (no context); scount even */
    geo_nv(&g, 30, 40, 23, 12);
    roundtrip("nv-plain", &g, 1, 0, 0, Gs, 7, fuzz);
    roundtrip("nv-cz-pb5", &g, 5, 1, 0, Gs, 7, fuzz);
    /* negative: CZ record with a coded nibble 8 must be refused; ceil underflow must be refused */
    {
        geo_mx(&g, 4, 40, 4, 40);
        unsigned char *src = malloc(g.raw), *out = malloc(g.raw);
        size_t cap = g.raw * 3u + 65536u, mb = wnr4_model_bytes(0, 1);
        unsigned char *rec = malloc(cap), *blob = malloc(mb);
        uint64_t *counts = calloc(wnr4_counts_words(0, 1), 8);
        Wnr4Model model;
        unsigned char a[3] = {120, 120, 120};
        make_slab(&g, src, 0, 0);
        src[g.woff[0]] = 0x08;   /* a -0 in the first cell */
        wnr4_count(&g, src, a, 1, counts);
        wnr4_model_build(0, 1, counts, blob);
        CHECK(!wnr4_model_open(blob, mb, &model), "neg model");
        size_t n = wnr4_encode(&model, &g, src, WNR4_FLAG_CZ, 7, 4, rec, cap);
        CHECK(n && (wnr4_decode(&model, &g, rec, n, 7, out, 1) & WNR4_ERR_CZ), "CZ nibble 8 not refused");
        /* ceil underflow: a coded block of max 5 with scale 0 under ceil flag of matrix 0 */
        memset(src + g.woff[0], 0x11, 16); src[g.woff[0]] = 0x15; src[g.soff[0]] = 0;
        for (uint32_t c = 1; c < g.cells[0]; c++) if (((src[g.woff[0] + c * 16u] >> 4) & 7u) == 5u) src[g.woff[0] + c * 16u] &= 0x0f;
        n = wnr4_encode(&model, &g, src, 1u, 7, 4, rec, cap);
        CHECK(n && (wnr4_decode(&model, &g, rec, n, 7, out, 1) & WNR4_ERR_CEIL), "ceil underflow not refused");
        /* another model must not decode it */
        counts[5]++;
        wnr4_model_build(0, 1, counts, blob);
        Wnr4Model other;
        CHECK(!wnr4_model_open(blob, mb, &other) && other.id != model.id, "other model");
        CHECK(wnr4_decode(&other, &g, rec, n, 7, out, 1), "record decoded under another model");
        /* corrupt model blob refused */
        blob[100] ^= 0xff; blob[101] ^= 0xff;
        CHECK(wnr4_model_open(blob, mb, &other), "corrupt model accepted");
        wnr4_model_free(&model);
        /* geometry: overlapping regions and huge counts are refused */
        {
            Wnr4Geo bad;
            uint32_t sc[3] = {160, 160, 160}, cols[3] = {40, 40, 40}, so[3] = {0, 160, 320}, wo[3] = {480, 480 + 2560, 480 + 5120};
            CHECK(!wnr4_geo(&bad, 0, sc, cols, so, wo), "valid geometry refused");
            wo[1] = 480 + 100;   /* overlaps matrix 0's nibbles */
            CHECK(wnr4_geo(&bad, 0, sc, cols, so, wo), "overlapping nibble regions accepted");
            wo[1] = 480 + 2560; so[2] = 100;   /* overlaps matrix 0's scales */
            CHECK(wnr4_geo(&bad, 0, sc, cols, so, wo), "overlapping scale regions accepted");
            so[2] = 320; sc[0] = 0xFFFFFFC0u; cols[0] = 0x40u;
            CHECK(wnr4_geo(&bad, 0, sc, cols, so, wo), "huge scale count accepted");
        }
        printf("negative cases done\n");
    }
    printf(failures ? "FAILED: %d\n" : "ALL OK\n", failures);
    return failures != 0;
}
