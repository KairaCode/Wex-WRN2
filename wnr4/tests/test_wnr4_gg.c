/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-GG CPU tests (wnr4_gg.h): round trips of synthetic GGUF expert records (1..4 slices mixing Q2_0, IQ4_NL
 * and verbatim types, partial tiles, single blocks, G 1..256) under a model trained on other synthetic records;
 * refusals (every header byte, every single-bit flip of small records, owner, model, geometry, model blobs, short
 * and misaligned records) and a mutation fuzz: every mutated record must be refused or decode to the expected bytes.
 * Verbatim slices are not covered by the tile hash: a flip there decodes without error to the flipped bytes (the
 * record SHA-256 of the WEX4 index covers them). The test checks that and prints it.
 * Known answers pin the format bytes: the model id and the FNV-1a of one record at G=1 and at G=32 (the data come
 * from a fixed xorshift seed in a fixed call order and the model build uses integers only, so they hold on every
 * compiler and platform). Not covered: two state rules of a tile, an initial lane state below L and a tail lane
 * that ends at L + 2^16 or above; removing either one is not detected, because the tile hash or the end state also
 * refuses every such record that flips make; a test would need a re-encoded tile.
 *   gcc -O2 -fopenmp wnr4/tests/test_wnr4_gg.c -o test_wnr4_gg && ./test_wnr4_gg [fuzz_iterations] */
#include "../wnr4_gg.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static uint32_t get16(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static void put16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static int g_ok(const unsigned char *rec) { return get16(rec + 14) >= 1u && get16(rec + 14) <= WNR4_MAX_G; }

/* a nibble: Q2_0 = two 2-bit codes leaning to 1 and 2, IQ4_NL = an index leaning to the middle of 0..15
 * (two rnd() calls in one expression only where their order does not matter: the known answers depend on it) */
static uint32_t nib(int sub) {
    static const unsigned char q[10] = {0, 1, 1, 1, 2, 2, 2, 3, 1, 2};
    uint32_t lo;
    if (sub) return 2u + rnd() % 5u + rnd() % 5u + rnd() % 5u;
    lo = q[rnd() % 10u];
    return lo | (uint32_t)q[rnd() % 10u] << 2;
}

/* a record's decoded bytes: per coded block an f16 scale whose high byte is one of a few (skew moves them; 3 %
 * outliers of any value, 2 % all-zero blocks) and 32 skewed nibbles; verbatim slices random */
static void make_record(const Wnr4GgGeo *g, unsigned char *slab, uint32_t skew) {
    uint32_t s, c, i;
    for (s = 0; s < g->nseg; s++) {
        unsigned char *p = slab + g->off[s];
        if (g->sub[s] < 0) { for (i = 0; i < g->bytes[s]; i++) p[i] = (unsigned char)rnd(); continue; }
        for (c = 0; c < g->cells[s]; c++, p += 18) {
            uint32_t r = rnd() % 100u;
            if (r < 2u) { memset(p, 0, 18); continue; }
            p[0] = (unsigned char)rnd();
            p[1] = (unsigned char)(r < 5u ? rnd() : 0x14u + skew + 4u * (uint32_t)g->sub[s] + rnd() % 3u + rnd() % 3u);
            for (i = 0; i < 16u; i++) { uint32_t lo = nib(g->sub[s]); p[2 + i] = (unsigned char)(lo | nib(g->sub[s]) << 4); }
        }
    }
}

/* a model from the counts of 4 other synthetic records (both subtypes and a verbatim slice) */
static void train(uint32_t skew, unsigned char *blob, Wnr4GgModel *m) {
    static const uint32_t type[3] = {WNR4_GG_Q2_0, 16, WNR4_GG_IQ4_NL}, bytes[3] = {2000 * 18, 100, 1500 * 18};
    uint64_t *counts = calloc(wnr4_gg_counts_words(), 8);
    unsigned char *slab;
    Wnr4GgGeo g;
    int k;
    if (wnr4_gg_geo(&g, 3, type, bytes)) { printf("FAIL: training geometry\n"); exit(1); }
    slab = malloc(g.raw);
    for (k = 0; k < 4; k++) { make_record(&g, slab, skew); wnr4_gg_count(&g, slab, counts); }
    if (wnr4_gg_model_build(counts, blob)) { printf("FAIL: model of skew %u not built\n", skew); exit(1); }
    if (wnr4_gg_model_open(blob, WNR4_GG_MODEL_BYTES, m)) { printf("FAIL: model of skew %u refused\n", skew); exit(1); }
    free(slab); free(counts);
}

/* what an accepted record decodes to: src with its verbatim slices taken from rec (raw area at rawoff) */
static void expect(const Wnr4GgGeo *g, const unsigned char *src, const unsigned char *rec, size_t rawoff, unsigned char *exp) {
    Wnr4GgRecord r;
    memset(&r, 0, sizeof(r));
    r.raw = rec + rawoff;
    memcpy(exp, src, g->raw);
    wnr4_gg_copy_raw(g, &r, exp);
}

/* every single-bit flip of the record, by region: refused, except in a verbatim slice (decodes without error to
 * the flipped bytes) and, for a record without tiles, in G when it stays in 1..4096 (decodes to the same bytes:
 * nothing else depends on G) */
static void flips(const char *name, uint32_t G, const Wnr4GgModel *m, const Wnr4GgGeo *g, const unsigned char *rec,
                  size_t n, uint32_t owner, const unsigned char *src, unsigned char *bad, unsigned char *out,
                  unsigned char *exp) {
    Wnr4GgRecord r;
    size_t end[6], b;
    int tried[6] = {0}, refused[6] = {0}, same = 0, k = 0;
    if (wnr4_gg_parse(rec, n, g, &r)) { CHECK(0, "%s G=%u: parse", name, G); return; }
    end[0] = WNR4_HEADER_BYTES; end[1] = (size_t)(r.raw - rec); end[2] = end[1] + g->raw_bytes;
    end[3] = (size_t)(r.data - rec); end[4] = end[3] + r.payload; end[5] = n;
    memcpy(bad, rec, n);
    for (b = 0; b < 8u * n; b++) {
        uint32_t err;
        while (b / 8u >= end[k]) k++;
        bad[b / 8u] ^= (unsigned char)(1u << (b % 8u));
        err = wnr4_gg_decode(m, g, bad, n, owner, out, 1);
        tried[k]++;
        if (err) refused[k]++;
        else if (k == 2) {
            expect(g, src, bad, end[1], exp);
            CHECK(!memcmp(out, exp, g->raw) && memcmp(out, src, g->raw), "%s G=%u: verbatim bit %u: wrong bytes", name, G,
                  (unsigned)(b - 8u * end[1]));
        } else {
            same++;
            CHECK(!r.ntiles && (b / 8u == 14u || b / 8u == 15u) && g_ok(bad) && !memcmp(out, src, g->raw),
                  "%s G=%u: flip of record bit %u accepted", name, G, (unsigned)b);
        }
        bad[b / 8u] ^= (unsigned char)(1u << (b % 8u));
    }
    CHECK(!refused[2], "%s G=%u: %d verbatim flips refused", name, G, refused[2]);
    printf("  %s G=%u single-bit flips: header %d, offsets %d, verbatim padding %d, tiles %d, end padding %d: %d refused,"
           " %d (G of a record without tiles) decoded to the same bytes; verbatim %d: decoded without error to the"
           " flipped bytes\n", name, G, tried[0], tried[1], tried[3], tried[4], tried[5],
           refused[0] + refused[1] + refused[3] + refused[4] + refused[5], same, tried[2]);
}

typedef struct { const char *name; uint32_t nseg, type[WNR4_GG_MAX_SEG], bytes[WNR4_GG_MAX_SEG]; } Case;

static void roundtrip(const Case *cs, const Wnr4GgModel *m, const Wnr4GgModel *other, const uint32_t *Gs, int nG,
                      int fuzz) {
    static const unsigned char mask[3] = {0x01, 0x40, 0xff};
    uint32_t owner = 0x0003002Au, gi, ntiles = 0;
    unsigned char *src, *out, *exp, *rec, *bad;
    Wnr4GgGeo g;
    size_t cap;
    int it;
    if (wnr4_gg_geo(&g, cs->nseg, cs->type, cs->bytes)) { CHECK(0, "%s: geometry", cs->name); return; }
    cap = (size_t)g.raw * 3u + 65536u;
    src = malloc(g.raw); out = malloc(g.raw); exp = malloc(g.raw); rec = malloc(cap); bad = malloc(cap);
    make_record(&g, src, 0);
    for (gi = 0; gi < (uint32_t)nG; gi++) {
        uint32_t G = Gs[gi], err;
        size_t n = wnr4_gg_encode(m, &g, src, owner, G, rec, cap);
        Wnr4GgRecord r;
        CHECK(n && !(n & 4095u), "%s G=%u: encode", cs->name, G);
        if (!n || wnr4_gg_parse(rec, n, &g, &r)) { CHECK(0, "%s G=%u: parse", cs->name, G); continue; }
        ntiles = r.ntiles;
        memset(out, 0xA5, g.raw);
        err = wnr4_gg_decode(m, &g, rec, n, owner, out, 4);
        CHECK(!err && !memcmp(out, src, g.raw), "%s G=%u: round trip (err %u)", cs->name, G, err);
        memset(out, 0x5A, g.raw);
        err = wnr4_gg_decode(m, &g, rec, n, owner, out, 1);
        CHECK(!err && !memcmp(out, src, g.raw), "%s G=%u: round trip on one thread (err %u)", cs->name, G, err);
        CHECK(wnr4_gg_decode(m, &g, rec, n, owner ^ 1u, out, 1), "%s G=%u: wrong owner accepted", cs->name, G);
        CHECK(wnr4_gg_decode(other, &g, rec, n, owner, out, 1), "%s G=%u: decoded under another model", cs->name, G);
        if (r.ntiles) {   /* relabelled in the header: the tile hash binds model id and owner */
            memcpy(bad, rec, n); wnr4_put32(bad + 4, other->id);
            CHECK(wnr4_gg_decode(other, &g, bad, n, owner, out, 1), "%s G=%u: relabelled to another model, accepted",
                  cs->name, G);
            memcpy(bad, rec, n); wnr4_put32(bad + 8, owner ^ 1u);
            CHECK(wnr4_gg_decode(m, &g, bad, n, owner ^ 1u, out, 1), "%s G=%u: relabelled owner accepted", cs->name, G);
        }
        /* every header byte XORed with 0x01, 0x40 and 0xff: refused (G of a record without tiles, still in
         * 1..4096: the same bytes) */
        for (it = 0; it < 64 * 3; it++) {
            memcpy(bad, rec, n); bad[it / 3] ^= mask[it % 3];
            err = wnr4_gg_decode(m, &g, bad, n, owner, out, 1);
            CHECK(err || (!r.ntiles && (it / 3 == 14 || it / 3 == 15) && g_ok(bad) && !memcmp(out, src, g.raw)),
                  "%s G=%u: header byte %d ^ 0x%02x accepted", cs->name, G, it / 3, mask[it % 3]);
        }
        if (gi == 0 && n <= 16384u) flips(cs->name, G, m, &g, rec, n, owner, src, bad, out, exp);
        /* fuzz: random flips anywhere in the used bytes and a little of the padding, 1..3 per trial */
        if (gi == 0 && fuzz) {
            int refused = 0, same = 0, ident = 0, verb = 0;
            size_t used = n;
            while (used && !rec[used - 1]) used--;
            used = used + 16u < n ? used + 16u : n;
            for (it = 0; it < fuzz; it++) {
                int k, f = 1 + (int)(rnd() % 3u);
                memcpy(bad, rec, n);
                for (k = 0; k < f; k++) bad[rnd() % used] ^= (unsigned char)(1u << (rnd() % 8u));
                if (!memcmp(bad, rec, n)) { same++; continue; }
                if (wnr4_gg_decode(m, &g, bad, n, owner, out, 1)) { refused++; continue; }
                expect(&g, src, bad, (size_t)(r.raw - rec), exp);
                CHECK(!memcmp(out, exp, g.raw), "%s: fuzz trial %d accepted with bytes other than its verbatim bytes",
                      cs->name, it);
                if (memcmp(out, src, g.raw)) { verb++; continue; }
                ident++;   /* the same bytes from other record bytes: only a valid G of a record without tiles */
                CHECK(!r.ntiles && g_ok(bad), "%s: fuzz trial %d: changed record accepted with the same bytes", cs->name, it);
            }
            printf("  %s fuzz: %d trials, %d refused, %d no-op, %d accepted with the same bytes, %d accepted with"
                   " changed verbatim bytes\n", cs->name, fuzz, refused, same, ident, verb);
        }
    }
    printf("%s: %u slices, %u verbatim bytes, %u tiles at G=%u, %d G values OK%s\n", cs->name, g.nseg, g.raw_bytes, ntiles,
           Gs[nG - 1], nG, ntiles ? "" : " (no tiles: model and owner are checked in the header only)");
    free(src); free(out); free(exp); free(rec); free(bad);
}

static size_t sub_at(uint32_t s) { return 32u + (size_t)s * WNR4_GG_SUB_BYTES; }
static size_t hi_at(uint32_t s, uint32_t i) { return sub_at(s) + 256u + 2u * i; }
static size_t lo_at(uint32_t s, uint32_t c, uint32_t i) { return sub_at(s) + 256u + 514u + 514u * c + 2u * i; }
static size_t nib_at(uint32_t s, uint32_t c, uint32_t i) { return sub_at(s) + 256u + 514u + 514u * WNR4_GG_CLASSES + 34u * c + 2u * i; }
static int opens(const unsigned char *b, size_t bytes) {
    Wnr4GgModel t;
    if (wnr4_gg_model_open(b, bytes, &t)) return 0;
    wnr4_gg_model_free(&t);
    return 1;
}

static void model_blobs(const unsigned char *blob) {
    const size_t MB = WNR4_GG_MODEL_BYTES;
    unsigned char *b = malloc(MB + 1u);
    uint32_t i;
    CHECK(opens(blob, MB), "model blob refused");
    CHECK(!get16(blob + hi_at(0, 0)) && get16(blob + hi_at(0, 256)) == WNR4_TOTAL && !get16(blob + lo_at(1, 0, 0)) &&
          get16(blob + nib_at(1, 31, 16)) == WNR4_TOTAL && nib_at(1, 31, 17) == MB, "model blob layout of this test");
    CHECK(!opens(blob, MB - 1u), "model blob one byte short accepted");
    memcpy(b, blob, MB); b[MB] = 0;
    CHECK(!opens(b, MB + 1u), "model blob one byte long accepted");
    CHECK(!opens(NULL, MB), "no model blob accepted");
    for (i = 0; i < 32u; i++) {
        memcpy(b, blob, MB); b[i] ^= 0x01;
        CHECK(!opens(b, MB), "model header byte %u changed, accepted", i);
    }
    memcpy(b, blob, MB); b[sub_at(0) + 7u] = 31;
    CHECK(opens(b, MB), "class 31 in the map refused");
    memcpy(b, blob, MB); b[sub_at(0) + 7u] = 32;
    CHECK(!opens(b, MB), "class 32 in the Q2_0 map accepted");
    memcpy(b, blob, MB); b[sub_at(1) + 255u] = 255;
    CHECK(!opens(b, MB), "class 255 in the IQ4_NL map accepted");
    memcpy(b, blob, MB); put16(b + hi_at(0, 10), get16(b + hi_at(0, 9)));
    CHECK(!opens(b, MB), "hi table with two equal entries accepted");
    memcpy(b, blob, MB); put16(b + lo_at(1, 31, 200), get16(b + lo_at(1, 31, 199)));
    CHECK(!opens(b, MB), "lo table with two equal entries accepted");
    memcpy(b, blob, MB); put16(b + nib_at(0, 5, 3), get16(b + nib_at(0, 5, 2)) - 1u);
    CHECK(!opens(b, MB), "decreasing nibble table accepted");
    memcpy(b, blob, MB); put16(b + hi_at(1, 0), 1);
    CHECK(!opens(b, MB), "table not starting at 0 accepted");
    memcpy(b, blob, MB); put16(b + nib_at(1, 31, 16), WNR4_TOTAL - 1u);
    CHECK(!opens(b, MB), "table not ending at 2^14 accepted");
    free(b);
    printf("model blobs: size, header, class map and table refusals done\n");
}

static void geometry(void) {
    uint32_t ty[5] = {16, 17, WNR4_GG_Q2_0, WNR4_GG_IQ4_NL, 8}, by[5] = {100, 101, 36, 18, 7};
    Wnr4GgGeo t;
    CHECK(!wnr4_gg_geo(&t, 4, ty, by) && t.raw_bytes == 201u && t.cells[2] == 2u && t.cells[3] == 1u && t.raw == 255u,
          "valid geometry refused");
    CHECK(wnr4_gg_geo(&t, 5, ty, by), "5 slices accepted");
    CHECK(wnr4_gg_geo(&t, 0, ty, by), "0 slices accepted");
    by[2] = 35; CHECK(wnr4_gg_geo(&t, 4, ty, by), "Q2_0 slice of 35 bytes accepted"); by[2] = 36;
    by[3] = 19; CHECK(wnr4_gg_geo(&t, 4, ty, by), "IQ4_NL slice of 19 bytes accepted"); by[3] = 18;
    by[2] = 0; CHECK(wnr4_gg_geo(&t, 4, ty, by), "Q2_0 slice of 0 bytes accepted"); by[2] = 36;
    by[1] = 0; CHECK(wnr4_gg_geo(&t, 4, ty, by), "verbatim slice of 0 bytes accepted"); by[1] = 101;
    by[1] = 19; CHECK(!wnr4_gg_geo(&t, 4, ty, by), "verbatim slice of 19 bytes refused"); by[1] = 101;
    by[2] = 18u << 27; CHECK(wnr4_gg_geo(&t, 4, ty, by), "Q2_0 slice of 2^27 blocks accepted"); by[2] = 36;
    by[0] = by[1] = 0x40000000u; CHECK(wnr4_gg_geo(&t, 4, ty, by), "2^31 verbatim bytes accepted"); by[0] = 100; by[1] = 101;
    by[2] = by[3] = 18u * ((1u << 27) - 1u); CHECK(wnr4_gg_geo(&t, 4, ty, by), "record of 2^32 bytes or more accepted");
    printf("geometry refusals done\n");
}

/* one record decoded under geometries other than its own, short and misaligned records, a longer last tile, and
 * the encoder's refusals */
static void record_shape(const Wnr4GgModel *m) {
    static const uint32_t ty0[3] = {16, 17, WNR4_GG_Q2_0}, by0[3] = {2112, 1650, 357 * 18};
    uint32_t ty[3], by[3], owner = 0x00010007u, err, d;
    Wnr4GgGeo g, h;
    Wnr4GgRecord r;
    size_t cap, n;
    unsigned char *src, *out, *rec, *small;
    if (wnr4_gg_geo(&g, 3, ty0, by0)) { CHECK(0, "shape geometry"); return; }
    cap = (size_t)g.raw * 3u + 65536u;
    src = malloc(g.raw); out = malloc(g.raw + 4096u); rec = calloc(cap + 4097u, 1);
    make_record(&g, src, 0);
    n = wnr4_gg_encode(m, &g, src, owner, 1, rec + 1, cap);
    CHECK(n > 4096u, "shape record: encode (%u bytes)", (unsigned)n);
    if (n <= 4096u) { free(src); free(out); free(rec); return; }
    err = wnr4_gg_decode(m, &g, rec + 1, n, owner, out, 2);
    CHECK(!err && !memcmp(out, src, g.raw), "record at an odd address (err %u)", err);
    memmove(rec, rec + 1, n); rec[n] = 0;
#define OTHER_GEO(what, edit, accept) do {                                                                 \
        memcpy(ty, ty0, sizeof(ty)); memcpy(by, by0, sizeof(by)); edit;                                    \
        if (wnr4_gg_geo(&h, 3, ty, by)) { CHECK(0, "geometry %s", what); break; }                          \
        err = wnr4_gg_decode(m, &h, rec, n, owner, out, 1);                                               \
        CHECK(accept ? !err && !memcmp(out, src, g.raw) : err != 0, "decoded as %s: err %u", what, err);    \
    } while (0)
    OTHER_GEO("its own geometry", (void)0, 1);
    OTHER_GEO("IQ4_NL instead of Q2_0", ty[2] = WNR4_GG_IQ4_NL, 0);
    OTHER_GEO("a verbatim slice instead of Q2_0", ty[2] = 22, 0);
    OTHER_GEO("one block more", by[2] += 18u, 0);
    OTHER_GEO("one block less", by[2] -= 18u, 0);
    OTHER_GEO("a verbatim slice 4 bytes longer", by[0] += 4u, 0);
    OTHER_GEO("bytes moved between verbatim slices", (by[0] += 4u, by[1] -= 4u), 1);
    OTHER_GEO("another ggml type of a verbatim slice", ty[1] = 22, 1);
#undef OTHER_GEO
    CHECK(wnr4_gg_decode(m, &g, rec, n - 4096u, owner, out, 1), "record one page short accepted");
    CHECK(wnr4_gg_decode(m, &g, rec, 63, owner, out, 1), "record of 63 bytes accepted");
    CHECK(wnr4_gg_decode(m, &g, rec, 0, owner, out, 1), "record of 0 bytes accepted");
    CHECK(wnr4_gg_decode(m, &g, NULL, n, owner, out, 1), "no record accepted");
    CHECK(wnr4_gg_decode(m, &g, rec, n - 1u, owner, out, 1), "record one byte short accepted");
    CHECK(wnr4_gg_decode(m, &g, rec, n + 1u, owner, out, 1), "record one byte long accepted");
    memset(rec + n, 0, 4096);
    CHECK(wnr4_gg_decode(m, &g, rec, n + 4096u, owner, out, 1), "record with an extra zero page accepted");
    /* the last tile 1 or 2 bytes longer (payload and last offset raised together, the bytes taken from the zero
     * padding): an odd tile length is refused, and so is a word the decoder does not use */
    if (wnr4_gg_parse(rec, n, &g, &r) || (size_t)(r.data - rec) + r.payload + 2u > n)
        CHECK(0, "shape record: no room to lengthen its last tile");
    else for (d = 1; d <= 2u; d++) {
        unsigned char *last = rec + WNR4_HEADER_BYTES + 4u * r.ntiles;
        wnr4_put32(rec + 24, r.payload + d); wnr4_put32(last, r.payload + d);
        CHECK(wnr4_gg_decode(m, &g, rec, n, owner, out, 1), "last tile %u byte(s) longer accepted", d);
        wnr4_put32(rec + 24, r.payload); wnr4_put32(last, r.payload);
    }
    small = malloc(n - 1u);   /* exactly n - 1 bytes: a sanitizer build checks that the encoder stays inside them */
    CHECK(small && !wnr4_gg_encode(m, &g, src, owner, 1, small, n - 1u), "record encoded into %u bytes, one too few",
          (unsigned)(n - 1u));
    CHECK(!wnr4_gg_encode(m, &g, src, owner, 0, rec, cap), "record encoded at G=0");
    CHECK(!wnr4_gg_encode(m, &g, src, owner, WNR4_MAX_G + 1u, rec, cap), "record encoded at G=%u", WNR4_MAX_G + 1u);
    printf("record shape: other geometries, short and misaligned records, longer last tiles and encoder refusals done;"
           " the split between verbatim slices and their ggml types are not part of the record (only their total"
           " size)\n");
    free(src); free(out); free(rec); free(small);
}

/* known answers (run right after training, so the rnd() state does not depend on the fuzz count): a change made
 * the same way in encoder and decoder (a lane seed, a table, the layout) changes these, other writers and the CUDA
 * decoder would not follow */
static void known_answers(const Wnr4GgModel *m) {
    static const uint32_t type[4] = {WNR4_GG_Q2_0, WNR4_GG_IQ4_NL, 8, WNR4_GG_IQ4_NL}, bytes[4] = {34 * 18, 65 * 18, 7, 18};
    static const uint32_t Gs[2] = {1, 32}, want[2] = {0x022c6e96u, 0xa71d5993u}, want_id = 0x28c021d1u;
    Wnr4GgGeo g;
    unsigned char *src, *rec;
    size_t n;
    int i;
    CHECK(m->id == want_id, "model id %08x, expected %08x", m->id, want_id);
    if (wnr4_gg_geo(&g, 4, type, bytes)) { CHECK(0, "known-answer geometry"); return; }
    src = malloc(g.raw); rec = malloc(65536);
    make_record(&g, src, 0);
    for (i = 0; i < 2; i++) {
        n = wnr4_gg_encode(m, &g, src, 0x0003002Au, Gs[i], rec, 65536);
        CHECK(n && wnr4_fnv1a(rec, n) == want[i], "record at G=%u: %u bytes, FNV-1a %08x, expected %08x", Gs[i],
              (unsigned)n, n ? wnr4_fnv1a(rec, n) : 0u, want[i]);
    }
    printf("known answers: model id %08x, one record at G=1 and G=32 checked\n", m->id);
    free(src); free(rec);
}

int main(int argc, char **argv) {
    int fuzz = argc > 1 ? atoi(argv[1]) : 2000;
    static const uint32_t Gs[] = {1, 2, 3, 7, 32, 128, 256};
    static const Case cases[] = {
        {"q2-single", 1, {WNR4_GG_Q2_0}, {18}},
        {"iq4-33", 1, {WNR4_GG_IQ4_NL}, {33 * 18}},
        {"q2-1000", 1, {WNR4_GG_Q2_0}, {1000 * 18}},
        {"iq2-iq2-q2", 3, {16, 17, WNR4_GG_Q2_0}, {2112, 1650, 357 * 18}},
        {"mixed-4", 4, {WNR4_GG_Q2_0, WNR4_GG_IQ4_NL, 8, WNR4_GG_IQ4_NL}, {34 * 18, 65 * 18, 7, 18}},
        {"verbatim-split", 4, {16, WNR4_GG_Q2_0, 22, WNR4_GG_IQ4_NL}, {5, 129 * 18, 1, 64 * 18}},
        {"verbatim-only", 1, {22}, {1001}},
        {"q2-9000", 1, {WNR4_GG_Q2_0}, {9000 * 18}},
    };
    unsigned char *blob = malloc(WNR4_GG_MODEL_BYTES), *blob2 = malloc(WNR4_GG_MODEL_BYTES);
    Wnr4GgModel model, other;
    size_t i;
    train(0, blob, &model);
    train(40, blob2, &other);
    CHECK(other.id != model.id, "the two models have the same id");
    known_answers(&model);
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) roundtrip(&cases[i], &model, &other, Gs, 7, fuzz);
    model_blobs(blob);
    geometry();
    record_shape(&model);
    printf("NOTE: verbatim slices are outside the tile hash: a flipped verbatim bit decodes without error to the"
           " flipped byte; the record SHA-256 of the WEX4 index covers those bytes\n");
    wnr4_gg_model_free(&model); wnr4_gg_model_free(&other);
    free(blob); free(blob2);
    printf(failures ? "FAILED: %d\n" : "ALL OK\n", failures);
    return failures != 0;
}
