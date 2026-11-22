/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4-I4 CPU tests (wnr4_i4.h): round trips of synthetic int4-g64 expert records (1..3 matrices, planar and pairs
 * packing, groups per row 1..48 with odd counts and partial 128-row blocks, single groups, G 1..256) under a model
 * trained on other synthetic records; hostile inputs (every scale random 32 bits, NaN, infinity, negative zero,
 * subnormals, random nibbles; an all-zero record); refusals (every header byte, every single-bit flip of small
 * records, owner, model, geometry, bases, model blobs, short and misaligned records) and a mutation fuzz: every
 * mutated record must be refused or decode to the original bytes. A record has no verbatim area: every decoded byte
 * is under the tile hash. Two kinds of change can decode to the original bytes: a base of a block whose scales all
 * escape (it is not used) changed to another valid base, and an anchor changed where every class stays the same
 * (the clamp saturates); the test counts them.
 * Known answers pin the format bytes: the model id and the FNV-1a of one record at G=1 and at G=32 (fixed xorshift
 * data; the model build uses integers only, the encoder's base search IEEE binary32 and binary64 arithmetic).
 *   gcc -O2 -fopenmp wnr4/tests/test_wnr4_i4.c -o test_wnr4_i4 && ./test_wnr4_i4 [fuzz_iterations] */
#include "../wnr4_i4.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static uint32_t get16(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static void put16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

/* A record as an int4 converter of an FP8 block-scaled source writes it: per 128 x 128 block a base B, per group
 * scale = lat(k, B) with k the group's largest E4M3 code (96..126, leaning high); 2 % of the groups get any 32 bits
 * (escapes). hostile: every scale random (NaN, infinity, negative zero, subnormals among them) and uniform nibbles.
 * Nibbles lean to 8 (int4 0) with a spread that follows k. */
static void make_record(const Wnr4I4Geo *g, unsigned char *slab, uint32_t skew, int hostile) {
    static const uint32_t odd[6] = {0x7fc00000u, 0x7f800000u, 0x80000000u, 0x00000001u, 0xff800000u, 0x007fffffu};
    uint32_t m, i, e, nb = g->nbases ? g->nbases : 1u, *B = malloc(nb * 4u);
    for (i = 0; i < g->nbases; i++) B[i] = 0x3a000000u + (rnd() % 0x01000000u) + skew * 0x00100000u;
    for (m = 0; m < g->nmat; m++)
        for (i = 0; i < g->cells[m]; i++) {
            uint32_t k = 126u - rnd() % 16u - rnd() % 16u, s, r = rnd() % 100u, v[64];
            unsigned char *grp = slab + g->coff[m] + 32u * (size_t)i;
            if (hostile) s = r < 6u ? odd[r] : rnd() << 16 ^ rnd();
            else if (r < 2u) s = rnd() << 16 ^ rnd();
            else s = wnr4_i4_lat(k, B[wnr4_i4_block(g->bbase[m], g->gc[m], g->nbc[m], i)]);
            wnr4_put32(slab + g->soff[m] + 4u * (size_t)i, s);
            for (e = 0; e < 64u; e++) {
                uint32_t sp = 1u + (k - 96u) / 8u;
                v[e] = hostile ? rnd() % 16u : 8u + rnd() % (2u * sp + 1u) - sp;
            }
            for (e = 0; e < 32u; e++)
                grp[e] = g->pairs ? (unsigned char)(v[2 * e] | v[2 * e + 1] << 4) : (unsigned char)(v[e] | v[e + 32] << 4);
        }
    free(B);
}

/* a model from the counts of 3 other synthetic records */
static void train(uint32_t skew, unsigned char *blob, Wnr4I4Model *m) {
    static const uint32_t rows[3] = {200, 200, 300}, cols[3] = {960, 960, 640};
    uint64_t *counts = calloc(wnr4_i4_counts_words(), 8);
    unsigned char *slab;
    Wnr4I4Geo g;
    int k;
    if (wnr4_i4_geo(&g, 3, rows, cols, 0)) { printf("FAIL: training geometry\n"); exit(1); }
    slab = malloc(g.raw);
    for (k = 0; k < 3; k++) { make_record(&g, slab, skew, 0); if (wnr4_i4_count(&g, slab, counts)) exit(1); }
    wnr4_i4_model_build(counts, blob);
    if (wnr4_i4_model_open(blob, WNR4_I4_MODEL_BYTES, m)) { printf("FAIL: model of skew %u refused\n", skew); exit(1); }
    free(slab); free(counts);
}

/* every single-bit flip of the record, by region: refused or decoded to the original bytes (allowed only in a base
 * or an anchor) */
static void flips(const char *name, uint32_t G, const Wnr4I4Model *m, const Wnr4I4Geo *g, const unsigned char *rec,
                  size_t n, uint32_t owner, const unsigned char *src, unsigned char *bad, unsigned char *out) {
    Wnr4I4Record r;
    size_t end[5], b;
    int tried[5] = {0}, refused[5] = {0}, same = 0, k = 0;
    if (wnr4_i4_parse(rec, n, g, &r)) { CHECK(0, "%s G=%u: parse", name, G); return; }
    end[0] = WNR4_HEADER_BYTES; end[1] = (size_t)(r.bases - rec); end[2] = (size_t)(r.data - rec);
    end[3] = end[2] + r.payload; end[4] = n;
    memcpy(bad, rec, n);
    for (b = 0; b < 8u * n; b++) {
        while (b / 8u >= end[k]) k++;
        bad[b / 8u] ^= (unsigned char)(1u << (b % 8u));
        tried[k]++;
        if (wnr4_i4_decode(m, g, bad, n, owner, out, 1)) refused[k]++;
        else {
            same++;
            CHECK(!memcmp(out, src, g->raw) && (k == 2 || (k == 0 && b / 8u >= 28u && b / 8u < 34u)),
                  "%s G=%u: flip of record bit %u accepted (region %d)", name, G, (unsigned)b, k);
        }
        bad[b / 8u] ^= (unsigned char)(1u << (b % 8u));
    }
    printf("  %s G=%u single-bit flips: header %d, offsets %d, bases %d, tiles %d, padding %d: %d refused, %d (a base"
           " or an anchor that is not used) decoded to the same bytes\n", name, G, tried[0], tried[1], tried[2], tried[3],
           tried[4], refused[0] + refused[1] + refused[2] + refused[3] + refused[4], same);
}

typedef struct { const char *name; uint32_t nmat, rows[3], cols[3], pairs; int hostile; } Case;

static void roundtrip(const Case *cs, const Wnr4I4Model *m, const Wnr4I4Model *other, const uint32_t *Gs, int nG, int fuzz) {
    static const unsigned char mask[3] = {0x01, 0x40, 0xff};
    uint32_t owner = 0x0003002Au, gi, ntiles = 0;
    unsigned char *src, *out, *rec, *bad;
    Wnr4I4Geo g;
    size_t cap;
    int it;
    if (wnr4_i4_geo(&g, cs->nmat, cs->rows, cs->cols, cs->pairs)) { CHECK(0, "%s: geometry", cs->name); return; }
    cap = (size_t)g.raw * 4u + 65536u;
    src = malloc(g.raw); out = malloc(g.raw); rec = malloc(cap); bad = malloc(cap);
    if (cs->hostile == 2) memset(src, 0, g.raw); else make_record(&g, src, 0, cs->hostile);
    for (gi = 0; gi < (uint32_t)nG; gi++) {
        uint32_t G = Gs[gi], err;
        size_t n = wnr4_i4_encode(m, &g, src, owner, G, rec, cap);
        Wnr4I4Record r;
        CHECK(n && !(n & 4095u), "%s G=%u: encode", cs->name, G);
        if (!n || wnr4_i4_parse(rec, n, &g, &r)) { CHECK(0, "%s G=%u: parse", cs->name, G); continue; }
        ntiles = r.ntiles;
        memset(out, 0xA5, g.raw);
        err = wnr4_i4_decode(m, &g, rec, n, owner, out, 4);
        CHECK(!err && !memcmp(out, src, g.raw), "%s G=%u: round trip (err %u)", cs->name, G, err);
        memset(out, 0x5A, g.raw);
        err = wnr4_i4_decode(m, &g, rec, n, owner, out, 1);
        CHECK(!err && !memcmp(out, src, g.raw), "%s G=%u: round trip on one thread (err %u)", cs->name, G, err);
        CHECK(wnr4_i4_decode(m, &g, rec, n, owner ^ 1u, out, 1), "%s G=%u: wrong owner accepted", cs->name, G);
        CHECK(wnr4_i4_decode(other, &g, rec, n, owner, out, 1), "%s G=%u: decoded under another model", cs->name, G);
        memcpy(bad, rec, n); wnr4_put32(bad + 4, other->id);   /* relabelled: the tile hash binds model id and owner */
        CHECK(wnr4_i4_decode(other, &g, bad, n, owner, out, 1), "%s G=%u: relabelled to another model, accepted", cs->name, G);
        memcpy(bad, rec, n); wnr4_put32(bad + 8, owner ^ 1u);
        CHECK(wnr4_i4_decode(m, &g, bad, n, owner ^ 1u, out, 1), "%s G=%u: relabelled owner accepted", cs->name, G);
        /* every header byte XORed with 0x01, 0x40 and 0xff: refused, or (an anchor) the same bytes */
        for (it = 0; it < 64 * 3; it++) {
            memcpy(bad, rec, n); bad[it / 3] ^= mask[it % 3];
            err = wnr4_i4_decode(m, &g, bad, n, owner, out, 1);
            CHECK(err || (it / 3 >= 28 && it / 3 < 28 + 2 * (int)g.nmat && !memcmp(out, src, g.raw)),
                  "%s G=%u: header byte %d ^ 0x%02x accepted", cs->name, G, it / 3, mask[it % 3]);
        }
        if (gi == 0 && n <= 16384u) flips(cs->name, G, m, &g, rec, n, owner, src, bad, out);
        if (gi == 0 && fuzz) {   /* random flips in the used bytes and a little of the padding, 1..3 per trial */
            int refused = 0, noop = 0, ident = 0;
            size_t used = n;
            while (used && !rec[used - 1]) used--;
            used = used + 16u < n ? used + 16u : n;
            for (it = 0; it < fuzz; it++) {
                int k, f = 1 + (int)(rnd() % 3u);
                memcpy(bad, rec, n);
                for (k = 0; k < f; k++) bad[rnd() % used] ^= (unsigned char)(1u << (rnd() % 8u));
                if (!memcmp(bad, rec, n)) { noop++; continue; }
                if (wnr4_i4_decode(m, &g, bad, n, owner, out, 1)) { refused++; continue; }
                ident++;
                CHECK(!memcmp(out, src, g.raw), "%s: fuzz trial %d accepted with other bytes", cs->name, it);
            }
            printf("  %s fuzz: %d trials, %d refused, %d no-op, %d accepted with the same bytes\n", cs->name, fuzz,
                   refused, noop, ident);
        }
    }
    printf("%s: %u matrices, %s, %u groups, %u bases, %u tiles at G=%u, %d G values OK\n", cs->name, g.nmat,
           g.pairs ? "pairs" : "planar", g.cells[0] + g.cells[1] + g.cells[2], g.nbases, ntiles, Gs[nG - 1], nG);
    free(src); free(out); free(rec); free(bad);
}

static size_t w_at(uint32_t row, uint32_t i) { return 32u + 34u * row + 2u * i; }
static size_t s_at(uint32_t row, uint32_t i) { return 32u + 34u * 3u * WNR4_I4_WROWS + 514u * row + 2u * i; }
static int opens(const unsigned char *b, size_t bytes) {
    Wnr4I4Model t;
    if (wnr4_i4_model_open(b, bytes, &t)) return 0;
    wnr4_i4_model_free(&t);
    return 1;
}
static void model_blobs(const unsigned char *blob) {
    const size_t MB = WNR4_I4_MODEL_BYTES;
    unsigned char *b = malloc(MB + 1u);
    uint32_t i;
    CHECK(MB == 196688u, "model blob size %u", (unsigned)MB);
    CHECK(opens(blob, MB), "model blob refused");
    CHECK(!get16(blob + w_at(0, 0)) && get16(blob + w_at(2699, 16)) == WNR4_TOTAL && !get16(blob + s_at(0, 0)) &&
          get16(blob + s_at(203, 256)) == WNR4_TOTAL && s_at(203, 257) == MB, "model blob layout of this test");
    CHECK(!opens(blob, MB - 1u), "model blob one byte short accepted");
    memcpy(b, blob, MB); b[MB] = 0;
    CHECK(!opens(b, MB + 1u), "model blob one byte long accepted");
    CHECK(!opens(NULL, MB), "no model blob accepted");
    for (i = 0; i < 32u; i++) { memcpy(b, blob, MB); b[i] ^= 0x01; CHECK(!opens(b, MB), "model header byte %u changed, accepted", i); }
    memcpy(b, blob, MB); put16(b + w_at(1234, 5), get16(b + w_at(1234, 4)));
    CHECK(!opens(b, MB), "nibble table with two equal entries accepted");
    memcpy(b, blob, MB); put16(b + s_at(100, 200), get16(b + s_at(100, 199)) - 1u);
    CHECK(!opens(b, MB), "decreasing scale table accepted");
    memcpy(b, blob, MB); put16(b + s_at(67, 0), 1);
    CHECK(!opens(b, MB), "table not starting at 0 accepted");
    memcpy(b, blob, MB); put16(b + w_at(899, 16), WNR4_TOTAL - 1u);
    CHECK(!opens(b, MB), "table not ending at 2^14 accepted");
    free(b);
    printf("model blobs: size, header and table refusals done\n");
}

static void geometry(void) {
    uint32_t rows[3] = {640, 640, 2560}, cols[3] = {2560, 2560, 640};
    Wnr4I4Geo t;
    CHECK(!wnr4_i4_geo(&t, 3, rows, cols, 0) && t.raw == 2764800u && t.nbases == 300u && t.soff[0] == 2457600u &&
          t.coff[2] == 1638400u && t.soff[2] == 2662400u, "Qwen3.8-like geometry");
    CHECK(wnr4_i4_geo(&t, 0, rows, cols, 0), "0 matrices accepted");
    CHECK(wnr4_i4_geo(&t, 4, rows, cols, 0), "4 matrices accepted");
    CHECK(wnr4_i4_geo(&t, 3, rows, cols, 2), "packing 2 accepted");
    cols[1] = 2559; CHECK(wnr4_i4_geo(&t, 3, rows, cols, 0), "cols not a multiple of 64 accepted"); cols[1] = 2560;
    cols[1] = 0; CHECK(wnr4_i4_geo(&t, 3, rows, cols, 0), "0 cols accepted"); cols[1] = 2560;
    rows[0] = 0; CHECK(wnr4_i4_geo(&t, 3, rows, cols, 0), "0 rows accepted"); rows[0] = 640;
    rows[0] = 1u << 21; cols[0] = 4096; CHECK(wnr4_i4_geo(&t, 1, rows, cols, 0), "2^27 groups accepted");
    rows[0] = 1864135u; CHECK(!wnr4_i4_geo(&t, 1, rows, cols, 0) && t.raw == 4294967040u, "record of 2^32 - 256 bytes refused");
    rows[0] = 1864136u; CHECK(wnr4_i4_geo(&t, 1, rows, cols, 0), "record of 2^32 + 2048 bytes accepted"); rows[0] = 640; cols[0] = 2560;
    rows[0] = rows[1] = rows[2] = (1u << 21) - 1u; cols[0] = cols[1] = cols[2] = 4096;
    CHECK(wnr4_i4_geo(&t, 2, rows, cols, 0), "record of 2^32 bytes or more accepted");
    printf("geometry refusals done\n");
}

/* other geometries, short and misaligned records, bad bases, a longer last tile, and the encoder's refusals */
static void record_shape(const Wnr4I4Model *m) {
    static const uint32_t rows0[2] = {130, 70}, cols0[2] = {192, 320};
    uint32_t rows[2], cols[2], owner = 0x00010007u, err, d;
    Wnr4I4Geo g, h;
    Wnr4I4Record r;
    size_t cap, n;
    unsigned char *src, *out, *rec, *small;
    if (wnr4_i4_geo(&g, 2, rows0, cols0, 1)) { CHECK(0, "shape geometry"); return; }
    cap = (size_t)g.raw * 4u + 65536u;
    src = malloc(g.raw); out = malloc(g.raw + 65536u); rec = calloc(cap + 4097u, 1);
    make_record(&g, src, 0, 0);
    n = wnr4_i4_encode(m, &g, src, owner, 1, rec + 1, cap);
    CHECK(n > 4096u, "shape record: encode (%u bytes)", (unsigned)n);
    if (n <= 4096u) { free(src); free(out); free(rec); return; }
    err = wnr4_i4_decode(m, &g, rec + 1, n, owner, out, 2);
    CHECK(!err && !memcmp(out, src, g.raw), "record at an odd address (err %u)", err);
    memmove(rec, rec + 1, n); rec[n] = 0;
#define OTHER_GEO(what, edit, pairs_) do {                                                                \
        memcpy(rows, rows0, sizeof(rows)); memcpy(cols, cols0, sizeof(cols)); edit;                         \
        if (wnr4_i4_geo(&h, 2, rows, cols, pairs_)) { CHECK(0, "geometry %s", what); break; }               \
        CHECK(wnr4_i4_decode(m, &h, rec, n, owner, out, 1) != 0, "decoded as %s", what);                    \
    } while (0)
    OTHER_GEO("planar packing", (void)0, 0);
    OTHER_GEO("one row more", rows[1]++, 1);
    OTHER_GEO("one row less", rows[0]--, 1);
    OTHER_GEO("rows and cols swapped", (rows[0] = 390, cols[0] = 64), 1);
    OTHER_GEO("one group per row more", cols[1] += 64, 1);
#undef OTHER_GEO
    CHECK(wnr4_i4_geo(&h, 1, rows0, cols0, 1) || wnr4_i4_decode(m, &h, rec, n, owner, out, 1), "decoded as one matrix");
    CHECK(wnr4_i4_decode(m, &g, rec, n - 4096u, owner, out, 1), "record one page short accepted");
    CHECK(wnr4_i4_decode(m, &g, rec, 63, owner, out, 1), "record of 63 bytes accepted");
    CHECK(wnr4_i4_decode(m, &g, NULL, n, owner, out, 1), "no record accepted");
    CHECK(wnr4_i4_decode(m, &g, rec, n - 1u, owner, out, 1), "record one byte short accepted");
    CHECK(wnr4_i4_decode(m, &g, rec, n + 1u, owner, out, 1), "record one byte long accepted");
    memset(rec + n, 0, 4096);
    CHECK(wnr4_i4_decode(m, &g, rec, n + 4096u, owner, out, 1), "record with an extra zero page accepted");
    if (wnr4_i4_parse(rec, n, &g, &r)) { CHECK(0, "shape record: parse"); }
    else {
        static const uint32_t badb[4] = {0x7f800000u, 0x7fc00000u, 0x80000000u, 0x00000001u};
        unsigned char *b0 = (unsigned char *)r.bases, keep[4];
        memcpy(keep, b0, 4);
        for (d = 0; d < 4u; d++) {   /* infinity, NaN, negative zero, a subnormal: refused as bases */
            wnr4_put32(b0, badb[d]);
            CHECK(wnr4_i4_decode(m, &g, rec, n, owner, out, 1), "base %08x accepted", badb[d]);
        }
        memcpy(b0, keep, 4);
        /* the last tile 1 or 2 bytes longer, taken from the zero padding: refused */
        if ((size_t)(r.data - rec) + r.payload + 2u > n) CHECK(0, "shape record: no room to lengthen its last tile");
        else for (d = 1; d <= 2u; d++) {
            unsigned char *last = rec + WNR4_HEADER_BYTES + 4u * r.ntiles;
            wnr4_put32(rec + 24, r.payload + d); wnr4_put32(last, r.payload + d);
            CHECK(wnr4_i4_decode(m, &g, rec, n, owner, out, 1), "last tile %u byte(s) longer accepted", d);
            wnr4_put32(rec + 24, r.payload); wnr4_put32(last, r.payload);
        }
        CHECK(!wnr4_i4_decode(m, &g, rec, n, owner, out, 1) && !memcmp(out, src, g.raw), "shape record restored");
    }
    small = malloc(n - 1u);   /* exactly n - 1 bytes: a sanitizer build checks that the encoder stays inside them */
    CHECK(small && !wnr4_i4_encode(m, &g, src, owner, 1, small, n - 1u), "record encoded into %u bytes, one too few", (unsigned)(n - 1u));
    CHECK(!wnr4_i4_encode(m, &g, src, owner, 0, rec, cap), "record encoded at G=0");
    CHECK(!wnr4_i4_encode(m, &g, src, owner, WNR4_MAX_G + 1u, rec, cap), "record encoded at G=%u", WNR4_MAX_G + 1u);
    printf("record shape: other geometries, short and misaligned records, bad bases, longer last tiles and encoder"
           " refusals done\n");
    free(src); free(out); free(rec); free(small);
}

/* the lattice: E4M3 codes, a value from the converter's arithmetic, monotone in k */
static void lattice(void) {
    uint32_t k, B = 0x3b9a5c3du, prev = 0;
    CHECK(wnr4_i4_e4m3(1) == 0x3b000000u && wnr4_i4_e4m3(7) == 0x3c600000u && wnr4_i4_e4m3(8) == 0x3c800000u &&
          wnr4_i4_e4m3(126) == 0x43e00000u, "E4M3 codes");
    for (k = 0; k < 127u; k++) {
        float want = (float)((float)((double)wnr4_i4_f(wnr4_i4_e4m3(k)) * (double)wnr4_i4_f(B)) / 7.0f);
        uint32_t v = wnr4_i4_lat(k, B);
        CHECK(v == wnr4_i4_b(want) && v >= prev && wnr4_i4_find_k(B, v) <= k, "lattice value %u", k);
        prev = v;
    }
    CHECK(wnr4_i4_lat(126, 0x7f000000u) == 0x7f800000u, "lattice overflow to infinity");
    CHECK(wnr4_i4_find_k(B, 0x80000000u) == WNR4_I4_ESC && wnr4_i4_find_k(0, 0) == 0 && wnr4_i4_find_k(0, 1) == WNR4_I4_ESC,
          "escapes");
    printf("lattice: E4M3 codes, 127 values, overflow and escapes checked\n");
}

static void known_answers(const Wnr4I4Model *m) {
    static const uint32_t rows[3] = {33, 7, 129}, cols[3] = {128, 2048, 64}, Gs[2] = {1, 32};
    static const uint32_t want[2] = {0x15079cc5u, 0xa79459acu}, want_id = 0x98ed56afu;
    Wnr4I4Geo g;
    unsigned char *src, *rec;
    size_t n, cap;
    int i;
    CHECK(m->id == want_id, "model id %08x, expected %08x", m->id, want_id);
    if (wnr4_i4_geo(&g, 3, rows, cols, 0)) { CHECK(0, "known-answer geometry"); return; }
    cap = (size_t)g.raw * 4u + 65536u;
    src = malloc(g.raw); rec = malloc(cap);
    make_record(&g, src, 0, 0);
    for (i = 0; i < 2; i++) {
        n = wnr4_i4_encode(m, &g, src, 0x0003002Au, Gs[i], rec, cap);
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
        {"single-group", 1, {1}, {64}, 0, 0},
        {"single-group-hostile", 1, {1}, {64}, 1, 1},
        {"gc1-pairs", 1, {300}, {64}, 1, 0},
        {"gc3-planar", 1, {130}, {192}, 0, 0},
        {"two-mats-pairs", 2, {129, 70}, {192, 320}, 1, 0},
        {"q38-small", 3, {192, 192, 640}, {640, 640, 192}, 0, 0},
        {"gc32-gc48", 3, {40, 3, 9}, {2048, 3072, 64}, 0, 0},
        {"hostile", 3, {50, 50, 130}, {256, 256, 128}, 0, 1},
        {"hostile-pairs", 2, {131, 2}, {448, 64}, 1, 1},
        {"all-zero", 3, {64, 64, 128}, {256, 256, 128}, 0, 2},
        {"q38-like-rows", 3, {256, 256, 1024}, {1024, 1024, 256}, 0, 0},
    };
    unsigned char *blob = malloc(WNR4_I4_MODEL_BYTES), *blob2 = malloc(WNR4_I4_MODEL_BYTES);
    Wnr4I4Model model, other;
    size_t i;
    train(0, blob, &model);
    train(5, blob2, &other);
    CHECK(other.id != model.id, "the two models have the same id");
    known_answers(&model);
    lattice();
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) roundtrip(&cases[i], &model, &other, Gs, 7, fuzz);
    model_blobs(blob);
    geometry();
    record_shape(&model);
    wnr4_i4_model_free(&model); wnr4_i4_model_free(&other);
    free(blob); free(blob2);
    printf(failures ? "FAILED: %d\n" : "ALL OK\n", failures);
    return failures != 0;
}
