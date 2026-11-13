/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* security-c review harness (not part of the product). Built into out/rv_harness.dll with
 *   gcc -O3 -march=haswell -fopenmp -static -shared -include wex_fast_decode.h -I c -I wex4 -I wex4/research/fast-cpu-decode
 * 1. rv_bp_guard: wex4_bp_decode with the record and the output both ending at an inaccessible page.
 * 2. rv_find: wex4_find on a validated view.
 * 3. rv_nr3_campaign: random valid WNR3 records (random shapes, chunk sizes, contents, real or random valid
 *    model), decoded by the reference decoder and by wex_fast_decode in every kernel/lane/fuse configuration,
 *    then byte mutations of each record. The record is decoded from a copy that ends exactly at the end of
 *    its weight payload, against a guard page; outputs are exactly `raw` bytes against a guard page. */
#include "wex4.h"
#include <windows.h>
#define RV_EXPORT __declspec(dllexport)

static unsigned char *galloc(size_t n) {
    size_t body = (n + 4095u) & ~(size_t)4095u;
    unsigned char *base = (unsigned char *)VirtualAlloc(NULL, body + 4096u, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    DWORD old;
    if (!base) return NULL;
    VirtualProtect(base + body, 4096u, PAGE_NOACCESS, &old);
    return base + (body - n);
}
static void gfree(unsigned char *p) { if (p) VirtualFree((void *)((uintptr_t)p & ~(uintptr_t)4095u), 0, MEM_RELEASE); }

RV_EXPORT int rv_bp_guard(const unsigned char *rec, size_t total, uint32_t decoded, uint32_t owner, unsigned char *copy_out) {
    unsigned char *r = galloc(total), *o = galloc(decoded);
    int st;
    if (!r || !o) return -100;
    memcpy(r, rec, total);
    st = wex4_bp_decode(r, total, decoded, owner, o);
    if (!st && copy_out) memcpy(copy_out, o, decoded);
    gfree(r); gfree(o);
    return st;
}

RV_EXPORT int64_t rv_find(const unsigned char *page, uint64_t file_bytes, const unsigned char *index, size_t index_len,
                          const unsigned char *name, size_t len) {
    Wex4Header h;
    Wex4View v;
    if (wex4_parse_header(page, file_bytes, &h) || wex4_open(&h, index, index_len, &v)) return -2;
    return wex4_find(&v, name, len);
}

static uint64_t rnd(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * UINT64_C(2685821657736338717); }

static void random_cdf(uint64_t *s, unsigned char *cdf, uint32_t symbols) {
    uint32_t f[256], i, left = WEX_NR2_TOTAL - symbols, at = 0, style = (uint32_t)(rnd(s) % 4u);
    for (i = 0; i < symbols; i++) f[i] = 1;
    while (left) {
        uint32_t k = (uint32_t)(rnd(s) % symbols), lump;
        if (style == 0) lump = left;                                  /* one symbol takes everything */
        else if (style == 1) lump = 1u + (uint32_t)(rnd(s) % 8u);     /* near flat */
        else lump = 1u + (uint32_t)(rnd(s) % (left < 4000u ? left : 4000u));
        if (lump > left) lump = left;
        f[k] += lump; left -= lump;
    }
    for (i = 0; i <= symbols; i++) {
        cdf[i * 2u] = (unsigned char)at; cdf[i * 2u + 1u] = (unsigned char)(at >> 8);
        if (i < symbols) at += f[i];
    }
}
static void random_model(uint64_t *s, unsigned char *blob) {
    const uint32_t head[8] = {WEX_NR2_MODEL_MAGIC, 1u, WEX_NR2_PRECISION, WEX_NR2_CONTEXTS, 3u, 0u, 0u, 0u};
    uint32_t i;
    for (i = 0; i < 32u; i++) blob[i] = (unsigned char)(head[i / 4u] >> ((i % 4u) * 8u));
    for (i = 0; i < 3u * WEX_NR2_CONTEXTS; i++) random_cdf(s, blob + 32u + (size_t)i * 34u, 16u);
    for (i = 0; i < 3u; i++) random_cdf(s, blob + 32u + WEX_NR2_WCDF_BYTES + (size_t)i * 514u, 256u);
}

static const WexFastOptions rv_opts[] = {
    {1, 1, 4, 1}, {1, 0, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}, {1, 1, 2, 0}, {1, 1, 2, 1}, {1, 1, 3, 0}, {1, 1, 3, 1},
    {1, 1, 4, 0}, {1, 1, 6, 0}, {1, 1, 6, 1}, {1, 1, 8, 0}, {1, 1, 8, 1}, {1, 2, 1, 0}, {1, 2, 1, 1}, {1, 2, 2, 0},
    {1, 2, 2, 1}, {1, 2, 3, 0}, {1, 2, 3, 1}};
#define RV_NOPTS (sizeof(rv_opts) / sizeof(rv_opts[0]))

/* counts: [0] cases, [1] encoder/reference round-trip failures, [2] fast != reference on a VALID record (status or
 * bytes), [3] mutations, [4] mutations refused by the parser, [5] refused by both decoders, [6] fast and reference
 * disagree on a mutated record (status, or bytes when both accept), [7] mutated record accepted with bytes that
 * differ from the original (a level-1 hash collision: expected about mutations * 2^-23), [8] mutated record
 * accepted with the original bytes, [9] the case and [10] the mutation of the first disagreement (+1),
 * [11] option index of it, [12..75] offsets of the edits counted in [8]. threads: OpenMP threads of the fast decoder (1 = faults are catchable by the caller). */
RV_EXPORT int rv_nr3_campaign(const unsigned char *real_model, uint64_t seed, uint32_t cases, uint32_t mutations,
                              int threads, uint32_t max_scales, uint64_t *counts) {
    uint64_t s = seed * UINT64_C(0x9e3779b97f4a7c15) + 1u;
    unsigned char *model_blob = (unsigned char *)malloc(WEX_NR2_MODEL_BYTES);
    WexFastModel *tables = (WexFastModel *)malloc(sizeof(*tables));
    uint32_t c;
    if (!model_blob || !tables || !counts) return -1;
    memset(counts, 0, 76u * sizeof(*counts));
    for (c = 0; c < cases; c++) {
        WexNr2Model model;
        WexNr2Layout layout;
        WexNr2Record view, gview;
        uint32_t m, i, flags, cz = (uint32_t)(rnd(&s) & 1u), wchunk = 512u << (rnd(&s) % 6u), schunk = 256u << (rnd(&s) % 7u);
        uint32_t sc[3], streams = 0, mu, mutations_left_guard = 0;
        size_t raw = 0, cap, written = 0, total, at;
        unsigned char *orig, *coded, *rec, *recg, *pristine, *oref, *ofast, *sp[3], *wp[3], *rs[3], *rw[3], *fs[3], *fw[3];
        if (real_model && (rnd(&s) & 1u)) memcpy(model_blob, real_model, WEX_NR2_MODEL_BYTES);
        else random_model(&s, model_blob);
        if (wex_nr2_model_open(model_blob, WEX_NR2_MODEL_BYTES, &model)) return -2;
        wex_fast_model_init(tables, &model);
        for (m = 0; m < 3u; m++) {
            uint32_t big = (uint32_t)(rnd(&s) % 8u) == 0u;
            sc[m] = 1u + (uint32_t)(rnd(&s) % (big ? max_scales : 96u));
            layout.scale_count[m] = sc[m]; layout.symbol_count[m] = sc[m] * 32u;
            raw += (size_t)sc[m] * 17u;
            streams += wex_nr2_streams(sc[m], schunk) + wex_nr2_streams(sc[m] * 32u, wchunk);
        }
        orig = (unsigned char *)malloc(raw); coded = (unsigned char *)malloc(raw);
        cap = raw * 4u + 64u + (size_t)streams * 8u;
        rec = (unsigned char *)calloc(1, cap + 8192u);
        oref = galloc(raw); ofast = galloc(raw);
        if (!orig || !coded || !rec || !oref || !ofast) return -3;
        for (m = 0, at = 0; m < 3u; m++) {   /* canonical slab: scales w1 w2 w3, weights w1 w2 w3 */
            sp[m] = coded + at; rs[m] = oref + at; fs[m] = ofast + at;
            {
                uint32_t style = (uint32_t)(rnd(&s) % 5u), center = (uint32_t)(rnd(&s) % 256u);
                for (i = 0; i < sc[m]; i++) {
                    int v = style == 0 ? (int)center + (int)(rnd(&s) % 7u) - 3 : style == 1 ? (int)(rnd(&s) % 256u)
                          : style == 2 ? 0 : style == 3 ? 255 : (int)(254u + rnd(&s) % 2u) - (int)(rnd(&s) % 16u == 0u) * 254;
                    orig[at + i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
            at += sc[m];
        }
        for (m = 0; m < 3u; m++) {
            uint32_t allow5 = (uint32_t)(rnd(&s) % 3u) == 0u, b, k;
            wp[m] = coded + at; rw[m] = oref + at; fw[m] = ofast + at;
            for (b = 0; b < sc[m]; b++) {
                uint32_t style = (uint32_t)(rnd(&s) % 5u), signs = (uint32_t)(rnd(&s) % 3u);
                static const unsigned char even[6] = {0, 2, 4, 5, 6, 7}, even5[5] = {0, 2, 4, 6, 7};
                if (style == 3u && !allow5) style = 2u;
                for (k = 0; k < 32u; k++) {
                    uint32_t a = style == 0 ? (uint32_t)(rnd(&s) % 8u) : style == 1 ? (uint32_t)(rnd(&s) % 2u)
                               : style == 2 ? (allow5 ? even[rnd(&s) % 6u] : even5[rnd(&s) % 5u])
                               : style == 3 ? (uint32_t)(rnd(&s) % 6u) : 0u;
                    uint32_t x;
                    if (style == 0 && !allow5 && a == 5u) a = 4u;
                    if (style == 2 && k == 7u) a = 7u;
                    if (style == 3 && k == 9u) a = 5u;
                    x = a | (signs == 0 ? 0u : signs == 1 ? (uint32_t)(rnd(&s) & 8u) : 8u);
                    if (!(k & 1u)) orig[at + (size_t)b * 16u + k / 2u] = (unsigned char)x;
                    else orig[at + (size_t)b * 16u + k / 2u] |= (unsigned char)(x << 4);
                }
            }
            at += (size_t)sc[m] * 16u;
        }
        memcpy(coded, orig, raw);
        if (cz) {
            wex_nr3_canonical_zero(wp[0], (size_t)(sc[0] + sc[1] + sc[2]) * 16u);
            wex_nr3_canonical_zero(orig + (wp[0] - coded), (size_t)(sc[0] + sc[1] + sc[2]) * 16u);   /* expected = Z(source) */
        }
        flags = wex_nr3_ceil_forward(&layout, sp, wp, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
        counts[0]++;
        if (wex_nr3_encode(&model, &layout, (const unsigned char *const *)sp, (const unsigned char *const *)wp, wchunk, schunk,
                           flags, (uint32_t)rnd(&s), rec, cap, &written)) { counts[1]++; goto next; }
        total = (written + 4095u) & ~(size_t)4095u;
        if (wex_nrx_parse(rec, total, &layout, model.id, WEX_NR3_MAGIC, &view)) { counts[1]++; goto next; }
        recg = galloc(written);   /* ends exactly at the end of the weight payload */
        pristine = (unsigned char *)malloc(written);
        if (!recg || !pristine) return -3;
        memcpy(recg, rec, written); memcpy(pristine, rec, written);
        gview = view; gview.lengths = recg + WEX_NR2_HEADER_BYTES; gview.scale_payload = recg + view.metadata_bytes;
        gview.weight_payload = gview.scale_payload + view.scale_payload_bytes;
        if (wex_nr2_decode(&gview, &layout, &model, rs, rw) || memcmp(oref, orig, raw)) { counts[1]++; gfree(recg); free(pristine); goto next; }
        for (i = 0; i < RV_NOPTS; i++) {
            WexFastOptions o = rv_opts[i];
            o.threads = i == 0 ? threads : 1;
            memset(ofast, 0xa5, raw);
            if (wex_fast_decode_with(i == 1 ? NULL : tables, &gview, &layout, &model, fs, fw, &o) || memcmp(ofast, orig, raw)) {
                counts[2]++;
                if (!counts[9]) { counts[9] = c + 1u; counts[11] = i; }
            }
        }
        for (mu = 0; mu < mutations; mu++) {
            unsigned char saved[3];
            size_t where[3];
            uint32_t edits = 1u + (uint32_t)(rnd(&s) % 3u), e, oi = mu & 1u ? (uint32_t)(rnd(&s) % RV_NOPTS) : 0u;
            int a, b;
            WexFastOptions o = rv_opts[oi];
            o.threads = oi == 0 ? threads : 1;
            for (e = 0; e < edits; e++) {
                uint64_t x = rnd(&s);
                /* half of the edits in the header and the length table, the rest anywhere in the record */
                where[e] = (size_t)(rnd(&s) % ((x & 1u) ? 32u + 2u * (size_t)streams : written));
                saved[e] = rec[where[e]];
                rec[where[e]] = (x >> 1 & 3u) == 0 ? (unsigned char)(rec[where[e]] ^ (1u << (x >> 8 & 7u)))
                              : (x >> 1 & 3u) == 1 ? (unsigned char)(x >> 8) : (unsigned char)(rec[where[e]] + ((x >> 8 & 1u) ? 1u : 255u));
            }
            if (!memcmp(rec, pristine, written)) { mu--; mutations_left_guard++; if (mutations_left_guard > 100000u) break; goto restore; }   /* a no-op edit */
            counts[3]++;
            if (wex_nrx_parse(rec, total, &layout, model.id, WEX_NR3_MAGIC, &view)) counts[4]++;
            else {
                size_t disk = (size_t)view.metadata_bytes + view.scale_payload_bytes + view.weight_payload_bytes;
                unsigned char *g = disk == written ? recg : galloc(disk);   /* the copy always ends at the payload end */
                if (!g) return -3;
                memcpy(g, rec, disk);
                gview = view; gview.lengths = g + WEX_NR2_HEADER_BYTES; gview.scale_payload = g + view.metadata_bytes;
                gview.weight_payload = gview.scale_payload + view.scale_payload_bytes;
                a = wex_nr2_decode(&gview, &layout, &model, rs, rw);
                b = wex_fast_decode_with(oi == 1 ? NULL : tables, &gview, &layout, &model, fs, fw, &o);
                if (g != recg) gfree(g);
                if ((a != 0) != (b != 0) || (!a && memcmp(oref, ofast, raw))) {
                    counts[6]++;
                    if (!counts[9]) { counts[9] = c + 1u; counts[10] = mu + 1u; counts[11] = oi; }
                } else if (a) counts[5]++;
                else if (memcmp(oref, orig, raw)) counts[7]++;
                else {   /* changed bytes, same output: histogram of the edited offsets ([12 + offset], 63 = beyond the header) */
                    counts[8]++;
                    for (e = 0; e < edits; e++) if (rec[where[e]] != pristine[where[e]]) counts[12u + (where[e] < 63u ? where[e] : 63u)]++;
                }
            }
restore:
            for (e = edits; e-- > 0;) rec[where[e]] = saved[e];
        }
        gfree(recg); free(pristine);
next:
        free(orig); free(coded); free(rec); gfree(oref); gfree(ofast);
    }
    free(model_blob); free(tables);
    return 0;
}

/* wex4_bp_parse alone: plane sizes the decoder would use (n[8]); 0 or -1. */
RV_EXPORT int rv_bp_parse(const unsigned char *rec, size_t total, uint32_t decoded, uint32_t owner, uint32_t *n) {
    Wex4Bp b;
    uint32_t p;
    int st = wex4_bp_parse(rec, total, decoded, owner, &b);
    for (p = 0; p < 8u; p++) n[p] = !st && p < b.planes ? b.n[p] : 0u;
    return st;
}
