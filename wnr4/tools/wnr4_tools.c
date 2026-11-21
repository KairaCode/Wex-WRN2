/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CPU side for the tools (ctypes): training counts, model blob, encoder,
 * reference decoder. geo = u32[13]: kind, scount[3], scols[3], soff[3], woff[3]. */
#include "../wnr4.h"

#ifdef _WIN32
#define WNR4_EXPORT __declspec(dllexport)
#else
#define WNR4_EXPORT __attribute__((visibility("default")))
#endif

static int geo_of(const uint32_t *v, Wnr4Geo *g) { return v ? wnr4_geo(g, v[0], v + 1, v + 4, v + 7, v + 10) : -1; }

WNR4_EXPORT uint32_t wnr4t_abi(void) { return 1u; }
/* pb = position buckets of the weight model: 1 or 5 */
WNR4_EXPORT size_t wnr4t_counts_words(uint32_t kind, uint32_t pb) { return kind > 1u ? 0 : wnr4_counts_words(kind, pb); }
WNR4_EXPORT size_t wnr4t_model_bytes(uint32_t kind, uint32_t pb) { return kind > 1u ? 0 : wnr4_model_bytes(kind, pb); }

/* adds the coded slab's counts (anchors: rule g of its coded scales) */
WNR4_EXPORT int wnr4t_count(const uint32_t *geo, const unsigned char *coded, uint32_t pb, uint64_t *counts) {
    Wnr4Geo g;
    unsigned char a[3];
    uint32_t m;
    if (geo_of(geo, &g) || !coded || !counts || (pb != 1u && pb != 5u)) return -1;
    for (m = 0; m < 3u; m++) a[m] = wnr4_anchor(coded + g.soff[m], g.scount[m]);
    wnr4_count(&g, coded, a, pb, counts);
    return 0;
}
WNR4_EXPORT uint32_t wnr4t_build(uint32_t kind, uint32_t pb, const uint64_t *counts, unsigned char *blob) {
    Wnr4Model m;
    uint32_t id;
    if (kind > 1u || !counts || !blob || (pb != 1u && pb != 5u)) return 0;
    wnr4_model_build(kind, pb, counts, blob);
    if (wnr4_model_open(blob, wnr4_model_bytes(kind, pb), &m)) return 0;
    id = m.id;
    wnr4_model_free(&m);
    return id;
}
WNR4_EXPORT void *wnr4t_model_open(const unsigned char *blob, size_t bytes) {
    Wnr4Model *m = (Wnr4Model *)malloc(sizeof(Wnr4Model));
    if (!m || wnr4_model_open(blob, bytes, m)) { free(m); return NULL; }
    return m;
}
WNR4_EXPORT void wnr4t_model_close(void *p) { if (p) { wnr4_model_free((Wnr4Model *)p); free(p); } }
WNR4_EXPORT uint32_t wnr4t_model_id(const void *p) { return p ? ((const Wnr4Model *)p)->id : 0u; }

/* record bytes, or 0 on error */
WNR4_EXPORT size_t wnr4t_encode(const void *model, const uint32_t *geo, const unsigned char *coded, uint32_t flags,
                                uint32_t owner, uint32_t G, unsigned char *out, size_t cap) {
    Wnr4Geo g;
    if (!model || geo_of(geo, &g) || !coded || !out) return 0;
    return wnr4_encode((const Wnr4Model *)model, &g, coded, flags, owner, G, out, cap);
}
/* 0 or error bits; out (g.raw bytes) is written only on success (decode goes through scratch) */
WNR4_EXPORT uint32_t wnr4t_decode(const void *model, const uint32_t *geo, const unsigned char *rec, size_t total,
                                  uint32_t owner, unsigned char *out, int threads) {
    Wnr4Geo g;
    unsigned char *tmp;
    uint32_t err;
    if (!model || geo_of(geo, &g) || !rec || !out) return WNR4_ERR_STRUCT;
    tmp = (unsigned char *)calloc(1, (size_t)g.raw);
    if (!tmp) return WNR4_ERR_STRUCT;
    err = wnr4_decode((const Wnr4Model *)model, &g, rec, total, owner, tmp, threads);
    if (!err) memcpy(out, tmp, (size_t)g.raw);
    free(tmp);
    return err;
}
/* decode straight into out (no scratch), for timing */
WNR4_EXPORT uint32_t wnr4t_decode_direct(const void *model, const uint32_t *geo, const unsigned char *rec,
                                         size_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4Geo g;
    if (!model || geo_of(geo, &g) || !rec || !out) return WNR4_ERR_STRUCT;
    return wnr4_decode((const Wnr4Model *)model, &g, rec, total, owner, out, threads);
}
/* n records of one geometry decoded in one parallel loop over all their tiles, straight into outs[i]
 * (partial data when status[i] != 0). Returns the number of failed records. */
WNR4_EXPORT int wnr4t_decode_batch(const void *model, const uint32_t *geo, int n, const unsigned char *const *recs,
                                   const size_t *totals, const uint32_t *owners, unsigned char *const *outs,
                                   uint32_t *status, int threads) {
    Wnr4Geo g;
    if (!model || geo_of(geo, &g) || n < 0 || (n && (!recs || !totals || !owners || !outs || !status))) return n;
    return wnr4_decode_batch((const Wnr4Model *)model, &g, n, recs, totals, owners, outs, status, threads);
}
