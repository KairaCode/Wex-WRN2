/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Ludovico Zanichelli */
/* ctypes API of wnr4_i4.h (WNR4 records of int4-g64 experts). A geometry is (nmat, rows[], cols[], pairs).
 *   gcc -O3 -march=haswell -fopenmp -static -shared -o wnr4_i4_tools.dll wnr4/tools/wnr4_i4_tools.c */
#include "../wnr4_i4.h"
#include <stdlib.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API uint32_t wnr4i4_abi(void) { return 1u; }
API uint64_t wnr4i4_counts_words(void) { return wnr4_i4_counts_words(); }
API uint32_t wnr4i4_model_bytes(void) { return WNR4_I4_MODEL_BYTES; }
/* decoded record bytes of a geometry, or 0 when it is refused */
API uint64_t wnr4i4_raw_bytes(uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs) {
    Wnr4I4Geo g;
    return wnr4_i4_geo(&g, nmat, rows, cols, pairs) ? 0u : g.raw;
}
/* 0, or -1 (geometry refused or out of memory) */
API int wnr4i4_count(uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs, const unsigned char *slab,
                     uint64_t *counts) {
    Wnr4I4Geo g;
    if (wnr4_i4_geo(&g, nmat, rows, cols, pairs)) return -1;
    return wnr4_i4_count(&g, slab, counts);
}
API void wnr4i4_build(const uint64_t *counts, unsigned char *blob) { wnr4_i4_model_build(counts, blob); }

API void *wnr4i4_model_open(const unsigned char *blob, uint64_t bytes) {
    Wnr4I4Model *m = (Wnr4I4Model *)calloc(1, sizeof(*m));
    if (!m || wnr4_i4_model_open(blob, (size_t)bytes, m)) { free(m); return NULL; }
    return m;
}
API void wnr4i4_model_close(void *m) { if (m) { wnr4_i4_model_free((Wnr4I4Model *)m); free(m); } }
API uint32_t wnr4i4_model_id(const void *m) { return ((const Wnr4I4Model *)m)->id; }

/* returns the record bytes or 0 */
API uint64_t wnr4i4_encode(const void *m, uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs,
                           const unsigned char *slab, uint32_t owner, uint32_t G, unsigned char *rec, uint64_t cap) {
    Wnr4I4Geo g;
    if (!m || wnr4_i4_geo(&g, nmat, rows, cols, pairs)) return 0;
    return wnr4_i4_encode((const Wnr4I4Model *)m, &g, slab, owner, G, rec, (size_t)cap);
}
/* returns 0 or the error bits of wnr4.h (geometry errors: WNR4_ERR_STRUCT) */
API uint32_t wnr4i4_decode(const void *m, uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs,
                           const unsigned char *rec, uint64_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4I4Geo g;
    if (!m || wnr4_i4_geo(&g, nmat, rows, cols, pairs)) return WNR4_ERR_STRUCT;
    return wnr4_i4_decode((const Wnr4I4Model *)m, &g, rec, (size_t)total, owner, out, threads);
}
