/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Ludovico Zanichelli */
/* ctypes API of wnr4_gg.h (WNR4 records of GGUF expert slices). A geometry is (nseg, types[], bytes[]).
 *   gcc -O3 -march=haswell -fopenmp -static -shared -o wnr4_gg_tools.dll wnr4/tools/wnr4_gg_tools.c */
#include "../wnr4_gg.h"
#include <stdlib.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API uint32_t wnr4gg_abi(void) { return 1u; }
API uint64_t wnr4gg_counts_words(void) { return wnr4_gg_counts_words(); }
API uint32_t wnr4gg_model_bytes(void) { return WNR4_GG_MODEL_BYTES; }

API int wnr4gg_count(uint32_t nseg, const uint32_t *types, const uint32_t *bytes, const unsigned char *slab,
                     uint64_t *counts) {
    Wnr4GgGeo g;
    if (wnr4_gg_geo(&g, nseg, types, bytes)) return -1;
    wnr4_gg_count(&g, slab, counts);
    return 0;
}
/* 0, or -1 when out of memory */
API int wnr4gg_build(const uint64_t *counts, unsigned char *blob) { return wnr4_gg_model_build(counts, blob); }

API void *wnr4gg_model_open(const unsigned char *blob, uint64_t bytes) {
    Wnr4GgModel *m = (Wnr4GgModel *)calloc(1, sizeof(*m));
    if (!m || wnr4_gg_model_open(blob, (size_t)bytes, m)) { free(m); return NULL; }
    return m;
}
API void wnr4gg_model_close(void *m) { if (m) { wnr4_gg_model_free((Wnr4GgModel *)m); free(m); } }
API uint32_t wnr4gg_model_id(const void *m) { return ((const Wnr4GgModel *)m)->id; }

/* returns the record bytes or 0 */
API uint64_t wnr4gg_encode(const void *m, uint32_t nseg, const uint32_t *types, const uint32_t *bytes,
                           const unsigned char *slab, uint32_t owner, uint32_t G, unsigned char *rec, uint64_t cap) {
    Wnr4GgGeo g;
    if (!m || wnr4_gg_geo(&g, nseg, types, bytes)) return 0;
    return wnr4_gg_encode((const Wnr4GgModel *)m, &g, slab, owner, G, rec, (size_t)cap);
}
/* returns 0 or the error bits of wnr4.h (geometry errors: WNR4_ERR_STRUCT) */
API uint32_t wnr4gg_decode(const void *m, uint32_t nseg, const uint32_t *types, const uint32_t *bytes,
                           const unsigned char *rec, uint64_t total, uint32_t owner, unsigned char *out, int threads) {
    Wnr4GgGeo g;
    if (!m || wnr4_gg_geo(&g, nseg, types, bytes)) return WNR4_ERR_STRUCT;
    return wnr4_gg_decode((const Wnr4GgModel *)m, &g, rec, (size_t)total, owner, out, threads);
}
