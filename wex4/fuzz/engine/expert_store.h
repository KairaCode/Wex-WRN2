/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/expert_store.h and c/tensor.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
#ifndef WEX_FUZZ_ENGINE_EXPERT_STORE_H
#define WEX_FUZZ_ENGINE_EXPERT_STORE_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only. c/expert_store_wex.c includes the engine's expert_store.h,
 * which is not in this repository. This file declares what the store uses of it, with the engine's names and
 * types: the tensor view, the expert view, the statistics and the store's function table. It is not the
 * engine's header and it is not kept in step with it. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    COLI_TENSOR_F32 = 0,
    COLI_TENSOR_Q8_ROW,
    COLI_TENSOR_Q4_ROW,
    COLI_TENSOR_Q2_ROW,
    COLI_TENSOR_FP8_E4M3_BLOCK,
    COLI_TENSOR_FP4_NATIVE_BLOCK
} ColiTensorFormat;

typedef enum {
    COLI_SCALE_NONE = 0,
    COLI_SCALE_F32,
    COLI_SCALE_UE8M0
} ColiScaleFormat;

typedef struct {
    ColiTensorFormat format;
    ColiScaleFormat scale_format;
    const void *data;
    const void *scales;
    size_t data_bytes;
    size_t scale_bytes;
    int64_t rows;
    int64_t columns;
    uint32_t block_rows;
    uint32_t block_columns;
    void *gpu;
} ColiTensorView;

typedef struct ColiExpertStore ColiExpertStore;

typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

typedef struct {
    ColiExpertKey key;
    ColiTensorView gate;
    ColiTensorView down;
    ColiTensorView up;
    void *lease;
} ColiExpertView;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t bytes_read;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;
} ColiExpertStoreStats;

typedef struct {
    int (*lookup)(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view);
    void (*release)(ColiExpertStore *store, ColiExpertView *view);
    int (*prefetch)(ColiExpertStore *store, const ColiExpertKey *keys, size_t count);
    void (*stats)(const ColiExpertStore *store, ColiExpertStoreStats *stats);
    void (*destroy)(ColiExpertStore *store);
} ColiExpertStoreOps;

struct ColiExpertStore {
    const ColiExpertStoreOps *ops;
    void *state;
    void *gpu;
};

static inline int coli_expert_lookup(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    int result;
    if (!store || !store->ops || !store->ops->lookup) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    result = store->ops->lookup(store, key, view);
    if (result != 0 && view) memset(view, 0, sizeof(*view));
    return result;
}

static inline void coli_expert_release(ColiExpertStore *store, ColiExpertView *view) {
    if (store && store->ops && store->ops->release) store->ops->release(store, view);
    if (view) memset(view, 0, sizeof(*view));
}

#endif /* WEX_FUZZ_ENGINE_EXPERT_STORE_H */
