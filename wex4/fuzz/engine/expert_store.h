/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/expert_store.h and c/tensor.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
#ifndef WEX_FUZZ_ENGINE_EXPERT_STORE_H
#define WEX_FUZZ_ENGINE_EXPERT_STORE_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only. c/expert_store_wex.c includes the engine's expert_store.h,
 * which is not in this repository. This file declares what the store uses of it and of the tensor.h it
 * includes: the tensor view, the expert view, the statistics and the store's function table.
 *
 * Checked against colibri dev 98327913f980514970173b6df7bf67210ad0dc16 (c/tensor.h, c/expert_store.h): every
 * declaration below has the names, types and order of the line it cites. What colibri's files have besides is
 * left out: their extern "C" guards and comments. `WEX_FUZZ_COLIBRI=fetch bash wex4/fuzz/run.sh --replay`
 * builds the harness against those files instead and compares the layouts (engine_layout.c). */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* c/tensor.h:13-26. The last three were missing here until 2026-11-15; the store names only
 * COLI_TENSOR_FP4_NATIVE_BLOCK, whose value (5) did not change */
typedef enum {
    COLI_TENSOR_F32 = 0,
    COLI_TENSOR_Q8_ROW,
    COLI_TENSOR_Q4_ROW,
    COLI_TENSOR_Q2_ROW,
    COLI_TENSOR_FP8_E4M3_BLOCK,
    COLI_TENSOR_FP4_NATIVE_BLOCK,
    COLI_TENSOR_INT8_BLOCK,
    COLI_TENSOR_INT4_BLOCK,
    COLI_TENSOR_GGML
} ColiTensorFormat;

/* c/tensor.h:28-32 */
typedef enum {
    COLI_SCALE_NONE = 0,
    COLI_SCALE_F32,
    COLI_SCALE_UE8M0
} ColiScaleFormat;

/* c/tensor.h:34-49 */
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

/* c/expert_store.h:14 */
typedef struct ColiExpertStore ColiExpertStore;

/* c/expert_store.h:16-19 */
typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

/* c/expert_store.h:21-27 */
typedef struct {
    ColiExpertKey key;
    ColiTensorView gate;
    ColiTensorView down;
    ColiTensorView up;
    void *lease;
} ColiExpertView;

/* c/expert_store.h:29-38 */
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

/* c/expert_store.h:59-69 (the lease contract is the comment at lines 40-58) */
typedef struct {
    int (*lookup)(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view);
    void (*release)(ColiExpertStore *store, ColiExpertView *view);
    int (*prefetch)(ColiExpertStore *store, const ColiExpertKey *keys, size_t count);
    void (*stats)(const ColiExpertStore *store, ColiExpertStoreStats *stats);
    void (*destroy)(ColiExpertStore *store);
} ColiExpertStoreOps;

/* c/expert_store.h:71-78 */
struct ColiExpertStore {
    const ColiExpertStoreOps *ops;
    void *state;
    void *gpu;
};

/* c/expert_store.h:80-90 */
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

/* c/expert_store.h:92-97 */
static inline void coli_expert_release(ColiExpertStore *store, ColiExpertView *view) {
    if (store && store->ops && store->ops->release) store->ops->release(store, view);
    if (view) memset(view, 0, sizeof(*view));
}

#endif /* WEX_FUZZ_ENGINE_EXPERT_STORE_H */
