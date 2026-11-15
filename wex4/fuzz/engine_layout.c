/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Prints the declarations of the engine headers that the two engine harnesses use: every member's offset and
 * size, whether an integer member is signed, the size of each type and the value of each enumerator. run.sh
 * builds it twice when WEX_FUZZ_COLIBRI names a colibri tree, once against the stand-ins of engine/ and once
 * against the tree's c/ folder (with -DWEX_FUZZ_COLIBRI), and requires the two listings to be equal: a stand-in
 * that has another width, signedness, order or enumerator value than colibri's declaration fails the replay.
 *
 * What it compares: the types c/expert_store_wex.c takes from expert_store.h, tensor.h, deepseek_v4.h and
 * deepseek_v4_internal.h, and st_tensor, which c/qwen38_wex.h takes from st.h. Not compared: the function
 * prototypes (building the harness against the tree checks them), and the model types of engine/q38_engine.h,
 * which no public colibri header declares in the form c/qwen38_wex.h needs. gcc or clang (__typeof__). */
#define _FILE_OFFSET_BITS 64
#include <stddef.h>
#include <stdio.h>
#include "expert_store.h"
#include "deepseek_v4_internal.h"
#include "wex_format.h"
#include "engine/q38_engine.h"

#define TYPE(T) printf("sizeof %s %u\n", #T, (unsigned)sizeof(T))
#define MEMBER(T, m) printf("%s.%s offset %u size %u\n", #T, #m, (unsigned)offsetof(T, m), (unsigned)sizeof(((T *)0)->m))
#define INTEGER(T, m) \
    printf("%s.%s offset %u size %u %s\n", #T, #m, (unsigned)offsetof(T, m), (unsigned)sizeof(((T *)0)->m), \
           (double)(__typeof__(((T *)0)->m))-1 < 0.0 ? "signed" : "unsigned")
#define VALUE(e) printf("%s = %ld\n", #e, (long)(e))

int main(void) {
    VALUE(COLI_TENSOR_F32); VALUE(COLI_TENSOR_Q8_ROW); VALUE(COLI_TENSOR_Q4_ROW); VALUE(COLI_TENSOR_Q2_ROW);
    VALUE(COLI_TENSOR_FP8_E4M3_BLOCK); VALUE(COLI_TENSOR_FP4_NATIVE_BLOCK); VALUE(COLI_TENSOR_INT8_BLOCK);
    VALUE(COLI_TENSOR_INT4_BLOCK); VALUE(COLI_TENSOR_GGML);
    VALUE(COLI_SCALE_NONE); VALUE(COLI_SCALE_F32); VALUE(COLI_SCALE_UE8M0);

    TYPE(ColiTensorView);
    INTEGER(ColiTensorView, format); INTEGER(ColiTensorView, scale_format); MEMBER(ColiTensorView, data);
    MEMBER(ColiTensorView, scales); INTEGER(ColiTensorView, data_bytes); INTEGER(ColiTensorView, scale_bytes);
    INTEGER(ColiTensorView, rows); INTEGER(ColiTensorView, columns); INTEGER(ColiTensorView, block_rows);
    INTEGER(ColiTensorView, block_columns); MEMBER(ColiTensorView, gpu);

    TYPE(ColiExpertKey);
    INTEGER(ColiExpertKey, layer); INTEGER(ColiExpertKey, expert);

    TYPE(ColiExpertView);
    MEMBER(ColiExpertView, key); MEMBER(ColiExpertView, gate); MEMBER(ColiExpertView, down);
    MEMBER(ColiExpertView, up); MEMBER(ColiExpertView, lease);

    TYPE(ColiExpertStoreStats);
    INTEGER(ColiExpertStoreStats, requests); INTEGER(ColiExpertStoreStats, hits);
    INTEGER(ColiExpertStoreStats, misses); INTEGER(ColiExpertStoreStats, prefetched);
    INTEGER(ColiExpertStoreStats, prefetch_hits); INTEGER(ColiExpertStoreStats, bytes_read);
    INTEGER(ColiExpertStoreStats, resident_bytes); INTEGER(ColiExpertStoreStats, capacity_bytes);

    TYPE(ColiExpertStoreOps);
    MEMBER(ColiExpertStoreOps, lookup); MEMBER(ColiExpertStoreOps, release); MEMBER(ColiExpertStoreOps, prefetch);
    MEMBER(ColiExpertStoreOps, stats); MEMBER(ColiExpertStoreOps, destroy);

    TYPE(ColiExpertStore);
    MEMBER(ColiExpertStore, ops); MEMBER(ColiExpertStore, state); MEMBER(ColiExpertStore, gpu);

    TYPE(ColiDeepSeekV4ExpertStoreOptions);
    MEMBER(ColiDeepSeekV4ExpertStoreOptions, model_dir); INTEGER(ColiDeepSeekV4ExpertStoreOptions, layers);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, experts_per_layer);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, cache_bytes);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, pin_slots_per_layer);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, repin_interval);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, skip_mirror_setup);
    INTEGER(ColiDeepSeekV4ExpertStoreOptions, keep_free_bytes);

    TYPE(ColiDeepSeekV4Config);
    VALUE(COLI_V4_MAX_LAYERS);
    INTEGER(ColiDeepSeekV4Config, hidden_size); INTEGER(ColiDeepSeekV4Config, n_routed_experts);
    INTEGER(ColiDeepSeekV4Config, moe_intermediate_size); INTEGER(ColiDeepSeekV4Config, compress_ratio_count);
    MEMBER(ColiDeepSeekV4Config, compress_ratios); MEMBER(ColiDeepSeekV4Config, rms_norm_eps);
    MEMBER(ColiDeepSeekV4Config, compress_rope_theta); INTEGER(ColiDeepSeekV4Config, rope_beta_slow);

    TYPE(st_tensor);
    VALUE(ST_MAX_RANK);
    MEMBER(st_tensor, name); INTEGER(st_tensor, fd); INTEGER(st_tensor, off); INTEGER(st_tensor, nbytes);
    INTEGER(st_tensor, dtype); INTEGER(st_tensor, numel); INTEGER(st_tensor, rank); MEMBER(st_tensor, shape);
    return 0;
}
