/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.h, c/deepseek_v4_internal.h and c/expert_store_registry.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
#ifndef WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H
#define WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only: what c/expert_store_wex.c uses of the engine's
 * deepseek_v4_internal.h and of the deepseek_v4.h and expert_store_registry.h it includes. The engine stays
 * opaque: the harness opens the store without an engine (engine = NULL), as c/tests/test_expert_store_wex.c
 * does.
 *
 * Checked against colibri dev 98327913f980514970173b6df7bf67210ad0dc16: every declaration below has the names,
 * types and order of the line it cites. Left out: the rest of those headers (the engine, its tensors, its
 * plans), which the store does not name, and the registry's backend_count, backend_lookup and
 * backend_open_selected (expert_store_registry.h:58-74), which it does not call. Until 2026-11-15 the
 * configuration was an incomplete struct here; the store only passes a pointer to it on. */
#include "expert_store.h"

/* c/deepseek_v4.h:13 and 15-53 */
#define COLI_V4_MAX_LAYERS 128

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int head_dim;
    int q_lora_rank;
    int qk_rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int sliding_window;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;
    int moe_intermediate_size;
    int num_hash_layers;
    int num_nextn_predict_layers;
    int dspark_block_size;
    int dspark_noise_token_id;
    int dspark_markov_rank;
    int hc_mult;
    int hc_sinkhorn_iters;
    int vocab_size;
    int max_position_embeddings;
    int original_max_position_embeddings;
    int compress_ratio_count;
    int compress_ratios[COLI_V4_MAX_LAYERS];
    float rms_norm_eps;
    float hc_eps;
    float routed_scaling_factor;
    float swiglu_limit;
    float rope_theta;
    float rope_factor;
    float compress_rope_theta;
    int rope_beta_fast;
    int rope_beta_slow;
} ColiDeepSeekV4Config;

/* c/deepseek_v4_internal.h:123 (the struct itself is at line 1278; the store never looks inside it) */
typedef struct ColiV4Engine ColiV4Engine;

/* c/deepseek_v4_internal.h:730-744 */
typedef struct ColiDeepSeekV4ExpertStoreOptions {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
    int pin_slots_per_layer;
    uint64_t repin_interval;
    int skip_mirror_setup;
    uint64_t keep_free_bytes;
} ColiDeepSeekV4ExpertStoreOptions;

/* c/expert_store_registry.h:42-47 */
typedef int (*ColiExpertStoreBackendOpenFn)(ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
                                            const ColiDeepSeekV4ExpertStoreOptions *options,
                                            ColiExpertStore **output, char *error, size_t error_size);

/* c/deepseek_v4_internal.h:1405-1411. In colibri it opens the RAM-tiered store of an engine; the harness
 * defines it to open nothing (the store calls it with an engine, or when the container cannot be opened on
 * Windows, and the harness passes no engine) */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *options,
                                      ColiExpertStore **store, char *error, size_t error_size);
/* c/expert_store_registry.h:55-56. In colibri it records the backend and returns 0, or -1 for a NULL argument
 * or a full registry; the harness defines it to record nothing and return 0 */
int coli_expert_store_backend_register(const char *name, ColiExpertStoreBackendOpenFn open_fn);

#endif /* WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H */
