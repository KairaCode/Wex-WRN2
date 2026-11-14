/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4_internal.h and c/expert_store_registry.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
#ifndef WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H
#define WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only: what c/expert_store_wex.c uses of the engine's
 * deepseek_v4_internal.h and expert_store_registry.h, with the engine's names and types. The engine and its
 * configuration stay opaque: the harness opens the store without an engine (engine = NULL), as
 * c/tests/test_expert_store_wex.c does. Not the engine's header, and not kept in step with it. */
#include "expert_store.h"

typedef struct ColiV4Engine ColiV4Engine;
typedef struct ColiDeepSeekV4Config ColiDeepSeekV4Config;

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

typedef int (*ColiExpertStoreBackendOpenFn)(ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
                                            const ColiDeepSeekV4ExpertStoreOptions *options,
                                            ColiExpertStore **output, char *error, size_t error_size);

/* the harness defines both: the first never opens anything, the second records nothing */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *options,
                                      ColiExpertStore **store, char *error, size_t error_size);
int coli_expert_store_backend_register(const char *name, ColiExpertStoreBackendOpenFn open_fn);

#endif /* WEX_FUZZ_ENGINE_DEEPSEEK_V4_INTERNAL_H */
