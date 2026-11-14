/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the standalone expert store c/expert_store_wex.c: the input is a whole legacy container
 * (WEX1, WEX2, WEX3 with scale tiles, palette, WNR, WNR2 or WNR3 records).
 *
 * The store is built without the engine, the way c/tests/test_expert_store_wex.c opens it (engine = NULL), in
 * one translation unit with c/wex_decode.c. The engine headers it includes are not in this repository:
 * wex4/fuzz/engine/ holds stand-ins that declare what the store uses of them (see the files there). Every
 * line of c/expert_store_wex.c, c/wex_decode.c and c/wex_format.h is the real one.
 *
 * The store works on a path, so every input is written to one temporary file, named by COLI_WEX_PATH. The
 * store opens it and the harness looks up its experts (all of them up to FUZZ_MAX_LOOKUPS, else that many
 * spread over the table), looks the first hit up again, prefetches and closes.
 *
 * Besides what the sanitizers catch:
 *  - the store opens the file exactly when wex_validate_header and wex_validate_table accept the same bytes
 *    in memory and, for a WNR2 or WNR3 container, its model is where the header says and opens under the id
 *    the header names;
 *  - a lookup succeeds exactly when the record's bytes, copied from the input, pass the owner check and
 *    decode with wex_decode_record or wex_decode_nr2_record, and the three matrices the store hands out are
 *    those decoded bytes;
 *  - the first expert that loaded loads again, and when that second lookup is a cache hit no byte is read;
 *    the store's request, hit and miss counters add up to the lookups made.
 *
 * Limits of this harness, not of the store. A header may claim any aligned_expert_bytes up to 4 GiB, and the
 * store allocates 24 slabs of that size per layer when it opens: an input that would make it allocate more
 * than FUZZ_MAX_SLABS is not opened here. The model registry of c/wex_decode.c keeps 16 models for the life
 * of the process; the harness empties it before every input, so that an input gives the same verdict whatever
 * ran before it. The log lines of the store and of the decoder go nowhere. No mirror file, no CUDA helper.
 *
 * The custom mutator: one call in two, when the input is a container that opens and has WNR3 records, one
 * of them is mutated with wnr3_mutator.h (decoded, edited, encoded again, so its stream hashes are right) and
 * written back if it still takes the same pages. Every other call is libFuzzer's own mutation. */
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
static int fuzz_quiet(FILE *stream, const char *format, ...) { (void)stream; (void)format; return 0; }
#define fprintf fuzz_quiet
#include "../../c/expert_store_wex.c"
#include "../../c/wex_decode.c"
#undef fprintf
#define FUZZ_MUTATOR_MAX_LEN 1048576u
#define FUZZ_MUTATOR_ROUNDS 100
#include "fuzz_common.h"
#include "wnr3_mutator.h"

#define FUZZ_MAX_SLABS (UINT64_C(64) << 20)
#define FUZZ_MAX_LOOKUPS 64u

static char temp_path[1024];

int coli_v4_expert_store_open_planned(ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **store,
                                      char *error, size_t error_size) {
    (void)engine; (void)config; (void)options; (void)store; (void)error; (void)error_size;
    return -1;
}
int coli_expert_store_backend_register(const char *name, ColiExpertStoreBackendOpenFn open_fn) {
    (void)name; (void)open_fn;
    return 0;
}

/* what the run reached, printed at exit */
static unsigned long long count_inputs, count_opened, count_lookups, count_served, count_served_coded;

static void fuzz_cleanup(void) {
    remove(temp_path);
    fprintf(stderr, "fuzz_store_wex: %llu inputs, %llu opened by the store, %llu lookups, %llu records served, %llu of "
            "them decoded from a coded record, %llu%s different ones (structured mutations %s)\n", count_inputs,
            count_opened, count_lookups, count_served, count_served_coded, fuzz_distinct,
            fuzz_distinct >= FUZZ_SEEN_MAX ? " or more" : "", fuzz_structured() ? "on" : "off");
}

static void fuzz_setup(void) {
    fuzz_temp_path(temp_path, sizeof temp_path, "store.wex");
    fuzz_setenv("COLI_WEX_PATH", temp_path);
    fuzz_setenv("COLI_WEX_CUDA_LIB", "0");      /* the CPU decoders: no helper is looked for */
    fuzz_setenv("COLI_MODEL_MIRROR", "");
    fuzz_setenv("DSV4_CUDA_MOE_MASKED", "");
    fuzz_setenv("WEX_READ_RETRY", "");
    atexit(fuzz_cleanup);
}

/* the registry of c/wex_decode.c, emptied: its models are heap copies that nothing else holds */
static void fuzz_forget_models(void) {
    int i;
    for (i = 0; i < wex_nr2_model_count; i++) free((void *)wex_nr2_models[i].blob);
    wex_nr2_model_count = 0;
#ifdef _WIN32
    memset(wex_nr2_helper_ids, 0, sizeof wex_nr2_helper_ids);
#endif
}

/* 1 when the store must open these bytes. *model is the container's model when *has_model. */
static int fuzz_opens(const unsigned char *file, size_t size, WexNr2Model *model, int *has_model) {
    const WexHeader *h = (const WexHeader *)file;
    size_t table_bytes, aligned_table;
    *has_model = 0;
    if (size < WEX_ALIGN || wex_validate_header(h, size, 0, 0, &table_bytes, &aligned_table) ||
        wex_validate_table(h, (const WexEntry *)(file + h->table_offset), size)) return 0;
    if (h->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)) {
        uint64_t offset, aligned;
        uint32_t bytes, id;
        if (wex_nr2_model_location(h, file, &offset, &bytes, &id) || wex_aligned_size(bytes, &aligned) ||
            wex_nr2_model_open(file + offset, bytes, model) || model->id != id) return 0;
        *has_model = 1;
    }
    return 1;
}

/* 1 when the header asks the store for more slab memory than the harness gives it */
static int fuzz_too_large(const unsigned char *file, size_t size) {
    const WexHeader *h = (const WexHeader *)file;
    return size >= sizeof(*h) &&
           (h->num_layers > 65536u || h->aligned_expert_bytes > FUZZ_MAX_SLABS ||
            (uint64_t)h->num_layers * 24u * h->aligned_expert_bytes > FUZZ_MAX_SLABS);
}

size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed) {
    if (fuzz_structured() && !(seed % 2u) && fuzz_wnr3_mutate_legacy(data, size, seed)) return size;
    return LLVMFuzzerMutate(data, size, max_size);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ColiDeepSeekV4ExpertStoreOptions options;
    ColiExpertStore *store = NULL;
    ColiExpertStoreStats before, after;
    ColiExpertKey first_hit = {-1, -1}, keys[4];
    WexNr2Model model;
    const WexHeader *h;
    const WexEntry *entries;
    unsigned char *file, *slab;
    char error[256] = {0};
    size_t count, step, i;
    uint64_t looked_up = 0;
    int opens, has_model, rc;
    if (!temp_path[0]) fuzz_setup();
    count_inputs++;
    file = fuzz_exact_copy(data, size);
    if (fuzz_too_large(file, size)) { free(file); return 0; }
    fuzz_forget_models();
    fuzz_write_file(temp_path, data, size);
    memset(&options, 0, sizeof options);
    options.model_dir = ".";
    rc = coli_v4_expert_store_open_wex(NULL, NULL, &options, &store, error, sizeof error);
    opens = fuzz_opens(file, size, &model, &has_model);
    FUZZ_REQUIRE(rc ? !store : store != NULL, "the store's open result and its output pointer disagree");
    FUZZ_REQUIRE((rc == 0) == opens, "the store and the validators of wex_format.h disagree on whether the container opens");
    if (rc) { free(file); return 0; }
    count_opened++;

    h = (const WexHeader *)file;
    entries = (const WexEntry *)(file + h->table_offset);
    count = (size_t)h->num_layers * h->num_experts;
    step = count > FUZZ_MAX_LOOKUPS ? count / FUZZ_MAX_LOOKUPS : 1u;
    slab = (unsigned char *)malloc((size_t)h->aligned_expert_bytes);
    if (!slab) abort();
    for (i = 0; i < count; i += step) {
        const WexEntry *e = &entries[i];
        ColiExpertKey key;
        ColiExpertView view;
        size_t at = wex_record_read_offset(e);
        int want = -1, got;
        unsigned int m;
        key.layer = (int)(i / h->num_experts);
        key.expert = (int)(i % h->num_experts);
        if (e->total_bytes) {
            /* wex_validate_table has bound the record to the file and its decoded size to the slab */
            FUZZ_REQUIRE(at <= h->aligned_expert_bytes && e->total_bytes <= h->aligned_expert_bytes - at,
                         "a validated entry is read past its slab");
            memcpy(slab + at, file + e->file_offset, e->total_bytes);
            if (!wex_nr3_owner_ok(e, slab + at, (uint32_t)key.layer, (uint32_t)key.expert, h->flags)) want = -1;
            else if (!wex_entry_nr_model(e)) want = wex_decode_record(e, slab, (size_t)h->aligned_expert_bytes);
            else if (has_model && wex_read_u32_le(slab + at + 4u) == model.id)
                want = wex_decode_nr2_record(e, slab, (size_t)h->aligned_expert_bytes, &model);
        }
        memset(&view, 0, sizeof view);
        got = coli_expert_lookup(store, key, &view);
        looked_up++;
        FUZZ_REQUIRE(!got == !want, "the store and the decoders of wex_format.h disagree on a record");
        if (got) continue;
        {
            const ColiTensorView *matrix[3];
            matrix[0] = &view.gate; matrix[1] = &view.down; matrix[2] = &view.up;
            for (m = 0; m < 3u; m++) {
                size_t scale_bytes = (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
                size_t weight_bytes = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
                FUZZ_REQUIRE(matrix[m]->scale_bytes == scale_bytes && matrix[m]->data_bytes == weight_bytes &&
                             !memcmp(matrix[m]->scales, slab + e->scale_offsets[m], scale_bytes) &&
                             !memcmp(matrix[m]->data, slab + e->weight_offsets[m], weight_bytes),
                             "the store hands out a matrix that is not the decoded record");
            }
        }
        coli_expert_release(store, &view);
        count_served++;
        if (e->flags) {
            count_served_coded++;
            fuzz_note_distinct(slab, (size_t)e->scale_bytes + e->weight_bytes);
        }
        if (first_hit.layer < 0) first_hit = key;
    }
    count_lookups += looked_up;
    if (first_hit.layer >= 0) {
        ColiExpertView view;
        memset(&view, 0, sizeof view);
        store->ops->stats(store, &before);
        FUZZ_REQUIRE(!coli_expert_lookup(store, first_hit, &view), "an expert that loaded does not load again");
        store->ops->stats(store, &after);
        /* a hit unless 24 later experts of its layer took every slot */
        FUZZ_REQUIRE(after.hits != before.hits + 1u || after.bytes_read == before.bytes_read, "a cache hit read bytes");
        coli_expert_release(store, &view);
        looked_up++;
    }
    keys[0].layer = 0; keys[0].expert = 0;
    keys[1].layer = (int)h->num_layers - 1; keys[1].expert = (int)h->num_experts - 1;
    keys[2].layer = -1; keys[2].expert = 0;
    keys[3].layer = 0; keys[3].expert = (int)h->num_experts;
    store->ops->prefetch(store, keys, 4u);
    store->ops->stats(store, &after);
    FUZZ_REQUIRE(after.requests == looked_up && after.hits + after.misses == looked_up,
                 "the store's counters do not add up to the lookups");
    store->ops->destroy(store);
    free(slab);
    free(file);
    return 0;
}
