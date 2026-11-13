/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Production HOT ExpertStore integration test against a small safetensors
 * fixture. This opens no model engine and performs no GPU computation.
 * Usage: test_wex_planned MODEL_DIR WEX_PATH [--expect-fallback]
 */
#include "../deepseek_v4_internal.h"
#include "../wex_format.h"
#include "../expert_store_wex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COLI_V4_TEST_HOOKS
#error "test_wex_planned must link the COLI_V4_TEST_HOOKS HOT unit"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void env_set(const char *name, const char *value) {
    CHECK(value && *value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0);
}

static int seek_file(FILE *file, uint64_t offset) {
    if (offset > INT64_MAX) return -1;
#ifdef _WIN32
    return _fseeki64(file, (int64_t)offset, SEEK_SET);
#else
    return fseeko(file, (off_t)offset, SEEK_SET);
#endif
}

static uint64_t inspect_wex(const char *path, int expect_fallback,
                             WexHeader *header, WexEntry entries[2]) {
    FILE *file = fopen(path, "rb");
    uint64_t length;
    int64_t position;
    CHECK(file);
#ifdef _WIN32
    CHECK(!_fseeki64(file, 0, SEEK_END));
    position = _ftelli64(file);
#else
    CHECK(!fseeko(file, 0, SEEK_END));
    position = ftello(file);
#endif
    CHECK(position >= 0);
    length = (uint64_t)position;
    CHECK(!seek_file(file, 0));
    memset(header, 0, sizeof(*header));
    memset(entries, 0, sizeof(WexEntry) * 2u);
    if (expect_fallback) {
        /* A rejected/truncated container need not contain a readable header. */
        (void)fread(header, 1, sizeof(*header), file);
    } else {
        CHECK(fread(header, 1, sizeof(*header), file) == sizeof(*header));
        CHECK(!wex_validate_header(header, length, 1, 2, NULL, NULL));
        CHECK(!seek_file(file, header->table_offset));
        CHECK(fread(entries, sizeof(*entries), 2, file) == 2);
        CHECK(!wex_validate_entry(header, entries, length));
        CHECK(!wex_validate_entry(header, entries + 1, length));
        CHECK(entries[0].total_bytes && entries[1].total_bytes);
    }
    CHECK(!fclose(file));
    return length;
}

static uint64_t compare_matrix(const ColiSafetensorsIndex *index,
                                const ColiTensorView *view,
                                int expert, const char *matrix) {
    char key[160];
    const ColiSafetensorsTensor *scale, *weight;
    unsigned char *reference;
    int shard;
    uint64_t bytes;
    snprintf(key, sizeof(key), "layers.0.ffn.experts.%d.%s.scale", expert, matrix);
    scale = coli_st_find(index, key);
    snprintf(key, sizeof(key), "layers.0.ffn.experts.%d.%s.weight", expert, matrix);
    weight = coli_st_find(index, key);
    CHECK(scale && weight && scale->nbytes > 0 && weight->nbytes > 0);
    CHECK((uint64_t)scale->nbytes <= SIZE_MAX && (uint64_t)weight->nbytes <= SIZE_MAX);
    CHECK(view->format == COLI_TENSOR_FP4_NATIVE_BLOCK);
    CHECK(view->scale_format == COLI_SCALE_UE8M0);
    CHECK(view->rows == weight->shape[0] && view->columns == weight->shape[1] * 2);
    CHECK(view->block_rows == 1 && view->block_columns == 32);
    CHECK(view->scale_bytes == (size_t)scale->nbytes);
    CHECK(view->data_bytes == (size_t)weight->nbytes);
    CHECK(view->scales && view->data);
    reference = (unsigned char *)malloc((size_t)(weight->nbytes > scale->nbytes
                                                 ? weight->nbytes : scale->nbytes));
    CHECK(reference);
    shard = coli_st_tensor_shard(index, scale);
    CHECK(shard >= 0 && !coli_st_read_at(index, shard, (uint64_t)scale->off,
                                        (size_t)scale->nbytes, reference));
    if (memcmp(reference, view->scales, view->scale_bytes)) {
        fprintf(stderr, "scale mismatch expert=%d matrix=%s\n", expert, matrix);
        exit(1);
    }
    shard = coli_st_tensor_shard(index, weight);
    CHECK(shard >= 0 && !coli_st_read_at(index, shard, (uint64_t)weight->off,
                                        (size_t)weight->nbytes, reference));
    if (memcmp(reference, view->data, view->data_bytes)) {
        fprintf(stderr, "weight mismatch expert=%d matrix=%s\n", expert, matrix);
        exit(1);
    }
    bytes = (uint64_t)scale->nbytes + (uint64_t)weight->nbytes;
    free(reference);
    return bytes;
}

static uint64_t compare_expert(const ColiSafetensorsIndex *index,
                                const ColiExpertView *view, int expert) {
    CHECK(view->lease && view->key.layer == 0 && view->key.expert == expert);
    return compare_matrix(index, &view->gate, expert, "w1") +
           compare_matrix(index, &view->down, expert, "w2") +
           compare_matrix(index, &view->up, expert, "w3");
}

int main(int argc, char **argv) {
    WexHeader header;
    WexEntry entries[2];
    ColiDeepSeekV4ExpertStoreOptions options;
    ColiSafetensorsIndex *reference = NULL;
    ColiExpertStore *store = NULL;
    ColiExpertView view;
    ColiExpertStoreStats start, first, hit, prefetched, second, final;
    ColiExpertKey key = {0, 0}, next = {0, 1};
    uint64_t raw_bytes[2], expected_first, expected_second;
    uint64_t direct_reads, direct_fallbacks;
    int expect_fallback, expect_cuda, accepted;
    char error[512] = {0};
    if (argc < 3 || argc > 4 || (argc == 4 && strcmp(argv[3], "--expect-fallback") && strcmp(argv[3], "--expect-cuda"))) {
        fprintf(stderr, "Usage: %s MODEL_DIR WEX_PATH [--expect-fallback|--expect-cuda]\n", argv[0]);
        return 2;
    }
    expect_fallback = argc == 4 && !strcmp(argv[3], "--expect-fallback");
    expect_cuda = argc == 4 && !strcmp(argv[3], "--expect-cuda");
    (void)inspect_wex(argv[2], expect_fallback, &header, entries);
    env_set("COLI_WEX_PATH", argv[2]);
    env_set("COLI_V4_AUTOPIN", "0");
    env_set("COLI_V4_SAVE_USAGE", "0");
    env_set("COLI_V4_ROWS16", "0");
    env_set("COLI_V4_PREWARM", "0");
    env_set("COLI_V4_TIER_PREWARM", "0");
    env_set("COLI_V4_ELASTIC_RAM", "0");
    env_set("COLI_MODEL_MIRROR", NULL);
    env_set("SNAP_MIRROR", NULL);
    env_set("V4_ROUTE_UDP", NULL);
    env_set("ROUTE_TRACE", NULL);
    CHECK(!coli_st_index_open(&reference, argv[1], error, sizeof(error)));
    memset(&options, 0, sizeof(options));
    options.model_dir = argv[1];
    options.layers = 1;
    options.experts_per_layer = 2;
    options.cache_bytes = expect_fallback ? UINT64_C(64) * 1024u * 1024u
                                         : 2u * header.aligned_expert_bytes;
    options.skip_mirror_setup = 1;
    options.repin_interval = UINT64_MAX;
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error))) {
        fprintf(stderr, "HOT store open: %s\n", error);
        return 1;
    }
    CHECK(store && store->ops && store->ops->prefetch);
    store->ops->stats(store, &start);
    CHECK(!start.requests && !start.bytes_read);
    coli_v4_test_reset_direct_io_stats();

    memset(&view, 0, sizeof(view));
    CHECK(!coli_expert_lookup(store, key, &view));
    raw_bytes[0] = compare_expert(reference, &view, 0);
    coli_expert_release(store, &view);
    CHECK(!view.lease);
    store->ops->stats(store, &first);
    uint64_t cuda_first = coli_wex_cuda_decode_count();
    if (expect_cuda) CHECK(cuda_first >= 1u);
    expected_first = expect_fallback ? raw_bytes[0] : entries[0].total_bytes;
    CHECK(first.misses == start.misses + 1u);
    CHECK(first.bytes_read == start.bytes_read + expected_first);

    CHECK(!coli_expert_lookup(store, key, &view));
    CHECK(compare_expert(reference, &view, 0) == raw_bytes[0]);
    coli_expert_release(store, &view);
    store->ops->stats(store, &hit);
    CHECK(hit.hits == first.hits + 1u && hit.bytes_read == first.bytes_read);
    CHECK(coli_wex_cuda_decode_count() == cuda_first);

    accepted = store->ops->prefetch(store, &next, 1);
    CHECK(accepted >= 0 && accepted <= 1);
    store->ops->stats(store, &prefetched);
    CHECK(prefetched.prefetched == hit.prefetched + (uint64_t)accepted);
    CHECK(prefetched.bytes_read == hit.bytes_read); /* advisory OS prefetch */
    CHECK(!coli_expert_lookup(store, next, &view));
    raw_bytes[1] = compare_expert(reference, &view, 1);
    coli_expert_release(store, &view);
    store->ops->stats(store, &second);
    uint64_t cuda_second = coli_wex_cuda_decode_count();
    if (expect_cuda) CHECK(cuda_second > cuda_first);
    expected_second = expect_fallback ? raw_bytes[1] : entries[1].total_bytes;
    CHECK(second.misses == first.misses + 1u);
    CHECK(second.bytes_read == first.bytes_read + expected_second);

    CHECK(!coli_expert_lookup(store, key, &view));
    CHECK(compare_expert(reference, &view, 0) == raw_bytes[0]);
    coli_expert_release(store, &view);
    CHECK(!coli_expert_lookup(store, next, &view));
    CHECK(compare_expert(reference, &view, 1) == raw_bytes[1]);
    coli_expert_release(store, &view);
    store->ops->stats(store, &final);
    CHECK(final.hits == second.hits + 2u && final.bytes_read == second.bytes_read);
    CHECK(coli_wex_cuda_decode_count() == cuda_second);
    CHECK(final.bytes_read == expected_first + expected_second);
    direct_reads = coli_v4_test_direct_reads();
    direct_fallbacks = coli_v4_test_direct_fallbacks();
    if (!expect_fallback) {
        CHECK(direct_reads >= 2u);
        CHECK(!direct_fallbacks);
    }
    store->ops->destroy(store);
    coli_st_index_close(reference);
    printf("WEX HOT planned: %s version=%u experts=2 raw_bytes=%llu read_bytes=%llu "
           "direct_reads=%llu direct_fallbacks=%llu cuda_decodes=%llu; all scale/weight bytes, prefetch and hits passed.\n",
           expect_fallback ? "safetensors-fallback" : "container", header.version,
           (unsigned long long)(raw_bytes[0] + raw_bytes[1]),
           (unsigned long long)final.bytes_read,
           (unsigned long long)direct_reads, (unsigned long long)direct_fallbacks,
           (unsigned long long)cuda_second);
    return 0;
}
