/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#include "../expert_store_wex.h"
#include "../expert_store_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Test the independent backend without loading the engine or model. */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine,
                                    const ColiDeepSeekV4Config *config,
                                    const ColiDeepSeekV4ExpertStoreOptions *opts,
                                    ColiExpertStore **out,
                                    char *error, size_t error_size) {
    (void)engine; (void)config; (void)opts; (void)out; (void)error; (void)error_size;
    return -1;
}

typedef struct Fixture {
    unsigned char *data;
    size_t bytes;
    WexHeader *header;
    WexEntry *entries;
} Fixture;

static void env_set(const char *name, const char *value) {
#ifdef _WIN32
    assert(_putenv_s(name, value ? value : "") == 0);
    assert(SetEnvironmentVariableA(name, value && *value ? value : NULL));
#else
    assert(value && *value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0);
#endif
}

static void put_u32(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value; p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16); p[3] = (unsigned char)(value >> 24);
}

static unsigned char scale_value(int index, size_t position) {
    return (unsigned char)((index * 7 + (int)(position / 256) * 3 + 93) % 200);
}

static unsigned char weight_value(int index, int matrix, size_t position) {
    return (unsigned char)(index * 7 + matrix * 31 + position * 13);
}

static Fixture fixture_make(int version, int layers, int experts, int mixed) {
    Fixture fixture = {0};
    size_t count = (size_t)layers * experts;
    size_t table_bytes = WEX_ALIGN_UP(count * sizeof(WexEntry), WEX_ALIGN);
    uint32_t scale_matrix_bytes = version == 1 ? 256u : 4096u;
    uint32_t weight_matrix_bytes = scale_matrix_bytes * 16u;
    uint32_t raw_scales = scale_matrix_bytes * 3u;
    uint32_t raw_weights = weight_matrix_bytes * 3u;
    uint32_t raw_record = WEX_ALIGN_UP(raw_scales + raw_weights, WEX_ALIGN);
    fixture.bytes = WEX_ALIGN + table_bytes;
    for (size_t i = 0; i < count; i++)
        fixture.bytes += version == 2 && (!mixed || i % 7 != 0)
            ? WEX_ALIGN + raw_weights : raw_record;
    fixture.data = (unsigned char *)calloc(1, fixture.bytes);
    assert(fixture.data);
    fixture.header = (WexHeader *)fixture.data;
    fixture.entries = (WexEntry *)(fixture.data + WEX_ALIGN);
    fixture.header->magic = version == 1 ? WEX_MAGIC_V1 : WEX_MAGIC_V2;
    fixture.header->version = (uint32_t)version;
    fixture.header->num_layers = (uint32_t)layers;
    fixture.header->num_experts = (uint32_t)experts;
    fixture.header->aligned_expert_bytes = raw_record;
    fixture.header->table_offset = WEX_ALIGN;
    fixture.header->flags = version == 2 ? WEX_HEADER_SCALE_TILES : 0;
    if (version == 2) memset(fixture.data + sizeof(WexHeader), 0x51, 16);
    size_t offset = WEX_ALIGN + table_bytes;
    for (size_t i = 0; i < count; i++) {
        WexEntry *entry = &fixture.entries[i];
        int compressed = version == 2 && (!mixed || i % 7 != 0);
        entry->file_offset = offset;
        entry->total_bytes = compressed ? WEX_ALIGN + raw_weights : raw_record;
        entry->scale_bytes = raw_scales;
        entry->weight_bytes = raw_weights;
        entry->flags = compressed ? (1u << WEX_ENTRY_SCALE_PAGES_SHIFT) | WEX_ENTRY_SCALE_TILES : 0;
        for (int m = 0; m < 3; m++) {
            entry->scale_shapes[m][0] = version == 1 ? 32 : 64;
            entry->scale_shapes[m][1] = version == 1 ? 8 : 64;
            entry->weight_shapes[m][0] = entry->scale_shapes[m][0];
            entry->weight_shapes[m][1] = entry->scale_shapes[m][1] * 16;
            entry->scale_offsets[m] = (uint32_t)m * scale_matrix_bytes;
            entry->weight_offsets[m] = raw_scales + (uint32_t)m * weight_matrix_bytes;
        }
        unsigned char *record = fixture.data + offset;
        size_t disk_weights = compressed ? WEX_ALIGN : raw_scales;
        if (compressed) {
            uint32_t tiles = raw_scales / WEX_SCALE_TILE_VALUES;
            put_u32(record, WEX_SCALE_STREAM_MAGIC);
            put_u32(record + 4, raw_scales);
            put_u32(record + 8, tiles * 2);
            for (uint32_t t = 0; t < tiles; t++) {
                record[12 + 2 * t] = scale_value((int)i, (size_t)t * 256);
                record[12 + 2 * t + 1] = 0;
            }
        } else {
            for (uint32_t j = 0; j < raw_scales; j++)
                record[j] = scale_value((int)i, j);
        }
        for (int m = 0; m < 3; m++)
            for (uint32_t j = 0; j < weight_matrix_bytes; j++)
                record[disk_weights + m * weight_matrix_bytes + j] = weight_value((int)i, m, j);
        offset += entry->total_bytes;
    }
    assert(offset == fixture.bytes);
    return fixture;
}

static void file_write(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    assert(file);
    assert(fwrite(data, 1, bytes, file) == bytes);
    assert(fclose(file) == 0);
}

static ColiExpertStore *fixture_open(const char *path, const Fixture *fixture, int success) {
    ColiDeepSeekV4ExpertStoreOptions options = {0};
    options.model_dir = ".";
    options.layers = (int)fixture->header->num_layers;
    options.experts_per_layer = (int)fixture->header->num_experts;
    env_set("COLI_WEX_PATH", path);
    char error[256] = {0};
    ColiExpertStore *store = NULL;
    int rc = coli_v4_expert_store_open_wex(NULL, NULL, &options, &store, error, sizeof(error));
    if (success && (rc || !store)) fprintf(stderr, "Open failed: %s\n", error);
    if (!success && (rc == 0 || store)) fprintf(stderr, "Unexpected acceptance: version=%u flags=%u offset=%llu entry_flags=%u\n",
        fixture->header->version, fixture->header->flags,
        (unsigned long long)fixture->header->table_offset, fixture->entries[0].flags);
    assert(success ? rc == 0 && store : rc != 0 && !store);
    if (!success) assert(error[0]);
    return store;
}

static void view_check(const Fixture *fixture, int index, const ColiExpertView *view) {
    const WexEntry *entry = &fixture->entries[index];
    const ColiTensorView *matrices[3] = { &view->gate, &view->down, &view->up };
    assert(view->lease);
    for (int m = 0; m < 3; m++) {
        const ColiTensorView *matrix = matrices[m];
        assert(matrix->format == COLI_TENSOR_FP4_NATIVE_BLOCK);
        assert(matrix->scale_format == COLI_SCALE_UE8M0);
        assert(matrix->rows == entry->weight_shapes[m][0]);
        assert(matrix->columns == entry->weight_shapes[m][1] * 2);
        assert(matrix->data_bytes == (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1]);
        assert(matrix->scale_bytes == (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1]);
        for (size_t j = 0; j < matrix->scale_bytes; j++)
            assert(((const unsigned char *)matrix->scales)[j] == scale_value(index, entry->scale_offsets[m] + j));
        for (size_t j = 0; j < matrix->data_bytes; j++)
            assert(((const unsigned char *)matrix->data)[j] == weight_value(index, m, j));
    }
}

static void test_roundtrip(int version) {
    const char *path = "test_wex_roundtrip.wex";
    Fixture fixture = fixture_make(version, 2, 32, 1);
    file_write(path, fixture.data, fixture.bytes);
    ColiExpertStore *store = fixture_open(path, &fixture, 1);
    ColiExpertView held[24] = {0};
    for (int e = 0; e < 24; e++) {
        assert(coli_expert_lookup(store, (ColiExpertKey){0,e}, &held[e]) == 0);
        view_check(&fixture, e, &held[e]);
    }
    ColiExpertView view = {0};
    assert(coli_expert_lookup(store, (ColiExpertKey){0,24}, &view) != 0);
    assert(!view.lease);
    for (int e = 0; e < 24; e++) {
        view_check(&fixture, e, &held[e]); /* leased buffers survived cache pressure */
        coli_expert_release(store, &held[e]);
    }
    assert(coli_expert_lookup(store, (ColiExpertKey){0,24}, &view) == 0);
    view_check(&fixture, 24, &view);
    coli_expert_release(store, &view);
    for (int e = 25; e < 32; e++) {
        assert(coli_expert_lookup(store, (ColiExpertKey){0,e}, &view) == 0);
        view_check(&fixture, e, &view);
        coli_expert_release(store, &view);
    }
    ColiExpertStoreStats before, after;
    store->ops->stats(store, &before);
    assert(coli_expert_lookup(store, (ColiExpertKey){0,31}, &view) == 0);
    coli_expert_release(store, &view);
    store->ops->stats(store, &after);
    assert(after.hits == before.hits + 1 && after.bytes_read == before.bytes_read);
    ColiExpertKey keys[] = {{1,1}, {1,2}, {-1,0}, {0,32}};
    assert(store->ops->prefetch(store, keys, 4) == 2);
    assert(store->ops->prefetch(store, keys, 4) == 0);
    for (int e = 1; e <= 2; e++) {
        assert(coli_expert_lookup(store, (ColiExpertKey){1,e}, &view) == 0);
        view_check(&fixture, 32 + e, &view);
        coli_expert_release(store, &view);
    }
    assert(coli_expert_lookup(store, (ColiExpertKey){-1,0}, &view) != 0);
    assert(coli_expert_lookup(store, (ColiExpertKey){0,32}, &view) != 0);
    store->ops->stats(store, &after);
    uint64_t expected_read = 0;
    for (int e = 0; e < 32; e++) expected_read += fixture.entries[e].total_bytes;
    expected_read += fixture.entries[33].total_bytes + fixture.entries[34].total_bytes;
    assert(after.bytes_read == expected_read);
    assert(after.prefetched == 2);
    assert(after.capacity_bytes == fixture.header->aligned_expert_bytes * 48);
    printf("WEX%d: exact scales/weights, eviction, leases, cache and prefetch passed (%llu disk bytes).\n",
           version, (unsigned long long)after.bytes_read);
    store->ops->destroy(store);
    free(fixture.data);
    assert(remove(path) == 0);
}

static void test_rejections(void) {
    const char *path = "test_wex_invalid.wex";
    Fixture fixture = fixture_make(2, 1, 1, 0);
    WexHeader header = *fixture.header;
    WexEntry entry = fixture.entries[0];
#define REJECT() do { file_write(path, fixture.data, fixture.bytes); fixture_open(path, &fixture, 0); } while (0)
    fixture.header->version = 99; REJECT(); *fixture.header = header;
    fixture.header->magic = WEX_MAGIC_V1; REJECT(); *fixture.header = header;
    fixture.header->flags |= 2; REJECT(); *fixture.header = header;
    fixture.header->reserved = 1; REJECT(); *fixture.header = header;
    fixture.header->aligned_expert_bytes--; REJECT(); *fixture.header = header;
    fixture.header->table_offset = UINT64_MAX - 4095; REJECT(); *fixture.header = header;
    fixture.entries[0].flags |= 2; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].flags = WEX_ENTRY_SCALE_TILES; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].file_offset = fixture.bytes; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].file_offset = 4096; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].total_bytes--; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].scale_bytes++; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].weight_bytes++; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].scale_offsets[1]++; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].weight_offsets[1]++; REJECT(); fixture.entries[0] = entry;
    fixture.entries[0].weight_shapes[0][1]++; REJECT(); fixture.entries[0] = entry;
    file_write(path, fixture.data, fixture.bytes - WEX_ALIGN); fixture_open(path, &fixture, 0);
    file_write(path, fixture.data, WEX_ALIGN + 100); fixture_open(path, &fixture, 0);
    file_write(path, fixture.data, 32); fixture_open(path, &fixture, 0);
#undef REJECT
    /* Metadata validates, but a malformed scale stream must not become resident. */
    unsigned char *record = fixture.data + entry.file_offset;
    unsigned char saved = record[13];
    record[13] = 9; /* impossible tile bit width */
    file_write(path, fixture.data, fixture.bytes);
    ColiExpertStore *store = fixture_open(path, &fixture, 1);
    ColiExpertView view = {0};
    ColiExpertKey key = {0,0};
    assert(coli_expert_lookup(store, key, &view) != 0 && !view.lease);
    assert(coli_expert_lookup(store, key, &view) != 0 && !view.lease);
    assert(store->ops->prefetch(store, &key, 1) == 0);
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    assert(stats.hits == 0 && stats.bytes_read == 0 && stats.prefetched == 0);
    store->ops->destroy(store);
    record[13] = saved;
    /* A mismatched expected model dimension is also rejected. */
    file_write(path, fixture.data, fixture.bytes);
    ColiDeepSeekV4ExpertStoreOptions options = {0};
    options.model_dir = "."; options.layers = 2; options.experts_per_layer = 1;
    store = NULL;
    char error[256] = {0};
    assert(coli_v4_expert_store_open_wex(NULL, NULL, &options, &store, error, sizeof(error)) != 0);
    assert(!store);
    free(fixture.data);
    assert(remove(path) == 0);
    puts("Malformed headers, entries, truncation, scale streams and model mismatches rejected.");
}

static void test_mirror(const char *filename, int backslash) {
    const char *primary_directory = "test_wex_primary";
    const char *directory = "test_wex_mirror";
    char primary[256], mirror[256];
#ifdef _WIN32
    assert(_mkdir(primary_directory) == 0);
    assert(_mkdir(directory) == 0);
    snprintf(primary, sizeof(primary), "%s%c%s", primary_directory, backslash ? '\\' : '/', filename);
#else
    (void)backslash;
    assert(mkdir(primary_directory, 0700) == 0);
    assert(mkdir(directory, 0700) == 0);
    snprintf(primary, sizeof(primary), "%s/%s", primary_directory, filename);
#endif
    snprintf(mirror, sizeof(mirror), "%s/%s", directory, filename);
    Fixture fixture = fixture_make(2, 1, 2, 0);
    file_write(primary, fixture.data, fixture.bytes);
    file_write(mirror, fixture.data, fixture.bytes);
    env_set("COLI_MODEL_MIRROR", directory);
    env_set("COLI_DISK_WEIGHTS", "1,1000000");
    ColiExpertStore *store = fixture_open(primary, &fixture, 1);
    ColiExpertView view = {0};
    assert(coli_expert_lookup(store, (ColiExpertKey){0,1}, &view) == 0);
    view_check(&fixture, 1, &view);
    coli_expert_release(store, &view);
    store->ops->destroy(store);
    fixture.data[40] ^= 1; /* identical layout, different WEX2 generation */
    file_write(mirror, fixture.data, fixture.bytes);
    fixture_open(primary, &fixture, 0);
    fixture.data[40] ^= 1;
    fixture.entries[0].file_offset += WEX_ALIGN;
    file_write(mirror, fixture.data, fixture.bytes);
    fixture_open(primary, &fixture, 0);
    fixture.entries[0].file_offset -= WEX_ALIGN;
    file_write(mirror, fixture.data, fixture.bytes - WEX_ALIGN);
    fixture_open(primary, &fixture, 0);
    env_set("COLI_MODEL_MIRROR", NULL);
    env_set("COLI_DISK_WEIGHTS", NULL);
    assert(remove(primary) == 0 && remove(mirror) == 0);
#ifdef _WIN32
    assert(_rmdir(primary_directory) == 0);
    assert(_rmdir(directory) == 0);
#else
    assert(rmdir(primary_directory) == 0);
    assert(rmdir(directory) == 0);
#endif
    free(fixture.data);
    printf("Mirror generation, table and file-size validation passed (%s).\n", filename);
}

static void test_external_v3(const char *reference_path, const char *container_path, int expect_cuda) {
    FILE *file = fopen(reference_path, "rb");
    WexHeader header;
    WexEntry entries[2];
    Fixture fixture = {0};
    ColiExpertStore *store;
    ColiExpertStoreStats before, after;
    assert(file && fread(&header, sizeof(header), 1, file) == 1);
    assert(header.version == 1 && header.num_layers == 1 && header.num_experts == 2);
    assert(fseek(file, (long)header.table_offset, SEEK_SET) == 0);
    assert(fread(entries, sizeof(entries), 1, file) == 1);
    fixture.header = &header;
    fixture.entries = entries;
    store = fixture_open(container_path, &fixture, 1);
    for (int e = 0; e < 2; e++) {
        unsigned char *expected = (unsigned char *)malloc(entries[e].total_bytes);
        ColiExpertView view = {0};
        assert(expected && entries[e].file_offset <= LONG_MAX);
        assert(!fseek(file, (long)entries[e].file_offset, SEEK_SET));
        assert(fread(expected, 1, entries[e].total_bytes, file) == entries[e].total_bytes);
        assert(!coli_expert_lookup(store, (ColiExpertKey){0,e}, &view));
        const ColiTensorView *matrices[3] = {&view.gate, &view.down, &view.up};
        for (int m = 0; m < 3; m++) {
            assert(!memcmp(matrices[m]->data, expected + entries[e].weight_offsets[m], matrices[m]->data_bytes));
            assert(!memcmp(matrices[m]->scales, expected + entries[e].scale_offsets[m], matrices[m]->scale_bytes));
        }
        coli_expert_release(store, &view);
        free(expected);
    }
    store->ops->stats(store, &before);
    uint64_t cuda_loads = coli_wex_cuda_decode_count();
    if (expect_cuda) assert(cuda_loads >= 2u);
    ColiExpertView view = {0};
    assert(!coli_expert_lookup(store, (ColiExpertKey){0,0}, &view));
    coli_expert_release(store, &view);
    store->ops->stats(store, &after);
    assert(after.hits == before.hits + 1 && after.bytes_read == before.bytes_read);
    assert(coli_wex_cuda_decode_count() == cuda_loads);
    store->ops->destroy(store);
    assert(!fclose(file));
    printf("WEX3 standalone: every scale/weight byte matches the reference; cached hit reads zero bytes; cuda_decodes=%llu.\n",
           (unsigned long long)cuda_loads);
}

/* WNR2 (profile 0) and WNR3 (1: flags 33, 2: flags 97 = canonical zero): a
 * container whose records need its model; the standalone store must register
 * the model at open, decode every record exactly (CZ: equal to the source with
 * -0 stored as +0), and refuse a corrupted model, a descriptor naming another
 * one and (WNR3) a record served for another expert. */
static uint32_t nr2_next(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static void test_nr2_container(int profile) {
    const char *path = "test_wex_nr2.wex";
    enum { EXPERTS = 4 };
    static const unsigned char mag[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    uint32_t scale_matrix = 4096u, weight_matrix = scale_matrix * 16u;
    uint32_t raw_scales = 3u * scale_matrix, raw_weights = 3u * weight_matrix;
    uint32_t raw_record = WEX_ALIGN_UP(raw_scales + raw_weights, WEX_ALIGN);
    uint64_t *wcounts = calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), *scounts = calloc(3u * 256u, sizeof(uint64_t));
    unsigned char *raw[EXPERTS], *blob = calloc(1, WEX_NR2_MODEL_BYTES);
    unsigned char *records[EXPERTS], *coded[EXPERTS] = {0};
    uint32_t rflags[EXPERTS] = {0};
    size_t sizes[EXPERTS];
    WexEntry entries[EXPERTS];
    WexNr2Layout layout;
    WexNr2Model model;
    uint32_t seed = 977u;
    assert(wcounts && scounts && blob);
    memset(entries, 0, sizeof(entries));
    for (int i = 0; i < EXPERTS; i++) {
        WexEntry *e = &entries[i];
        e->scale_bytes = raw_scales; e->weight_bytes = raw_weights;
        for (int m = 0; m < 3; m++) {
            e->scale_shapes[m][0] = e->weight_shapes[m][0] = 64;
            e->scale_shapes[m][1] = 64; e->weight_shapes[m][1] = 1024;
            e->scale_offsets[m] = (uint32_t)m * scale_matrix;
            e->weight_offsets[m] = raw_scales + (uint32_t)m * weight_matrix;
        }
        raw[i] = malloc(raw_scales + raw_weights);
        assert(raw[i]);
        for (uint32_t j = 0; j < raw_scales; j++) raw[i][j] = (unsigned char)(119u + nr2_next(&seed) % 4u);
        for (uint32_t j = 0; j < raw_weights; j++) {
            uint32_t r = nr2_next(&seed);
            raw[i][raw_scales + j] = (unsigned char)((mag[r & 15u] | ((r >> 4) & 8u)) |
                                                     (mag[(r >> 8) & 15u] | ((r >> 12) & 8u)) << 4);
        }
        if (profile)   /* every 4th block "even" (magnitudes 0,2,4,5,6,7, max 7): ceil form */
            for (uint32_t blk = 0; blk < raw_weights / 16u; blk += 4u)
                for (uint32_t k = 0; k < 16u; k++) {
                    static const unsigned char even[8] = {0, 2, 4, 5, 6, 7, 7, 0};
                    uint32_t r = nr2_next(&seed);
                    raw[i][raw_scales + blk * 16u + k] =
                        (unsigned char)(even[r & 7u] | (r & 8u) | (even[(r >> 4) & 7u] | (r & 128u)) << 4);
                }
        const unsigned char *scales[3], *weights[3];
        wex_nr2_layout(e, &layout);
        if (profile) {   /* coded domain: source -> Z -> ceil form */
            unsigned char *cs[3], *cw[3];
            coded[i] = malloc(raw_scales + raw_weights);
            assert(coded[i]);
            memcpy(coded[i], raw[i], raw_scales + raw_weights);
            if (profile == 2) wex_nr3_canonical_zero(coded[i] + raw_scales, raw_weights);
            for (int m = 0; m < 3; m++) { cs[m] = coded[i] + e->scale_offsets[m]; cw[m] = coded[i] + e->weight_offsets[m]; }
            rflags[i] = wex_nr3_ceil_forward(&layout, cs, cw, NULL) | (profile == 2 ? WEX_NR3_FLAG_CZ : 0u);
            for (int m = 0; m < 3; m++) { scales[m] = cs[m]; weights[m] = cw[m]; }
            wex_nr3_count(&layout, scales, weights, wcounts, scounts);
            if (profile == 2) wex_nr3_canonical_zero(raw[i] + raw_scales, raw_weights);   /* expected = Z(source) */
            continue;
        }
        for (int m = 0; m < 3; m++) { scales[m] = raw[i] + e->scale_offsets[m]; weights[m] = raw[i] + e->weight_offsets[m]; }
        wex_nr2_count(&layout, scales, weights, wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, blob);
    assert(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model));
    for (int i = 0; i < EXPERTS; i++) {
        const unsigned char *scales[3], *weights[3];
        const unsigned char *src = profile ? coded[i] : raw[i];
        for (int m = 0; m < 3; m++) { scales[m] = src + entries[i].scale_offsets[m]; weights[m] = src + entries[i].weight_offsets[m]; }
        records[i] = calloc(1, raw_record);
        assert(records[i]);
        if (profile) assert(!wex_nr3_encode(&model, &layout, scales, weights, 2048u, 512u, rflags[i], (uint32_t)i,
                                            records[i], raw_record, &sizes[i]));
        else assert(!wex_nr2_encode(&model, &layout, scales, weights, 4096u, 2048u, records[i], raw_record, &sizes[i]));
        sizes[i] = WEX_ALIGN_UP(sizes[i], WEX_ALIGN);
        assert(sizes[i] < raw_record);
    }
    /* layout: header page, model at 4096 (61440 bytes reserved), table, records */
    size_t table_offset = WEX_ALIGN + WEX_ALIGN_UP(WEX_NR2_MODEL_BYTES, WEX_ALIGN);
    size_t offset = table_offset + WEX_ALIGN_UP(EXPERTS * sizeof(WexEntry), WEX_ALIGN), bytes = offset;
    for (int i = 0; i < EXPERTS; i++) bytes += sizes[i];
    Fixture fixture = {0};
    fixture.data = calloc(1, bytes);
    fixture.bytes = bytes;
    assert(fixture.data);
    fixture.header = (WexHeader *)fixture.data;
    fixture.entries = (WexEntry *)(fixture.data + table_offset);
    fixture.header->magic = WEX_MAGIC_V3; fixture.header->version = 3;
    fixture.header->num_layers = 1; fixture.header->num_experts = EXPERTS;
    fixture.header->aligned_expert_bytes = raw_record;
    fixture.header->table_offset = table_offset;
    fixture.header->flags = !profile ? WEX_HEADER_NEURAL_RANS2 :
        WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3 | (profile == 2 ? WEX_HEADER_CANONICAL_ZERO : 0u);
    memset(fixture.data + sizeof(WexHeader), 0x77, 16);
    unsigned char *d = fixture.data + WEX_NR2_DESCRIPTOR_OFFSET;
    put_u32(d, WEX_NR2_DESCRIPTOR_MAGIC); put_u32(d + 4, WEX_NR2_MODEL_BYTES);
    put_u32(d + 8, WEX_ALIGN); put_u32(d + 12, 0); put_u32(d + 16, model.id); put_u32(d + 20, 0);
    memcpy(fixture.data + WEX_ALIGN, blob, WEX_NR2_MODEL_BYTES);
    for (int i = 0; i < EXPERTS; i++) {
        entries[i].file_offset = offset;
        entries[i].total_bytes = (uint32_t)sizes[i];
        entries[i].flags = profile ? WEX_ENTRY_NEURAL_RANS3 : WEX_ENTRY_NEURAL_RANS2;
        memcpy(fixture.data + offset, records[i], sizes[i]);
        offset += sizes[i];
    }
    memcpy(fixture.entries, entries, sizeof(entries));
    env_set("COLI_WEX_CUDA_LIB", "0");   /* the CPU reference path */
    file_write(path, fixture.data, fixture.bytes);
    ColiExpertStore *store = fixture_open(path, &fixture, 1);
    for (int i = 0; i < EXPERTS; i++) {
        ColiExpertView view = {0};
        assert(!coli_expert_lookup(store, (ColiExpertKey){0, i}, &view));
        const ColiTensorView *matrices[3] = {&view.gate, &view.down, &view.up};
        for (int m = 0; m < 3; m++) {
            assert(!memcmp(matrices[m]->scales, raw[i] + entries[i].scale_offsets[m], matrices[m]->scale_bytes));
            assert(!memcmp(matrices[m]->data, raw[i] + entries[i].weight_offsets[m], matrices[m]->data_bytes));
        }
        coli_expert_release(store, &view);
    }
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    store->ops->destroy(store);
    /* a corrupted model is refused before any record is read */
    fixture.data[WEX_ALIGN] ^= 0xffu;   /* model magic */
    file_write(path, fixture.data, fixture.bytes);
    fixture_open(path, &fixture, 0);
    fixture.data[WEX_ALIGN] ^= 0xffu;
    /* a descriptor naming another model is refused */
    put_u32(d + 16, model.id ^ 1u);
    file_write(path, fixture.data, fixture.bytes);
    fixture_open(path, &fixture, 0);
    put_u32(d + 16, model.id);
    if (profile) {
        /* a record served for another expert (records 0 and 1 swapped in the
         * file, the table still contiguous): the owner check refuses it */
        uint32_t keep = fixture.header->flags;
        size_t at0 = entries[0].file_offset;
        memcpy(fixture.data + at0, records[1], sizes[1]);
        memcpy(fixture.data + at0 + sizes[1], records[0], sizes[0]);
        fixture.entries[0].total_bytes = (uint32_t)sizes[1];
        fixture.entries[1].file_offset = at0 + sizes[1];
        fixture.entries[1].total_bytes = (uint32_t)sizes[0];
        file_write(path, fixture.data, fixture.bytes);
        store = fixture_open(path, &fixture, 1);
        for (int i = 0; i < 2; i++) {
            ColiExpertView view = {0};
            assert(coli_expert_lookup(store, (ColiExpertKey){0, i}, &view) != 0);
        }
        {
            ColiExpertView view = {0};   /* the others still decode */
            assert(!coli_expert_lookup(store, (ColiExpertKey){0, 2}, &view));
            assert(!memcmp(view.gate.data, raw[2] + entries[2].weight_offsets[0], view.gate.data_bytes));
            coli_expert_release(store, &view);
        }
        store->ops->destroy(store);
        memcpy(fixture.entries, entries, sizeof(entries));
        memcpy(fixture.data + at0, records[0], sizes[0]);
        memcpy(fixture.data + at0 + sizes[0], records[1], sizes[1]);
        /* every record's canonical-zero flag differs from the container's: all refused */
        env_set("DSV4_CUDA_MOE_MASKED", NULL);
        fixture.header->flags = keep ^ WEX_HEADER_CANONICAL_ZERO;
        file_write(path, fixture.data, fixture.bytes);
        store = fixture_open(path, &fixture, 1);
        for (int i = 0; i < EXPERTS; i++) {
            ColiExpertView view = {0};
            assert(coli_expert_lookup(store, (ColiExpertKey){0, i}, &view) != 0);
        }
        store->ops->destroy(store);
        fixture.header->flags = keep;
        /* a canonical-zero container is refused on the masked DeepGEMM MoE path */
        env_set("DSV4_CUDA_MOE_MASKED", "1");
        file_write(path, fixture.data, fixture.bytes);
        store = fixture_open(path, &fixture, profile != 2);
        if (store) store->ops->destroy(store);
        env_set("DSV4_CUDA_MOE_MASKED", "0");
        store = fixture_open(path, &fixture, 1);
        store->ops->destroy(store);
        env_set("DSV4_CUDA_MOE_MASKED", NULL);
        /* WNR3 entries under a WNR2 header, and a CZ flag without WNR3, are invalid */
        fixture.header->flags = WEX_HEADER_NEURAL_RANS2;
        file_write(path, fixture.data, fixture.bytes);
        fixture_open(path, &fixture, 0);
        fixture.header->flags = WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_CANONICAL_ZERO;
        file_write(path, fixture.data, fixture.bytes);
        fixture_open(path, &fixture, 0);
        fixture.header->flags = keep;
    }
    /* without the header flag the WNR2/WNR3 entries are invalid */
    fixture.header->flags = 0;
    file_write(path, fixture.data, fixture.bytes);
    fixture_open(path, &fixture, 0);
    printf("%s: model registered at open, %d experts decoded exactly (%llu disk bytes of %u raw each), "
           "corrupted or foreign models%s refused.\n", profile == 0 ? "WNR2" : profile == 1 ? "WNR3" : "WNR3 canonical-zero",
           EXPERTS, (unsigned long long)stats.bytes_read, raw_record, profile ? " and foreign-owner records" : "");
    for (int i = 0; i < EXPERTS; i++) { free(raw[i]); free(records[i]); free(coded[i]); }
    free(fixture.data); free(blob); free(wcounts); free(scounts);
    env_set("COLI_WEX_CUDA_LIB", NULL);
    assert(remove(path) == 0);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    assert(sizeof(WexHeader) == 40 && sizeof(WexEntry) == 72);
    assert(coli_expert_store_backend_lookup("wex"));
    env_set("COLI_MODEL_MIRROR", NULL);
    test_roundtrip(1);
    test_roundtrip(2);
    test_rejections();
    test_mirror("v4_flash_experts.v2.wex", 0);
    test_mirror("custom_expert_container.wex", 1);
    if (argc == 3 || (argc == 4 && !strcmp(argv[3], "--expect-cuda")))
        test_external_v3(argv[1], argv[2], argc == 4);
    else assert(argc == 1);
    test_nr2_container(0);   /* after: it may pin the CUDA helper off */
    test_nr2_container(1);
    test_nr2_container(2);
    env_set("COLI_WEX_PATH", NULL);
    puts("ALL WEX STANDALONE TESTS PASSED");
    return 0;
}
