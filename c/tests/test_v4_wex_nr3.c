/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR3 containers on the engine's own expert store (deepseek_v4.c, the V4/V4.1
 * path), CPU only: tests/fake_wex_cuda.dll stands in for kaira_wex_cuda.dll and
 * decodes on the CPU. One layer of four 64x2048 FP4 experts, profiles 33 (WNR3)
 * and 97 (WNR3 canonical zero). Checks the engine's reaction to records it must
 * not serve: a corrupted stream (hash), a CZ flag that differs from the
 * container's, a record of another expert (owner); each one falls back to the
 * safetensors bytes. A canonical-zero container is refused on
 * DSV4_CUDA_MOE_MASKED. The GPU-direct fill (through the RAM cache of
 * compressed records) drops a cached record that fails to decode, rereads it
 * from disk next time, and keeps it when the helper only reports 1.
 * NR3_REAL_HELPER=<kaira_wex_cuda.dll> runs it on a real helper (GPU decode)
 * instead, without that direct-fill check: it hands host buffers as device
 * memory and relies on FAKE_WEX_UNAVAILABLE, which only the fake answers.
 * Built with -DV4_WEX_PREWARM_TEST (tests/test_v4_wex_prewarm), it runs only the
 * V4.1 tier prewarm checks instead: the default loop against V4_PREWARM_BATCH=1
 * (check_prewarm). */
#define main unused_merged_test_main
#include "test_deepseek_v4.c"
#undef main

#ifndef _WIN32
int main(void) {
    puts("test_v4_wex_nr3: skipped (the engine's WEX direct I/O is Windows only)");
    return 0;
}
#else
#include <malloc.h>
#include <direct.h>
#include "../wex_decode.h"
#include "../wex4.h"

int coli_v4_test_wex_active(ColiExpertStore *store);
int coli_v4_test_wex_seen(ColiExpertStore *store, ColiExpertKey key);
int coli_v4_expert_store_prewarm_tiers(ColiExpertStore *store);
int coli_v4_gpu_expert_peek(ColiExpertStore *store, ColiExpertView *view);
/* deepseek_v4_internal.h declares these only for the GPU unit */
const WexEntry *coli_v4_expert_store_wex_entry(ColiExpertStore *store, ColiExpertKey key, size_t capacity);
int coli_v4_wex_crc_get(ColiExpertStore *store, ColiExpertKey key, const WexEntry *entry,
                        unsigned char **record, int *hit);
void coli_v4_wex_crc_done(ColiExpertStore *store, int slot, int filled);
int coli_v4_expert_store_wex_read(ColiExpertStore *store, ColiExpertKey key, const WexEntry *entry,
                                  unsigned char *dst);
int coli_v4_wex_first_read_ok_n(ColiExpertStore *store, int n, const ColiExpertKey *keys,
                                const WexEntry *const *entries, unsigned char *const *records,
                                const int *striped, int *ok, int parallel);
int coli_v4_test_wex_fill(ColiExpertStore *store, ColiExpertKey key, unsigned char *staging,
                          size_t staging_bytes, unsigned char *const scale[3], unsigned char *const weight[3]);
int coli_v4_wex_first_read_ok(ColiExpertStore *store, ColiExpertKey key, const WexEntry *entry,
                              const unsigned char *record, int striped);
void *coli_v4_expert_store_wex_read_issue(ColiExpertStore *store, int n, const ColiExpertKey *keys,
                                          const WexEntry *const *entries, unsigned char *const *dst,
                                          int stripe);
int coli_v4_expert_store_wex_read_wait(void *reads, int *ok);
HANDLE coli_v4_test_wex_mirror(ColiExpertStore *store, int r, HANDLE file);
uint64_t coli_v4_test_wex_sub_reads(void);
int coli_v4_test_sha_defer(ColiExpertStore *store, int on);
void coli_v4_test_sha_busy(int on);

enum { EXPERTS = 4, ROWS = 64, SCALE_COLS = 64, WEIGHT_COLS = 1024 };
#define SCALE_M ((size_t)ROWS * SCALE_COLS)
#define WEIGHT_M ((size_t)ROWS * WEIGHT_COLS)
#define RAW ((size_t)3 * (SCALE_M + WEIGHT_M))   /* 208896 = 51 pages */

static unsigned char source[EXPERTS][RAW], wnr3[EXPERTS][RAW];   /* safetensors bytes / what WNR3 decodes */
static char directory[] = "colibri-v4-wnr3-XXXXXX", st_path[512], wex_path[512];

static uint32_t next(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    assert(f && fwrite(data, 1, bytes, f) == bytes && fclose(f) == 0);
}

/* model.safetensors: per expert w1/w2/w3 scales then w1/w2/w3 weights, contiguous */
static void write_safetensors(void) {
    static const char *names[3] = {"w1", "w2", "w3"};
    char header[16384];
    size_t used = (size_t)snprintf(header, sizeof header, "{");
    for (int e = 0; e < EXPERTS; e++)
        for (int m = 0; m < 3; m++) {
            size_t s = (size_t)e * RAW + m * SCALE_M, w = (size_t)e * RAW + 3 * SCALE_M + m * WEIGHT_M;
            used += (size_t)snprintf(header + used, sizeof header - used,
                "%s\"layers.0.ffn.experts.%d.%s.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
                "\"layers.0.ffn.experts.%d.%s.weight\":{\"dtype\":\"I8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                used > 1 ? "," : "", e, names[m], ROWS, SCALE_COLS, s, s + SCALE_M,
                e, names[m], ROWS, WEIGHT_COLS, w, w + WEIGHT_M);
            assert(used < sizeof header);
        }
    header[used++] = '}';
    uint64_t length = used;
    FILE *f = fopen(st_path, "wb");
    assert(f && fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, used, f) == used);
    assert(fwrite(source, 1, sizeof source, f) == sizeof source && fclose(f) == 0);
}

typedef struct {
    unsigned char *data;
    size_t bytes, used[EXPERTS];
    WexEntry entries[EXPERTS];
    unsigned char *records[EXPERTS];
} Container;

/* v4_flash_experts.wex next to the safetensors: header page, model at 4096, table, records */
static void build_container(int cz, Container *c) {
    static const unsigned char mag[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    static unsigned char coded[EXPERTS][RAW], blob[WEX_NR2_MODEL_BYTES];
    uint64_t *wcounts = calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), *scounts = calloc(3u * 256u, sizeof(uint64_t));
    uint32_t seed = 2026111u, flags[EXPERTS];
    WexNr2Layout layout;
    WexNr2Model model;
    assert(wcounts && scounts);
    memset(c, 0, sizeof *c);
    for (int i = 0; i < EXPERTS; i++) {
        WexEntry *e = &c->entries[i];
        e->scale_bytes = (uint32_t)(3 * SCALE_M); e->weight_bytes = (uint32_t)(3 * WEIGHT_M);
        for (int m = 0; m < 3; m++) {
            e->scale_shapes[m][0] = e->weight_shapes[m][0] = ROWS;
            e->scale_shapes[m][1] = SCALE_COLS; e->weight_shapes[m][1] = WEIGHT_COLS;
            e->scale_offsets[m] = (uint32_t)(m * SCALE_M);
            e->weight_offsets[m] = (uint32_t)(3 * SCALE_M + m * WEIGHT_M);
        }
        unsigned char *raw = source[i];
        for (size_t j = 0; j < 3 * SCALE_M; j++) raw[j] = (unsigned char)(119u + next(&seed) % 4u);
        for (size_t j = 0; j < 3 * WEIGHT_M; j++) {   /* sign bits on zero magnitudes: -0 */
            uint32_t r = next(&seed);
            raw[3 * SCALE_M + j] = (unsigned char)((mag[r & 15u] | ((r >> 4) & 8u)) |
                                                   (mag[(r >> 8) & 15u] | ((r >> 12) & 8u)) << 4);
        }
        for (size_t blk = 0; blk < 3 * WEIGHT_M / 16u; blk += 4u)   /* "even" blocks: ceil form */
            for (int k = 0; k < 16; k++) {
                static const unsigned char even[8] = {0, 2, 4, 5, 6, 7, 7, 0};
                uint32_t r = next(&seed);
                raw[3 * SCALE_M + blk * 16u + k] =
                    (unsigned char)(even[r & 7u] | (r & 8u) | (even[(r >> 4) & 7u] | (r & 128u)) << 4);
            }
        memcpy(wnr3[i], raw, RAW);
        if (cz) assert(wex_nr3_canonical_zero(wnr3[i] + 3 * SCALE_M, 3 * WEIGHT_M) > 0);
        memcpy(coded[i], wnr3[i], RAW);
        unsigned char *cs[3], *cw[3];
        for (int m = 0; m < 3; m++) { cs[m] = coded[i] + e->scale_offsets[m]; cw[m] = coded[i] + e->weight_offsets[m]; }
        wex_nr2_layout(e, &layout);
        flags[i] = wex_nr3_ceil_forward(&layout, cs, cw, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
        wex_nr3_count(&layout, (const unsigned char *const *)cs, (const unsigned char *const *)cw, wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, blob);
    assert(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model));
    size_t table_offset = WEX_ALIGN + WEX_ALIGN_UP(WEX_NR2_MODEL_BYTES, WEX_ALIGN);
    size_t offset = table_offset + WEX_ALIGN_UP(sizeof c->entries, WEX_ALIGN), bytes = offset, sizes[EXPERTS];
    for (int i = 0; i < EXPERTS; i++) {
        const unsigned char *s[3], *w[3];
        for (int m = 0; m < 3; m++) { s[m] = coded[i] + c->entries[i].scale_offsets[m]; w[m] = coded[i] + c->entries[i].weight_offsets[m]; }
        c->records[i] = calloc(1, RAW);
        assert(c->records[i] && !wex_nr3_encode(&model, &layout, s, w, 2048u, 512u, flags[i], (uint32_t)i,
                                                c->records[i], RAW, &c->used[i]));
        sizes[i] = WEX_ALIGN_UP(c->used[i], WEX_ALIGN);
        assert(sizes[i] < RAW);
        bytes += sizes[i];
    }
    c->data = calloc(1, bytes);
    c->bytes = bytes;
    assert(c->data);
    WexHeader *h = (WexHeader *)c->data;
    h->magic = WEX_MAGIC_V3; h->version = 3; h->num_layers = 1; h->num_experts = EXPERTS;
    h->aligned_expert_bytes = RAW; h->table_offset = table_offset;
    h->flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3 | (cz ? WEX_HEADER_CANONICAL_ZERO : 0u);
    unsigned char *d = c->data + WEX_NR2_DESCRIPTOR_OFFSET;
    uint32_t words[6] = {WEX_NR2_DESCRIPTOR_MAGIC, WEX_NR2_MODEL_BYTES, WEX_ALIGN, 0, model.id, 0};
    memcpy(d, words, sizeof words);
    memcpy(c->data + WEX_ALIGN, blob, WEX_NR2_MODEL_BYTES);
    for (int i = 0; i < EXPERTS; i++) {
        c->entries[i].file_offset = offset;
        c->entries[i].total_bytes = (uint32_t)sizes[i];
        c->entries[i].flags = WEX_ENTRY_NEURAL_RANS3;
        memcpy(c->data + offset, c->records[i], sizes[i]);
        offset += sizes[i];
    }
    memcpy(c->data + table_offset, c->entries, sizeof c->entries);
    free(wcounts); free(scounts);
}

/* the container with records 0 and 1 swapped in the file (the table still contiguous) */
static unsigned char *swapped_copy(const Container *c) {
    unsigned char *copy = malloc(c->bytes);
    assert(copy);
    memcpy(copy, c->data, c->bytes);
    WexEntry *table = (WexEntry *)(copy + ((const WexHeader *)copy)->table_offset);
    size_t at0 = c->entries[0].file_offset;
    memcpy(copy + at0, c->data + c->entries[1].file_offset, c->entries[1].total_bytes);
    memcpy(copy + at0 + c->entries[1].total_bytes, c->data + at0, c->entries[0].total_bytes);
    table[0].total_bytes = c->entries[1].total_bytes;
    table[1].file_offset = at0 + c->entries[1].total_bytes;
    table[1].total_bytes = c->entries[0].total_bytes;
    return copy;
}

/* The WEX4 form of a WEX3/WNR3 container, as the in-place upgrade writes it: the same bytes, then the old
 * header page and the index appended, and a new header page. *root gets the 64 hex digits of its root. */
static unsigned char *upgrade_wex4(const unsigned char *old, size_t old_bytes, size_t *bytes, char root[65]) {
    const WexHeader *h3 = (const WexHeader *)old;
    const WexEntry *table = (const WexEntry *)(old + h3->table_offset);
    int cz = (h3->flags & WEX_HEADER_CANONICAL_ZERO) != 0;
    uint32_t model_id;
    char names[EXPERTS * 6][48];
    size_t name_bytes = 0, at = 0;
    for (int i = 0; i < EXPERTS * 6; i++)
        name_bytes += (size_t)snprintf(names[i], sizeof names[i], "layers.0.ffn.experts.%d.w%d.%s", i / 6, i % 3 + 1,
                                       i % 6 < 3 ? "scale" : "weight");
    size_t index_bytes = 3 * WEX4_BLOB_BYTES + EXPERTS * WEX4_RECORD_BYTES + EXPERTS * 6 * (WEX4_TENSOR_BYTES + 4) + name_bytes;
    size_t index_offset = old_bytes + WEX_ALIGN, total = index_offset + WEX_ALIGN_UP(index_bytes, WEX_ALIGN);
    unsigned char *out = calloc(1, total), *x;
    assert(out);
    memcpy(out, old, old_bytes);
    memcpy(out + old_bytes, old, WEX_ALIGN);                       /* the old header page */
    memcpy(&model_id, old + WEX_NR2_DESCRIPTOR_OFFSET + 16u, 4u);
    x = out + index_offset;
#define PUT(value, type) do { type t_ = (type)(value); memcpy(x + at, &t_, sizeof t_); at += sizeof t_; } while (0)
    struct { uint32_t kind, id; uint64_t offset, size; } blobs[3] = {
        {WEX4_BLOB_WNM2, model_id, WEX_ALIGN, WEX_NR2_MODEL_BYTES},
        {17u, 0u, h3->table_offset, EXPERTS * sizeof(WexEntry)},
        {16u, 0u, old_bytes, WEX_ALIGN}};
    for (int b = 0; b < 3; b++) {
        PUT(blobs[b].kind, uint32_t); PUT(blobs[b].id, uint32_t); PUT(blobs[b].offset, uint64_t); PUT(blobs[b].size, uint64_t);
        wex4_sha256(out + blobs[b].offset, (size_t)blobs[b].size, x + at); at += 32; PUT(0, uint64_t);
    }
    for (int i = 0; i < EXPERTS; i++) {
        PUT(table[i].file_offset, uint64_t); PUT(table[i].total_bytes, uint32_t); PUT(RAW, uint32_t);
        PUT(WEX4_CODEC_WNR3, uint16_t); PUT(WEX4_REC_EXPERT | (cz ? WEX4_REC_VALUE_EXACT : 0), uint16_t);
        PUT(1, uint32_t); PUT(i, uint32_t); PUT(i * 6, uint32_t); PUT(6, uint32_t);
        wex4_sha256(out + table[i].file_offset, table[i].total_bytes, x + at); at += 32; PUT(0, uint32_t);
    }
    for (int i = 0, name_at = 0; i < EXPERTS * 6; i++) {
        int m = i % 3, weight = i % 6 >= 3;
        PUT(name_at, uint32_t); PUT(strlen(names[i]), uint16_t); PUT(weight ? 3 : 6, uint8_t); PUT(2, uint8_t);
        PUT(ROWS, uint64_t); PUT(weight ? WEIGHT_COLS : SCALE_COLS, uint64_t);
        for (int d = 2; d < 8; d++) PUT(0, uint64_t);
        PUT(i / 6, uint32_t); PUT(1, uint32_t);
        PUT(weight ? 3 * SCALE_M + m * WEIGHT_M : m * SCALE_M, uint64_t); PUT(weight ? WEIGHT_M : SCALE_M, uint64_t);
        name_at += (int)strlen(names[i]);
    }
    for (int i = 0; i < EXPERTS * 6; i++) {   /* name order: w1.scale < w1.weight < w2.scale ... */
        static const int by_name[6] = {0, 3, 1, 4, 2, 5};
        PUT(i / 6 * 6 + by_name[i % 6], uint32_t);
    }
    for (int i = 0; i < EXPERTS * 6; i++) { memcpy(x + at, names[i], strlen(names[i])); at += strlen(names[i]); }
    assert(at == index_bytes);
    x = out; at = 0;
    memset(out, 0, WEX_ALIGN);
    PUT(WEX4_MAGIC, uint32_t); PUT(4, uint16_t); PUT(0, uint16_t); PUT(cz, uint64_t); PUT(0, uint64_t);
    PUT(total, uint64_t); PUT(index_offset, uint64_t); PUT(index_bytes, uint64_t);
    PUT(3, uint32_t); PUT(EXPERTS, uint32_t); PUT(EXPERTS * 6, uint32_t); PUT(1, uint32_t); PUT(EXPERTS, uint32_t);
    PUT(RAW, uint32_t); PUT(name_bytes, uint64_t); PUT(0, uint64_t); PUT((uint64_t)EXPERTS * RAW, uint64_t);
    wex4_sha256(out + index_offset, index_bytes, out + at);
    wex4_sha256(out, WEX4_HEADER_HASHED, out + WEX4_HEADER_HASHED);
#undef PUT
    for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", out[WEX4_HEADER_HASHED + i]);
    *bytes = total;
    return out;
}

static void check_host(const int want[EXPERTS], int active);
static ColiExpertStore *open_store(void);

/* the batch first-read check (the WEX batch path's): a corrupted record is refused for the
 * process (not hashed again, even when reread intact), a passed one is not hashed again, an
 * unread one is skipped; two or more due records hash on parallel threads only when asked */
static void check_first_read_n(const Container *c) {
    ColiExpertStore *store = open_store();
    ColiExpertKey keys[EXPERTS];
    const WexEntry *entries[EXPERTS];
    unsigned char *records[EXPERTS];
    int ok[EXPERTS];
    for (int i = 0; i < EXPERTS; i++) {
        keys[i] = (ColiExpertKey){0, i};
        entries[i] = coli_v4_expert_store_wex_entry(store, keys[i], RAW);
        records[i] = _aligned_malloc(RAW, WEX_ALIGN);
        assert(entries[i] && records[i] && !coli_v4_expert_store_wex_read(store, keys[i], entries[i], records[i]));
        ok[i] = i < 2;                                      /* records 2 and 3 "not read": left alone */
    }
    records[1][c->used[1] / 2] ^= 0x10u;
    int team = coli_v4_wex_first_read_ok_n(store, EXPERTS, keys, entries, records, NULL, ok, 0);
    assert(team == 1);                                      /* two due, serial asked (prefetch) */
    assert(ok[0] && !ok[1] && !ok[2] && !ok[3]);
    records[1][c->used[1] / 2] ^= 0x10u;                   /* reread intact: still refused, not hashed */
    records[3][c->used[3] / 2] ^= 0x10u;                   /* first read of 3, corrupted: refused */
    records[0][c->used[0] / 2] ^= 0x10u;                   /* 0 passed before: not hashed again */
    for (int i = 0; i < EXPERTS; i++) ok[i] = 1;
    team = coli_v4_wex_first_read_ok_n(store, EXPERTS, keys, entries, records, NULL, ok, 1);
#ifdef _OPENMP
    assert(team == 2);                                      /* 2 and 3 due: the region really forked */
#endif
    assert(ok[0] && !ok[1] && ok[2] && !ok[3]);
    assert(!coli_v4_wex_first_read_ok(store, keys[1], entries[1], records[1], 0));   /* single check too */
    for (int i = 0; i < EXPERTS; i++) _aligned_free(records[i]);
    store->ops->destroy(store);
}

/* one batch of every record, whole (stripe 0) or head from the routed drive + tail from the next
 * one (stripe 1); reported[i]: the ok[] read_wait must give (1 whole, 2 striped); good[i]: 1 = record
 * i holds the bytes of w4 and passes that layout's first-read check, 0 = it differs and fails it */
static void batch_check(ColiExpertStore *store, const unsigned char *w4, int stripe,
                        const int reported[EXPERTS], const int good[EXPERTS]) {
    ColiExpertKey keys[EXPERTS];
    const WexEntry *entries[EXPERTS];
    unsigned char *dst[EXPERTS];
    int ok[EXPERTS];
    for (int i = 0; i < EXPERTS; i++) {
        keys[i] = (ColiExpertKey){0, i};
        entries[i] = coli_v4_expert_store_wex_entry(store, keys[i], RAW);
        assert(entries[i] && entries[i]->total_bytes >= 4 * WEX_ALIGN);   /* room for a 1:3 cut */
        dst[i] = _aligned_malloc(RAW, WEX_ALIGN);
        assert(dst[i]);
        memset(dst[i], 0xee, RAW);
    }
    void *io = coli_v4_expert_store_wex_read_issue(store, EXPERTS, keys, entries, dst, stripe);
    assert(io && coli_v4_expert_store_wex_read_wait(io, ok) == EXPERTS);
    for (int i = 0; i < EXPERTS; i++) {
        assert(ok[i] == reported[i]);
        int same = !memcmp(dst[i], w4 + entries[i]->file_offset, entries[i]->total_bytes);
        assert(same == good[i]);
        assert(coli_v4_wex_first_read_ok(store, keys[i], entries[i], dst[i], ok[i] == 2) == good[i]);
        _aligned_free(dst[i]);
    }
}

static void stripe_batch(ColiExpertStore *store, const unsigned char *w4, int stripe, int reported,
                         const int good[EXPERTS]) {
    int each[EXPERTS];
    for (int i = 0; i < EXPERTS; i++) each[i] = reported;
    batch_check(store, w4, stripe, each, good);
}

/* <dir>\v4_flash_experts.wex = w4 with the last byte (in a striped read: the tail) of every
 * record in the damaged mask flipped */
static void write_mirror(const char *dir, unsigned char *w4, size_t bytes, const Container *c, unsigned damaged) {
    char path[700];
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", dir);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < EXPERTS; i++)
            if (damaged >> i & 1u) w4[c->entries[i].file_offset + c->entries[i].total_bytes - 1u] ^= 1u;
        if (!pass) write_file(path, w4, bytes);
    }
}

/* V4_WEX_STRIPE: a demand batch reads part of each record from a second drive. The first-read
 * SHA-256 is kept per layout, so a damaged mirror tail is caught even after the whole read of
 * the same record passed. Covers the dual route (head primary, tail mirror), a failing tail
 * (reread from the routed drive, reported whole), COLI_WEX_WEIGHTS cuts with the head on the
 * mirror and the tail on the primary, and three drives where the last one's tail wraps to the
 * primary. Without COLI_MODEL_MIRROR the dual route sends every record to the primary. */
static void check_stripe(unsigned char *w4, size_t bytes, const Container *c) {
    char m1[600], m2[600], list[1300], path[700];
    int all[EXPERTS] = {1, 1, 1, 1}, one_bad[EXPERTS] = {1, 0, 1, 1};
    int last_bad[EXPERTS] = {1, 1, 1, 0}, two_bad[EXPERTS] = {1, 1, 0, 1};
    ColiExpertStore *store;
    snprintf(m1, sizeof m1, "%s\\mirror", directory);
    snprintf(m2, sizeof m2, "%s\\mirror2", directory);
    assert(!_mkdir(m1) && !_mkdir(m2));
    setenv("COLI_WEX_EXTRA_MIRRORS", m1, 1);
    setenv("COLI_WEX_VERIFY", "1", 1);
    write_mirror(m1, w4, bytes, c, 0);
    store = open_store();
    HANDLE mirror = coli_v4_test_wex_mirror(store, 0, INVALID_HANDLE_VALUE);   /* every tail read fails */
    stripe_batch(store, w4, 1, 1, all);
    coli_v4_test_wex_mirror(store, 0, mirror);
    stripe_batch(store, w4, 1, 2, all);
    stripe_batch(store, w4, 0, 1, all);
    store->ops->destroy(store);
    write_mirror(m1, w4, bytes, c, 1u << 1);   /* record 1 on the primary: only its striped tail is damaged */
    store = open_store();
    stripe_batch(store, w4, 0, 1, all);
    stripe_batch(store, w4, 1, 2, one_bad);
    store->ops->destroy(store);
    /* 1:3 -> records 0, 1 on the primary, 2, 3 on the mirror: record 3's whole read is damaged, its
     * striped read (3/4 head from the mirror, tail from the primary) is not */
    setenv("COLI_WEX_WEIGHTS", "1,3", 1);
    write_mirror(m1, w4, bytes, c, 1u << 1 | 1u << 3);
    store = open_store();
    stripe_batch(store, w4, 0, 1, last_bad);
    stripe_batch(store, w4, 1, 2, one_bad);
    store->ops->destroy(store);
    /* three drives 1:1:2 -> records 0, 1 primary, 2 mirror, 3 mirror2; record 2's tail comes from
     * mirror2 (damaged), record 3's wraps to the primary */
    snprintf(list, sizeof list, "%s;%s", m1, m2);
    setenv("COLI_WEX_EXTRA_MIRRORS", list, 1);
    setenv("COLI_WEX_WEIGHTS", "1,1,2", 1);
    write_mirror(m1, w4, bytes, c, 0);
    write_mirror(m2, w4, bytes, c, 1u << 2 | 1u << 3);
    store = open_store();
    stripe_batch(store, w4, 0, 1, last_bad);
    stripe_batch(store, w4, 1, 2, two_bad);
    store->ops->destroy(store);
    unsetenv("COLI_WEX_EXTRA_MIRRORS");
    unsetenv("COLI_WEX_WEIGHTS");
    unsetenv("COLI_WEX_VERIFY");
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", m1);
    assert(remove(path) == 0 && _rmdir(m1) == 0);
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", m2);
    assert(remove(path) == 0 && _rmdir(m2) == 0);
    puts("WEX4 striped batch reads: same bytes as whole reads on 2 and 3 drives, head on the primary or "
         "the mirror; a failed tail is reread whole; a damaged tail fails the striped layout's first-read "
         "SHA-256 after the whole layout passed.");
}

/* the sub-reads one record read goes out as: up to n a part, one a page when a part has fewer
 * pages; cut = 1: head (half, on a page) from the routed drive, tail from the mirror (1:1 shares) */
static int sub_count(uint32_t bytes, int n) {
    uint32_t pages = bytes / WEX_ALIGN;
    return pages >= (uint32_t)n ? n : pages ? (int)pages : 1;
}
static uint64_t sub_reads(const Container *c, int n, int cut) {
    uint64_t sum = 0;
    for (int i = 0; i < EXPERTS; i++) {
        uint32_t bytes = c->entries[i].total_bytes, head = cut ? (bytes / 2u) & ~(WEX_ALIGN - 1u) : bytes;
        if (head < WEX_ALIGN || head >= bytes) head = bytes;
        sum += (uint64_t)sub_count(head, n) + (head < bytes ? (uint64_t)sub_count(bytes - head, n) : 0u);
    }
    return sum;
}

/* every record through the single GPU-direct read: the bytes of w4 (it stays on the routed drive) */
static void single_reads(ColiExpertStore *store, const unsigned char *w4) {
    unsigned char *dst = _aligned_malloc(RAW, WEX_ALIGN);
    assert(dst);
    for (int i = 0; i < EXPERTS; i++) {
        ColiExpertKey key = {0, i};
        const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, RAW);
        memset(dst, 0xee, RAW);
        assert(entry && !coli_v4_expert_store_wex_read(store, key, entry, dst));
        assert(!memcmp(dst, w4 + entry->file_offset, entry->total_bytes));
        assert(dst[entry->total_bytes] == 0xee);   /* nothing past the record */
    }
    _aligned_free(dst);
}

/* V4_WEX_SUBREADS: each drive's part of a record goes out as up to N sub-reads in flight, cut on
 * pages and reassembled in place. Without a mirror: the batch, the single GPU-direct read and the
 * host path give the same bytes as one read, N reads a record (fewer pages: one a page). With a
 * mirror the batch (even stripe 0) and the host path cut each record across both drives and report
 * it striped, so a damaged mirror tail fails the striped layout's first-read SHA-256 although the
 * whole layout is fine; the next batch reads that record whole. Unset: one read a record, as before. */
static void check_subreads(unsigned char *w4, size_t bytes, const Container *c) {
    int all[EXPERTS] = {1, 1, 1, 1}, one_bad[EXPERTS] = {1, 0, 1, 1}, whole[EXPERTS] = {1, 1, 1, 1};
    int cut[EXPERTS] = {2, 2, 2, 2}, cut_but_1[EXPERTS] = {2, 1, 2, 2};
    char m1[600], path[700];
    uint64_t before;
    setenv("COLI_WEX_VERIFY", "1", 1);
    ColiExpertStore *store = open_store();
    for (int t = 0; t < 4; t++) {
        int n = (int[]){1, 3, 4, 16}[t];   /* 16: more than a record's pages, one a page */
        char text[8];
        snprintf(text, sizeof text, "%d", n);
        setenv("V4_WEX_SUBREADS", text, 1);
        before = coli_v4_test_wex_sub_reads();
        batch_check(store, w4, 0, whole, all);
        single_reads(store, w4);
        assert(coli_v4_test_wex_sub_reads() - before == 2 * sub_reads(c, n, 0));
    }
    setenv("V4_WEX_SUBREADS", "3", 1);
    before = coli_v4_test_wex_sub_reads();
    check_host(all, 1);
    assert(coli_v4_test_wex_sub_reads() - before == sub_reads(c, 3, 0));
    store->ops->destroy(store);
    /* a mirror whose copy of record 1 has a damaged last byte (in the tail of a cut read) */
    snprintf(m1, sizeof m1, "%s\\mirror", directory);
    assert(!_mkdir(m1));
    setenv("COLI_WEX_EXTRA_MIRRORS", m1, 1);
    write_mirror(m1, w4, bytes, c, 1u << 1);
    unsetenv("V4_WEX_SUBREADS");
    store = open_store();
    before = coli_v4_test_wex_sub_reads();
    batch_check(store, w4, 0, whole, all);                 /* unset: whole from the primary, as before */
    assert(coli_v4_test_wex_sub_reads() - before == EXPERTS);
    store->ops->destroy(store);
    setenv("V4_WEX_SUBREADS", "4", 1);
    store = open_store();
    before = coli_v4_test_wex_sub_reads();
    batch_check(store, w4, 0, cut, one_bad);               /* cut although stripe 0: record 1 refused */
    assert(coli_v4_test_wex_sub_reads() - before == sub_reads(c, 4, 1));
    batch_check(store, w4, 0, cut_but_1, all);             /* its striped layout failed: read whole */
    single_reads(store, w4);
    store->ops->destroy(store);
    before = coli_v4_test_wex_sub_reads();
    check_host(one_bad, 1);                                /* host path cut too: record 1 to safetensors */
    assert(coli_v4_test_wex_sub_reads() - before == sub_reads(c, 4, 1));
    write_mirror(m1, w4, bytes, c, 0);
    check_host(all, 1);
    unsetenv("V4_WEX_SUBREADS");
    unsetenv("COLI_WEX_EXTRA_MIRRORS");
    unsetenv("COLI_WEX_VERIFY");
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", m1);
    assert(remove(path) == 0 && _rmdir(m1) == 0);
    puts("V4_WEX_SUBREADS: 1/3/4/16 sub-reads a part reassemble to the same bytes on the batch, single "
         "and host paths; with a mirror each record is cut across both drives and judged by the striped "
         "layout's first-read SHA-256 (a damaged mirror tail refused, then read whole); unset = one read.");
}

/* The I/O gate on real reads (V4_NP_MAX_INFLIGHT_PER_DRIVE / _DEMAND_FIRST / _CANCEL), records
 * 0, 1 on the primary and 2, 3 on the mirror (COLI_WEX_WEIGHTS 1,3). The trace keeps the most
 * prefetch reads in flight per drive and counts each gate result. */
enum { IO_DEMAND, IO_PREFETCH, IO_WAIT, IO_DROP };
static int io_events[8][4], io_most[4];
static void io_trace(int event, int drive, int layer, int expert, int prefetch, int demand) {
    (void)layer; (void)expert; (void)demand;   /* called under the gate's mutex */
    __atomic_add_fetch(&io_events[event][drive], 1, __ATOMIC_RELEASE);
    if (prefetch > io_most[drive]) io_most[drive] = prefetch;
}
static int io_count(int event, int drive) { return __atomic_load_n(&io_events[event][drive], __ATOMIC_ACQUIRE); }
static void io_reset(int cap, int first, int cancel) {
    coli_v4_test_io_setup(cap, first, cancel);
    memset(io_events, 0, sizeof io_events);
    memset(io_most, 0, sizeof io_most);
}
static void io_await(int event, int drive, int want) {
    for (int i = 0; i < 5000 && io_count(event, drive) < want; i++) Sleep(1);
    assert(io_count(event, drive) >= want);
}

/* a prefetch reader of layer 0 (queued when it had routed `seen` times): one batch of every
 * record, or (lookup >= 0) a host-path lookup of that expert */
typedef struct {
    ColiExpertStore *store;
    const WexEntry **entries;
    unsigned char **dst;
    uint32_t seen;
    int lookup, ok[EXPERTS], count, dropped;
} IoReader;
static void *io_reader(void *arg) {
    IoReader *r = arg;
    coli_v4_io_prefetch_item(0, -1, r->seen);
    if (r->lookup >= 0) {
        ColiExpertView view = {0};
        coli_v4_expert_lookup_set_prefetch(1);
        r->count = coli_expert_lookup(r->store, (ColiExpertKey){0, r->lookup}, &view);
        coli_v4_expert_lookup_set_prefetch(0);
        if (!r->count) coli_expert_release(r->store, &view);
    } else {
        ColiExpertKey keys[EXPERTS];
        for (int i = 0; i < EXPERTS; i++) keys[i] = (ColiExpertKey){0, i};
        void *io = coli_v4_expert_store_wex_read_issue(r->store, EXPERTS, keys, r->entries, r->dst, 0);
        assert(io);
        r->count = coli_v4_expert_store_wex_read_wait(io, r->ok);
    }
    r->dropped = coli_v4_io_dropped(0, r->lookup);   /* a batch (-1): any of its records */
    coli_v4_io_prefetch_item(-1, -1, 0);
    return NULL;
}

static void check_io_gate(unsigned char *w4, size_t bytes, const Container *c) {
    char m1[600], path[700];
    const WexEntry *entries[EXPERTS];
    unsigned char *dst[EXPERTS];
    ColiExpertKey keys[EXPERTS];
    unsigned long long s[5];
    pthread_t thread;
    IoReader r;
    snprintf(m1, sizeof m1, "%s\\mirror", directory);
    assert(!_mkdir(m1));
    write_mirror(m1, w4, bytes, c, 0);
    setenv("COLI_WEX_EXTRA_MIRRORS", m1, 1);
    setenv("COLI_WEX_WEIGHTS", "1,3", 1);
    setenv("COLI_WEX_VERIFY", "1", 1);
    coli_v4_test_io_trace = io_trace;
    ColiExpertStore *store = open_store();
    for (int i = 0; i < EXPERTS; i++) {
        keys[i] = (ColiExpertKey){0, i};
        entries[i] = coli_v4_expert_store_wex_entry(store, keys[i], RAW);
        dst[i] = _aligned_malloc(RAW, WEX_ALIGN);
        assert(entries[i] && dst[i]);
    }
    /* cap 1: a prefetch batch keeps one read in flight per drive, all four land intact */
    io_reset(1, 0, 0);
    r = (IoReader){store, entries, dst, 0, -1, {0}, 0, 0};
    for (int i = 0; i < EXPERTS; i++) memset(dst[i], 0xee, RAW);
    io_reader(&r);
    assert(r.count == EXPERTS && !r.dropped);
    for (int i = 0; i < EXPERTS; i++) {
        assert(r.ok[i] == 1 && !memcmp(dst[i], w4 + entries[i]->file_offset, entries[i]->total_bytes));
        assert(coli_v4_wex_first_read_ok(store, keys[i], entries[i], dst[i], 0));
    }
    assert(io_count(IO_PREFETCH, 0) == 2 && io_count(IO_PREFETCH, 1) == 2);
    assert(io_most[0] == 1 && io_most[1] == 1);
    /* demand first + cancel: a demand read in flight on each drive holds the batch; layer 0 routes
     * {1, 3}: those two go at once as demand reads, 0 and 2 are dropped unread */
    store->ops->destroy(store);
    store = open_store();   /* a fresh first-read state */
    for (int i = 0; i < EXPERTS; i++) assert((entries[i] = coli_v4_expert_store_wex_entry(store, keys[i], RAW)));
    io_reset(0, 1, 1);
    assert(coli_v4_test_io_begin(0, 0, 0, 1) == IO_DEMAND && coli_v4_test_io_begin(1, 0, 0, 1) == IO_DEMAND);
    r = (IoReader){store, entries, dst, coli_v4_io_route_seq(0), -1, {0}, 0, 0};
    for (int i = 0; i < EXPERTS; i++) memset(dst[i], 0xee, RAW);
    assert(!pthread_create(&thread, NULL, io_reader, &r));
    io_await(IO_WAIT, 0, 1);   /* the batch blocks on the gate, holding no read */
    assert(io_count(IO_PREFETCH, 0) + io_count(IO_PREFETCH, 1) == 0);
    Sleep(20);
    coli_v4_io_routed(0, (int[]){1, 3}, 2);
    assert(!pthread_join(thread, NULL));
    assert(coli_v4_test_io_wait_ms() >= 60.0);   /* each of the 4 parts held >= 20 ms, summed */
    assert(r.count == 2 && r.dropped && !r.ok[0] && r.ok[1] == 1 && !r.ok[2] && r.ok[3] == 1);
    for (int i = 0; i < EXPERTS; i++) {
        int read = i == 1 || i == 3;
        if (read) assert(!memcmp(dst[i], w4 + entries[i]->file_offset, entries[i]->total_bytes));
        else for (size_t j = 0; j < RAW; j++) assert(dst[i][j] == 0xee);   /* nothing wrote it */
    }
    coli_v4_test_io_stats(s);
    assert(s[3] == 2 && s[4] == 2 && s[1] == 0);   /* promoted, dropped, no prefetch read issued */
    /* a dropped record's first-read SHA-256 never ran: a corrupted first read of 0 still fails,
     * an intact one of 2 still passes */
    memcpy(dst[0], w4 + entries[0]->file_offset, entries[0]->total_bytes);
    dst[0][c->used[0] / 2] ^= 0x10u;
    assert(!coli_v4_wex_first_read_ok(store, keys[0], entries[0], dst[0], 0));
    memcpy(dst[2], w4 + entries[2]->file_offset, entries[2]->total_bytes);
    assert(coli_v4_wex_first_read_ok(store, keys[2], entries[2], dst[2], 0));
    /* the GPU-direct record read of a dropped guess: -1 at once, no read; a picked one reads */
    coli_v4_io_prefetch_item(0, -1, r.seen);   /* the context only, as a batch sets it */
    assert(coli_v4_expert_store_wex_read(store, keys[0], entries[0], dst[0]) == -1 && coli_v4_io_dropped(0, 0));
    assert(coli_v4_io_prefetch_item(0, 0, r.seen) == 1);   /* v4_np_work's check: dropped already */
    assert(coli_v4_io_prefetch_item(0, 1, r.seen) == 0);
    assert(!coli_v4_expert_store_wex_read(store, keys[1], entries[1], dst[1]) && !coli_v4_io_dropped(0, 1));
    coli_v4_test_io_stats(s);
    assert(s[4] == 3);   /* 0 and 2 in the batch, 0 once more on this thread, not again by its check */
    coli_v4_io_prefetch_item(-1, -1, 0);
    /* the host path (expert 2, its first read passed above; 0's was refused, so the host path
     * would read 0 from safetensors): a prefetch lookup holds its RAM slot while it waits on the
     * gate; layer 0 routes without it: the lookup fails quietly, the slot is free, no safetensors
     * read, and a demand lookup then serves the WNR3 bytes */
    r = (IoReader){store, entries, dst, coli_v4_io_route_seq(0), 2, {0}, 0, 0};
    memset(io_events, 0, sizeof io_events);
    assert(!pthread_create(&thread, NULL, io_reader, &r));
    io_await(IO_WAIT, 1, 1);
    coli_v4_io_routed(0, (int[]){3}, 1);
    assert(!pthread_join(thread, NULL));
    assert(r.count == -1 && r.dropped);
    coli_v4_test_io_end(0, IO_DEMAND);
    coli_v4_test_io_end(1, IO_DEMAND);
    uint64_t decoded = coli_wex_cuda_decode_count();
    ColiExpertView view = {0};
    assert(!coli_expert_lookup(store, keys[2], &view));
    assert(!memcmp(view.gate.scales, wnr3[2], SCALE_M) && !memcmp(view.gate.data, wnr3[2] + 3 * SCALE_M, WEIGHT_M));
    coli_expert_release(store, &view);
    assert(coli_wex_cuda_decode_count() - decoded == 1);
    /* V4_WEX_SUBREADS cuts a prefetch batch across both drives (records 0, 1: head on the primary,
     * tail on the mirror; 2, 3 the other way). Demand first on the mirror only: the primary's parts
     * go, the mirror's wait; layer 0 routes {1, 3}: record 0's tail is dropped after its head was
     * read, so record 0 is dropped, not reread whole off the gate as a failed tail would be */
    setenv("V4_WEX_SUBREADS", "2", 1);
    io_reset(0, 1, 1);
    assert(coli_v4_test_io_begin(1, 0, 0, 1) == IO_DEMAND);
    r = (IoReader){store, entries, dst, coli_v4_io_route_seq(0), -1, {0}, 0, 0};
    for (int i = 0; i < EXPERTS; i++) memset(dst[i], 0xee, RAW);
    assert(!pthread_create(&thread, NULL, io_reader, &r));
    io_await(IO_WAIT, 1, 1);   /* the primary's parts landed, the batch blocks on the mirror */
    coli_v4_io_routed(0, (int[]){1, 3}, 2);
    assert(!pthread_join(thread, NULL));
    assert(r.count == 2 && r.dropped && !r.ok[0] && r.ok[1] == 2 && !r.ok[2] && r.ok[3] == 2);
    for (int i = 1; i < EXPERTS; i += 2)
        assert(!memcmp(dst[i], w4 + entries[i]->file_offset, entries[i]->total_bytes));
    coli_v4_test_io_end(1, IO_DEMAND);
    /* the host path's tail probe (expert 3: head on the mirror, tail on the primary): demand first
     * on the primary would hold the tail, so the record is read whole from the mirror; nothing was
     * held, so the gate's held count stays 0 */
    io_reset(0, 1, 1);
    assert(coli_v4_test_io_begin(0, 0, 0, 1) == IO_DEMAND);
    r = (IoReader){store, entries, dst, coli_v4_io_route_seq(0), 3, {0}, 0, 0};
    io_reader(&r);
    coli_v4_test_io_stats(s);
    assert(r.count == 0 && !r.dropped && s[1] == 1 && s[2] == 0);   /* one prefetch read, none held */
    coli_v4_test_io_end(0, IO_DEMAND);
    unsetenv("V4_WEX_SUBREADS");
    store->ops->destroy(store);
    io_reset(0, 0, 0);
    coli_v4_test_io_trace = NULL;
    for (int i = 0; i < EXPERTS; i++) _aligned_free(dst[i]);
    unsetenv("COLI_WEX_EXTRA_MIRRORS");
    unsetenv("COLI_WEX_WEIGHTS");
    unsetenv("COLI_WEX_VERIFY");
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", m1);
    assert(remove(path) == 0 && _rmdir(m1) == 0);
    puts("I/O gate on WEX reads: cap 1 keeps one prefetch read in flight per drive; demand first holds "
         "a batch until the router picks; picked records read as demand, unpicked ones dropped unread "
         "(first-read SHA-256 untouched, host slot freed, no safetensors fallback).");
}

static void check_views(ColiExpertStore *store, const int want[EXPERTS]);
static int fill(ColiExpertStore *store, int expert, unsigned char *staging, unsigned char *buffer);
static int cached(ColiExpertStore *store, int expert, unsigned char **record);
static int real_helper;

/* V4_WEX_SHA_ASYNC=1 on a WEX4 container whose record 1 decodes fine but has the wrong SHA-256 in the
 * index (bad = 1) or on an intact one (bad = 0): the host path (hash of a copy, overlapped with the
 * in-place decode), the tier prewarm's deferred verdicts and the GPU-direct fill (hash overlapped with
 * the device decode) refuse exactly what the inline check refuses, after the decode, before publishing. */
static void check_sha_async(int bad) {
    int all[EXPERTS] = {1, 1, 1, 1}, one_bad[EXPERTS] = {1, 0, 1, 1};
    const int *want = bad ? one_bad : all;
    setenv("V4_WEX_SHA_ASYNC", "1", 1);
    ColiExpertStore *store = open_store();                 /* joined before the read returns (record 1 */
    check_views(store, want);                              /* is decoded, then refused: no decode count) */
    store->ops->destroy(store);
    store = open_store();                                  /* deferred: every read serves its decode ... */
    assert(!coli_v4_test_sha_defer(store, 1));
    check_views(store, all);
    assert(coli_v4_test_sha_defer(store, 0) == bad);       /* ... until the drain refuses record 1: */
    check_views(store, want);                              /* its RAM copy went, reread from safetensors */
    store->ops->destroy(store);
    coli_v4_test_sha_busy(1);                              /* every hash thread busy: the readers hash */
    store = open_store();                                  /* inline, so even deferred, record 1 is */
    assert(!coli_v4_test_sha_defer(store, 1));             /* refused at its read, nothing left to drain */
    check_views(store, want);
    assert(coli_v4_test_sha_defer(store, 0) == 0);
    store->ops->destroy(store);
    coli_v4_test_sha_busy(0);
    if (!real_helper) {
        unsigned char *staging = _aligned_malloc(RAW, WEX_ALIGN), *buffer = malloc(RAW);
        assert(staging && buffer);
        setenv("COLI_WEX_GPU_DIRECT", "1", 1);
        setenv("COLI_WEX_RAM_CACHE_GB", "0.01", 1);
        store = open_store();
        for (int round = 0; round < 2; round++)            /* the second: rejected before, never hashed */
            for (int i = 0; i < EXPERTS; i++) {
                memset(buffer, 0xee, RAW);
                int status = fill(store, i, staging, buffer);
                assert(status == (want[i] ? 0 : -1) && cached(store, i, NULL) == want[i]);
                if (want[i]) assert(!memcmp(buffer, wnr3[i], RAW));
            }
        store->ops->destroy(store);
        /* the RAM cache filled as coli_v4_wex_crc_prewarm fills it (read, never hashed): a hit is
         * hashed before it is served, so a bad record is refused and leaves the cache */
        store = open_store();
        for (int i = 0; i < EXPERTS; i++) {
            ColiExpertKey key = {0, i};
            const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, RAW);
            unsigned char *r = NULL;
            int hit = 0, slot = entry ? coli_v4_wex_crc_get(store, key, entry, &r, &hit) : -1;
            assert(slot >= 0 && !hit);
            coli_v4_wex_crc_done(store, slot, !coli_v4_expert_store_wex_read(store, key, entry, r));
            assert(cached(store, i, NULL));
        }
        for (int i = 0; i < EXPERTS; i++) {
            memset(buffer, 0xee, RAW);
            assert(fill(store, i, staging, buffer) == (want[i] ? 0 : -1) && cached(store, i, NULL) == want[i]);
            if (want[i]) assert(!memcmp(buffer, wnr3[i], RAW));
        }
        store->ops->destroy(store);
        unsetenv("COLI_WEX_GPU_DIRECT");
        unsetenv("COLI_WEX_RAM_CACHE_GB");
        _aligned_free(staging);
        free(buffer);
    }
    unsetenv("V4_WEX_SHA_ASYNC");
    printf("V4_WEX_SHA_ASYNC=1, %s: host path, deferred prewarm verdicts%s refuse what the inline check "
           "refuses.\n", bad ? "wrong SHA-256 for record 1" : "intact container",
           real_helper ? "" : " and the GPU-direct fill");
}

/* The same container as WEX4: the engine must serve the same bytes and fall back in the same cases. */
static void run_wex4(const Container *c, int cz) {
    int all[EXPERTS] = {1, 1, 1, 1}, none[EXPERTS] = {0}, two_bad[EXPERTS] = {0, 0, 1, 1}, one_bad[EXPERTS] = {1, 0, 1, 1};
    char root[65], other[65];
    size_t bytes;
    unsigned char *w4 = upgrade_wex4(c->data, c->bytes, &bytes, root);
    assert(!memcmp(w4 + WEX_ALIGN, c->data + WEX_ALIGN, c->bytes - WEX_ALIGN));   /* nothing moved */
    write_file(wex_path, w4, bytes);
    check_host(all, 1);
    setenv("COLI_WEX_ROOT", root, 1);                      /* the right root pinned */
    check_host(all, 1);
    write_file(wex_path, c->data, c->bytes);               /* W06: the legacy form has no root: refused */
    check_host(none, 0);
    w4[c->entries[1].file_offset + c->used[1] / 2] ^= 0x10u;   /* a flipped byte inside record 1 */
    write_file(wex_path, w4, bytes);
    check_host(one_bad, 1);
    w4[c->entries[1].file_offset + c->used[1] / 2] ^= 0x10u;
    w4[bytes - WEX_ALIGN + 20] ^= 1u;                      /* a bit of the index: the open fails */
    write_file(wex_path, w4, bytes);
    check_host(none, 0);
    w4[bytes - WEX_ALIGN + 20] ^= 1u;
    w4[30] ^= 1u;                                          /* a bit of the header page */
    write_file(wex_path, w4, bytes);
    check_host(none, 0);
    w4[30] ^= 1u;
    free(w4);
    unsigned char *copy = swapped_copy(c);                 /* records 0 and 1 swapped: the owner check */
    w4 = upgrade_wex4(copy, c->bytes, &bytes, other);
    write_file(wex_path, w4, bytes);
    check_host(none, 0);                                   /* another container than the pinned one */
    unsetenv("COLI_WEX_ROOT");
    check_host(two_bad, 1);
    free(copy); free(w4);
    w4 = upgrade_wex4(c->data, c->bytes, &bytes, root);
    {   /* a wrong SHA-256 for record 1 in the index (hashes recomputed): only the first-read check sees it */
        size_t index_offset = c->bytes + WEX_ALIGN, index_bytes;
        memcpy(&index_bytes, w4 + 40, 8);
        w4[index_offset + 3 * WEX4_BLOB_BYTES + WEX4_RECORD_BYTES + 36] ^= 1u;
        wex4_sha256(w4 + index_offset, index_bytes, w4 + 96);
        wex4_sha256(w4, WEX4_HEADER_HASHED, w4 + WEX4_HEADER_HASHED);
        write_file(wex_path, w4, bytes);
        check_host(all, 1);                                /* verification off: level 1 alone accepts it */
        setenv("COLI_WEX_VERIFY", "1", 1);
        check_host(one_bad, 1);                            /* on: that record falls back to safetensors */
        check_sha_async(1);
        free(w4);
        w4 = upgrade_wex4(c->data, c->bytes, &bytes, root);
        write_file(wex_path, w4, bytes);
        check_host(all, 1);                                /* on, intact container: every record passes */
        check_sha_async(0);
        check_first_read_n(c);
        unsetenv("COLI_WEX_VERIFY");
        check_stripe(w4, bytes, c);
        check_subreads(w4, bytes, c);
        check_io_gate(w4, bytes, c);
    }
    setenv("DSV4_CUDA_MOE_MASKED", "1", 1);                /* canonical zero on the masked path: refused */
    check_host(cz ? none : all, !cz);
    unsetenv("DSV4_CUDA_MOE_MASKED");
    free(w4);
    printf("WEX4%s on the engine store: same bytes served; pinned root (a legacy file refused under it), first-read SHA-256, damaged header / index, corrupted and "
           "foreign records and the masked path handled as for WEX3.\n", cz ? " canonical-zero" : "");
}

static ColiExpertStore *open_store(void) {
    char error[256] = {0};
    ColiDeepSeekV4ExpertStoreOptions options = {directory, 1, EXPERTS, (uint64_t)EXPERTS * RAW, -1, UINT64_MAX, 0, 0, 0};
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof error))
        fprintf(stderr, "open failed: %s\n", error);
    assert(store);
    return store;
}

/* every expert through the host path; want[i]: 1 = the WNR3 bytes, 0 = safetensors */
static void check_views(ColiExpertStore *store, const int want[EXPERTS]) {
    for (int i = 0; i < EXPERTS; i++) {
        ColiExpertView view = {0};
        assert(!coli_expert_lookup(store, (ColiExpertKey){0, i}, &view));
        const ColiTensorView *t[3] = {&view.gate, &view.down, &view.up};
        const unsigned char *expect = want[i] ? wnr3[i] : source[i];
        for (int m = 0; m < 3; m++) {
            assert(t[m]->scale_bytes == SCALE_M && t[m]->data_bytes == WEIGHT_M);
            assert(!memcmp(t[m]->scales, expect + m * SCALE_M, SCALE_M));
            assert(!memcmp(t[m]->data, expect + 3 * SCALE_M + m * WEIGHT_M, WEIGHT_M));
        }
        coli_expert_release(store, &view);
    }
}

static void check_host(const int want[EXPERTS], int active) {
    /* the fake helper's host decodes count the experts WNR3 served */
    uint64_t decoded = coli_wex_cuda_decode_count();
    int served = 0;
    ColiExpertStore *store = open_store();
    assert(coli_v4_test_wex_active(store) == active);
    check_views(store, want);
    for (int i = 0; i < EXPERTS; i++) served += want[i];
    assert(coli_wex_cuda_decode_count() - decoded == (uint64_t)served);
    store->ops->destroy(store);
}

static int fill(ColiExpertStore *store, int expert, unsigned char *staging, unsigned char *buffer) {
    unsigned char *scale[3], *weight[3];
    for (int m = 0; m < 3; m++) { scale[m] = buffer + m * SCALE_M; weight[m] = buffer + 3 * SCALE_M + m * WEIGHT_M; }
    return coli_v4_test_wex_fill(store, (ColiExpertKey){0, expert}, staging, RAW, scale, weight);
}

/* 1 when the RAM cache holds the expert's record (a hit), the lease released */
static int cached(ColiExpertStore *store, int expert, unsigned char **record) {
    ColiExpertKey key = {0, expert};
    const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, RAW);
    unsigned char *r = NULL;
    int hit = 0, slot;
    assert(entry);
    slot = coli_v4_wex_crc_get(store, key, entry, &r, &hit);
    assert(slot >= 0);
    if (record) *record = r;
    coli_v4_wex_crc_done(store, slot, hit);   /* a miss's claim is released unfilled */
    return hit;
}

/* real_helper (above), NR3_REAL_HELPER: no check_direct */

static void check_direct(const Container *c) {
    unsigned char *staging = _aligned_malloc(RAW, WEX_ALIGN), *buffer = malloc(RAW), *record;
    assert(staging && buffer);
    setenv("COLI_WEX_GPU_DIRECT", "1", 1);
    setenv("COLI_WEX_RAM_CACHE_GB", "0.01", 1);
    ColiExpertStore *store = open_store();
    for (int i = 0; i < EXPERTS; i++) {   /* misses: disk -> RAM cache -> "device" */
        memset(buffer, 0xee, RAW);
        assert(fill(store, i, staging, buffer) == 0 && !memcmp(buffer, wnr3[i], RAW));
        assert(cached(store, i, NULL));
    }
    /* a cached record that no longer decodes: -1, the mirror untouched, the slot dropped */
    assert(cached(store, 1, &record));
    record[c->used[1] / 2] ^= 0x10u;
    memset(buffer, 0xee, RAW);
    assert(fill(store, 1, staging, buffer) == -1);
    for (size_t j = 0; j < RAW; j++) assert(buffer[j] == 0xee);
    assert(!cached(store, 1, NULL));
    /* the next use rereads the (intact) disk record */
    assert(fill(store, 1, staging, buffer) == 0 && !memcmp(buffer, wnr3[1], RAW));
    assert(cached(store, 1, NULL));
    /* helper unavailable (1): the record's bytes stay cached */
    setenv("FAKE_WEX_UNAVAILABLE", "1", 1);
    assert(fill(store, 2, staging, buffer) == 1);
    unsetenv("FAKE_WEX_UNAVAILABLE");
    assert(cached(store, 2, NULL));
    store->ops->destroy(store);
    /* records 0 and 1 swapped on disk: the owner check refuses both before decoding */
    unsigned char *copy = swapped_copy(c);
    write_file(wex_path, copy, c->bytes);
    store = open_store();
    assert(fill(store, 0, staging, buffer) == -1 && fill(store, 1, staging, buffer) == -1);
    assert(!cached(store, 0, NULL) && !cached(store, 1, NULL));
    assert(fill(store, 2, staging, buffer) == 0 && !memcmp(buffer, wnr3[2], RAW));
    store->ops->destroy(store);
    /* the CZ flag of every record differs from the container's: all refused */
    memcpy(copy, c->data, c->bytes);
    ((WexHeader *)copy)->flags ^= WEX_HEADER_CANONICAL_ZERO;
    write_file(wex_path, copy, c->bytes);
    store = open_store();
    for (int i = 0; i < EXPERTS; i++) assert(fill(store, i, staging, buffer) == -1);
    store->ops->destroy(store);
    write_file(wex_path, c->data, c->bytes);
    free(copy);
    unsetenv("COLI_WEX_GPU_DIRECT");
    unsetenv("COLI_WEX_RAM_CACHE_GB");
    _aligned_free(staging);
    free(buffer);
}

static void run_profile(int cz) {
    Container c;
    build_container(cz, &c);
    write_safetensors();
    write_file(wex_path, c.data, c.bytes);
    int all[EXPERTS] = {1, 1, 1, 1}, none[EXPERTS] = {0}, two_bad[EXPERTS] = {0, 0, 1, 1}, one_bad[EXPERTS] = {1, 0, 1, 1};
    if (cz) assert(memcmp(wnr3, source, sizeof source));   /* CZ bytes show which path served */
    check_host(all, 1);
    /* a flipped byte inside record 1: hash failure, safetensors for that expert */
    c.data[c.entries[1].file_offset + c.used[1] / 2] ^= 0x10u;
    write_file(wex_path, c.data, c.bytes);
    check_host(one_bad, 1);
    c.data[c.entries[1].file_offset + c.used[1] / 2] ^= 0x10u;
    /* the container's canonical-zero flag differs from every record's */
    ((WexHeader *)c.data)->flags ^= WEX_HEADER_CANONICAL_ZERO;
    write_file(wex_path, c.data, c.bytes);
    check_host(none, 1);
    ((WexHeader *)c.data)->flags ^= WEX_HEADER_CANONICAL_ZERO;
    /* records 0 and 1 swapped: the owner check refuses both */
    unsigned char *copy = swapped_copy(&c);
    write_file(wex_path, copy, c.bytes);
    check_host(two_bad, 1);
    free(copy);
    /* canonical zero on the masked DeepGEMM path: the container is refused at open */
    write_file(wex_path, c.data, c.bytes);
    setenv("DSV4_CUDA_MOE_MASKED", "1", 1);
    check_host(cz ? none : all, !cz);
    unsetenv("DSV4_CUDA_MOE_MASKED");
    if (real_helper) puts("NR3_REAL_HELPER: the direct-fill check needs the fake helper, skipped");
    else check_direct(&c);
    run_wex4(&c, cz);
    write_file(wex_path, c.data, c.bytes);
    printf("WNR3%s on the engine store: host path exact, corrupted / wrong-profile / foreign records and the "
           "masked path fall back to safetensors%s.\n", cz ? " canonical-zero" : "",
           real_helper ? "" : "; direct fill drops a bad cached record and rereads it");
    for (int i = 0; i < EXPERTS; i++) free(c.records[i]);
    free(c.data);
}

#ifdef V4_WEX_PREWARM_TEST
/* Host stand-ins for the fp4 mirror buffers (coli_v4_test_fp4): a tensor is its bytes. At most
 * fake_mirrors mirrors fit: past them the free-VRAM probe answers 0 MB. */
typedef struct { int rows, columns; unsigned char *w, *s; } FakeTensor;
static int fake_live, fake_mirrors = 1 << 20;
static int fake_alloc(Dsv4CudaTensor **t, int rows, int columns, int device) {
    FakeTensor *f = calloc(1, sizeof *f);
    (void)device;
    assert(f && rows > 0 && columns % 32 == 0);
    f->rows = rows; f->columns = columns;
    f->w = malloc((size_t)rows * columns / 2); f->s = malloc((size_t)rows * columns / 32);
    assert(f->w && f->s);
    memset(f->w, 0xcc, (size_t)rows * columns / 2); memset(f->s, 0xcc, (size_t)rows * columns / 32);
    fake_live++;
    *t = (Dsv4CudaTensor *)f;
    return 1;
}
static int fake_buffers(const Dsv4CudaTensor *t, void **w, void **s, int *rows, int *columns) {
    const FakeTensor *f = (const FakeTensor *)t;
    *w = f->w; *s = f->s; *rows = f->rows; *columns = f->columns;
    return 1;
}
static int fake_refill(Dsv4CudaTensor *t, const uint8_t *w, const uint8_t *s, int rows, int columns, int sync) {
    FakeTensor *f = (FakeTensor *)t;
    (void)sync;
    if (f->rows != rows || f->columns != columns) return 0;
    memcpy(f->w, w, (size_t)rows * columns / 2);
    memcpy(f->s, s, (size_t)rows * columns / 32);
    return 1;
}
static int fake_upload(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *s, int rows, int columns, int device) {
    return fake_alloc(t, rows, columns, device) && fake_refill(*t, w, s, rows, columns, 1);
}
static void fake_free(Dsv4CudaTensor *t) {
    FakeTensor *f = (FakeTensor *)t;
    free(f->w); free(f->s); free(f);
    fake_live--;
}
static long long fake_free_mb(int device) {
    (void)device;
    return fake_live / 3 < fake_mirrors ? 100000 : 0;
}
static const ColiV4TestFp4 fake_fp4 = {fake_alloc, fake_buffers, fake_upload, fake_refill, fake_free, fake_free_mb};

typedef struct {
    char signature[2048];
    int mirrored[EXPERTS];            /* expert i has a VRAM mirror */
    int bytes[EXPERTS];               /* its bytes: 1 the WNR3 decode, 0 safetensors, -1 neither */
    int sha[EXPERTS];                 /* first-read SHA-256 (coli_v4_test_wex_seen): 1 passed, 4 refused, 0 never */
    uint64_t host_decodes, device_decodes;
} PrewarmResult;

static int same_expert(const FakeTensor *t[3], const unsigned char *expect) {
    for (int m = 0; m < 3; m++)
        if (memcmp(t[m]->s, expect + m * SCALE_M, SCALE_M) || memcmp(t[m]->w, expect + 3 * SCALE_M + m * WEIGHT_M, WEIGHT_M))
            return 0;
    return 1;
}

/* one tier prewarm on a fresh store with a V4.1 mirror cache; batch: V4_PREWARM_BATCH=1 with
 * `batch` records per batch, 0: the default loop */
static void prewarm_once(int batch, PrewarmResult *r) {
    memset(r, 0, sizeof *r);
    if (batch) setenv("V4_PREWARM_BATCH", "1", 1); else unsetenv("V4_PREWARM_BATCH");
    coli_v4_test_prewarm_batch = batch;
    uint64_t host = coli_wex_cuda_decode_count(), device = coli_wex_cuda_device_decode_count();
    ColiExpertStore *store = open_store();
    assert(!coli_v4_test_store_mirrors(store, 16, 1));
    coli_v4_expert_store_prewarm_tiers(store);
    r->host_decodes = coli_wex_cuda_decode_count() - host;
    r->device_decodes = coli_wex_cuda_device_decode_count() - device;
    assert(!coli_v4_test_mirror_signature(store, r->signature, sizeof r->signature));
    for (int i = 0; i < EXPERTS; i++) {
        ColiExpertKey key = {0, i};
        ColiExpertView view = {.key = key};
        r->bytes[i] = -1;
        r->mirrored[i] = coli_v4_gpu_expert_peek(store, &view) == 0;
        if (r->mirrored[i]) {
            const FakeTensor *t[3] = {view.gate.gpu, view.down.gpu, view.up.gpu};
            r->bytes[i] = same_expert(t, wnr3[i]) ? 1 : same_expert(t, source[i]) ? 0 : -1;
        }
        r->sha[i] = coli_v4_test_wex_seen(store, key);
    }
    assert(!coli_v4_test_store_mirrors(store, 0, 0));
    store->ops->destroy(store);
    unsetenv("V4_PREWARM_BATCH");
    coli_v4_test_prewarm_batch = 0;
}

/* the default loop (a) against V4_PREWARM_BATCH (b): the same mirrors, entry by entry (key,
 * order, epoch, frequency), the same bytes in them, the same first-read results (want_sha NULL:
 * the same as a's); b decoded `direct` records in batches, a every WNR3 record it read on the
 * host, b only the others */
static void prewarm_compare(int batch, const int want_bytes[EXPERTS], const int *want_sha, int direct) {
    PrewarmResult a, b;
    prewarm_once(0, &a);
    prewarm_once(batch, &b);
    if (strcmp(a.signature, b.signature))
        fprintf(stderr, "default loop:\n%sbatched (%d a batch):\n%s", a.signature, batch, b.signature);
    assert(!strcmp(a.signature, b.signature));
    for (int i = 0; i < EXPERTS; i++) {
        assert(a.bytes[i] == want_bytes[i] && b.bytes[i] == want_bytes[i]);
        assert(a.mirrored[i] == (want_bytes[i] >= 0) && b.mirrored[i] == a.mirrored[i]);
        assert(b.sha[i] == a.sha[i] && (!want_sha || a.sha[i] == want_sha[i]));
    }
    assert(a.device_decodes == 0 && b.device_decodes == (uint64_t)direct);
    assert(a.host_decodes == b.host_decodes + b.device_decodes);
}

static void check_prewarm(const Container *c) {
    int all[EXPERTS] = {1, 1, 1, 1}, passed[EXPERTS] = {1, 1, 1, 1};
    int one_bad[EXPERTS] = {1, 0, 1, 1}, one_refused[EXPERTS] = {1, 4, 1, 1};
    int three_bad[EXPERTS] = {1, 1, 1, 0}, three_refused[EXPERTS] = {1, 1, 1, 4};
    int two_fit[EXPERTS] = {1, 1, -1, -1};
    char root[65], m1[600], path[700];
    size_t bytes;
    unsigned char *w4 = upgrade_wex4(c->data, c->bytes, &bytes, root);
    write_file(wex_path, w4, bytes);
    setenv("COLI_WEX_ROOT", root, 1);                       /* pinned: the first-read SHA-256 is on */
    setenv("COLI_V4_TIER_PREWARM", "1", 1);
    unsetenv("COLI_V4_PREWARM");
    coli_v4_test_fp4 = &fake_fp4;
    for (int batch = 1; batch <= COLI_V4_WEX_BATCH_MAX; batch *= 2)   /* 4, 2, 1, 1 batches */
        prewarm_compare(batch, all, passed, EXPERTS);
    puts("V4_PREWARM_BATCH: same VRAM mirrors (keys, order, epochs, bytes) and the same first-read "
         "SHA-256 results as the default loop, every record decoded in batches, 1 to 4 a batch.");
    /* record 1 corrupted: refused by its first-read SHA-256 in the batch, filled from safetensors
     * through the host path into the mirror the batch held for it, as the default loop does */
    w4[c->entries[1].file_offset + c->used[1] / 2] ^= 0x10u;
    write_file(wex_path, w4, bytes);
    prewarm_compare(2, one_bad, one_refused, EXPERTS - 1);
    prewarm_compare(8, one_bad, one_refused, EXPERTS - 1);
    w4[c->entries[1].file_offset + c->used[1] / 2] ^= 0x10u;
    write_file(wex_path, w4, bytes);
    /* a mirror drive (COLI_WEX_WEIGHTS 1,1: records 2 and 3 read from it) with record 3 damaged */
    snprintf(m1, sizeof m1, "%s\\mirror", directory);
    assert(!_mkdir(m1));
    write_mirror(m1, w4, bytes, c, 1u << 3);
    setenv("COLI_WEX_EXTRA_MIRRORS", m1, 1);
    setenv("COLI_WEX_WEIGHTS", "1,1", 1);
    prewarm_compare(2, three_bad, three_refused, EXPERTS - 1);
    unsetenv("COLI_WEX_EXTRA_MIRRORS");
    unsetenv("COLI_WEX_WEIGHTS");
    snprintf(path, sizeof path, "%s\\v4_flash_experts.wex", m1);
    assert(remove(path) == 0 && _rmdir(m1) == 0);
    puts("V4_PREWARM_BATCH: a corrupted record (primary or mirror drive) is refused by its first-read "
         "SHA-256 and never published from the batch; the host path fills its held mirror from "
         "safetensors, the same mirrors as the default loop.");
    /* VRAM for two mirrors: both stop at the third, which stays in RAM */
    fake_mirrors = 2;
    prewarm_compare(2, two_fit, NULL, 2);
    prewarm_compare(8, two_fit, NULL, 2);
    fake_mirrors = 1 << 20;
    puts("V4_PREWARM_BATCH: the VRAM reserve stops it at the same record as the default loop.");
    {   /* a wrong SHA-256 for record 2 in the index (hashes recomputed; root unpinned, the check
         * forced on): its bytes decode fine, only the first-read check keeps them out of VRAM */
        size_t index_offset = c->bytes + WEX_ALIGN, index_bytes;
        int two_bad[EXPERTS] = {1, 1, 0, 1}, two_refused[EXPERTS] = {1, 1, 4, 1};
        memcpy(&index_bytes, w4 + 40, 8);
        w4[index_offset + 3 * WEX4_BLOB_BYTES + 2 * WEX4_RECORD_BYTES + 36] ^= 1u;
        wex4_sha256(w4 + index_offset, index_bytes, w4 + 96);
        wex4_sha256(w4, WEX4_HEADER_HASHED, w4 + WEX4_HEADER_HASHED);
        write_file(wex_path, w4, bytes);
        unsetenv("COLI_WEX_ROOT");
        setenv("COLI_WEX_VERIFY", "1", 1);
        prewarm_compare(2, two_bad, two_refused, EXPERTS - 1);
        prewarm_compare(8, two_bad, two_refused, EXPERTS - 1);
        unsetenv("COLI_WEX_VERIFY");
        puts("V4_PREWARM_BATCH: a record that decodes but fails its first-read SHA-256 is not published "
             "from the batch.");
    }
    assert(fake_live == 0);
    coli_v4_test_fp4 = NULL;
    unsetenv("COLI_WEX_ROOT");
    unsetenv("COLI_V4_TIER_PREWARM");
    free(w4);
}
#endif

int main(int argc, char **argv) {
    char helper[1024], *slash;
    const char *real = getenv("NR3_REAL_HELPER");
    (void)argc;
    setbuf(stdout, NULL);
    /* the fake helper sits next to this test; LoadLibraryEx wants an absolute path */
    real_helper = real && *real;
    if (real_helper) assert(_fullpath(helper, real, sizeof helper));
    else {
        assert(_fullpath(helper, argv[0], sizeof helper));
        slash = strrchr(helper, '\\');
        assert(slash && (size_t)(slash - helper) + sizeof "\\fake_wex_cuda.dll" < sizeof helper);
        strcpy(slash, "\\fake_wex_cuda.dll");
    }
    setenv("COLI_WEX_CUDA_LIB", helper, 1);
#ifdef V4_WEX_PREWARM_TEST
    /* before the helper loads and the GPU-direct switches are read once: both arms run with the
     * WEX batch path available, as the V4.1 profile does */
    setenv("FAKE_WEX_BATCH_DEVICE", "1", 1);
    setenv("COLI_WEX_GPU_DIRECT", "1", 1);
    setenv("V4_WEX_BATCH", "1", 1);
    coli_v4_test_fp4 = &fake_fp4;
#endif
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    setenv("COLI_V4_PREWARM", "0", 1);
    setenv("COLI_V4_AUTOPIN", "0", 1);
    unsetenv("COLI_V4_DIRECT");
    unsetenv("COLI_WEX_PATH");
    unsetenv("COLI_MODEL_MIRROR");
    unsetenv("COLI_WEX_EXTRA_MIRRORS");
    unsetenv("COLI_WEX_WEIGHTS");
    unsetenv("DSV4_CUDA_MOE_MASKED");
    unsetenv("COLI_WEX_ROOT");
    unsetenv("COLI_WEX_VERIFY");
    unsetenv("V4_WEX_SHA_ASYNC");
    assert(mkdtemp(directory));
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", directory);
    snprintf(wex_path, sizeof wex_path, "%s\\v4_flash_experts.wex", directory);
#ifdef V4_WEX_PREWARM_TEST
    {   /* canonical zero: the WNR3 bytes differ from safetensors, so a mirror shows its source */
        Container c;
        build_container(1, &c);
        write_safetensors();
        check_prewarm(&c);
        for (int i = 0; i < EXPERTS; i++) free(c.records[i]);
        free(c.data);
    }
    assert(remove(wex_path) == 0);
    assert(!scratch_remove(directory, st_path));
    puts("ALL V4 TIER PREWARM BATCH TESTS PASSED");
    return 0;
#endif
    run_profile(0);
    run_profile(1);
    assert(remove(wex_path) == 0);
    assert(!scratch_remove(directory, st_path));
    puts("ALL V4 WNR3 ENGINE STORE TESTS PASSED");
    return 0;
}
#endif
