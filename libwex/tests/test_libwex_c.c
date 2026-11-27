/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

/* Comprehensive self-contained C test suite for libwex:
 *   - RAW, WBP1, WNR3 (exact & CZ), WNR4-MX, WNR4-NV, WNR4-I4, WNR4-GG containers
 *   - wex_open / wex_info / wex_record_layout / wex_has_expert / wex_fetch_host
 *   - L2 Compressed-RAM cache (disk miss -> L2 hit, BYPASS_L2, tight-cap LRU eviction,
 *     multi-thread concurrent stress)
 *   - Background wex_prefetch warming L2
 *   - Root pinning, SHA-256 refusal lattice (WEX_E_HASH -> WEX_E_REFUSED), and
 *     Level-1 stream/tile refusal under WEX_VERIFY_OFF (WEX_E_DECODE)
 *   - Dynamic loader wex_loader.h check
 *
 * Usage: test_libwex_c [--cuda PATH] [--lib PATH] */

#include "../include/wex_store.h"
#include "../include/wex_loader.h"
#include "../src/wex_cache.h"

#include "../../c/wex4.h"
#include "../../wnr4/wnr4.h"
#include "../../wnr4/wnr4_gg.h"
#include "../../wnr4/wnr4_i4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
static void test_sleep_ms(unsigned ms) { Sleep(ms); }
#else
#include <pthread.h>
#include <time.h>
static void test_sleep_ms(unsigned ms) {
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    nanosleep(&ts, NULL);
}
#endif

static int g_checks = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

/* Deterministic PRNG (xorshift64*) */
static uint64_t rng_next(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * UINT64_C(2685821657736338717);
}

/* ---- Generic WEX4 synthetic container builder ---------------------------- */

typedef struct {
    char     name[96];
    uint8_t  dtype;
    uint8_t  rank;
    uint64_t shape[8];
    uint64_t offset;
    uint64_t nbytes;
} test_tensor_t;

typedef struct {
    const uint8_t *stored_data;
    uint32_t       stored_bytes;
    uint32_t       decoded_bytes;
    uint16_t       codec;
    uint16_t       flags;
    uint32_t       blob_1based;
    uint32_t       owner;
    test_tensor_t  tensors[8];
    uint32_t       tensor_count;
} test_record_t;

typedef struct {
    uint32_t       kind;
    uint32_t       id;
    const uint8_t *data;
    uint64_t       bytes;
} test_blob_t;

static const char *g_sort_names;
static const uint32_t *g_sort_at;
static const uint32_t *g_sort_len;

static int cmp_tensor_order(const void *a, const void *b) {
    uint32_t i = *(const uint32_t *)a;
    uint32_t j = *(const uint32_t *)b;
    uint32_t li = g_sort_len[i], lj = g_sort_len[j];
    int c = memcmp(g_sort_names + g_sort_at[i], g_sort_names + g_sort_at[j], li < lj ? li : lj);
    return c ? c : ((li > lj) - (li < lj));
}

static void put_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static void put_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u64_le(uint8_t *p, uint64_t v) {
    put_u32_le(p, (uint32_t)v); put_u32_le(p + 4, (uint32_t)(v >> 32));
}

static void write_wex4_file(const char *path,
                            const test_blob_t *blobs, uint32_t nblobs,
                            const test_record_t *recs, uint32_t nrecs,
                            uint32_t layers, uint32_t experts,
                            uint64_t required_features,
                            char root_hex_out[65]) {
    uint32_t ntensors = 0, r, b, k, t = 0;
    uint64_t names_bytes = 0, total_decoded = 0, max_decoded = 0;
    uint64_t pos = WEX4_PAGE;
    uint64_t *blob_off = NULL, *rec_off = NULL;
    char *names = NULL;
    uint32_t *name_at = NULL, *name_len = NULL, *order = NULL;
    uint64_t index_bytes, index_len, index_offset, file_bytes;
    uint8_t *file_buf = NULL, *idx;
    FILE *fp;

    for (r = 0; r < nrecs; r++) {
        ntensors += recs[r].tensor_count;
        total_decoded += recs[r].decoded_bytes;
        if (recs[r].decoded_bytes > max_decoded) max_decoded = recs[r].decoded_bytes;
        for (k = 0; k < recs[r].tensor_count; k++) {
            names_bytes += strlen(recs[r].tensors[k].name);
        }
    }

    blob_off = (uint64_t *)calloc(nblobs ? nblobs : 1u, sizeof(uint64_t));
    rec_off  = (uint64_t *)calloc(nrecs ? nrecs : 1u, sizeof(uint64_t));
    names    = (char *)malloc((size_t)(names_bytes ? names_bytes : 1u));
    name_at  = (uint32_t *)calloc(ntensors ? ntensors : 1u, sizeof(uint32_t));
    name_len = (uint32_t *)calloc(ntensors ? ntensors : 1u, sizeof(uint32_t));
    order    = (uint32_t *)calloc(ntensors ? ntensors : 1u, sizeof(uint32_t));
    CHECK(blob_off && rec_off && names && name_at && name_len && order);

    {
        uint32_t nat = 0;
        for (r = 0; r < nrecs; r++) {
            for (k = 0; k < recs[r].tensor_count; k++) {
                uint32_t len = (uint32_t)strlen(recs[r].tensors[k].name);
                name_at[t]  = nat;
                name_len[t] = len;
                memcpy(names + nat, recs[r].tensors[k].name, len);
                nat += len;
                order[t] = t;
                t++;
            }
        }
        CHECK(nat == names_bytes);
    }

    g_sort_names = names;
    g_sort_at    = name_at;
    g_sort_len   = name_len;
    qsort(order, ntensors, sizeof(uint32_t), cmp_tensor_order);

    for (b = 0; b < nblobs; b++) {
        blob_off[b] = pos;
        pos += wex4_align(blobs[b].bytes);
    }
    for (r = 0; r < nrecs; r++) {
        CHECK((recs[r].stored_bytes & (WEX4_PAGE - 1u)) == 0u);
        rec_off[r] = pos;
        pos += recs[r].stored_bytes;
    }

    index_bytes  = (uint64_t)nblobs * WEX4_BLOB_BYTES +
                   (uint64_t)nrecs * WEX4_RECORD_BYTES +
                   (uint64_t)ntensors * (WEX4_TENSOR_BYTES + 4u) +
                   names_bytes;
    index_len    = wex4_align(index_bytes);
    index_offset = pos;
    file_bytes   = index_offset + index_len;

    file_buf = (uint8_t *)calloc(1, (size_t)file_bytes);
    CHECK(file_buf != NULL);

    for (b = 0; b < nblobs; b++) {
        memcpy(file_buf + blob_off[b], blobs[b].data, (size_t)blobs[b].bytes);
    }
    for (r = 0; r < nrecs; r++) {
        memcpy(file_buf + rec_off[r], recs[r].stored_data, recs[r].stored_bytes);
    }

    idx = file_buf + index_offset;
    {
        size_t at = 0;
        uint32_t first_t = 0;
        /* Blobs */
        for (b = 0; b < nblobs; b++) {
            put_u32_le(idx + at + 0,  blobs[b].kind);
            put_u32_le(idx + at + 4,  blobs[b].id);
            put_u64_le(idx + at + 8,  blob_off[b]);
            put_u64_le(idx + at + 16, blobs[b].bytes);
            wex4_sha256(blobs[b].data, (size_t)blobs[b].bytes, idx + at + 24);
            at += WEX4_BLOB_BYTES;
        }
        /* Records */
        for (r = 0; r < nrecs; r++) {
            put_u64_le(idx + at + 0,  rec_off[r]);
            put_u32_le(idx + at + 8,  recs[r].stored_bytes);
            put_u32_le(idx + at + 12, recs[r].decoded_bytes);
            put_u16_le(idx + at + 16, recs[r].codec);
            put_u16_le(idx + at + 18, recs[r].flags);
            put_u32_le(idx + at + 20, recs[r].blob_1based);
            put_u32_le(idx + at + 24, recs[r].owner);
            put_u32_le(idx + at + 28, first_t);
            put_u32_le(idx + at + 32, recs[r].tensor_count);
            wex4_sha256(recs[r].stored_data, recs[r].stored_bytes, idx + at + 36);
            first_t += recs[r].tensor_count;
            at += WEX4_RECORD_BYTES;
        }
        /* Tensors */
        t = 0;
        for (r = 0; r < nrecs; r++) {
            for (k = 0; k < recs[r].tensor_count; k++) {
                const test_tensor_t *ten = &recs[r].tensors[k];
                uint32_t d;
                put_u32_le(idx + at + 0, name_at[t]);
                put_u16_le(idx + at + 4, (uint16_t)name_len[t]);
                idx[at + 6] = ten->dtype;
                idx[at + 7] = ten->rank;
                for (d = 0; d < 8u; d++) put_u64_le(idx + at + 8 + d * 8u, ten->shape[d]);
                put_u32_le(idx + at + 72, r);
                put_u32_le(idx + at + 76, 1u);
                put_u64_le(idx + at + 80, ten->offset);
                put_u64_le(idx + at + 88, ten->nbytes);
                at += WEX4_TENSOR_BYTES;
                t++;
            }
        }
        /* Name order */
        for (t = 0; t < ntensors; t++) {
            put_u32_le(idx + at, order[t]);
            at += 4u;
        }
        /* Name pool */
        memcpy(idx + at, names, (size_t)names_bytes);
        at += (size_t)names_bytes;
        CHECK(at == index_bytes);
    }

    /* Header page */
    put_u32_le(file_buf + 0,  WEX4_MAGIC);
    put_u16_le(file_buf + 4,  4u);
    put_u16_le(file_buf + 6,  0u);
    put_u64_le(file_buf + 8,  required_features);
    put_u64_le(file_buf + 16, 0u);
    put_u64_le(file_buf + 24, file_bytes);
    put_u64_le(file_buf + 32, index_offset);
    put_u64_le(file_buf + 40, index_bytes);
    put_u32_le(file_buf + 48, nblobs);
    put_u32_le(file_buf + 52, nrecs);
    put_u32_le(file_buf + 56, ntensors);
    put_u32_le(file_buf + 60, layers);
    put_u32_le(file_buf + 64, experts);
    put_u32_le(file_buf + 68, (uint32_t)max_decoded);
    put_u64_le(file_buf + 72, names_bytes);
    put_u64_le(file_buf + 80, 0u);
    put_u64_le(file_buf + 88, total_decoded);
    wex4_sha256(file_buf + index_offset, (size_t)index_bytes, file_buf + 96);
    wex4_sha256(file_buf, WEX4_HEADER_HASHED, file_buf + WEX4_HEADER_HASHED);

    if (root_hex_out) {
        static const char hex[] = "0123456789abcdef";
        uint32_t i;
        for (i = 0; i < 32u; i++) {
            root_hex_out[i * 2u]      = hex[file_buf[WEX4_HEADER_HASHED + i] >> 4];
            root_hex_out[i * 2u + 1u] = hex[file_buf[WEX4_HEADER_HASHED + i] & 0x0fu];
        }
        root_hex_out[64] = '\0';
    }

    fp = fopen(path, "wb");
    CHECK(fp != NULL);
    CHECK(fwrite(file_buf, 1, (size_t)file_bytes, fp) == (size_t)file_bytes);
    CHECK(fclose(fp) == 0);

    free(file_buf);
    free(blob_off);
    free(rec_off);
    free(names);
    free(name_at);
    free(name_len);
    free(order);
}

/* ---- Synthetic data generators for each codec ---------------------------- */

/* Generate compressible MXFP4 expert slab: 3 matrices, rows=64, scols=8 (wcols=128)
 * Scales per matrix = 512 B (total 1536 B), Weights per matrix = 8192 B (total 24576 B),
 * Total decoded = 26112 B -> aligned raw = 28672 B (7 pages). Compressible to ~4-5 pages! */
#define EXP_ROWS  64u
#define EXP_SCOLS 8u
#define EXP_WCOLS (EXP_SCOLS * 16u)
#define EXP_SC_BYTES (3u * EXP_ROWS * EXP_SCOLS)
#define EXP_WT_BYTES (3u * EXP_ROWS * EXP_WCOLS)
#define EXP_DEC_BYTES (EXP_SC_BYTES + EXP_WT_BYTES)
#define EXP_RAW_ALIGNED 28672u

static void fill_mxfp4_slab(uint8_t *slab, uint64_t *rng) {
    uint32_t i;
    /* Concentrated scales around 120..123 */
    for (i = 0; i < EXP_SC_BYTES; i++) {
        slab[i] = (uint8_t)(120u + (rng_next(rng) % 4u));
    }
    /* Skewed small FP4 nibbles (mostly 0, 1, 2, 9, 10 -> highly compressible) */
    static const uint8_t nibs[8] = {0, 0, 1, 1, 2, 9, 9, 10};
    for (i = 0; i < EXP_WT_BYTES; i++) {
        uint8_t lo = nibs[rng_next(rng) & 7u];
        uint8_t hi = nibs[rng_next(rng) & 7u];
        slab[EXP_SC_BYTES + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void fill_mxfp4_entry_and_tensors(WexEntry *e, test_record_t *tr, uint32_t layer, uint32_t eid) {
    uint32_t m;
    memset(e, 0, sizeof(*e));
    e->scale_bytes  = EXP_SC_BYTES;
    e->weight_bytes = EXP_WT_BYTES;
    for (m = 0; m < 3u; m++) {
        e->scale_shapes[m][0]  = EXP_ROWS;
        e->scale_shapes[m][1]  = EXP_SCOLS;
        e->scale_offsets[m]    = m * EXP_ROWS * EXP_SCOLS;
        e->weight_shapes[m][0] = EXP_ROWS;
        e->weight_shapes[m][1] = EXP_WCOLS;
        e->weight_offsets[m]   = EXP_SC_BYTES + m * EXP_ROWS * EXP_WCOLS;
    }
    tr->tensor_count = 6u;
    for (m = 0; m < 6u; m++) {
        test_tensor_t *ten = &tr->tensors[m];
        memset(ten, 0, sizeof(*ten));
        snprintf(ten->name, sizeof(ten->name), "layers.%u.ffn.experts.%u.w%u.%s",
                 layer, eid, (m % 3u) + 1u, m < 3u ? "scale" : "weight");
        ten->dtype    = (m < 3u) ? WEX4_DT_F8_E8M0 : WEX4_DT_I8;
        ten->rank     = 2u;
        ten->shape[0] = EXP_ROWS;
        ten->shape[1] = (m < 3u) ? EXP_SCOLS : EXP_WCOLS;
        ten->offset   = (m < 3u) ? e->scale_offsets[m] : e->weight_offsets[m - 3u];
        ten->nbytes   = ten->shape[0] * ten->shape[1];
    }
}

/* ---- Test 1: RAW container + basic API & error checks -------------------- */

static void test_raw_and_api(const char *cuda_path) {
    const char *path = "wex_test_tmp_raw.wex4";
    uint8_t slabs[8][EXP_RAW_ALIGNED];
    test_record_t recs[8];
    uint64_t rng = UINT64_C(0x1122334455667788);
    char root_hex[65];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    wex_layout_t lay;
    char err[256];
    uint32_t i;

    printf("[test] 1. RAW container & core API...\n");
    CHECK(wex_abi_version() == WEX_STORE_ABI_VERSION);
    CHECK(wex_version_string() != NULL && strlen(wex_version_string()) > 0);

    memset(slabs, 0, sizeof(slabs));
    for (i = 0; i < 8u; i++) {
        WexEntry e;
        uint32_t l = i / 4u, eid = i % 4u;
        fill_mxfp4_slab(slabs[i], &rng);
        memset(&recs[i], 0, sizeof(recs[i]));
        fill_mxfp4_entry_and_tensors(&e, &recs[i], l, eid);
        recs[i].stored_data   = slabs[i];
        recs[i].stored_bytes  = EXP_RAW_ALIGNED;
        recs[i].decoded_bytes = EXP_DEC_BYTES;
        recs[i].codec         = WEX4_CODEC_RAW;
        recs[i].flags         = WEX4_REC_EXPERT;
        recs[i].blob_1based   = 0u;
        recs[i].owner         = (l << 16) | eid;
    }
    write_wex4_file(path, NULL, 0, recs, 8, 2, 4, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.root_hex        = root_hex;
    opts.verify          = WEX_VERIFY_AUTO;

    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.abi_version == WEX_STORE_ABI_VERSION);
    CHECK(info.layers == 2u && info.experts == 4u);
    CHECK(info.num_records == 8u && info.num_tensors == 48u);
    CHECK(info.codec_kind == WEX_CODEC_RAW);
    CHECK(info.verified_root == 1u && info.verify_on == 1u);
    CHECK(strcmp(info.root_hex, root_hex) == 0);
    CHECK(info.max_record_decoded_bytes == EXP_DEC_BYTES);
    CHECK(info.max_record_stored_bytes == EXP_RAW_ALIGNED);

    CHECK(wex_has_expert(s, 0, 0) == 1);
    CHECK(wex_has_expert(s, 1, 3) == 1);
    CHECK(wex_has_expert(s, 2, 0) == 0);
    CHECK(wex_has_expert(s, 0, 4) == 0);
    CHECK(wex_has_expert(s, -1, 0) == 0);

    CHECK(wex_record_layout(s, 1, 2, 0, &lay) == WEX_OK);
    CHECK(lay.record_index == 6u);
    CHECK(lay.codec_kind == WEX_CODEC_RAW);
    CHECK(lay.tensor_count == 6u);
    CHECK(lay.decoded_bytes == EXP_DEC_BYTES);
    CHECK(lay.scale_shapes[0][0] == EXP_ROWS && lay.scale_shapes[0][1] == EXP_SCOLS);
    CHECK(lay.weight_shapes[2][0] == EXP_ROWS && lay.weight_shapes[2][1] == EXP_WCOLS);

    /* Fetch layer 1 experts [0, 3] and error cases */
    {
        uint8_t out0[EXP_DEC_BYTES], out1[EXP_DEC_BYTES];
        void *dsts[3] = { out0, out1, out0 };
        size_t caps[3] = { EXP_DEC_BYTES, 128u, EXP_DEC_BYTES };
        int eids[3] = { 0, 3, 99 };
        int st[3] = { 0, 0, 0 };
        int failed = wex_fetch_host(s, 1, eids, 3, dsts, caps, 0, st);
        CHECK(failed == 2);
        CHECK(st[0] == WEX_OK);
        CHECK(st[1] == WEX_E_CAP);
        CHECK(st[2] == WEX_E_NOT_FOUND);
        CHECK(memcmp(out0, slabs[4], EXP_DEC_BYTES) == 0);
    }

    /* Fetch by flat record ID */
    {
        uint8_t out[EXP_DEC_BYTES];
        void *dsts[1] = { out };
        size_t caps[1] = { EXP_DEC_BYTES };
        int rids[1] = { 7 };
        int st[1] = { 0 };
        CHECK(wex_fetch_host(s, 0, rids, 1, dsts, caps, WEX_FETCH_BY_RECORD_ID, st) == 0);
        CHECK(st[0] == WEX_OK);
        CHECK(memcmp(out, slabs[7], EXP_DEC_BYTES) == 0);
    }

    wex_close(s);
    remove(path);
}

/* ---- Test 2: WBP1 codec 6 container -------------------------------------- */

static void test_wbp1(void) {
    const char *path = "wex_test_tmp_wbp1.wex4";
    uint32_t dec_bytes = 32768u;
    uint8_t *raw = (uint8_t *)malloc(dec_bytes);
    uint8_t *enc = (uint8_t *)malloc(dec_bytes);
    size_t written = 0;
    test_record_t rec;
    char root_hex[65];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    uint32_t i;
    char err[256];

    printf("[test] 2. WBP1 codec 6...\n");
    CHECK(raw && enc);
    /* Compressible float-like 4-byte pattern */
    for (i = 0; i < dec_bytes / 4u; i++) {
        float f = 1.0f + (float)(i & 15u) * 0.0625f;
        memcpy(raw + i * 4u, &f, 4u);
    }
    CHECK(wex4_bp_encode(raw, dec_bytes, 4u, 12u, 1u, 0u, enc, dec_bytes, &written) == 0);
    CHECK(written < dec_bytes && (written & (WEX4_PAGE - 1u)) == 0u);

    memset(&rec, 0, sizeof(rec));
    rec.stored_data   = enc;
    rec.stored_bytes  = (uint32_t)written;
    rec.decoded_bytes = dec_bytes;
    rec.codec         = WEX4_CODEC_WBP1;
    rec.flags         = 0u;
    rec.blob_1based   = 0u;
    rec.owner         = 0u;
    rec.tensor_count  = 1u;
    snprintf(rec.tensors[0].name, sizeof(rec.tensors[0].name), "norm.weight");
    rec.tensors[0].dtype    = 13u; /* F32 */
    rec.tensors[0].rank     = 1u;
    rec.tensors[0].shape[0] = dec_bytes / 4u;
    rec.tensors[0].offset   = 0u;
    rec.tensors[0].nbytes   = dec_bytes;

    write_wex4_file(path, NULL, 0, &rec, 1, 0, 0, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device = -2;
    opts.verify     = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WBP1);
    {
        uint8_t *out = (uint8_t *)calloc(1, dec_bytes);
        void *dsts[1] = { out };
        size_t caps[1] = { dec_bytes };
        int rids[1] = { 0 };
        int st[1] = { 0 };
        CHECK(out != NULL);
        CHECK(wex_fetch_host(s, 0, rids, 1, dsts, caps, WEX_FETCH_BY_RECORD_ID, st) == 0);
        CHECK(st[0] == WEX_OK);
        CHECK(memcmp(out, raw, dec_bytes) == 0);
        free(out);
    }
    wex_close(s);
    remove(path);
    free(raw);
    free(enc);
}

/* ---- Helper: build a real WNR3 container (8 experts) --------------------- */

static void build_wnr3_container(const char *path, int cz,
                                 uint8_t expected_out[8][EXP_DEC_BYTES],
                                 uint32_t stored_out[8],
                                 char root_hex[65]) {
    uint64_t rng = UINT64_C(0x20261127ABCDEF01);
    uint8_t raw[8][EXP_DEC_BYTES];
    uint8_t coded[EXP_DEC_BYTES];
    uint8_t enc_buf[8][EXP_RAW_ALIGNED];
    uint8_t *wnm2_blob = (uint8_t *)malloc(WEX_NR2_MODEL_BYTES);
    uint64_t *wcounts = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t));
    uint64_t scounts[3u * 256u];
    WexNr2Model model;
    WexEntry e;
    WexNr2Layout layout;
    test_record_t recs[8];
    test_blob_t blob;
    uint32_t i;

    CHECK(wnm2_blob && wcounts);
    memset(scounts, 0, sizeof(scounts));
    fill_mxfp4_entry_and_tensors(&e, &recs[0], 0, 0);
    wex_nr2_layout(&e, &layout);

    for (i = 0; i < 8u; i++) {
        unsigned char *sc[3], *wt[3];
        uint32_t m;
        fill_mxfp4_slab(raw[i], &rng);
        memcpy(expected_out[i], raw[i], EXP_DEC_BYTES);
        if (cz) wex_nr3_canonical_zero(expected_out[i] + EXP_SC_BYTES, EXP_WT_BYTES);
        memcpy(coded, expected_out[i], EXP_DEC_BYTES);
        for (m = 0; m < 3u; m++) { sc[m] = coded + e.scale_offsets[m]; wt[m] = coded + e.weight_offsets[m]; }
        (void)wex_nr3_ceil_forward(&layout, sc, wt, NULL);
        wex_nr3_count(&layout, (const unsigned char *const *)sc, (const unsigned char *const *)wt,
                      wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, wnm2_blob);
    CHECK(wex_nr2_model_open(wnm2_blob, WEX_NR2_MODEL_BYTES, &model) == 0);

    for (i = 0; i < 8u; i++) {
        unsigned char *sc[3], *wt[3];
        uint32_t l = i / 4u, eid = i % 4u, m, flags;
        size_t written = 0, stored;
        memset(&recs[i], 0, sizeof(recs[i]));
        fill_mxfp4_entry_and_tensors(&e, &recs[i], l, eid);
        memcpy(coded, expected_out[i], EXP_DEC_BYTES);
        for (m = 0; m < 3u; m++) { sc[m] = coded + e.scale_offsets[m]; wt[m] = coded + e.weight_offsets[m]; }
        flags = wex_nr3_ceil_forward(&layout, sc, wt, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
        memset(enc_buf[i], 0, sizeof(enc_buf[i]));
        CHECK(wex_nr3_encode(&model, &layout, (const unsigned char *const *)sc,
                             (const unsigned char *const *)wt,
                             4096u, 256u, flags, (l << 16) | eid,
                             enc_buf[i], EXP_RAW_ALIGNED, &written) == 0);
        stored = (written + (WEX4_PAGE - 1u)) & ~(size_t)(WEX4_PAGE - 1u);
        CHECK(stored > 0 && stored < EXP_RAW_ALIGNED);
        recs[i].stored_data   = enc_buf[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = EXP_DEC_BYTES;
        recs[i].codec         = WEX4_CODEC_WNR3;
        recs[i].flags         = (uint16_t)(WEX4_REC_EXPERT | (cz ? WEX4_REC_VALUE_EXACT : 0u));
        recs[i].blob_1based   = 1u;
        recs[i].owner         = (l << 16) | eid;
        if (stored_out) stored_out[i] = (uint32_t)stored;
    }

    blob.kind  = WEX4_BLOB_WNM2;
    blob.id    = model.id;
    blob.data  = wnm2_blob;
    blob.bytes = WEX_NR2_MODEL_BYTES;
    write_wex4_file(path, &blob, 1, recs, 8, 2, 4, cz ? WEX4_REQ_VALUE_EXACT : 0u, root_hex);

    free(wnm2_blob);
    free(wcounts);
}

/* ---- Test 3: WNR3 + L2 Cache + Prefetch + Multi-thread Stress ------------ */

typedef struct {
    wex_store_t *store;
    const uint8_t (*expected)[EXP_DEC_BYTES];
    int tid;
    int ok;
} mt_stress_arg_t;

#ifdef _WIN32
static unsigned __stdcall mt_stress_worker(void *arg)
#else
static void *mt_stress_worker(void *arg)
#endif
{
    mt_stress_arg_t *a = (mt_stress_arg_t *)arg;
    uint64_t rng = UINT64_C(0x9E3779B97F4A7C15) ^ (uint64_t)(a->tid + 1) * UINT64_C(0xBF58476D1CE4E5B9);
    uint8_t out[4][EXP_DEC_BYTES];
    void *dsts[4] = { out[0], out[1], out[2], out[3] };
    size_t caps[4] = { EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES };
    int iter;
    a->ok = 1;
    for (iter = 0; iter < 150; iter++) {
        int l = (int)(rng_next(&rng) & 1u);
        int k = 1 + (int)(rng_next(&rng) & 3u);
        int eids[4], st[4], j;
        for (j = 0; j < k; j++) eids[j] = (int)(rng_next(&rng) & 3u);
        if (wex_fetch_host(a->store, l, eids, k, dsts, caps, 0, st) != 0) {
            a->ok = 0;
            break;
        }
        for (j = 0; j < k; j++) {
            if (st[j] != WEX_OK || memcmp(out[j], a->expected[l * 4 + eids[j]], EXP_DEC_BYTES) != 0) {
                a->ok = 0;
                break;
            }
        }
        if (!a->ok) break;
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void test_wnr3_l2_prefetch_mt(const char *cuda_path) {
    const char *path = "wex_test_tmp_wnr3.wex4";
    uint8_t expected[8][EXP_DEC_BYTES];
    uint32_t stored[8];
    char root_hex[65];
    char err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    wex_stats_t st0, st1;
    uint8_t out[4][EXP_DEC_BYTES];
    void *dsts[4] = { out[0], out[1], out[2], out[3] };
    size_t caps[4] = { EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES };
    int eids[4] = { 0, 1, 2, 3 };
    int status[4];
    int i;

    printf("[test] 3. WNR3 (CZ), L2 cache, eviction, prefetch & MT stress...\n");
    build_wnr3_container(path, 1, expected, stored, root_hex);

    /* 3a. L2 hit on 2nd fetch + BYPASS_L2 */
    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.l2_cache_bytes  = 16u * EXP_RAW_ALIGNED;
    opts.verify          = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WNR3);
    CHECK(info.value_exact == 1u);
    CHECK(info.l2_capacity_bytes == opts.l2_cache_bytes);

    /* 1st fetch -> disk */
    CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, status) == 0);
    for (i = 0; i < 4; i++) {
        CHECK(status[i] == WEX_OK);
        CHECK(memcmp(out[i], expected[i], EXP_DEC_BYTES) == 0);
    }
    wex_stats(s, &st0);
    CHECK(st0.records_from_disk == 4u && st0.records_from_l2 == 0u && st0.l2_entries == 4u);

    /* 2nd fetch -> L2 hit (with WEX_FETCH_SCRATCH) */
    memset(out, 0, sizeof(out));
    CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, WEX_FETCH_SCRATCH, status) == 0);
    for (i = 0; i < 4; i++) {
        CHECK(status[i] == WEX_OK);
        CHECK(memcmp(out[i], expected[i], EXP_DEC_BYTES) == 0);
    }
    wex_stats(s, &st1);
    CHECK(st1.records_from_l2 == 4u && st1.records_from_disk == 4u);

    /* BYPASS_L2 must not insert layer 1 records into L2 */
    CHECK(wex_fetch_host(s, 1, eids, 4, dsts, caps, WEX_FETCH_BYPASS_L2, status) == 0);
    for (i = 0; i < 4; i++) CHECK(memcmp(out[i], expected[4 + i], EXP_DEC_BYTES) == 0);
    wex_stats(s, &st1);
    CHECK(st1.l2_entries == 4u && st1.records_from_disk == 8u);
    wex_close(s);

    /* 3b. Tight-cap LRU eviction (holds exactly 2 records) */
    wex_open_opts_init(&opts);
    opts.gpu_device     = -2;
    opts.l2_cache_bytes = (uint64_t)stored[0] + (uint64_t)stored[1];
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    {
        int p01[2] = { 0, 1 }, p23[2] = { 2, 3 };
        CHECK(wex_fetch_host(s, 0, p01, 2, dsts, caps, 0, status) == 0);
        wex_stats(s, &st0);
        CHECK(st0.records_from_disk == 2u && st0.l2_entries == 2u);

        CHECK(wex_fetch_host(s, 0, p23, 2, dsts, caps, 0, status) == 0);
        wex_stats(s, &st0);
        CHECK(st0.records_from_disk == 4u && st0.l2_entries == 2u);

        /* 2 & 3 should be L2 hits */
        CHECK(wex_fetch_host(s, 0, p23, 2, dsts, caps, 0, status) == 0);
        wex_stats(s, &st1);
        CHECK(st1.records_from_l2 == 2u && st1.records_from_disk == 4u);

        /* 0 & 1 were evicted -> disk misses */
        CHECK(wex_fetch_host(s, 0, p01, 2, dsts, caps, 0, status) == 0);
        wex_stats(s, &st1);
        CHECK(st1.records_from_disk == 6u);
    }
    wex_close(s);

    /* 3c. Background wex_prefetch warming L2 */
    wex_open_opts_init(&opts);
    opts.gpu_device     = -2;
    opts.l2_cache_bytes = 16u * EXP_RAW_ALIGNED;
    opts.verify         = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_prefetch(s, 0, eids, 4, 0) == 4);
    for (i = 0; i < 200; i++) {
        wex_stats(s, &st0);
        if (st0.l2_entries >= 4u) break;
        test_sleep_ms(10);
    }
    CHECK(st0.l2_entries >= 4u);
    CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, status) == 0);
    for (i = 0; i < 4; i++) {
        CHECK(status[i] == WEX_OK);
        CHECK(memcmp(out[i], expected[i], EXP_DEC_BYTES) == 0);
    }
    wex_stats(s, &st1);
    CHECK(st1.records_from_l2 == 4u && st1.records_from_disk == 0u && st1.prefetch_hits == 4u);
    wex_close(s);

    /* 3d. Concurrent multi-thread stress under tight L2 eviction */
    wex_open_opts_init(&opts);
    opts.gpu_device     = -2;
    opts.l2_cache_bytes = 3u * (uint64_t)stored[0];
    opts.verify         = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    {
        mt_stress_arg_t args[4];
#ifdef _WIN32
        HANDLE th[4];
        for (i = 0; i < 4; i++) {
            args[i].store    = s;
            args[i].expected = (const uint8_t (*)[EXP_DEC_BYTES])expected;
            args[i].tid      = i;
            args[i].ok       = 0;
            th[i] = (HANDLE)_beginthreadex(NULL, 0, mt_stress_worker, &args[i], 0, NULL);
            CHECK(th[i] != NULL);
        }
        for (i = 0; i < 4; i++) {
            WaitForSingleObject(th[i], INFINITE);
            CloseHandle(th[i]);
            CHECK(args[i].ok == 1);
        }
#else
        pthread_t th[4];
        for (i = 0; i < 4; i++) {
            args[i].store    = s;
            args[i].expected = (const uint8_t (*)[EXP_DEC_BYTES])expected;
            args[i].tid      = i;
            args[i].ok       = 0;
            CHECK(pthread_create(&th[i], NULL, mt_stress_worker, &args[i]) == 0);
        }
        for (i = 0; i < 4; i++) {
            pthread_join(th[i], NULL);
            CHECK(args[i].ok == 1);
        }
#endif
    }
    wex_close(s);
    remove(path);
}

/* ---- Test 4: WNR4-MX (codec 8, kind 0) ----------------------------------- */

static void test_wnr4_mx(const char *cuda_path) {
    const char *path = "wex_test_tmp_wnr4_mx.wex4";
    uint64_t rng = UINT64_C(0x4444444412345678);
    uint8_t raw[4][EXP_DEC_BYTES];
    uint8_t enc_buf[4][EXP_RAW_ALIGNED];
    uint32_t scount[3], scols[3], soff[3], woff[3];
    Wnr4Geo geo;
    size_t cwords = wnr4_counts_words(WNR4_KIND_MX, 1u);
    size_t mbytes = wnr4_model_bytes(WNR4_KIND_MX, 1u);
    uint64_t *counts = (uint64_t *)calloc(cwords, sizeof(uint64_t));
    uint8_t *mblob = (uint8_t *)malloc(mbytes);
    Wnr4Model model;
    test_record_t recs[4];
    test_blob_t blob;
    char root_hex[65], err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    uint32_t i, m;

    printf("[test] 4. WNR4-MX (codec 8, kind 0)...\n");
    CHECK(counts && mblob);
    for (m = 0; m < 3u; m++) {
        scount[m] = EXP_ROWS * EXP_SCOLS;
        scols[m]  = EXP_SCOLS;
        soff[m]   = m * EXP_ROWS * EXP_SCOLS;
        woff[m]   = EXP_SC_BYTES + m * EXP_ROWS * EXP_WCOLS;
    }
    CHECK(wnr4_geo(&geo, WNR4_KIND_MX, scount, scols, soff, woff) == 0);

    for (i = 0; i < 4u; i++) {
        unsigned char anchor[3];
        fill_mxfp4_slab(raw[i], &rng);
        for (m = 0; m < 3u; m++) anchor[m] = wnr4_anchor(raw[i] + soff[m], scount[m]);
        wnr4_count(&geo, raw[i], anchor, 1u, counts);
    }
    wnr4_model_build(WNR4_KIND_MX, 1u, counts, mblob);
    CHECK(wnr4_model_open(mblob, mbytes, &model) == 0);

    for (i = 0; i < 4u; i++) {
        WexEntry e;
        size_t stored;
        memset(&recs[i], 0, sizeof(recs[i]));
        fill_mxfp4_entry_and_tensors(&e, &recs[i], 0, i);
        stored = wnr4_encode(&model, &geo, raw[i], 0u, i, 4u, enc_buf[i], EXP_RAW_ALIGNED);
        CHECK(stored > 0 && stored < EXP_RAW_ALIGNED);
        recs[i].stored_data   = enc_buf[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = EXP_DEC_BYTES;
        recs[i].codec         = 8u;
        recs[i].flags         = WEX4_REC_EXPERT;
        recs[i].blob_1based   = 1u;
        recs[i].owner         = i;
    }
    blob.kind  = 3u;
    blob.id    = model.id;
    blob.data  = mblob;
    blob.bytes = mbytes;
    write_wex4_file(path, &blob, 1, recs, 4, 1, 4, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.verify          = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_MX);
    {
        uint8_t out[4][EXP_DEC_BYTES];
        void *dsts[4] = { out[0], out[1], out[2], out[3] };
        size_t caps[4] = { EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES, EXP_DEC_BYTES };
        int eids[4] = { 0, 1, 2, 3 };
        int st[4];
        CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, st) == 0);
        for (i = 0; i < 4u; i++) {
            CHECK(st[i] == WEX_OK);
            CHECK(memcmp(out[i], raw[i], EXP_DEC_BYTES) == 0);
        }
    }
    wex_close(s);
    remove(path);
    wnr4_model_free(&model);
    free(counts);
    free(mblob);
}

/* ---- Test 5: WNR4-I4 (codec 8, kind 3) ----------------------------------- */

static void test_wnr4_i4(const char *cuda_path) {
    const char *path = "wex_test_tmp_wnr4_i4.wex4";
    uint32_t rows[3] = { 64u, 64u, 64u };
    uint32_t cols[3] = { 256u, 256u, 256u };
    Wnr4I4Geo geo;
    size_t raw_bytes, aligned_raw;
    uint8_t *raw[4], *enc[4];
    uint64_t *counts;
    uint8_t *mblob;
    Wnr4I4Model model;
    test_record_t recs[4];
    test_blob_t blob;
    uint64_t rng = UINT64_C(0xCAFEBABE20261127);
    char root_hex[65], err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    uint32_t i, m;

    printf("[test] 5. WNR4-I4 (codec 8, kind 3, planar & pairs)...\n");
    CHECK(wnr4_i4_geo(&geo, 3u, rows, cols, 0u) == 0);
    raw_bytes   = (size_t)geo.raw;
    aligned_raw = (size_t)wex4_align(raw_bytes);
    counts      = (uint64_t *)calloc(wnr4_i4_counts_words(), sizeof(uint64_t));
    mblob       = (uint8_t *)malloc(WNR4_I4_MODEL_BYTES);
    CHECK(counts && mblob);

    for (i = 0; i < 4u; i++) {
        raw[i] = (uint8_t *)malloc(raw_bytes);
        enc[i] = (uint8_t *)malloc(aligned_raw);
        CHECK(raw[i] && enc[i]);
        /* Skewed int4 codes centered near 8 */
        {
            static const uint8_t cvals[8] = {7, 8, 8, 8, 8, 9, 7, 9};
            size_t off_sc = geo.soff[0];
            size_t b;
            for (b = 0; b < off_sc; b++) {
                uint8_t lo = cvals[rng_next(&rng) & 7u];
                uint8_t hi = cvals[rng_next(&rng) & 7u];
                raw[i][b] = (uint8_t)(lo | (hi << 4));
            }
            /* Lattice-friendly f32 scales from fixed base 0.01f */
            {
                float base_f = 0.01f;
                uint32_t B = wnr4_i4_b(base_f);
                uint32_t gidx;
                for (m = 0; m < 3u; m++) {
                    for (gidx = 0; gidx < geo.cells[m]; gidx++) {
                        uint32_t kv = 110u + (uint32_t)(rng_next(&rng) & 7u);
                        uint32_t sv = wnr4_i4_lat(kv, B);
                        wnr4_put32(raw[i] + geo.soff[m] + (size_t)gidx * 4u, sv);
                    }
                }
            }
        }
        CHECK(wnr4_i4_count(&geo, raw[i], counts) == 0);
    }
    wnr4_i4_model_build(counts, mblob);
    CHECK(wnr4_i4_model_open(mblob, WNR4_I4_MODEL_BYTES, &model) == 0);

    for (i = 0; i < 4u; i++) {
        static const char *mnames[3] = { "gate_proj", "up_proj", "down_proj" };
        Wnr4I4Geo rgeo;
        uint32_t pairs = (i & 1u);
        size_t stored;
        CHECK(wnr4_i4_geo(&rgeo, 3u, rows, cols, pairs) == 0);
        stored = wnr4_i4_encode(&model, &rgeo, raw[i], i, 4u, enc[i], aligned_raw);
        CHECK(stored > 0 && stored < aligned_raw);
        memset(&recs[i], 0, sizeof(recs[i]));
        recs[i].stored_data   = enc[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = (uint32_t)raw_bytes;
        recs[i].codec         = 8u;
        recs[i].flags         = 0u;
        recs[i].blob_1based   = 1u;
        recs[i].owner         = i;
        recs[i].tensor_count  = 6u;
        for (m = 0; m < 3u; m++) {
            test_tensor_t *tc = &recs[i].tensors[m];
            test_tensor_t *ts = &recs[i].tensors[3u + m];
            snprintf(tc->name, sizeof(tc->name), "layers.0.experts.%u.%s.weight", i, mnames[m]);
            tc->dtype = WEX4_DT_U8; tc->rank = 2u;
            tc->shape[0] = rows[m]; tc->shape[1] = cols[m] / 2u;
            tc->offset = rgeo.coff[m]; tc->nbytes = (uint64_t)rows[m] * (cols[m] / 2u);

            snprintf(ts->name, sizeof(ts->name), "layers.0.experts.%u.%s.weight.qs", i, mnames[m]);
            ts->dtype = 13u; /* F32 */ ts->rank = 2u;
            ts->shape[0] = rows[m]; ts->shape[1] = cols[m] / 64u;
            ts->offset = rgeo.soff[m]; ts->nbytes = (uint64_t)rows[m] * (cols[m] / 64u) * 4u;
        }
    }
    blob.kind  = 3u;
    blob.id    = model.id;
    blob.data  = mblob;
    blob.bytes = WNR4_I4_MODEL_BYTES;
    write_wex4_file(path, &blob, 1, recs, 4, 0, 0, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.verify          = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_I4);
    CHECK(info.layers == 1u && info.experts == 4u);
    CHECK(wex_has_expert(s, 0, 2) == 1);
    {
        uint8_t *outs[4];
        void *dsts[4];
        size_t caps[4];
        int eids[4] = { 0, 1, 2, 3 };
        int st[4];
        for (i = 0; i < 4u; i++) {
            outs[i] = (uint8_t *)calloc(1, raw_bytes);
            CHECK(outs[i] != NULL);
            dsts[i] = outs[i];
            caps[i] = raw_bytes;
        }
        CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, st) == 0);
        for (i = 0; i < 4u; i++) {
            CHECK(st[i] == WEX_OK);
            CHECK(memcmp(outs[i], raw[i], raw_bytes) == 0);
            free(outs[i]);
        }
    }
    wex_close(s);
    remove(path);
    wnr4_i4_model_free(&model);
    free(counts);
    free(mblob);
    for (i = 0; i < 4u; i++) { free(raw[i]); free(enc[i]); }
}

/* ---- Test 6: WNR4-GG (codec 8, kind 2) ----------------------------------- */

static void test_wnr4_gg(const char *cuda_path) {
    const char *path = "wex_test_tmp_wnr4_gg.wex4";
    uint32_t types[2] = { 16u, WNR4_GG_Q2_0 };  /* slice 0 verbatim IQ2_XXS, slice 1 coded Q2_0 */
    uint32_t sizes[2] = { 1056u, 18u * 1024u }; /* 1056 B verbatim + 18432 B Q2_0 = 19488 B */
    Wnr4GgGeo geo;
    size_t raw_bytes, aligned_raw;
    uint8_t *raw[4], *enc[4];
    uint64_t *counts;
    uint8_t *mblob;
    Wnr4GgModel model;
    test_record_t recs[4];
    test_blob_t blob;
    uint64_t rng = UINT64_C(0x55AA55AA11223344);
    char root_hex[65], err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    uint32_t i;

    printf("[test] 6. WNR4-GG (codec 8, kind 2, verbatim + Q2_0)...\n");
    CHECK(wnr4_gg_geo(&geo, 2u, types, sizes) == 0);
    raw_bytes   = (size_t)geo.raw;
    aligned_raw = (size_t)wex4_align(raw_bytes);
    counts      = (uint64_t *)calloc(wnr4_gg_counts_words(), sizeof(uint64_t));
    mblob       = (uint8_t *)malloc(WNR4_GG_MODEL_BYTES);
    CHECK(counts && mblob);

    for (i = 0; i < 4u; i++) {
        uint32_t k, c;
        raw[i] = (uint8_t *)malloc(raw_bytes);
        enc[i] = (uint8_t *)malloc(aligned_raw);
        CHECK(raw[i] && enc[i]);
        for (k = 0; k < sizes[0]; k++) raw[i][k] = (uint8_t)(rng_next(&rng) & 0xffu);
        for (c = 0; c < geo.cells[1]; c++) {
            uint8_t *blk = raw[i] + geo.off[1] + (size_t)c * 18u;
            static const uint8_t hi_choices[4] = { 0x2c, 0x2c, 0x2d, 0x2e };
            static const uint8_t code_choices[4] = { 0x00, 0x01, 0x10, 0x11 };
            blk[0] = (uint8_t)(rng_next(&rng) & 0xffu);
            blk[1] = hi_choices[rng_next(&rng) & 3u];
            for (k = 0; k < 16u; k++) blk[2u + k] = code_choices[rng_next(&rng) & 3u];
        }
        wnr4_gg_count(&geo, raw[i], counts);
    }
    CHECK(wnr4_gg_model_build(counts, mblob) == 0);
    CHECK(wnr4_gg_model_open(mblob, WNR4_GG_MODEL_BYTES, &model) == 0);

    for (i = 0; i < 4u; i++) {
        size_t stored = wnr4_gg_encode(&model, &geo, raw[i], i, 4u, enc[i], aligned_raw);
        CHECK(stored > 0 && stored < aligned_raw);
        memset(&recs[i], 0, sizeof(recs[i]));
        recs[i].stored_data   = enc[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = (uint32_t)raw_bytes;
        recs[i].codec         = 8u;
        recs[i].flags         = 0u;
        recs[i].blob_1based   = 1u;
        recs[i].owner         = i;
        recs[i].tensor_count  = 2u;

        snprintf(recs[i].tensors[0].name, sizeof(recs[i].tensors[0].name),
                 "blk.0.ffn_gate_exps.weight#%u", i);
        recs[i].tensors[0].dtype    = (uint8_t)(128u + types[0]);
        recs[i].tensors[0].rank     = 2u;
        recs[i].tensors[0].shape[0] = 16u;
        recs[i].tensors[0].shape[1] = 256u;
        recs[i].tensors[0].offset   = geo.off[0];
        recs[i].tensors[0].nbytes   = sizes[0];

        snprintf(recs[i].tensors[1].name, sizeof(recs[i].tensors[1].name),
                 "blk.0.ffn_down_exps.weight#%u", i);
        recs[i].tensors[1].dtype    = (uint8_t)(128u + types[1]);
        recs[i].tensors[1].rank     = 2u;
        recs[i].tensors[1].shape[0] = 64u;
        recs[i].tensors[1].shape[1] = 1024u;
        recs[i].tensors[1].offset   = geo.off[1];
        recs[i].tensors[1].nbytes   = sizes[1];
    }
    blob.kind  = 3u;
    blob.id    = model.id;
    blob.data  = mblob;
    blob.bytes = WNR4_GG_MODEL_BYTES;
    write_wex4_file(path, &blob, 1, recs, 4, 0, 0, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.verify          = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_GG);
    CHECK(info.layers == 1u && info.experts == 4u);
    CHECK(wex_has_expert(s, 0, 3) == 1);
    {
        uint8_t *outs[4];
        void *dsts[4];
        size_t caps[4];
        int eids[4] = { 0, 1, 2, 3 };
        int st[4];
        for (i = 0; i < 4u; i++) {
            outs[i] = (uint8_t *)calloc(1, raw_bytes);
            CHECK(outs[i] != NULL);
            dsts[i] = outs[i];
            caps[i] = raw_bytes;
        }
        CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, st) == 0);
        for (i = 0; i < 4u; i++) {
            CHECK(st[i] == WEX_OK);
            CHECK(memcmp(outs[i], raw[i], raw_bytes) == 0);
            free(outs[i]);
        }
    }
    wex_close(s);
    remove(path);
    wnr4_gg_model_free(&model);
    free(counts);
    free(mblob);
    for (i = 0; i < 4u; i++) { free(raw[i]); free(enc[i]); }
}

/* ---- Test 7: Root pinning & corruption refusal --------------------------- */

static void test_root_and_corruption(void) {
    const char *path = "wex_test_tmp_corrupt.wex4";
    uint8_t expected[8][EXP_DEC_BYTES];
    uint32_t stored[8];
    char root_hex[65], bad_root[65], err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_layout_t lay;
    uint8_t out[EXP_DEC_BYTES], sentinel[EXP_DEC_BYTES];
    void *dsts[1] = { out };
    size_t caps[1] = { EXP_DEC_BYTES };
    int eid0[1] = { 0 }, eid1[1] = { 1 }, st[1];
    FILE *fp;
    uint8_t byte_val = 0;

    printf("[test] 7. Root pinning & corruption refusal (VERIFY_ON & VERIFY_OFF)...\n");
    build_wnr3_container(path, 0, expected, stored, root_hex);
    memset(sentinel, 0xA5, sizeof(sentinel));

    /* Wrong root_hex must refuse open */
    memcpy(bad_root, root_hex, 65);
    bad_root[0] = (bad_root[0] == 'a') ? 'b' : 'a';
    wex_open_opts_init(&opts);
    opts.gpu_device = -2;
    opts.root_hex   = bad_root;
    err[0] = '\0';
    CHECK(wex_open(path, &opts, err, sizeof(err)) == NULL);
    CHECK(err[0] != '\0');

    /* Locate record 0 file offset before corrupting 1 bit inside its compressed payload */
    opts.root_hex = root_hex;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_record_layout(s, 0, 0, 0, &lay) == WEX_OK);
    wex_close(s);

    /* Flip 1 bit in the middle of record 0's stored bytes on disk */
    fp = fopen(path, "r+b");
    CHECK(fp != NULL);
    CHECK(fseek(fp, (long)(lay.file_offset + 512u), SEEK_SET) == 0);
    CHECK(fread(&byte_val, 1, 1, fp) == 1);
    byte_val ^= 0x01u;
    CHECK(fseek(fp, (long)(lay.file_offset + 512u), SEEK_SET) == 0);
    CHECK(fwrite(&byte_val, 1, 1, fp) == 1);
    CHECK(fclose(fp) == 0);

    /* Case A: WEX_VERIFY_ON -> 1st fetch = WEX_E_HASH, 2nd fetch = WEX_E_REFUSED, dst untouched */
    wex_open_opts_init(&opts);
    opts.gpu_device     = -2;
    opts.verify         = WEX_VERIFY_ON;
    opts.l2_cache_bytes = 8u * EXP_RAW_ALIGNED;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    memcpy(out, sentinel, sizeof(out));
    CHECK(wex_fetch_host(s, 0, eid0, 1, dsts, caps, 0, st) == 1);
    CHECK(st[0] == WEX_E_HASH);
    CHECK(memcmp(out, sentinel, sizeof(out)) == 0);

    CHECK(wex_fetch_host(s, 0, eid0, 1, dsts, caps, 0, st) == 1);
    CHECK(st[0] == WEX_E_REFUSED);

    /* Uncorrupted record 1 still succeeds */
    CHECK(wex_fetch_host(s, 0, eid1, 1, dsts, caps, 0, st) == 0);
    CHECK(st[0] == WEX_OK);
    CHECK(memcmp(out, expected[1], EXP_DEC_BYTES) == 0);
    wex_close(s);

    /* Case B: WEX_VERIFY_OFF -> Level-1 stream hash catches corruption -> WEX_E_DECODE */
    wex_open_opts_init(&opts);
    opts.gpu_device = -2;
    opts.verify     = WEX_VERIFY_OFF;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    memcpy(out, sentinel, sizeof(out));
    CHECK(wex_fetch_host(s, 0, eid0, 1, dsts, caps, 0, st) == 1);
    CHECK(st[0] == WEX_E_DECODE);
    CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
    wex_close(s);

    remove(path);
}

/* ---- Test 8: Dynamic loader wex_loader.h --------------------------------- */

static void test_dynamic_loader(const char *lib_path) {
    wex_lib_t lib;
    printf("[test] 8. Dynamic loader wex_loader.h...\n");
    CHECK(wex_lib_open(&lib, "nonexistent_libwex_99999.dll") < 0);
    CHECK(lib.handle == NULL);

    if (lib_path && lib_path[0]) {
        const char *path = "wex_test_tmp_dyn.wex4";
        uint8_t expected[8][EXP_DEC_BYTES];
        char root_hex[65], err[256];
        wex_open_opts_t opts;
        wex_store_t *s;
        uint8_t out[EXP_DEC_BYTES];
        void *dsts[1] = { out };
        size_t caps[1] = { EXP_DEC_BYTES };
        int eid[1] = { 2 }, st[1];

        build_wnr3_container(path, 0, expected, NULL, root_hex);
        CHECK(wex_lib_open(&lib, lib_path) == WEX_OK);
        CHECK(lib.abi_version() == WEX_STORE_ABI_VERSION);
        wex_open_opts_init(&opts);
        opts.gpu_device = -2;
        opts.root_hex   = root_hex;
        s = lib.open(path, &opts, err, sizeof(err));
        CHECK(s != NULL);
        CHECK(lib.fetch_host(s, 0, eid, 1, dsts, caps, 0, st) == 0);
        CHECK(st[0] == WEX_OK && memcmp(out, expected[2], EXP_DEC_BYTES) == 0);
        lib.close(s);
        wex_lib_close(&lib);
        remove(path);
        printf("       dynamic DLL round-trip OK (%s)\n", lib_path);
    }
}

/* ---- Test 9: L2 fixed-slot pinned arena cache unit test ------------------ */

static int g_mock_pins = 0;
static int g_mock_unpins = 0;
static int mock_pin_cb(void *ptr, size_t bytes) {
    if (!ptr || !bytes) return -1;
    g_mock_pins++;
    return 0;
}
static int mock_unpin_cb(void *ptr) {
    if (!ptr) return -1;
    g_mock_unpins++;
    return 0;
}

static void test_l2_pinned_arena(void) {
    uint8_t b0[4096], b1[4096], b2[4096];
    const uint8_t *p = NULL;
    size_t len = 0;
    wex_cache_handle_t h0, h1;
    wex_cache_t *c;
    uint64_t used = 0, cnt = 0;

    printf("[test] 9. L2 fixed-slot pinned arena & refcount borrow protection...\n");
    memset(b0, 0x11, sizeof(b0));
    memset(b1, 0x22, sizeof(b1));
    memset(b2, 0x33, sizeof(b2));
    g_mock_pins = g_mock_unpins = 0;

    c = wex_cache_create(8192u, 4096u, 4096u, mock_pin_cb, mock_unpin_cb);
    CHECK(c != NULL);
    CHECK(wex_cache_is_pinned(c) == 1 && g_mock_pins == 1);
    CHECK(wex_cache_capacity(c) == 8192u);

    CHECK(wex_cache_put(c, 10u, b0, 4096u) == 1);
    CHECK(wex_cache_put(c, 20u, b1, 4096u) == 1);
    wex_cache_stats(c, &used, &cnt);
    CHECK(used == 8192u && cnt == 2u);

    /* Borrow BOTH entries: inserting a 3rd must NOT evict pinned borrows! */
    CHECK(wex_cache_get(c, 10u, &p, &len, &h0) == 1 && len == 4096u && memcmp(p, b0, 4096u) == 0);
    CHECK(wex_cache_get(c, 20u, &p, &len, &h1) == 1 && len == 4096u && memcmp(p, b1, 4096u) == 0);
    CHECK(wex_cache_put(c, 30u, b2, 4096u) == 0);

    /* Release entry 10 (oldest): now inserting 30 evicts 10 while keeping borrowed 20 intact! */
    wex_cache_release(c, h0);
    CHECK(wex_cache_put(c, 30u, b2, 4096u) == 1);
    CHECK(wex_cache_contains(c, 10u) == 0);
    CHECK(wex_cache_contains(c, 20u) == 1);
    CHECK(wex_cache_contains(c, 30u) == 1);
    CHECK(memcmp(p, b1, 4096u) == 0);
    wex_cache_release(c, h1);

    wex_cache_destroy(c);
    CHECK(g_mock_unpins == 1);
}

/* ---- Test 10: WNR4-NV (codec 8, kind 1) ---------------------------------- */

static void test_wnr4_nv(const char *cuda_path) {
    const char *path = "wex_test_tmp_wnr4_nv.wex4";
    /* NVFP4: 1 E4M3 scale per 16 nibbles = 8 weight bytes -> wcols = 8 * scols.
     * rows = 64, scols = 16 -> wcols = 128; scale bytes/mat = 1024, weight bytes/mat = 8192.
     * 3 matrices: scales = 3072, weights = 24576, decoded = 27648 -> aligned = 28672. */
    const uint32_t nv_rows = 64u, nv_scols = 16u, nv_wcols = 128u;
    const uint32_t nv_sc_mat = nv_rows * nv_scols, nv_wt_mat = nv_rows * nv_wcols;
    const uint32_t nv_sc_tot = 3u * nv_sc_mat, nv_wt_tot = 3u * nv_wt_mat;
    const uint32_t nv_dec = nv_sc_tot + nv_wt_tot;
    const uint32_t nv_aln = (uint32_t)wex4_align(nv_dec);
    uint64_t rng = UINT64_C(0x13579BDF2468ACE0);
    uint8_t *raw[4], *enc[4];
    uint32_t scount[3], scols[3], soff[3], woff[3];
    Wnr4Geo geo;
    size_t cwords = wnr4_counts_words(WNR4_KIND_NV, 1u);
    size_t mbytes = wnr4_model_bytes(WNR4_KIND_NV, 1u);
    uint64_t *counts = (uint64_t *)calloc(cwords, sizeof(uint64_t));
    uint8_t *mblob = (uint8_t *)malloc(mbytes);
    Wnr4Model model;
    test_record_t recs[4];
    test_blob_t blob;
    char root_hex[65], err[256];
    wex_open_opts_t opts;
    wex_store_t *s;
    wex_info_t info;
    uint32_t i, m;

    printf("[test] 10. WNR4-NV (codec 8, kind 1)...\n");
    CHECK(counts && mblob);
    for (m = 0; m < 3u; m++) {
        scount[m] = nv_sc_mat; scols[m] = nv_scols;
        soff[m] = m * nv_sc_mat; woff[m] = nv_sc_tot + m * nv_wt_mat;
    }
    CHECK(wnr4_geo(&geo, WNR4_KIND_NV, scount, scols, soff, woff) == 0);

    for (i = 0; i < 4u; i++) {
        unsigned char anchor[3];
        uint32_t k;
        static const uint8_t nibs[8] = {0, 0, 1, 1, 2, 9, 9, 10};
        raw[i] = (uint8_t *)malloc(nv_dec);
        enc[i] = (uint8_t *)malloc(nv_aln);
        CHECK(raw[i] && enc[i]);
        for (k = 0; k < nv_sc_tot; k++) raw[i][k] = (uint8_t)(48u + (rng_next(&rng) & 3u));
        for (k = 0; k < nv_wt_tot; k++) {
            raw[i][nv_sc_tot + k] = (uint8_t)(nibs[rng_next(&rng) & 7u] | (nibs[rng_next(&rng) & 7u] << 4));
        }
        for (m = 0; m < 3u; m++) anchor[m] = wnr4_anchor(raw[i] + soff[m], scount[m]);
        wnr4_count(&geo, raw[i], anchor, 1u, counts);
    }
    wnr4_model_build(WNR4_KIND_NV, 1u, counts, mblob);
    CHECK(wnr4_model_open(mblob, mbytes, &model) == 0);

    for (i = 0; i < 4u; i++) {
        size_t stored = wnr4_encode(&model, &geo, raw[i], 0u, i, 4u, enc[i], nv_aln);
        CHECK(stored > 0 && stored < nv_aln);
        memset(&recs[i], 0, sizeof(recs[i]));
        recs[i].stored_data   = enc[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = nv_dec;
        recs[i].codec         = 8u;
        recs[i].flags         = WEX4_REC_EXPERT;
        recs[i].blob_1based   = 1u;
        recs[i].owner         = i;
        recs[i].tensor_count  = 6u;
        for (m = 0; m < 6u; m++) {
            test_tensor_t *ten = &recs[i].tensors[m];
            snprintf(ten->name, sizeof(ten->name), "layers.0.ffn.experts.%u.w%u.%s",
                     i, (m % 3u) + 1u, m < 3u ? "scale" : "weight");
            ten->dtype    = (m < 3u) ? 5u /* F8_E4M3 */ : WEX4_DT_U8;
            ten->rank     = 2u;
            ten->shape[0] = nv_rows;
            ten->shape[1] = (m < 3u) ? nv_scols : nv_wcols;
            ten->offset   = (m < 3u) ? soff[m] : woff[m - 3u];
            ten->nbytes   = ten->shape[0] * ten->shape[1];
        }
    }
    blob.kind  = 3u;
    blob.id    = model.id;
    blob.data  = mblob;
    blob.bytes = mbytes;
    write_wex4_file(path, &blob, 1, recs, 4, 1, 4, 0, root_hex);

    wex_open_opts_init(&opts);
    opts.gpu_device      = cuda_path ? 0 : -2;
    opts.gpu_helper_path = cuda_path;
    opts.verify          = WEX_VERIFY_ON;
    s = wex_open(path, &opts, err, sizeof(err));
    CHECK(s != NULL);
    CHECK(wex_info(s, &info) == WEX_OK);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_NV);
    {
        uint8_t *outs[4];
        void *dsts[4];
        size_t caps[4];
        int eids[4] = { 0, 1, 2, 3 };
        int st[4];
        for (i = 0; i < 4u; i++) {
            outs[i] = (uint8_t *)calloc(1, nv_dec);
            CHECK(outs[i] != NULL);
            dsts[i] = outs[i];
            caps[i] = nv_dec;
        }
        CHECK(wex_fetch_host(s, 0, eids, 4, dsts, caps, 0, st) == 0);
        for (i = 0; i < 4u; i++) {
            CHECK(st[i] == WEX_OK);
            CHECK(memcmp(outs[i], raw[i], nv_dec) == 0);
            free(outs[i]);
        }
    }
    wex_close(s);
    remove(path);
    wnr4_model_free(&model);
    free(counts);
    free(mblob);
    for (i = 0; i < 4u; i++) { free(raw[i]); free(enc[i]); }
}

int main(int argc, char **argv) {
    const char *cuda_path = NULL;
    const char *lib_path  = getenv("WEX_TEST_DLL");
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cuda") == 0 && i + 1 < argc) cuda_path = argv[++i];
        else if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc) lib_path = argv[++i];
    }

    test_raw_and_api(cuda_path);
    test_wbp1();
    test_wnr3_l2_prefetch_mt(cuda_path);
    test_wnr4_mx(cuda_path);
    test_wnr4_i4(cuda_path);
    test_wnr4_gg(cuda_path);
    test_root_and_corruption();
    test_dynamic_loader(lib_path);
    test_l2_pinned_arena();
    test_wnr4_nv(cuda_path);

    printf("ALL LIBWEX C TESTS PASSED (%d checks)\n", g_checks);
    return 0;
}
