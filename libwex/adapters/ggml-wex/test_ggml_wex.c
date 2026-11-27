/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

/* Live verification suite for the ggml-wex dynamic plugin (libggml-wex.{dll,so,dylib}):
 *   - Builds a genuine synthetic WEX4 WNR4-GG MoE container (2 layers x 8 experts,
 *     coded Q2_0 18-byte/64-weight blocks + WNM4 kind-2 model blob)
 *   - Dynamically loads libggml-wex and libwex (zero link-time dependencies)
 *   - Verifies ggml_wex_backend_score() == 1, WEX_VERIFY_ON + root pinning + auto L2
 *   - Verifies ggml_wex_dequant_row_q2_0() exact parity vs reference fp16+Q2_0 decoder
 *   - Executes deduplicated GGML_OP_MUL_MAT_ID (n_tokens=4, top_k=2 with duplicates)
 *     against exact reference dot products (max_abs_err < 1e-6)
 *   - Verifies L2 cache hits on second pass, deduplicated ggml_wex_fetch_experts,
 *     and ggml_wex_prefetch_layer() warming layer 1
 *
 * Usage: test_ggml_wex [--plugin PATH] [--lib PATH] [--cuda PATH|0] */

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "ggml_wex.h"

#include "../../../c/wex4.h"
#include "../../../wnr4/wnr4.h"
#include "../../../wnr4/wnr4_gg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void test_sleep_ms(unsigned ms) { Sleep(ms); }
static int test_setenv(const char *k, const char *v) { return _putenv_s(k, v ? v : ""); }
static void *test_dlopen(const char *p) {
    if (strchr(p, '\\') || strchr(p, '/')) {
        HMODULE m = LoadLibraryExA(p, NULL,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (m) return (void *)m;
    }
    return (void *)LoadLibraryA(p);
}
static void test_dlclose(void *h) { if (h) FreeLibrary((HMODULE)h); }
static int test_dlsym(void *h, const char *name, void *out) {
    FARPROC p = GetProcAddress((HMODULE)h, name);
    if (!p) return -1;
    memcpy(out, &p, sizeof(p));
    return 0;
}
#else
#include <dlfcn.h>
#include <time.h>
static void test_sleep_ms(unsigned ms) {
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    nanosleep(&ts, NULL);
}
static int test_setenv(const char *k, const char *v) { return v ? setenv(k, v, 1) : unsetenv(k); }
static void *test_dlopen(const char *p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void test_dlclose(void *h) { if (h) dlclose(h); }
static int test_dlsym(void *h, const char *name, void *out) {
    void *p = dlsym(h, name);
    if (!p) return -1;
    memcpy(out, &p, sizeof(p));
    return 0;
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

static uint64_t rng_next(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * UINT64_C(2685821657736338717);
}

static void put_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static void put_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u64_le(uint8_t *p, uint64_t v) {
    put_u32_le(p, (uint32_t)v); put_u32_le(p + 4, (uint32_t)(v >> 32));
}

/* ---- Independent reference Q2_0 dequantizer (18 bytes / 64 weights) ------ */

static float ref_fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 31;
    int mant = h & 1023;
    double v;
    if (exp == 0) {
        v = ldexp((double)mant, -24);
    } else if (exp == 31) {
        v = mant ? 0.0 : HUGE_VAL;
    } else {
        v = ldexp((double)(1024 + mant), exp - 25);
    }
    return (float)(sign ? -v : v);
}

static void ref_dequant_row_q2_0(const uint8_t *src, float *dst, int64_t k) {
    int64_t nb = k / GGML_WEX_QK2_0;
    int64_t b;
    for (b = 0; b < nb; b++) {
        const uint8_t *blk = src + (size_t)b * GGML_WEX_Q2_0_BYTES;
        uint16_t d_bits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float d = ref_fp16_to_fp32(d_bits);
        int j;
        for (j = 0; j < 16; j++) {
            uint8_t byte = blk[2 + j];
            dst[b * 64 + j * 4 + 0] = (float)((int)((byte >> 0) & 3u) - 1) * d;
            dst[b * 64 + j * 4 + 1] = (float)((int)((byte >> 2) & 3u) - 1) * d;
            dst[b * 64 + j * 4 + 2] = (float)((int)((byte >> 4) & 3u) - 1) * d;
            dst[b * 64 + j * 4 + 3] = (float)((int)((byte >> 6) & 3u) - 1) * d;
        }
    }
}

/* ---- Synthetic WEX4 WNR4-GG container builder (2 layers x 8 experts) ----- */

#define GG_LAYERS    2u
#define GG_EXPERTS   8u
#define GG_NRECS     (GG_LAYERS * GG_EXPERTS)
#define GG_ROWS      128u
#define GG_COLS      256u
#define GG_ROW_BLKS  (GG_COLS / GGML_WEX_QK2_0)
#define GG_SLICE_B   (GG_ROWS * GG_ROW_BLKS * GGML_WEX_Q2_0_BYTES) /* 9216 B */
#define GG_SLAB_B    (2u * GG_SLICE_B)                             /* 18432 B */
#define GG_ALN_B     20480u                                        /* 5 pages */

typedef struct {
    char     name[96];
    uint8_t  dtype;
    uint8_t  rank;
    uint64_t shape[8];
    uint64_t offset;
    uint64_t nbytes;
} gg_tensor_t;

typedef struct {
    const uint8_t *stored_data;
    uint32_t       stored_bytes;
    uint32_t       decoded_bytes;
    uint16_t       codec;
    uint16_t       flags;
    uint32_t       blob_1based;
    uint32_t       owner;
    gg_tensor_t    tensors[2];
    uint32_t       tensor_count;
} gg_record_t;

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

static void write_wex4_gg_file(const char *path,
                               uint32_t blob_id, const uint8_t *blob_data, uint64_t blob_bytes,
                               const gg_record_t *recs, uint32_t nrecs,
                               char root_hex_out[65]) {
    uint32_t ntensors = 0, r, k, t = 0;
    uint64_t names_bytes = 0, total_decoded = 0, max_decoded = 0;
    uint64_t pos = WEX4_PAGE, blob_off = WEX4_PAGE;
    uint64_t *rec_off = NULL;
    char *names = NULL;
    uint32_t *name_at = NULL, *name_len = NULL, *order = NULL;
    uint64_t index_bytes, index_len, index_offset, file_bytes;
    uint8_t *file_buf = NULL, *idx;
    FILE *fp;

    for (r = 0; r < nrecs; r++) {
        ntensors += recs[r].tensor_count;
        total_decoded += recs[r].decoded_bytes;
        if (recs[r].decoded_bytes > max_decoded) max_decoded = recs[r].decoded_bytes;
        for (k = 0; k < recs[r].tensor_count; k++) names_bytes += strlen(recs[r].tensors[k].name);
    }

    rec_off  = (uint64_t *)calloc(nrecs, sizeof(uint64_t));
    names    = (char *)malloc((size_t)names_bytes);
    name_at  = (uint32_t *)calloc(ntensors, sizeof(uint32_t));
    name_len = (uint32_t *)calloc(ntensors, sizeof(uint32_t));
    order    = (uint32_t *)calloc(ntensors, sizeof(uint32_t));
    CHECK(rec_off && names && name_at && name_len && order);

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

    pos += wex4_align(blob_bytes);
    for (r = 0; r < nrecs; r++) {
        CHECK((recs[r].stored_bytes & (WEX4_PAGE - 1u)) == 0u);
        rec_off[r] = pos;
        pos += recs[r].stored_bytes;
    }

    index_bytes  = (uint64_t)WEX4_BLOB_BYTES + (uint64_t)nrecs * WEX4_RECORD_BYTES +
                   (uint64_t)ntensors * (WEX4_TENSOR_BYTES + 4u) + names_bytes;
    index_len    = wex4_align(index_bytes);
    index_offset = pos;
    file_bytes   = index_offset + index_len;

    file_buf = (uint8_t *)calloc(1, (size_t)file_bytes);
    CHECK(file_buf != NULL);

    memcpy(file_buf + blob_off, blob_data, (size_t)blob_bytes);
    for (r = 0; r < nrecs; r++) {
        memcpy(file_buf + rec_off[r], recs[r].stored_data, recs[r].stored_bytes);
    }

    idx = file_buf + index_offset;
    {
        size_t at = 0;
        uint32_t first_t = 0;
        put_u32_le(idx + at + 0,  3u); /* WNM4 blob kind */
        put_u32_le(idx + at + 4,  blob_id);
        put_u64_le(idx + at + 8,  blob_off);
        put_u64_le(idx + at + 16, blob_bytes);
        wex4_sha256(blob_data, (size_t)blob_bytes, idx + at + 24);
        at += WEX4_BLOB_BYTES;

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
        t = 0;
        for (r = 0; r < nrecs; r++) {
            for (k = 0; k < recs[r].tensor_count; k++) {
                const gg_tensor_t *ten = &recs[r].tensors[k];
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
        for (t = 0; t < ntensors; t++) {
            put_u32_le(idx + at, order[t]);
            at += 4u;
        }
        memcpy(idx + at, names, (size_t)names_bytes);
        at += (size_t)names_bytes;
        CHECK(at == index_bytes);
    }

    put_u32_le(file_buf + 0,  WEX4_MAGIC);
    put_u16_le(file_buf + 4,  4u);
    put_u16_le(file_buf + 6,  0u);
    put_u64_le(file_buf + 8,  0u);
    put_u64_le(file_buf + 16, 0u);
    put_u64_le(file_buf + 24, file_bytes);
    put_u64_le(file_buf + 32, index_offset);
    put_u64_le(file_buf + 40, index_bytes);
    put_u32_le(file_buf + 48, 1u);
    put_u32_le(file_buf + 52, nrecs);
    put_u32_le(file_buf + 56, ntensors);
    put_u32_le(file_buf + 60, 0u); /* non-EXPERT GGUF container: grid discovered from blk.<L>...#<E> */
    put_u32_le(file_buf + 64, 0u);
    put_u32_le(file_buf + 68, (uint32_t)max_decoded);
    put_u64_le(file_buf + 72, names_bytes);
    put_u64_le(file_buf + 80, 0u);
    put_u64_le(file_buf + 88, total_decoded);
    wex4_sha256(file_buf + index_offset, (size_t)index_bytes, file_buf + 96);
    wex4_sha256(file_buf, WEX4_HEADER_HASHED, file_buf + WEX4_HEADER_HASHED);

    {
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
    free(rec_off);
    free(names);
    free(name_at);
    free(name_len);
    free(order);
}

static void build_wnr4_gg_container(const char *path,
                                    uint8_t *raw_slabs[GG_NRECS],
                                    char root_hex_out[65]) {
    uint32_t types[2] = { WNR4_GG_Q2_0, WNR4_GG_Q2_0 };
    uint32_t sizes[2] = { GG_SLICE_B, GG_SLICE_B };
    Wnr4GgGeo geo;
    uint64_t *counts;
    uint8_t *mblob;
    uint8_t *enc[GG_NRECS];
    Wnr4GgModel model;
    gg_record_t recs[GG_NRECS];
    uint64_t rng = UINT64_C(0x661120261127AA55);
    uint32_t r;

    CHECK(wnr4_gg_geo(&geo, 2u, types, sizes) == 0);
    CHECK(geo.raw == GG_SLAB_B);
    counts = (uint64_t *)calloc(wnr4_gg_counts_words(), sizeof(uint64_t));
    mblob  = (uint8_t *)malloc(WNR4_GG_MODEL_BYTES);
    CHECK(counts && mblob);

    for (r = 0; r < GG_NRECS; r++) {
        uint32_t s, c, k;
        /* Skewed realistic Q2_0 scales (small positive & negative fp16 values) and codes */
        static const uint8_t hi_choices[4] = { 0x30, 0x34, 0x38, 0xb4 };
        static const uint8_t code_choices[8] = { 0x55, 0x54, 0x56, 0x45, 0x65, 0x15, 0x59, 0x95 };
        raw_slabs[r] = (uint8_t *)malloc(GG_SLAB_B);
        enc[r]       = (uint8_t *)malloc(GG_ALN_B);
        CHECK(raw_slabs[r] && enc[r]);
        for (s = 0; s < 2u; s++) {
            for (c = 0; c < geo.cells[s]; c++) {
                uint8_t *blk = raw_slabs[r] + geo.off[s] + (size_t)c * GGML_WEX_Q2_0_BYTES;
                blk[0] = (uint8_t)(rng_next(&rng) & 0x3fu);
                blk[1] = hi_choices[rng_next(&rng) & 3u];
                for (k = 0; k < 16u; k++) blk[2u + k] = code_choices[rng_next(&rng) & 7u];
            }
        }
        wnr4_gg_count(&geo, raw_slabs[r], counts);
    }

    CHECK(wnr4_gg_model_build(counts, mblob) == 0);
    CHECK(wnr4_gg_model_open(mblob, WNR4_GG_MODEL_BYTES, &model) == 0);

    for (r = 0; r < GG_NRECS; r++) {
        uint32_t layer = r / GG_EXPERTS;
        uint32_t eid   = r % GG_EXPERTS;
        size_t stored  = wnr4_gg_encode(&model, &geo, raw_slabs[r], r, 4u, enc[r], GG_ALN_B);
        CHECK(stored > 0 && stored < GG_ALN_B);

        memset(&recs[r], 0, sizeof(recs[r]));
        recs[r].stored_data   = enc[r];
        recs[r].stored_bytes  = (uint32_t)stored;
        recs[r].decoded_bytes = GG_SLAB_B;
        recs[r].codec         = 8u;
        recs[r].flags         = 0u;
        recs[r].blob_1based   = 1u;
        recs[r].owner         = r;
        recs[r].tensor_count  = 2u;

        snprintf(recs[r].tensors[0].name, sizeof(recs[r].tensors[0].name),
                 "blk.%u.ffn_gate_exps.weight#%u", layer, eid);
        recs[r].tensors[0].dtype    = (uint8_t)(128u + WNR4_GG_Q2_0);
        recs[r].tensors[0].rank     = 2u;
        recs[r].tensors[0].shape[0] = GG_ROWS;
        recs[r].tensors[0].shape[1] = GG_COLS;
        recs[r].tensors[0].offset   = geo.off[0];
        recs[r].tensors[0].nbytes   = GG_SLICE_B;

        snprintf(recs[r].tensors[1].name, sizeof(recs[r].tensors[1].name),
                 "blk.%u.ffn_down_exps.weight#%u", layer, eid);
        recs[r].tensors[1].dtype    = (uint8_t)(128u + WNR4_GG_Q2_0);
        recs[r].tensors[1].rank     = 2u;
        recs[r].tensors[1].shape[0] = GG_ROWS;
        recs[r].tensors[1].shape[1] = GG_COLS;
        recs[r].tensors[1].offset   = geo.off[1];
        recs[r].tensors[1].nbytes   = GG_SLICE_B;
    }

    write_wex4_gg_file(path, model.id, mblob, WNR4_GG_MODEL_BYTES, recs, GG_NRECS, root_hex_out);

    wnr4_gg_model_free(&model);
    free(counts);
    free(mblob);
    for (r = 0; r < GG_NRECS; r++) free(enc[r]);
}

/* ---- Dynamic plugin symbol table ----------------------------------------- */

typedef struct {
    void *handle;
    uint32_t        (*abi_version)(void);
    int             (*backend_score)(void);
    ggml_wex_ctx_t *(*open)(const char *, const ggml_wex_opts_t *, char *, size_t);
    int             (*info)(const ggml_wex_ctx_t *, wex_info_t *);
    void            (*stats)(const ggml_wex_ctx_t *, wex_stats_t *);
    void            (*close)(ggml_wex_ctx_t *);
    int             (*prefetch_layer)(ggml_wex_ctx_t *, int, const int *, int);
    int             (*fetch_experts)(ggml_wex_ctx_t *, int, const int *, int,
                                     void *const *, const size_t *, int *);
    void            (*dequant_row_q2_0)(const void *, float *, int64_t);
    int             (*mul_mat_id_q2_0_f32)(ggml_wex_ctx_t *, int, int, const int32_t *,
                                           int, int, const float *, int64_t, int64_t, float *);
} ggml_wex_plugin_t;

static void load_plugin(ggml_wex_plugin_t *p, const char *path) {
    memset(p, 0, sizeof(*p));
    p->handle = test_dlopen(path);
    CHECK(p->handle != NULL);
    CHECK(test_dlsym(p->handle, "ggml_wex_abi_version",         &p->abi_version) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_backend_score",       &p->backend_score) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_open",                &p->open) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_info",                &p->info) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_stats",               &p->stats) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_close",               &p->close) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_prefetch_layer",      &p->prefetch_layer) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_fetch_experts",       &p->fetch_experts) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_dequant_row_q2_0",    &p->dequant_row_q2_0) == 0);
    CHECK(test_dlsym(p->handle, "ggml_wex_mul_mat_id_q2_0_f32", &p->mul_mat_id_q2_0_f32) == 0);
}

static float verify_mul_mat_id(const uint8_t *const raw_slabs[GG_NRECS],
                               int layer, int tensor_slot,
                               const int32_t *ids, int n_tokens, int n_used,
                               const float *x_f32, const float *got_out) {
    float row_f32[GG_COLS];
    float max_err = 0.0f;
    int tok, slot;
    size_t slice_off = (size_t)tensor_slot * GG_SLICE_B;
    size_t row_bytes = GG_ROW_BLKS * GGML_WEX_Q2_0_BYTES;

    for (tok = 0; tok < n_tokens; tok++) {
        const float *xt = x_f32 + (size_t)tok * GG_COLS;
        for (slot = 0; slot < n_used; slot++) {
            int eid = (int)ids[tok * n_used + slot];
            const uint8_t *mat = raw_slabs[layer * (int)GG_EXPERTS + eid] + slice_off;
            const float *got_vec = got_out + ((size_t)tok * (size_t)n_used + (size_t)slot) * GG_ROWS;
            uint32_t row;
            for (row = 0; row < GG_ROWS; row++) {
                float acc = 0.0f;
                uint32_t c;
                float diff;
                ref_dequant_row_q2_0(mat + (size_t)row * row_bytes, row_f32, GG_COLS);
                for (c = 0; c < GG_COLS; c++) acc += row_f32[c] * xt[c];
                diff = fabsf(got_vec[row] - acc);
                if (diff > max_err) max_err = diff;
            }
        }
    }
    return max_err;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    const char *plugin_path = "libggml-wex.dll";
#elif defined(__APPLE__)
    const char *plugin_path = "libggml-wex.dylib";
#else
    const char *plugin_path = "libggml-wex.so";
#endif
    const char *lib_path  = getenv("WEX_LIB_PATH");
    const char *cuda_path = NULL;
    const char *wex_file  = "ggml_wex_test_wnr4gg.wex4";
    uint8_t *raw_slabs[GG_NRECS];
    char root_hex[65], bad_root[65], err[256];
    ggml_wex_plugin_t pl;
    ggml_wex_opts_t opts;
    ggml_wex_ctx_t *ctx;
    wex_info_t info;
    wex_stats_t st0, st1, st2;
    float x_f32[4 * GG_COLS];
    float out_f32[4 * 2 * GG_ROWS];
    uint64_t rng = UINT64_C(0x123456789ABCDEF0);
    int i, expect_cuda;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--plugin") == 0 && i + 1 < argc)     plugin_path = argv[++i];
        else if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc)   lib_path    = argv[++i];
        else if (strcmp(argv[i], "--cuda") == 0 && i + 1 < argc)  cuda_path   = argv[++i];
        else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc)  wex_file    = argv[++i];
    }
    if (!cuda_path || !cuda_path[0]) cuda_path = "0";
    expect_cuda = (strcmp(cuda_path, "0") != 0);

    if (lib_path && lib_path[0]) CHECK(test_setenv("WEX_LIB_PATH", lib_path) == 0);

    printf("[test_ggml_wex] building synthetic WEX4 WNR4-GG container (2 layers x 8 experts)...\n");
    build_wnr4_gg_container(wex_file, raw_slabs, root_hex);

    /* 1. Dynamically load libggml-wex plugin & check GGML_BACKEND_DL score */
    load_plugin(&pl, plugin_path);
    CHECK(pl.abi_version() == GGML_WEX_ABI_VERSION);
    CHECK(pl.backend_score() == 1);

    /* 2. Direct Q2_0 dequantization parity check */
    {
        float got_row[GG_COLS], ref_row[GG_COLS];
        float max_d = 0.0f;
        pl.dequant_row_q2_0(raw_slabs[0], got_row, GG_COLS);
        ref_dequant_row_q2_0(raw_slabs[0], ref_row, GG_COLS);
        for (i = 0; i < (int)GG_COLS; i++) {
            float d = fabsf(got_row[i] - ref_row[i]);
            if (d > max_d) max_d = d;
        }
        CHECK(max_d == 0.0f);
    }

    /* 3. Open with auto L2 Compressed-RAM cache + WEX_VERIFY_ON + root pinning */
    ggml_wex_opts_init(&opts);
    opts.libwex_path         = lib_path;
    opts.root_hex            = root_hex;
    opts.verify              = WEX_VERIFY_ON;
    opts.l2_cache_bytes      = UINT64_MAX; /* single-NVMe auto L2 */
    opts.l2_pin_gpu          = expect_cuda ? 1 : 0;
    opts.gpu_device          = expect_cuda ? 0 : -2;
    opts.gpu_helper_path     = cuda_path;
    opts.prefetch_next_layer = 0; /* controlled explicitly to test L2/prefetch counters precisely */

    ctx = pl.open(wex_file, &opts, err, sizeof(err));
    CHECK(ctx != NULL);
    CHECK(pl.info(ctx, &info) == WEX_OK);
    CHECK(info.layers == GG_LAYERS && info.experts == GG_EXPERTS);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_GG);
    CHECK(info.verified_root == 1u && info.verify_on == 1u);
    CHECK(info.l2_capacity_bytes > 0u);
    if (expect_cuda) CHECK(strcmp(info.backend_name, "cuda") == 0);
    else             CHECK(strncmp(info.backend_name, "cpu-", 4) == 0);

    /* Populate synthetic F32 token activations (n_tokens = 4, k_cols = 256) */
    for (i = 0; i < (int)(4u * GG_COLS); i++) {
        int32_t v = (int32_t)(rng_next(&rng) % 2001u) - 1000;
        x_f32[i] = (float)v * 0.001f;
    }

    /* 4. Cold GGML_OP_MUL_MAT_ID (n_tokens=4, top_k=2, 8 pairs -> 4 unique experts: 1,3,5,7) */
    {
        const int32_t ids_pass1[4 * 2] = { 1, 3,  3, 5,  1, 5,  3, 7 };
        float max_err;
        memset(out_f32, 0, sizeof(out_f32));
        CHECK(pl.mul_mat_id_q2_0_f32(ctx, 0, 0, ids_pass1, 4, 2, x_f32,
                                     GG_COLS, GG_ROWS, out_f32) == WEX_OK);
        max_err = verify_mul_mat_id((const uint8_t *const *)raw_slabs, 0, 0,
                                    ids_pass1, 4, 2, x_f32, out_f32);
        printf("  pass 1 (cold disk, layer 0 gate, 8 pairs -> 4 unique): max_abs_err=%.6g\n",
               (double)max_err);
        CHECK(max_err < 1e-6f);

        pl.stats(ctx, &st0);
        CHECK(st0.records_requested == 4u); /* deduplication verified */
        CHECK(st0.records_from_disk == 4u);
        CHECK(st0.records_from_l2 == 0u);
        if (expect_cuda) CHECK(st0.gpu_decodes == 4u);
        else             CHECK(st0.cpu_decodes == 4u);
    }

    /* 5. Second pass over overlapping experts (layer 0 down slice, slot 1) -> 100% L2 hit */
    {
        const int32_t ids_pass2[4 * 2] = { 1, 3,  5, 7,  1, 7,  3, 5 };
        float max_err;
        memset(out_f32, 0, sizeof(out_f32));
        CHECK(pl.mul_mat_id_q2_0_f32(ctx, 0, 1, ids_pass2, 4, 2, x_f32,
                                     GG_COLS, GG_ROWS, out_f32) == WEX_OK);
        max_err = verify_mul_mat_id((const uint8_t *const *)raw_slabs, 0, 1,
                                    ids_pass2, 4, 2, x_f32, out_f32);
        printf("  pass 2 (L2 warm, layer 0 down, 8 pairs -> 4 unique): max_abs_err=%.6g\n",
               (double)max_err);
        CHECK(max_err < 1e-6f);

        pl.stats(ctx, &st1);
        CHECK(st1.records_from_l2 == 4u && st1.records_from_disk == 4u);
    }

    /* 6. Deduplicated ggml_wex_fetch_experts with duplicates [2, 6, 2, 6] */
    {
        uint8_t *slabs[4];
        void *dsts[4];
        size_t caps[4] = { GG_SLAB_B, GG_SLAB_B, GG_SLAB_B, GG_SLAB_B };
        int eids[4] = { 2, 6, 2, 6 };
        int st[4];
        for (i = 0; i < 4; i++) {
            slabs[i] = (uint8_t *)calloc(1, GG_SLAB_B);
            CHECK(slabs[i] != NULL);
            dsts[i] = slabs[i];
        }
        CHECK(pl.fetch_experts(ctx, 0, eids, 4, dsts, caps, st) == 0);
        for (i = 0; i < 4; i++) {
            CHECK(st[i] == WEX_OK);
            CHECK(memcmp(slabs[i], raw_slabs[eids[i]], GG_SLAB_B) == 0);
            free(slabs[i]);
        }
        pl.stats(ctx, &st1);
        CHECK(st1.records_requested == 10u); /* 4 + 4 + 2 unique (not +4) */
        CHECK(st1.records_from_disk == 6u);
    }

    /* 7. Prefetch layer 1 experts [0, 2, 4, 6] -> verify L2 prefetch hits during MUL_MAT_ID */
    {
        const int pf_eids[4] = { 0, 2, 4, 6 };
        const int32_t ids_l1[4 * 2] = { 0, 2,  4, 6,  0, 4,  2, 6 };
        float max_err;
        CHECK(pl.prefetch_layer(ctx, 1, pf_eids, 4) == 4);
        for (i = 0; i < 200; i++) {
            pl.stats(ctx, &st2);
            if (st2.l2_entries >= 10u) break;
            test_sleep_ms(10);
        }
        CHECK(st2.l2_entries >= 10u);

        CHECK(pl.mul_mat_id_q2_0_f32(ctx, 1, 0, ids_l1, 4, 2, x_f32,
                                     GG_COLS, GG_ROWS, out_f32) == WEX_OK);
        max_err = verify_mul_mat_id((const uint8_t *const *)raw_slabs, 1, 0,
                                    ids_l1, 4, 2, x_f32, out_f32);
        printf("  pass 3 (prefetched layer 1 gate): max_abs_err=%.6g\n", (double)max_err);
        CHECK(max_err < 1e-6f);

        pl.stats(ctx, &st2);
        CHECK(st2.records_from_l2 == 8u && st2.records_from_disk == 6u && st2.prefetch_hits == 4u);
    }

    pl.close(ctx);

    /* 8. Root-pinning mismatch refusal */
    memcpy(bad_root, root_hex, sizeof(bad_root));
    bad_root[0] = (bad_root[0] == '0') ? '1' : '0';
    opts.root_hex = bad_root;
    err[0] = '\0';
    CHECK(pl.open(wex_file, &opts, err, sizeof(err)) == NULL);
    CHECK(err[0] != '\0');
    printf("  root-mismatch refused as expected: %s\n", err);

    test_dlclose(pl.handle);
    for (i = 0; i < (int)GG_NRECS; i++) free(raw_slabs[i]);
    remove(wex_file);

    printf("ALL GGML-WEX PLUGIN TESTS PASSED (%d checks, backend=%s)\n",
           g_checks, info.backend_name);
    return 0;
}
