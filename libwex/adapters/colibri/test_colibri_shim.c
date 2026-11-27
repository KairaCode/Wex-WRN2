/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

/* Live verification test for colibri_wex_shim.h:
 *   - Generates a spec-valid synthetic WEX4 WNR4-MX container (2 layers x 4 experts)
 *   - Configures COLI_WEX_ROOT, COLI_WEX_VERIFY=1, COLI_WEX_L2_MB=16, COLI_WEX_LIB,
 *     and optional COLI_WEX_CUDA_LIB
 *   - Opens via coli_libwex_open_env(), checks info, has_expert, layout
 *   - Verifies cold disk fetch, warm L2 hit fetch (scratch=1), async prefetch -> L2 hit,
 *     single-line format_stats output, and root-mismatch refusal
 *
 * Usage: test_colibri_shim [--lib PATH] [--cuda PATH|0] */

#include "colibri_wex_shim.h"

#include "../../../c/wex4.h"
#include "../../../wnr4/wnr4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
static void shim_sleep_ms(unsigned ms) { Sleep(ms); }
static int shim_setenv(const char *k, const char *v) {
    return _putenv_s(k, v ? v : "");
}
#else
#include <time.h>
static void shim_sleep_ms(unsigned ms) {
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    nanosleep(&ts, NULL);
}
static int shim_setenv(const char *k, const char *v) {
    if (!v) return unsetenv(k);
    return setenv(k, v, 1);
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

/* ---- Minimal WEX4 WNR4-MX container builder (2 layers x 4 experts) ------- */

#define SHIM_LAYERS     2u
#define SHIM_EXPERTS    4u
#define SHIM_NRECS      (SHIM_LAYERS * SHIM_EXPERTS)
#define SHIM_ROWS       64u
#define SHIM_SCOLS      8u
#define SHIM_WCOLS      (SHIM_SCOLS * 16u)
#define SHIM_SC_BYTES   (3u * SHIM_ROWS * SHIM_SCOLS)
#define SHIM_WT_BYTES   (3u * SHIM_ROWS * SHIM_WCOLS)
#define SHIM_DEC_BYTES  (SHIM_SC_BYTES + SHIM_WT_BYTES)
#define SHIM_RAW_ALN    28672u

typedef struct {
    char     name[96];
    uint8_t  dtype;
    uint8_t  rank;
    uint64_t shape[8];
    uint64_t offset;
    uint64_t nbytes;
} shim_tensor_t;

typedef struct {
    const uint8_t *stored_data;
    uint32_t       stored_bytes;
    uint32_t       decoded_bytes;
    uint16_t       codec;
    uint16_t       flags;
    uint32_t       blob_1based;
    uint32_t       owner;
    shim_tensor_t  tensors[6];
    uint32_t       tensor_count;
} shim_record_t;

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

static void fill_mxfp4_slab(uint8_t *slab, uint64_t *rng) {
    static const uint8_t nibs[8] = {0, 0, 1, 1, 2, 9, 9, 10};
    uint32_t i;
    for (i = 0; i < SHIM_SC_BYTES; i++) {
        slab[i] = (uint8_t)(120u + (rng_next(rng) % 4u));
    }
    for (i = 0; i < SHIM_WT_BYTES; i++) {
        uint8_t lo = nibs[rng_next(rng) & 7u];
        uint8_t hi = nibs[rng_next(rng) & 7u];
        slab[SHIM_SC_BYTES + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void write_wex4_single_blob(const char *path,
                                   uint32_t blob_kind, uint32_t blob_id,
                                   const uint8_t *blob_data, uint64_t blob_bytes,
                                   const shim_record_t *recs, uint32_t nrecs,
                                   uint32_t layers, uint32_t experts,
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
        put_u32_le(idx + at + 0,  blob_kind);
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
                const shim_tensor_t *ten = &recs[r].tensors[k];
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
    put_u32_le(file_buf + 60, layers);
    put_u32_le(file_buf + 64, experts);
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

static void build_wnr4_mx_container(const char *path,
                                    uint8_t raw_out[SHIM_NRECS][SHIM_DEC_BYTES],
                                    char root_hex_out[65]) {
    uint64_t rng = UINT64_C(0xC011B21202611270);
    uint8_t enc[SHIM_NRECS][SHIM_RAW_ALN];
    uint32_t scount[3], scols[3], soff[3], woff[3];
    Wnr4Geo geo;
    size_t cwords = wnr4_counts_words(WNR4_KIND_MX, 1u);
    size_t mbytes = wnr4_model_bytes(WNR4_KIND_MX, 1u);
    uint64_t *counts = (uint64_t *)calloc(cwords, sizeof(uint64_t));
    uint8_t *mblob = (uint8_t *)malloc(mbytes);
    Wnr4Model model;
    shim_record_t recs[SHIM_NRECS];
    uint32_t i, m;

    CHECK(counts && mblob);
    for (m = 0; m < 3u; m++) {
        scount[m] = SHIM_ROWS * SHIM_SCOLS;
        scols[m]  = SHIM_SCOLS;
        soff[m]   = m * SHIM_ROWS * SHIM_SCOLS;
        woff[m]   = SHIM_SC_BYTES + m * SHIM_ROWS * SHIM_WCOLS;
    }
    CHECK(wnr4_geo(&geo, WNR4_KIND_MX, scount, scols, soff, woff) == 0);

    for (i = 0; i < SHIM_NRECS; i++) {
        unsigned char anchor[3];
        fill_mxfp4_slab(raw_out[i], &rng);
        for (m = 0; m < 3u; m++) anchor[m] = wnr4_anchor(raw_out[i] + soff[m], scount[m]);
        wnr4_count(&geo, raw_out[i], anchor, 1u, counts);
    }
    wnr4_model_build(WNR4_KIND_MX, 1u, counts, mblob);
    CHECK(wnr4_model_open(mblob, mbytes, &model) == 0);

    for (i = 0; i < SHIM_NRECS; i++) {
        uint32_t layer = i / SHIM_EXPERTS;
        uint32_t eid   = i % SHIM_EXPERTS;
        uint32_t owner = (layer << 16) | eid;
        size_t stored = wnr4_encode(&model, &geo, raw_out[i], 0u, owner, 4u, enc[i], SHIM_RAW_ALN);
        CHECK(stored > 0 && stored < SHIM_RAW_ALN);

        memset(&recs[i], 0, sizeof(recs[i]));
        recs[i].stored_data   = enc[i];
        recs[i].stored_bytes  = (uint32_t)stored;
        recs[i].decoded_bytes = SHIM_DEC_BYTES;
        recs[i].codec         = 8u;
        recs[i].flags         = WEX4_REC_EXPERT;
        recs[i].blob_1based   = 1u;
        recs[i].owner         = owner;
        recs[i].tensor_count  = 6u;
        for (m = 0; m < 6u; m++) {
            shim_tensor_t *ten = &recs[i].tensors[m];
            snprintf(ten->name, sizeof(ten->name), "layers.%u.ffn.experts.%u.w%u.%s",
                     layer, eid, (m % 3u) + 1u, m < 3u ? "scale" : "weight");
            ten->dtype    = (m < 3u) ? WEX4_DT_F8_E8M0 : WEX4_DT_I8;
            ten->rank     = 2u;
            ten->shape[0] = SHIM_ROWS;
            ten->shape[1] = (m < 3u) ? SHIM_SCOLS : SHIM_WCOLS;
            ten->offset   = (m < 3u) ? soff[m] : woff[m - 3u];
            ten->nbytes   = ten->shape[0] * ten->shape[1];
        }
    }

    write_wex4_single_blob(path, 3u, model.id, mblob, mbytes, recs, SHIM_NRECS,
                           SHIM_LAYERS, SHIM_EXPERTS, root_hex_out);
    wnr4_model_free(&model);
    free(counts);
    free(mblob);
}

int main(int argc, char **argv) {
    const char *lib_path  = getenv("COLI_WEX_LIB");
    const char *cuda_path = getenv("COLI_WEX_CUDA_LIB");
    const char *wex_file  = "colibri_shim_test_mx.wex4";
    uint8_t raw[SHIM_NRECS][SHIM_DEC_BYTES];
    char root_hex[65], bad_root[65], err[256], stats_line[256];
    coli_libwex_t cw;
    wex_info_t info;
    wex_layout_t lay;
    wex_stats_t st;
    int i, expect_cuda;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lib") == 0 && i + 1 < argc)       lib_path  = argv[++i];
        else if (strcmp(argv[i], "--cuda") == 0 && i + 1 < argc) cuda_path = argv[++i];
        else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) wex_file  = argv[++i];
    }
    if (!cuda_path || !cuda_path[0]) cuda_path = "0";
    expect_cuda = (strcmp(cuda_path, "0") != 0);

    printf("[test_colibri_shim] building synthetic WEX4 WNR4-MX container (%s)...\n", wex_file);
    build_wnr4_mx_container(wex_file, raw, root_hex);

    /* Configure colibri environment variables */
    CHECK(shim_setenv("COLI_WEX_PATH", wex_file) == 0);
    CHECK(shim_setenv("COLI_WEX_ROOT", root_hex) == 0);
    CHECK(shim_setenv("COLI_WEX_VERIFY", "1") == 0);
    CHECK(shim_setenv("COLI_WEX_L2_MB", "16") == 0);
    CHECK(shim_setenv("COLI_WEX_L2_PIN", "1") == 0);
    CHECK(shim_setenv("COLI_WEX_QD", "4") == 0);
    CHECK(shim_setenv("COLI_WEX_DIRECT", "-1") == 0);
    CHECK(shim_setenv("COLI_WEX_CUDA_LIB", cuda_path) == 0);
    CHECK(shim_setenv("COLI_MODEL_MIRROR", "") == 0);
    if (lib_path && lib_path[0]) CHECK(shim_setenv("COLI_WEX_LIB", lib_path) == 0);

    /* 1. Open via COLI_WEX_PATH env var */
    CHECK(coli_libwex_open_env(&cw, NULL, err, sizeof(err)) == WEX_OK);
    CHECK(cw.open == 1);

    /* 2. Inspect info, has_expert, layout */
    CHECK(coli_libwex_info(&cw, &info) == WEX_OK);
    CHECK(info.layers == SHIM_LAYERS && info.experts == SHIM_EXPERTS);
    CHECK(info.codec_kind == WEX_CODEC_WNR4_MX);
    CHECK(info.verified_root == 1u && info.verify_on == 1u);
    CHECK(strcmp(info.root_hex, root_hex) == 0);
    CHECK(info.l2_capacity_bytes == (UINT64_C(16) << 20));
    if (expect_cuda) {
        CHECK(strcmp(info.backend_name, "cuda") == 0);
    } else {
        CHECK(strncmp(info.backend_name, "cpu-", 4) == 0);
    }

    CHECK(coli_libwex_has_expert(&cw, 0, 0) == 1);
    CHECK(coli_libwex_has_expert(&cw, 1, 3) == 1);
    CHECK(coli_libwex_has_expert(&cw, 2, 0) == 0);
    CHECK(coli_libwex_has_expert(&cw, 0, 4) == 0);
    CHECK(coli_libwex_has_expert(&cw, -1, 0) == 0);

    CHECK(coli_libwex_layout(&cw, 1, 2, &lay) == WEX_OK);
    CHECK(lay.codec_kind == WEX_CODEC_WNR4_MX);
    CHECK(lay.tensor_count == 6u);
    CHECK(lay.decoded_bytes == SHIM_DEC_BYTES);
    CHECK(lay.scale_shapes[0][0] == SHIM_ROWS && lay.scale_shapes[0][1] == SHIM_SCOLS);
    CHECK(lay.weight_shapes[2][0] == SHIM_ROWS && lay.weight_shapes[2][1] == SHIM_WCOLS);

    /* 3. Cold disk fetch + warm L2 hit fetch + async prefetch */
    {
        uint8_t out[4][SHIM_DEC_BYTES];
        void *dsts[4] = { out[0], out[1], out[2], out[3] };
        size_t caps[4] = { SHIM_DEC_BYTES, SHIM_DEC_BYTES, SHIM_DEC_BYTES, SHIM_DEC_BYTES };
        int eids[4] = { 0, 1, 2, 3 };
        int status[4];

        /* Cold disk fetch (layer 0) */
        memset(out, 0, sizeof(out));
        CHECK(coli_libwex_fetch_host(&cw, 0, eids, 4, dsts, caps, 0, status) == 0);
        for (i = 0; i < 4; i++) {
            CHECK(status[i] == WEX_OK);
            CHECK(memcmp(out[i], raw[i], SHIM_DEC_BYTES) == 0);
        }
        cw.lib.stats(cw.store, &st);
        CHECK(st.records_from_disk == 4u && st.records_from_l2 == 0u);
        if (expect_cuda) CHECK(st.gpu_decodes == 4u);
        else             CHECK(st.cpu_decodes == 4u);

        /* Warm L2 hit fetch (layer 0, scratch=1 fast path) */
        memset(out, 0, sizeof(out));
        CHECK(coli_libwex_fetch_host(&cw, 0, eids, 4, dsts, caps, 1, status) == 0);
        for (i = 0; i < 4; i++) {
            CHECK(status[i] == WEX_OK);
            CHECK(memcmp(out[i], raw[i], SHIM_DEC_BYTES) == 0);
        }
        cw.lib.stats(cw.store, &st);
        CHECK(st.records_from_l2 == 4u && st.records_from_disk == 4u);

        /* Async prefetch for layer 1 -> verify L2 warms and serves 4 prefetch hits */
        CHECK(coli_libwex_prefetch(&cw, 1, eids, 4) == 4);
        for (i = 0; i < 200; i++) {
            cw.lib.stats(cw.store, &st);
            if (st.l2_entries >= 8u) break;
            shim_sleep_ms(10);
        }
        CHECK(st.l2_entries >= 8u);

        memset(out, 0, sizeof(out));
        CHECK(coli_libwex_fetch_host(&cw, 1, eids, 4, dsts, caps, 1, status) == 0);
        for (i = 0; i < 4; i++) {
            CHECK(status[i] == WEX_OK);
            CHECK(memcmp(out[i], raw[4 + i], SHIM_DEC_BYTES) == 0);
        }
        cw.lib.stats(cw.store, &st);
        CHECK(st.records_from_l2 == 8u && st.records_from_disk == 4u && st.prefetch_hits == 4u);

        /* Device fetch on CPU-only mode must cleanly report WEX_E_DEVICE */
        if (!expect_cuda) {
            void *dummy_sc[3] = { out[0], out[0], out[0] };
            void *dummy_wt[3] = { out[1], out[1], out[1] };
            int dst_st[1] = { 0 };
            CHECK(coli_libwex_fetch_device(&cw, 0, eids, 1, dummy_sc, dummy_wt, 0, NULL, dst_st) == 1);
            CHECK(dst_st[0] == WEX_E_DEVICE);
        }
    }

    /* 4. Single-line stats formatter */
    CHECK(coli_libwex_format_stats(&cw, stats_line, sizeof(stats_line)) > 0);
    CHECK(strncmp(stats_line, "[libwex] backend=", 17) == 0);
    CHECK(strstr(stats_line, "io=") != NULL);
    CHECK(strstr(stats_line, "l2_hit=66.7%") != NULL);
    CHECK(strstr(stats_line, "disk=") != NULL);
    CHECK(strstr(stats_line, "dec=") != NULL);
    printf("  %s\n", stats_line);

    coli_libwex_close(&cw);
    CHECK(cw.open == 0 && cw.store == NULL);

    /* 5. Root-mismatch refusal via explicit path_override */
    memcpy(bad_root, root_hex, sizeof(bad_root));
    bad_root[0] = (bad_root[0] == '0') ? '1' : '0';
    CHECK(shim_setenv("COLI_WEX_ROOT", bad_root) == 0);
    err[0] = '\0';
    CHECK(coli_libwex_open_env(&cw, wex_file, err, sizeof(err)) == WEX_E_ROOT);
    CHECK(cw.open == 0 && cw.store == NULL && err[0] != '\0');
    printf("  root-mismatch refused as expected: %s\n", err);

    /* 6. Mirror parser unit check (Windows drive colons preserved, separators split) */
    {
        char *mbuf = NULL;
        const char *ptrs[COLI_LIBWEX_MAX_MIRRORS];
        int nm = coli_libwex_parse_mirrors_("D:\\m1\\a.wex4;E:/m2/b.wex4, /mnt/nvme2/c.wex4",
                                            &mbuf, ptrs);
        CHECK(nm == 3);
        CHECK(strcmp(ptrs[0], "D:\\m1\\a.wex4") == 0);
        CHECK(strcmp(ptrs[1], "E:/m2/b.wex4") == 0);
        CHECK(strcmp(ptrs[2], "/mnt/nvme2/c.wex4") == 0);
        free(mbuf);
    }

    remove(wex_file);
    printf("ALL COLIBRI SHIM TESTS PASSED (%d checks, mode=%s)\n",
           g_checks, expect_cuda ? "CUDA" : "CPU");
    return 0;
}
