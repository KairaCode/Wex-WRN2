/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "ggml_wex.h"

#if defined(__has_include)
#  if __has_include("wex_loader.h")
#    include "wex_loader.h"
#  else
#    include "../../include/wex_loader.h"
#  endif
#else
#  include "wex_loader.h"
#endif

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

struct ggml_wex_ctx {
    wex_lib_t    lib;
    wex_store_t *store;
    wex_info_t   info;
    int          prefetch_next_layer;
};

static void ggml_wex_set_err(char *buf, size_t cap, const char *fmt, ...) {
    va_list ap;
    if (!buf || !cap) return;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
    buf[cap - 1u] = '\0';
}

/* Locate libwex beside libggml-wex when deployed in a plugin directory. */
static int ggml_wex_self_sibling(char *out, size_t cap, const char *filename) {
    char self_path[4096];
    const char *sep;
    size_t flen = strlen(filename);
#ifdef _WIN32
    HMODULE hmod = NULL;
    DWORD n;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)(uintptr_t)&ggml_wex_abi_version, &hmod)) {
        return -1;
    }
    n = GetModuleFileNameA(hmod, self_path, (DWORD)sizeof(self_path));
    if (!n || (size_t)n >= sizeof(self_path)) return -1;
    self_path[n] = '\0';
#else
    Dl_info dli;
    if (!dladdr((const void *)(uintptr_t)&ggml_wex_abi_version, &dli) || !dli.dli_fname) {
        return -1;
    }
    if (strlen(dli.dli_fname) >= sizeof(self_path)) return -1;
    strcpy(self_path, dli.dli_fname);
#endif
    sep = strrchr(self_path, '/');
#ifdef _WIN32
    {
        const char *bsep = strrchr(self_path, '\\');
        if (bsep && (!sep || bsep > sep)) sep = bsep;
    }
#endif
    if (!sep) return -1;
    if ((size_t)(sep - self_path) + 1u + flen + 1u > cap) return -1;
    memcpy(out, self_path, (size_t)(sep - self_path) + 1u);
    memcpy(out + (size_t)(sep - self_path) + 1u, filename, flen + 1u);
    return 0;
}

static int ggml_wex_load_lib(wex_lib_t *lib, const char *explicit_path) {
    if (explicit_path && explicit_path[0]) {
        return wex_lib_open(lib, explicit_path);
    }
    if (wex_lib_open(lib, NULL) == WEX_OK) return WEX_OK;
    {
        char cand[4096];
#ifdef _WIN32
        const char *libname = "libwex.dll";
#elif defined(__APPLE__)
        const char *libname = "libwex.dylib";
#else
        const char *libname = "libwex.so";
#endif
        if (ggml_wex_self_sibling(cand, sizeof(cand), libname) == 0) {
            return wex_lib_open(lib, cand);
        }
    }
    return WEX_E_NOT_FOUND;
}

/* Exact IEEE 754 binary16 -> binary32 conversion (normals, subnormals, +/-0, Inf/NaN). */
static inline float ggml_wex_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1fu);
    uint32_t mant = (uint32_t)(h & 0x03ffu);
    uint32_t bits;
    float f;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            uint32_t e = 113u; /* 127 - 14 */
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03ffu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Exact Q2_0 (18 bytes / 64 weights) x F32 dot product matching ggml_wex_dequant_row_q2_0. */
static float ggml_wex_dot_q2_0_f32(const uint8_t *src_blocks, const float *x_f32, int64_t k) {
    int64_t nb = k / GGML_WEX_QK2_0;
    int64_t b;
    float acc = 0.0f;

    for (b = 0; b < nb; b++) {
        const uint8_t *blk = src_blocks + (size_t)b * GGML_WEX_Q2_0_BYTES;
        uint16_t d_bits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float d = ggml_wex_fp16_to_fp32(d_bits);
        const uint8_t *qs = blk + 2u;
        const float *xb = x_f32 + (size_t)b * GGML_WEX_QK2_0;
        int j;
        for (j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            int q0 = (int)((byte >> 0) & 3u) - 1;
            int q1 = (int)((byte >> 2) & 3u) - 1;
            int q2 = (int)((byte >> 4) & 3u) - 1;
            int q3 = (int)((byte >> 6) & 3u) - 1;
            acc += ((float)q0 * d) * xb[j * 4 + 0];
            acc += ((float)q1 * d) * xb[j * 4 + 1];
            acc += ((float)q2 * d) * xb[j * 4 + 2];
            acc += ((float)q3 * d) * xb[j * 4 + 3];
        }
    }
    return acc;
}

/* ---- Public API ---------------------------------------------------------- */

GGML_WEX_API uint32_t ggml_wex_abi_version(void) {
    return GGML_WEX_ABI_VERSION;
}

GGML_WEX_API int ggml_wex_backend_score(void) {
    wex_lib_t lib;
    if (ggml_wex_load_lib(&lib, NULL) != WEX_OK) return 0;
    wex_lib_close(&lib);
    return 1;
}

GGML_WEX_API ggml_wex_ctx_t *ggml_wex_open(const char *wex_path,
                                           const ggml_wex_opts_t *opts,
                                           char *err, size_t err_cap) {
    ggml_wex_opts_t def;
    const ggml_wex_opts_t *o;
    wex_open_opts_t wopts;
    ggml_wex_ctx_t *ctx;
    int rc;

    if (err && err_cap) err[0] = '\0';
    if (!wex_path || !wex_path[0]) {
        ggml_wex_set_err(err, err_cap, "ggml_wex_open: wex_path is NULL or empty");
        return NULL;
    }
    if (opts) {
        if (opts->struct_size != (uint32_t)sizeof(ggml_wex_opts_t)) {
            ggml_wex_set_err(err, err_cap, "ggml_wex_open: opts->struct_size mismatch (%u != %u)",
                             opts->struct_size, (uint32_t)sizeof(ggml_wex_opts_t));
            return NULL;
        }
        o = opts;
    } else {
        ggml_wex_opts_init(&def);
        o = &def;
    }

    ctx = (ggml_wex_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        ggml_wex_set_err(err, err_cap, "ggml_wex_open: out of memory");
        return NULL;
    }

    rc = ggml_wex_load_lib(&ctx->lib, o->libwex_path);
    if (rc != WEX_OK) {
        ggml_wex_set_err(err, err_cap, "ggml_wex_open: failed to load libwex shared library (rc=%d)", rc);
        free(ctx);
        return NULL;
    }

    wex_open_opts_init(&wopts);
    wopts.root_hex        = o->root_hex;
    wopts.verify          = o->verify;
    wopts.mirrors         = o->mirrors;
    wopts.num_mirrors     = o->num_mirrors;
    wopts.l2_cache_bytes  = o->l2_cache_bytes;
    wopts.l2_pin_gpu      = o->l2_pin_gpu;
    wopts.io_direct       = o->io_direct;
    wopts.io_queue_depth  = o->io_queue_depth;
    wopts.cpu_threads     = o->cpu_threads;
    wopts.gpu_device      = o->gpu_device;
    wopts.gpu_helper_path = o->gpu_helper_path;

    ctx->store = ctx->lib.open(wex_path, &wopts, err, err_cap);
    if (!ctx->store) {
        wex_lib_close(&ctx->lib);
        free(ctx);
        return NULL;
    }

    if (ctx->lib.info(ctx->store, &ctx->info) != WEX_OK) {
        ggml_wex_set_err(err, err_cap, "ggml_wex_open: wex_info failed");
        ctx->lib.close(ctx->store);
        wex_lib_close(&ctx->lib);
        free(ctx);
        return NULL;
    }
    ctx->prefetch_next_layer = o->prefetch_next_layer ? 1 : 0;
    return ctx;
}

GGML_WEX_API int ggml_wex_info(const ggml_wex_ctx_t *ctx, wex_info_t *out) {
    if (!ctx || !ctx->store || !out) return WEX_E_ARGS;
    return ctx->lib.info(ctx->store, out);
}

GGML_WEX_API void ggml_wex_stats(const ggml_wex_ctx_t *ctx, wex_stats_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx || !ctx->store) return;
    ctx->lib.stats(ctx->store, out);
}

GGML_WEX_API void ggml_wex_close(ggml_wex_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->store && ctx->lib.close) ctx->lib.close(ctx->store);
    wex_lib_close(&ctx->lib);
    free(ctx);
}

GGML_WEX_API int ggml_wex_prefetch_layer(ggml_wex_ctx_t *ctx, int layer,
                                         const int *eids, int n_eids) {
    if (!ctx || !ctx->store || !eids || n_eids <= 0) return 0;
    return ctx->lib.prefetch(ctx->store, layer, eids, n_eids, WEX_FETCH_DEFAULT);
}

GGML_WEX_API int ggml_wex_fetch_experts(ggml_wex_ctx_t *ctx, int layer,
                                        const int *eids, int n_eids,
                                        void *const *dst_slabs,
                                        const size_t *dst_caps,
                                        int *status_out) {
    int *uniq_eids = NULL;
    int *slot_to_uniq = NULL;
    int *primary_slot = NULL;
    void **uniq_dsts = NULL;
    size_t *uniq_caps = NULL;
    size_t *uniq_decoded = NULL;
    int *uniq_status = NULL;
    int i, n_uniq = 0, failed = 0;

    if (!ctx || !ctx->store || n_eids < 0) return WEX_E_ARGS;
    if (n_eids == 0) return 0;
    if (!eids || !dst_slabs || !dst_caps) {
        if (status_out) {
            for (i = 0; i < n_eids; i++) status_out[i] = WEX_E_ARGS;
        }
        return n_eids;
    }

    uniq_eids    = (int *)malloc((size_t)n_eids * sizeof(int));
    slot_to_uniq = (int *)malloc((size_t)n_eids * sizeof(int));
    primary_slot = (int *)malloc((size_t)n_eids * sizeof(int));
    uniq_dsts    = (void **)malloc((size_t)n_eids * sizeof(void *));
    uniq_caps    = (size_t *)malloc((size_t)n_eids * sizeof(size_t));
    uniq_decoded = (size_t *)calloc((size_t)n_eids, sizeof(size_t));
    uniq_status  = (int *)malloc((size_t)n_eids * sizeof(int));
    if (!uniq_eids || !slot_to_uniq || !primary_slot || !uniq_dsts ||
        !uniq_caps || !uniq_decoded || !uniq_status) {
        free(uniq_eids); free(slot_to_uniq); free(primary_slot);
        free(uniq_dsts); free(uniq_caps); free(uniq_decoded); free(uniq_status);
        if (status_out) {
            for (i = 0; i < n_eids; i++) status_out[i] = WEX_E_OOM;
        }
        return n_eids;
    }

    /* Deduplicate repeated expert IDs while respecting each slot's buffer capacity. */
    for (i = 0; i < n_eids; i++) {
        int u, found = -1;
        for (u = 0; u < n_uniq; u++) {
            if (uniq_eids[u] == eids[i]) { found = u; break; }
        }
        if (found < 0) {
            wex_layout_t lay;
            found = n_uniq++;
            uniq_eids[found]    = eids[i];
            primary_slot[found] = i;
            uniq_dsts[found]    = dst_slabs[i];
            uniq_caps[found]    = dst_caps[i];
            uniq_status[found]  = WEX_OK;
            if (ctx->lib.record_layout(ctx->store, layer, eids[i], 0u, &lay) == WEX_OK) {
                uniq_decoded[found] = (size_t)lay.decoded_bytes;
            }
        } else if ((!uniq_dsts[found] || uniq_caps[found] < uniq_decoded[found]) &&
                   dst_slabs[i] && dst_caps[i] >= uniq_decoded[found]) {
            /* Promote a valid slot to primary if an earlier duplicate had an invalid buffer. */
            primary_slot[found] = i;
            uniq_dsts[found]    = dst_slabs[i];
            uniq_caps[found]    = dst_caps[i];
        }
        slot_to_uniq[i] = found;
    }

    (void)ctx->lib.fetch_host(ctx->store, layer, uniq_eids, n_uniq,
                              (void *const *)uniq_dsts, uniq_caps,
                              WEX_FETCH_DEFAULT, uniq_status);

    for (i = 0; i < n_eids; i++) {
        int u = slot_to_uniq[i];
        int st = uniq_status[u];
        if (i != primary_slot[u]) {
            if (!dst_slabs[i]) {
                st = WEX_E_ARGS;
            } else if (st == WEX_OK) {
                if (dst_caps[i] < uniq_decoded[u]) {
                    st = WEX_E_CAP;
                } else if (uniq_decoded[u] > 0u) {
                    memcpy(dst_slabs[i], uniq_dsts[u], uniq_decoded[u]);
                }
            }
        }
        if (status_out) status_out[i] = st;
        if (st != WEX_OK) failed++;
    }

    free(uniq_eids); free(slot_to_uniq); free(primary_slot);
    free(uniq_dsts); free(uniq_caps); free(uniq_decoded); free(uniq_status);
    return failed;
}

GGML_WEX_API void ggml_wex_dequant_row_q2_0(const void *src_blocks,
                                            float *dst_f32, int64_t k) {
    const uint8_t *src = (const uint8_t *)src_blocks;
    int64_t nb, b;
    if (!src || !dst_f32 || k <= 0 || (k % GGML_WEX_QK2_0) != 0) return;

    nb = k / GGML_WEX_QK2_0;
    for (b = 0; b < nb; b++) {
        const uint8_t *blk = src + (size_t)b * GGML_WEX_Q2_0_BYTES;
        uint16_t d_bits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float d = ggml_wex_fp16_to_fp32(d_bits);
        const uint8_t *qs = blk + 2u;
        float *out = dst_f32 + (size_t)b * GGML_WEX_QK2_0;
        int j;
        for (j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            out[j * 4 + 0] = (float)((int)((byte >> 0) & 3u) - 1) * d;
            out[j * 4 + 1] = (float)((int)((byte >> 2) & 3u) - 1) * d;
            out[j * 4 + 2] = (float)((int)((byte >> 4) & 3u) - 1) * d;
            out[j * 4 + 3] = (float)((int)((byte >> 6) & 3u) - 1) * d;
        }
    }
}

GGML_WEX_API int ggml_wex_mul_mat_id_q2_0_f32(ggml_wex_ctx_t *ctx,
                                              int layer, int tensor_slot,
                                              const int32_t *ids,
                                              int n_tokens, int n_used,
                                              const float *x_f32,
                                              int64_t k_cols, int64_t n_rows,
                                              float *out_f32) {
    int total_pairs;
    int *uniq_eids = NULL;
    int *pair_to_uniq = NULL;
    wex_layout_t *layouts = NULL;
    uint8_t *slab_arena = NULL;
    void **dst_slabs = NULL;
    size_t *dst_caps = NULL;
    int *statuses = NULL;
    size_t row_bytes, need_mat_bytes, total_arena = 0;
    int i, u, n_uniq = 0, rc = WEX_OK;

    if (!ctx || !ctx->store || !ids || !x_f32 || !out_f32 ||
        layer < 0 || tensor_slot < 0 || n_tokens <= 0 || n_used <= 0 ||
        k_cols <= 0 || (k_cols % GGML_WEX_QK2_0) != 0 || n_rows <= 0) {
        return WEX_E_ARGS;
    }
    if ((uint64_t)n_tokens * (uint64_t)n_used > (uint64_t)INT_MAX) return WEX_E_ARGS;
    total_pairs = n_tokens * n_used;

    row_bytes      = (size_t)(k_cols / GGML_WEX_QK2_0) * GGML_WEX_Q2_0_BYTES;
    need_mat_bytes = (size_t)n_rows * row_bytes;

    uniq_eids    = (int *)malloc((size_t)total_pairs * sizeof(int));
    pair_to_uniq = (int *)malloc((size_t)total_pairs * sizeof(int));
    if (!uniq_eids || !pair_to_uniq) {
        rc = WEX_E_OOM;
        goto cleanup;
    }

    /* 1. Deduplicate active expert IDs across the token batch. */
    for (i = 0; i < total_pairs; i++) {
        int eid = (int)ids[i];
        int found = -1;
        if (eid < 0) {
            rc = WEX_E_ARGS;
            goto cleanup;
        }
        for (u = 0; u < n_uniq; u++) {
            if (uniq_eids[u] == eid) { found = u; break; }
        }
        if (found < 0) {
            found = n_uniq++;
            uniq_eids[found] = eid;
        }
        pair_to_uniq[i] = found;
    }

    layouts   = (wex_layout_t *)calloc((size_t)n_uniq, sizeof(*layouts));
    dst_slabs = (void **)malloc((size_t)n_uniq * sizeof(void *));
    dst_caps  = (size_t *)malloc((size_t)n_uniq * sizeof(size_t));
    statuses  = (int *)calloc((size_t)n_uniq, sizeof(int));
    if (!layouts || !dst_slabs || !dst_caps || !statuses) {
        rc = WEX_E_OOM;
        goto cleanup;
    }

    /* 2. Inspect each unique expert's layout and size a contiguous scratch arena. */
    for (u = 0; u < n_uniq; u++) {
        rc = ctx->lib.record_layout(ctx->store, layer, uniq_eids[u], 0u, &layouts[u]);
        if (rc != WEX_OK) goto cleanup;
        if ((uint32_t)tensor_slot >= layouts[u].tensor_count ||
            (uint32_t)tensor_slot >= WEX_MAX_RECORD_TENSORS) {
            rc = WEX_E_NOT_FOUND;
            goto cleanup;
        }
        if (layouts[u].tensor_bytes[tensor_slot] < need_mat_bytes ||
            layouts[u].tensor_offsets[tensor_slot] + need_mat_bytes > layouts[u].decoded_bytes) {
            rc = WEX_E_FORMAT;
            goto cleanup;
        }
        dst_caps[u] = (size_t)layouts[u].decoded_bytes;
        total_arena += (dst_caps[u] + 63u) & ~(size_t)63u;
    }

    slab_arena = (uint8_t *)malloc(total_arena ? total_arena : 64u);
    if (!slab_arena) {
        rc = WEX_E_OOM;
        goto cleanup;
    }
    {
        size_t off = 0;
        for (u = 0; u < n_uniq; u++) {
            dst_slabs[u] = slab_arena + off;
            off += (dst_caps[u] + 63u) & ~(size_t)63u;
        }
    }

    /* 3. Fetch only unique active experts in a single libwex batch call. */
    {
        int failed = ctx->lib.fetch_host(ctx->store, layer, uniq_eids, n_uniq,
                                         (void *const *)dst_slabs, dst_caps,
                                         WEX_FETCH_SCRATCH, statuses);
        if (failed != 0) {
            rc = WEX_E_IO;
            for (u = 0; u < n_uniq; u++) {
                if (statuses[u] != WEX_OK) { rc = statuses[u]; break; }
            }
            goto cleanup;
        }
    }

    /* 4. Optional non-blocking prefetch hint for layer + 1. */
    if (ctx->prefetch_next_layer && (uint32_t)(layer + 1) < ctx->info.layers) {
        (void)ctx->lib.prefetch(ctx->store, layer + 1, uniq_eids, n_uniq, WEX_FETCH_DEFAULT);
    }

    /* 5. Compute exact Q2_0 x F32 dot products for each (tok, slot, row). */
    for (i = 0; i < total_pairs; i++) {
        int tok  = i / n_used;
        int uidx = pair_to_uniq[i];
        const uint8_t *mat_q2 = (const uint8_t *)dst_slabs[uidx] +
                                (size_t)layouts[uidx].tensor_offsets[tensor_slot];
        const float *x_tok    = x_f32 + (size_t)tok * (size_t)k_cols;
        float *out_vec        = out_f32 + (size_t)i * (size_t)n_rows;
        int64_t row;

        for (row = 0; row < n_rows; row++) {
            out_vec[row] = ggml_wex_dot_q2_0_f32(mat_q2 + (size_t)row * row_bytes,
                                                 x_tok, k_cols);
        }
    }

cleanup:
    free(slab_arena);
    free(statuses);
    free(dst_caps);
    free(dst_slabs);
    free(layouts);
    free(pair_to_uniq);
    free(uniq_eids);
    return rc;
}

/* ---- Optional llama.cpp / ik_llama.cpp GGML_BACKEND_DL bridge ------------ */
#ifdef GGML_WEX_WITH_GGML
#include "ggml-backend-impl.h"

static const char *ggml_wex_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "WEX";
}

static size_t ggml_wex_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return (size_t)(ggml_wex_backend_score() > 0 ? 1 : 0);
}

static ggml_backend_dev_t ggml_wex_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    (void)index;
    return NULL;
}

static void *ggml_wex_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    (void)reg;
    if (!name) return NULL;
    if (strcmp(name, "ggml_wex_open") == 0)                 return (void *)(uintptr_t)&ggml_wex_open;
    if (strcmp(name, "ggml_wex_close") == 0)                return (void *)(uintptr_t)&ggml_wex_close;
    if (strcmp(name, "ggml_wex_info") == 0)                 return (void *)(uintptr_t)&ggml_wex_info;
    if (strcmp(name, "ggml_wex_stats") == 0)                return (void *)(uintptr_t)&ggml_wex_stats;
    if (strcmp(name, "ggml_wex_prefetch_layer") == 0)       return (void *)(uintptr_t)&ggml_wex_prefetch_layer;
    if (strcmp(name, "ggml_wex_fetch_experts") == 0)        return (void *)(uintptr_t)&ggml_wex_fetch_experts;
    if (strcmp(name, "ggml_wex_dequant_row_q2_0") == 0)     return (void *)(uintptr_t)&ggml_wex_dequant_row_q2_0;
    if (strcmp(name, "ggml_wex_mul_mat_id_q2_0_f32") == 0)  return (void *)(uintptr_t)&ggml_wex_mul_mat_id_q2_0_f32;
    return NULL;
}

static struct ggml_backend_reg_i g_ggml_wex_reg_i = {
    /* .get_name         = */ ggml_wex_reg_get_name,
    /* .get_device_count = */ ggml_wex_reg_get_device_count,
    /* .get_device       = */ ggml_wex_reg_get_device,
    /* .get_proc_address = */ ggml_wex_reg_get_proc_address,
};

static struct ggml_backend_reg g_ggml_wex_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ {0},
    /* .context     = */ NULL,
};

GGML_WEX_API ggml_backend_reg_t ggml_backend_wex_reg(void) {
    g_ggml_wex_reg.iface = g_ggml_wex_reg_i;
    return &g_ggml_wex_reg;
}

GGML_WEX_API int ggml_backend_score(void) {
    return ggml_wex_backend_score();
}

GGML_WEX_API ggml_backend_reg_t ggml_backend_init(void) {
    return ggml_backend_wex_reg();
}
#endif /* GGML_WEX_WITH_GGML */
