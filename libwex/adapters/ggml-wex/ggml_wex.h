/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_GGML_WEX_H
#define LIBWEX_GGML_WEX_H

/* ggml-wex: dynamic MoE expert-streaming plugin for llama.cpp / ik_llama.cpp.
 *
 * Compiles standalone with zero compile-time dependency on libwex, CUDA, or
 * llama.cpp headers (libwex is loaded dynamically at runtime via wex_loader.h).
 * Define GGML_WEX_WITH_GGML when building inside llama.cpp / ik_llama.cpp to
 * expose standard GGML_BACKEND_DL registration entry points. */

#if defined(__has_include)
#  if __has_include("wex_store.h")
#    include "wex_store.h"
#  else
#    include "../../include/wex_store.h"
#  endif
#else
#  include "wex_store.h"
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(GGML_WEX_BUILDING_DLL)
#define GGML_WEX_API __declspec(dllexport)
#elif defined(_WIN32) && defined(GGML_WEX_USE_DLL)
#define GGML_WEX_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define GGML_WEX_API __attribute__((visibility("default")))
#else
#define GGML_WEX_API
#endif

#define GGML_WEX_ABI_VERSION UINT32_C(1)

/* GGUF / WNR4-GG block constants (matching wnr4/wnr4_gg.h):
 *   Q2_0 (ggml type 42): 64 weights in 18 bytes = 2-byte LE fp16 scale d +
 *   16 packed code bytes (4 x 2-bit codes per byte, levels {-1, 0, +1, +2} * d).
 *   IQ4_NL (ggml type 20): 32 weights in 18 bytes. */
#define GGML_WEX_TYPE_IQ4_NL 20u
#define GGML_WEX_TYPE_Q2_0   42u
#define GGML_WEX_QK2_0       64
#define GGML_WEX_Q2_0_BYTES  18

typedef struct ggml_wex_ctx ggml_wex_ctx_t;

typedef struct {
    uint32_t           struct_size;         /* must be sizeof(ggml_wex_opts_t) */
    const char        *libwex_path;         /* NULL = auto-discover libwex */
    const char        *root_hex;            /* NULL or 64-hex WEX4 root SHA-256 pin */
    wex_verify_mode_t  verify;              /* WEX_VERIFY_AUTO / OFF / ON */
    const char *const *mirrors;             /* NULL (single-NVMe default) or mirror paths */
    int                num_mirrors;         /* 0 for single-NVMe */
    uint64_t           l2_cache_bytes;      /* UINT64_MAX = auto (default), 0 = off */
    int                l2_pin_gpu;          /* 1 = page-lock L2 arena for fast GPU H2D */
    int                io_direct;           /* -1 = auto, 0 = buffered, 1 = direct I/O */
    int                io_queue_depth;      /* 0 = auto */
    int                cpu_threads;         /* 0 = auto */
    int                gpu_device;          /* -1 = auto, -2 = CPU-only, >=0 = explicit GPU */
    const char        *gpu_helper_path;     /* NULL = auto-discover libwex_cuda; "0" = off */
    int                prefetch_next_layer; /* 1 = prefetch layer + 1 during mul_mat_id */
} ggml_wex_opts_t;

static inline void ggml_wex_opts_init(ggml_wex_opts_t *o) {
    if (!o) return;
    o->struct_size         = (uint32_t)sizeof(*o);
    o->libwex_path         = NULL;
    o->root_hex            = NULL;
    o->verify              = WEX_VERIFY_AUTO;
    o->mirrors             = NULL;
    o->num_mirrors         = 0;
    o->l2_cache_bytes      = UINT64_MAX; /* single-NVMe-first auto L2 compressed-RAM cache */
    o->l2_pin_gpu          = 0;
    o->io_direct           = -1;
    o->io_queue_depth      = 0;
    o->cpu_threads         = 0;
    o->gpu_device          = -1;
    o->gpu_helper_path     = NULL;
    o->prefetch_next_layer = 1;
}

GGML_WEX_API uint32_t        ggml_wex_abi_version(void);

/* Standard GGML_BACKEND_DL probing contract: returns 1 if libwex (ABI v1) can
 * be dynamically loaded at runtime, 0 otherwise. */
GGML_WEX_API int             ggml_wex_backend_score(void);

GGML_WEX_API ggml_wex_ctx_t *ggml_wex_open(const char *wex_path,
                                           const ggml_wex_opts_t *opts,
                                           char *err, size_t err_cap);
GGML_WEX_API int             ggml_wex_info(const ggml_wex_ctx_t *ctx, wex_info_t *out);
GGML_WEX_API void            ggml_wex_stats(const ggml_wex_ctx_t *ctx, wex_stats_t *out);
GGML_WEX_API void            ggml_wex_close(ggml_wex_ctx_t *ctx);

/* Non-blocking L2 compressed-RAM warm for upcoming layer experts. */
GGML_WEX_API int             ggml_wex_prefetch_layer(ggml_wex_ctx_t *ctx, int layer,
                                                     const int *eids, int n_eids);

/* Deduplicated batch fetch through libwex (L2 cache + async NVMe + GPU/CPU decode).
 * Duplicate expert IDs in `eids` are decoded once and copied to matching slots.
 * Returns count of failed items (0 = all succeeded). */
GGML_WEX_API int             ggml_wex_fetch_experts(ggml_wex_ctx_t *ctx, int layer,
                                                    const int *eids, int n_eids,
                                                    void *const *dst_slabs,
                                                    const size_t *dst_caps,
                                                    int *status_out);

/* Dequantize a row of k Q2_0 weights (k must be a multiple of GGML_WEX_QK2_0 = 64,
 * 18 bytes per 64-weight block matching wnr4/wnr4_gg.h) to float32. */
GGML_WEX_API void            ggml_wex_dequant_row_q2_0(const void *src_blocks,
                                                       float *dst_f32, int64_t k);

/* Streaming MoE GGML_OP_MUL_MAT_ID for Q2_0 expert slices:
 *   - `ids` has shape [n_tokens, n_used] (row-major expert IDs per token)
 *   - `x_f32` has shape [n_tokens, k_cols] (row-major float32 activations)
 *   - `out_f32` has shape [n_tokens, n_used, n_rows]
 * Deduplicates active expert IDs across the token batch, fetches unique experts
 * in one libwex batch call, computes exact Q2_0 x F32 dot products for
 * `tensor_slot` inside each expert record, and optionally prefetches layer + 1.
 * Returns WEX_OK (0) on success or negative WEX_E_* error code. */
GGML_WEX_API int             ggml_wex_mul_mat_id_q2_0_f32(ggml_wex_ctx_t *ctx,
                                                          int layer, int tensor_slot,
                                                          const int32_t *ids,
                                                          int n_tokens, int n_used,
                                                          const float *x_f32,
                                                          int64_t k_cols, int64_t n_rows,
                                                          float *out_f32);

#ifdef GGML_WEX_WITH_GGML
#include "ggml-backend.h"
GGML_WEX_API ggml_backend_reg_t ggml_backend_wex_reg(void);
GGML_WEX_API int                ggml_backend_score(void);
GGML_WEX_API ggml_backend_reg_t ggml_backend_init(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_GGML_WEX_H */
