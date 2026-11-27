/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_STORE_H
#define LIBWEX_WEX_STORE_H

/* libwex public C99 ABI v1: single-NVMe-first WEX4 expert & tensor store.
 *
 * Designed for zero-friction adoption by inference engines (colibri, llama.cpp,
 * ik_llama.cpp, vLLM/PyTorch via pywex, Strata, Hermes, LM Studio runtimes):
 *   - Pure C99 opaque handle; zero CUDA/Vulkan/OS headers required by callers.
 *   - Dynamic runtime loading via wex_loader.h (~150 lines, zero build deps).
 *   - Built-in L2 Compressed-RAM sharded LRU cache (huge speedup on 1 NVMe).
 *   - Adaptive async I/O: Windows OVERLAPPED batch (16 MiB cap), Linux io_uring
 *     + O_DIRECT, macOS F_NOCACHE; optional N-mirror routing when >=2 drives.
 *   - Unified lossless decode: RAW, WBP1, WNR3, WNR4 (MX, NV, GG, I4) on CPU
 *     (AVX2/NEON/SVE/scalar) or optional dynamically loaded GPU backend (CUDA).
 *   - Streaming chunked 2 MiB SHA-256 verification (SHA-NI / ARMv8 Crypto). */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(LIBWEX_BUILDING_DLL)
#define WEX_API __declspec(dllexport)
#elif defined(_WIN32) && defined(LIBWEX_USE_DLL)
#define WEX_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define WEX_API __attribute__((visibility("default")))
#else
#define WEX_API
#endif

#define WEX_STORE_ABI_VERSION UINT32_C(1)
#define WEX_MAX_RECORD_TENSORS 16u

/* Error / status codes (0 = OK, negative = error, positive = transient/fallback) */
enum {
    WEX_OK              =  0,
    WEX_E_ARGS          = -1,
    WEX_E_IO            = -2,
    WEX_E_FORMAT        = -3,
    WEX_E_ROOT          = -4,
    WEX_E_HASH          = -5,
    WEX_E_DECODE        = -6,
    WEX_E_UNSUPPORTED   = -7,
    WEX_E_OOM           = -8,
    WEX_E_NOT_FOUND     = -9,
    WEX_E_CAP           = -10,
    WEX_E_DEVICE        = -11,
    WEX_E_REFUSED       = -12
};

typedef enum {
    WEX_VERIFY_AUTO = 0,   /* ON when root_hex is provided, OFF otherwise */
    WEX_VERIFY_OFF  = 1,
    WEX_VERIFY_ON   = 2
} wex_verify_mode_t;

typedef enum {
    WEX_CODEC_RAW     = 0,
    WEX_CODEC_WNR3    = 5,
    WEX_CODEC_WBP1    = 6,
    WEX_CODEC_WNR4_MX = 8,
    WEX_CODEC_WNR4_NV = 9,
    WEX_CODEC_WNR4_GG = 10,
    WEX_CODEC_WNR4_I4 = 11
} wex_codec_kind_t;

/* Fetch flags for wex_fetch_host / wex_fetch_device */
enum {
    WEX_FETCH_DEFAULT        = 0u,
    /* Caller passes scratch destination buffers that may be overwritten on error;
     * enables single-sync GPU D2H fast path (halves cudaStreamSynchronize count). */
    WEX_FETCH_SCRATCH        = 1u << 0,
    /* Do not insert newly read records into L2 compressed-RAM cache. */
    WEX_FETCH_BYPASS_L2      = 1u << 1,
    /* Record-id mode: `eids` holds flat record indices instead of (layer, eid). */
    WEX_FETCH_BY_RECORD_ID   = 1u << 2,
    /* For MXFP4 device destinations: expand E8M0 scale bytes to float32 on device. */
    WEX_FETCH_EXPAND_E8M0_F32 = 1u << 3
};

typedef struct wex_store wex_store_t;

typedef struct {
    uint32_t struct_size;        /* must be sizeof(wex_open_opts_t) */
    const char *root_hex;        /* NULL or 64-char hex root SHA-256 to pin */
    wex_verify_mode_t verify;    /* first-read record SHA-256 verification */
    const char *const *mirrors;  /* NULL (single-NVMe default) or array of mirror file paths */
    int num_mirrors;             /* 0 for single-NVMe */
    uint64_t l2_cache_bytes;     /* 0 = off, UINT64_MAX = auto (up to 25% free RAM, max 16 GiB) */
    int l2_pin_gpu;              /* 1 = page-lock L2 cache arena for fast GPU H2D (0 = normal RAM) */
    int io_direct;               /* -1 = auto, 0 = buffered, 1 = O_DIRECT / NO_BUFFERING */
    int io_queue_depth;          /* 0 = auto (4-8 on 1 NVMe, 8*(1+mirrors) on multi-NVMe) */
    int cpu_threads;             /* 0 = auto */
    int gpu_device;              /* -1 = auto, -2 = force CPU-only, >=0 = explicit GPU ordinal */
    const char *gpu_helper_path; /* NULL = auto-discover libwex_cuda beside libwex; "0" = disable */
    const uint32_t *disk_weights;/* NULL = auto (equal/probed), else 1+num_mirrors positive weights */
} wex_open_opts_t;

static inline void wex_open_opts_init(wex_open_opts_t *o) {
    if (!o) return;
    o->struct_size = (uint32_t)sizeof(*o);
    o->root_hex = NULL;
    o->verify = WEX_VERIFY_AUTO;
    o->mirrors = NULL;
    o->num_mirrors = 0;
    o->l2_cache_bytes = 0;
    o->l2_pin_gpu = 0;
    o->io_direct = -1;
    o->io_queue_depth = 0;
    o->cpu_threads = 0;
    o->gpu_device = -1;
    o->gpu_helper_path = NULL;
    o->disk_weights = NULL;
}

typedef struct {
    uint32_t abi_version;
    uint32_t layers;
    uint32_t experts;
    uint32_t num_records;
    uint32_t num_tensors;
    uint32_t codec_kind;          /* primary wex_codec_kind_t */
    uint32_t value_exact;         /* 1 if canonical-zero value-exact container */
    uint32_t verified_root;       /* 1 if root_hex matched */
    uint32_t verify_on;           /* 1 if first-read SHA-256 verification is active */
    uint32_t num_replicas;        /* 1 = single drive, >1 = primary + verified mirrors */
    uint64_t file_bytes;
    uint64_t total_decoded_bytes;
    uint64_t max_record_decoded_bytes;
    uint64_t max_record_stored_bytes;
    uint64_t l2_capacity_bytes;
    char root_hex[65];
    char backend_name[32];        /* "cuda", "cpu-avx2", "cpu-neon", "cpu-scalar" */
    char io_engine[32];           /* "win-overlapped", "linux-iouring", "posix-pread" */
} wex_info_t;

typedef struct {
    uint32_t record_index;
    uint32_t codec_kind;          /* wex_codec_kind_t */
    uint32_t flags;               /* WEX4 record flags */
    uint32_t tensor_count;
    uint64_t file_offset;
    uint64_t stored_bytes;
    uint64_t decoded_bytes;
    /* Offsets & byte sizes of up to WEX_MAX_RECORD_TENSORS tensors inside decoded slab */
    uint64_t tensor_offsets[WEX_MAX_RECORD_TENSORS];
    uint64_t tensor_bytes[WEX_MAX_RECORD_TENSORS];
    /* Expert convenience fields (for 6-tensor MXFP4/NVFP4 expert records; 0 otherwise):
     * matrix order 0=w1/gate, 1=w2/down, 2=w3/up */
    uint32_t scale_offsets[3];
    uint32_t weight_offsets[3];
    uint32_t scale_shapes[3][2];
    uint32_t weight_shapes[3][2];
} wex_layout_t;

typedef struct {
    uint64_t fetches;
    uint64_t records_requested;
    uint64_t records_from_l2;
    uint64_t records_from_disk;
    uint64_t gpu_decodes;
    uint64_t cpu_decodes;
    uint64_t prefetch_hits;
    uint64_t refused;
    uint64_t disk_bytes;
    uint64_t decoded_bytes;
    uint64_t l2_used_bytes;
    uint64_t l2_entries;
    double io_seconds;
    double decode_seconds;
    double verify_seconds;
} wex_stats_t;

WEX_API uint32_t     wex_abi_version(void);
WEX_API const char  *wex_version_string(void);

WEX_API wex_store_t *wex_open(const char *path, const wex_open_opts_t *opts,
                              char *err_buf, size_t err_cap);
WEX_API int          wex_info(const wex_store_t *s, wex_info_t *out);
WEX_API int          wex_has_expert(const wex_store_t *s, int layer, int eid);
WEX_API int          wex_record_layout(const wex_store_t *s, int layer, int eid,
                                       uint32_t flags, wex_layout_t *out);

/* Fetch n experts of `layer` (or n flat record indices with WEX_FETCH_BY_RECORD_ID)
 * into caller host buffers dst_slabs[0..n-1] (capacities dst_caps[0..n-1]).
 * Returns count of failed items (0 = all succeeded); status_out[i] = WEX_OK or < 0. */
WEX_API int          wex_fetch_host(wex_store_t *s, int layer, const int *eids, int n,
                                    void *const *dst_slabs, const size_t *dst_caps,
                                    uint32_t flags, int *status_out);

/* Fetch n experts directly into caller device buffers:
 * - For MXFP4/NVFP4 expert records: dst_scales[3*i+m] and dst_weights[3*i+m]
 *   (WEX matrix order m=0:w1, 1:w2, 2:w3); dst_slabs may be NULL.
 * - For contiguous non-expert records (GG/I4/RAW): dst_slabs[i] holds the device
 *   slab pointer; dst_scales/dst_weights may be NULL.
 * Returns count of failed items (0 = all succeeded); WEX_E_DEVICE in status_out[i]
 * if no GPU backend is active. */
WEX_API int          wex_fetch_device(wex_store_t *s, int layer, const int *eids, int n,
                                      void *const *dst_scales, void *const *dst_weights,
                                      void *const *dst_slabs, void *device_stream,
                                      uint32_t flags, int *status_out);

/* Non-blocking hint to warm L2 compressed-RAM cache with upcoming experts.
 * Safe no-op when L2 cache is disabled. Returns number of items queued. */
WEX_API int          wex_prefetch(wex_store_t *s, int layer, const int *eids, int n,
                                  uint32_t flags);

WEX_API void         wex_stats(const wex_store_t *s, wex_stats_t *out);
WEX_API void         wex_close(wex_store_t *s);

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_WEX_STORE_H */
