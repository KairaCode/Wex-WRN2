/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_DEV_OPS_H
#define LIBWEX_WEX_DEV_OPS_H

/* Internal pluggable device-backend vtable for libwex.
 * Today backed by libwex_cuda (CUDA: WNR3 + WNR4-MX/NV + WNR4-I4 + WNR4-GG);
 * Vulkan, Metal, and HIP backends implement this same table without changing
 * libwex's public wex_store.h ABI. */

#include <stddef.h>
#include <stdint.h>
#include "../../c/wex_format.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WEX_DEV_ABI_VERSION UINT32_C(1)

/* Bitmask of codecs supported by a device backend */
enum {
    WEX_DEV_CAP_PIN_HOST     = 1u << 0,
    WEX_DEV_CAP_WNR3         = 1u << 1,
    WEX_DEV_CAP_WNR4_MX      = 1u << 2,
    WEX_DEV_CAP_WNR4_NV      = 1u << 3,
    WEX_DEV_CAP_WNR4_I4      = 1u << 4,
    WEX_DEV_CAP_WNR4_GG      = 1u << 5,
    WEX_DEV_CAP_DIRECT_DEV   = 1u << 6,
    WEX_DEV_CAP_SCRATCH_HOST = 1u << 7,
    WEX_DEV_CAP_E8M0_EXPAND  = 1u << 8
};

/* Geometry descriptor for WNR4-I4 batch decode */
typedef struct {
    uint32_t nmat;
    uint32_t pairs;
    uint32_t rows[4];
    uint32_t cols[4];
} wex_dev_i4_geo_t;

/* Geometry descriptor for WNR4-GG batch decode (1..4 GGUF slices per record) */
typedef struct {
    uint32_t nslice;
    uint32_t ggml_type[4];
    uint64_t offsets[4];
    uint64_t nbytes[4];
    uint32_t rows[4];
    uint32_t cols[4];
} wex_dev_gg_geo_t;

typedef struct wex_dev_ops {
    uint32_t abi_version;
    uint32_t caps;
    const char *name; /* e.g. "cuda" */

    /* Device query & page-locked host memory registration */
    int (*device_count)(void);
    int (*host_register)(void *ptr, size_t bytes);
    int (*host_unregister)(void *ptr);

    /* Model blob registration:
     *   kind = 0 -> WNM2 (WNR2/WNR3)
     *   kind = 1 -> WNM4 MX/NV (WNR4 kinds 0,1)
     *   kind = 2 -> WNM4 GG (WNR4 kind 2)
     *   kind = 3 -> WNM4 I4 (WNR4 kind 3) */
    int (*register_model)(int kind, const uint8_t *blob, size_t bytes);

    /* WNR3 & WNR4-MX/NV expert batch decode -> host slabs.
     * scratch_dst != 0 allows single-sync D2H (may overwrite dst[i] on error). */
    int (*decode_expert_host)(int n, const WexEntry *const *entries,
                              const uint8_t *const *records, int device,
                              uint8_t *const *dst_slabs, int scratch_dst, int *status);

    /* WNR3 & WNR4-MX/NV expert batch decode -> device pointers (3 per record: w1,w2,w3).
     * expand_e8m0_f32 != 0 expands raw uint8 E8M0 bytes in dst_scale[3*i+m] into float32
     * in place (caller guarantees dst_scale[3*i+m] has 4 * nscales bytes). */
    int (*decode_expert_device)(int n, const WexEntry *const *entries,
                                const uint8_t *const *records, int device,
                                uint8_t *const *dst_scale, uint8_t *const *dst_weight,
                                void *stream, int expand_e8m0_f32, int *status);

    /* WNR4-I4 batch decode -> host or device slabs (dst_is_device=0/1) */
    int (*decode_i4_batch)(int n, const wex_dev_i4_geo_t *geo,
                           const uint8_t *const *records, const size_t *totals,
                           const uint32_t *owners, int device,
                           uint8_t *const *dst, int dst_is_device,
                           int scratch_dst, void *stream, int *status);

    /* WNR4-GG batch decode -> host or device slabs (dst_is_device=0/1) */
    int (*decode_gg_batch)(int n, const wex_dev_gg_geo_t *geos,
                           const uint8_t *const *records, const size_t *totals,
                           const uint32_t *owners, int device,
                           uint8_t *const *dst, int dst_is_device,
                           int scratch_dst, void *stream, int *status);
} wex_dev_ops_t;

/* Exported symbol name inside libwex_cuda.dll / libwex_cuda.so */
#define WEX_DEV_INIT_SYMBOL "wex_dev_get_ops"
typedef const wex_dev_ops_t *(*wex_dev_get_ops_fn)(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_WEX_DEV_OPS_H */
