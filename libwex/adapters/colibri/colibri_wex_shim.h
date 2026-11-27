/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef COLIBRI_WEX_SHIM_H
#define COLIBRI_WEX_SHIM_H

/* Thin header-only C99/C++11 adapter for colibri engines backed by libwex.
 * Depends only on wex_store.h + wex_loader.h (zero link-time dependency on
 * libwex or CUDA). Maps colibri's COLI_WEX_* environment variables directly
 * into wex_open_opts_t and forwards host/device fetches, L2 prefetch, and
 * single-line telemetry formatting. */

#if defined(__has_include)
#  if __has_include("wex_loader.h")
#    include "wex_loader.h"
#  else
#    include "../../include/wex_loader.h"
#  endif
#else
#  include "wex_loader.h"
#endif

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_LIBWEX_MAX_MIRRORS 16

typedef struct coli_libwex {
    wex_lib_t    lib;
    wex_store_t *store;
    wex_info_t   info;
    int          open;
} coli_libwex_t;

static inline int coli_libwex_stricmp_(const char *a, const char *b) {
    while (*a && *b) {
        int ca = tolower((unsigned char)*a++);
        int cb = tolower((unsigned char)*b++);
        if (ca != cb) return ca - cb;
    }
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}

static inline void coli_libwex_set_err_(char *err, size_t cap, const char *msg) {
    if (!err || !cap) return;
    snprintf(err, cap, "%s", msg ? msg : "unknown error");
    err[cap - 1u] = '\0';
}

static inline int coli_libwex_parse_int_(const char *s, int def_val) {
    char *end = NULL;
    long v;
    if (!s || !s[0]) return def_val;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || v < INT_MIN || v > INT_MAX) return def_val;
    return (int)v;
}

static inline int coli_libwex_parse_mirrors_(const char *raw, char **buf_out,
                                             const char *out_ptrs[COLI_LIBWEX_MAX_MIRRORS]) {
    size_t len;
    char *copy, *p;
    int count = 0;
    *buf_out = NULL;
    if (!raw || !raw[0]) return 0;

    len = strlen(raw);
    copy = (char *)malloc(len + 1u);
    if (!copy) return -1;
    memcpy(copy, raw, len + 1u);

    p = copy;
    while (*p && count < COLI_LIBWEX_MAX_MIRRORS) {
        char *tok, *end;
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ';') p++;
        if (!*p) break;
        tok = p;
        while (*p) {
            if (*p == ',' || *p == ';') break;
            if (*p == ':') {
                int is_win_drive = (p == tok + 1) && isalpha((unsigned char)tok[0]) &&
                                   (p[1] == '\\' || p[1] == '/');
                if (!is_win_drive) break;
            }
            p++;
        }
        end = p;
        if (*p) *p++ = '\0';
        while (end > tok && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
        if (*tok) out_ptrs[count++] = tok;
    }

    *buf_out = copy;
    return count;
}

static inline void coli_libwex_close(coli_libwex_t *cw) {
    if (!cw) return;
    if (cw->store && cw->lib.close) cw->lib.close(cw->store);
    wex_lib_close(&cw->lib);
    memset(cw, 0, sizeof(*cw));
}

static inline int coli_libwex_open_env(coli_libwex_t *cw, const char *path_override,
                                       char *err, size_t err_cap) {
    const char *path;
    const char *env_lib;
    const char *env_root;
    const char *env_verify;
    const char *env_mirror;
    const char *env_cuda;
    const char *env_l2_mb;
    const char *env_l2_pin;
    const char *env_qd;
    const char *env_direct;
    char *mirror_buf = NULL;
    const char *mirror_ptrs[COLI_LIBWEX_MAX_MIRRORS];
    int n_mirrors = 0;
    wex_open_opts_t opts;
    int rc;

    if (err && err_cap) err[0] = '\0';
    if (!cw) {
        coli_libwex_set_err_(err, err_cap, "coli_libwex_open_env: cw is NULL");
        return WEX_E_ARGS;
    }
    memset(cw, 0, sizeof(*cw));

    path = (path_override && path_override[0]) ? path_override : getenv("COLI_WEX_PATH");
    if (!path || !path[0]) {
        coli_libwex_set_err_(err, err_cap, "coli_libwex_open_env: no path and COLI_WEX_PATH unset");
        return WEX_E_ARGS;
    }

    env_lib = getenv("COLI_WEX_LIB");
    if (!env_lib || !env_lib[0]) env_lib = getenv("WEX_LIB_PATH");
    rc = wex_lib_open(&cw->lib, (env_lib && env_lib[0]) ? env_lib : NULL);
    if (rc != WEX_OK) {
        coli_libwex_set_err_(err, err_cap, "coli_libwex_open_env: failed to load libwex shared library");
        return rc;
    }

    wex_open_opts_init(&opts);

    env_root = getenv("COLI_WEX_ROOT");
    if (env_root && env_root[0]) opts.root_hex = env_root;

    env_verify = getenv("COLI_WEX_VERIFY");
    if (env_verify && env_verify[0]) {
        if (strcmp(env_verify, "0") == 0 || coli_libwex_stricmp_(env_verify, "off") == 0 ||
            coli_libwex_stricmp_(env_verify, "false") == 0) {
            opts.verify = WEX_VERIFY_OFF;
        } else if (strcmp(env_verify, "1") == 0 || coli_libwex_stricmp_(env_verify, "on") == 0 ||
                   coli_libwex_stricmp_(env_verify, "true") == 0) {
            opts.verify = WEX_VERIFY_ON;
        } else {
            opts.verify = WEX_VERIFY_AUTO;
        }
    }

    env_mirror = getenv("COLI_MODEL_MIRROR");
    if (env_mirror && env_mirror[0]) {
        n_mirrors = coli_libwex_parse_mirrors_(env_mirror, &mirror_buf, mirror_ptrs);
        if (n_mirrors < 0) {
            wex_lib_close(&cw->lib);
            coli_libwex_set_err_(err, err_cap, "coli_libwex_open_env: out of memory parsing COLI_MODEL_MIRROR");
            return WEX_E_OOM;
        }
        if (n_mirrors > 0) {
            opts.mirrors     = mirror_ptrs;
            opts.num_mirrors = n_mirrors;
        }
    }

    env_cuda = getenv("COLI_WEX_CUDA_LIB");
    if (env_cuda && env_cuda[0]) {
        opts.gpu_helper_path = env_cuda;
        if (strcmp(env_cuda, "0") == 0) opts.gpu_device = -2;
    }

    env_l2_mb = getenv("COLI_WEX_L2_MB");
    if (env_l2_mb && env_l2_mb[0]) {
        if (coli_libwex_stricmp_(env_l2_mb, "auto") == 0) {
            opts.l2_cache_bytes = UINT64_MAX;
        } else {
            char *end = NULL;
            unsigned long long mb;
            errno = 0;
            mb = strtoull(env_l2_mb, &end, 10);
            if (errno == 0 && end != env_l2_mb) {
                if (mb == 0ULL) opts.l2_cache_bytes = 0u;
                else if (mb >= (UINT64_MAX >> 20)) opts.l2_cache_bytes = UINT64_MAX - 1u;
                else opts.l2_cache_bytes = (uint64_t)mb << 20;
            }
        }
    } else if (opts.num_mirrors == 0) {
        opts.l2_cache_bytes = UINT64_MAX;
    }

    env_l2_pin = getenv("COLI_WEX_L2_PIN");
    if (env_l2_pin && env_l2_pin[0]) {
        opts.l2_pin_gpu = (coli_libwex_parse_int_(env_l2_pin, 0) != 0) ? 1 : 0;
    }

    env_qd = getenv("COLI_WEX_QD");
    if (env_qd && env_qd[0]) {
        int qd = coli_libwex_parse_int_(env_qd, 0);
        if (qd >= 0) opts.io_queue_depth = qd;
    }

    env_direct = getenv("COLI_WEX_DIRECT");
    if (env_direct && env_direct[0]) {
        int d = coli_libwex_parse_int_(env_direct, -1);
        opts.io_direct = (d < 0) ? -1 : (d > 0 ? 1 : 0);
    }

    cw->store = cw->lib.open(path, &opts, err, err_cap);
    free(mirror_buf);
    if (!cw->store) {
        wex_lib_close(&cw->lib);
        if (err && err_cap && strstr(err, "root") != NULL) return WEX_E_ROOT;
        return WEX_E_IO;
    }

    if (cw->lib.info(cw->store, &cw->info) != WEX_OK) {
        coli_libwex_close(cw);
        coli_libwex_set_err_(err, err_cap, "coli_libwex_open_env: wex_info failed");
        return WEX_E_FORMAT;
    }
    cw->open = 1;
    return WEX_OK;
}

static inline int coli_libwex_info(const coli_libwex_t *cw, wex_info_t *out) {
    if (!cw || !cw->open || !cw->store || !out) return WEX_E_ARGS;
    return cw->lib.info(cw->store, out);
}

static inline int coli_libwex_has_expert(const coli_libwex_t *cw, int layer, int eid) {
    if (!cw || !cw->open || !cw->store) return 0;
    return cw->lib.has_expert(cw->store, layer, eid);
}

static inline int coli_libwex_layout(const coli_libwex_t *cw, int layer, int eid,
                                     wex_layout_t *out) {
    if (!cw || !cw->open || !cw->store || !out) return WEX_E_ARGS;
    return cw->lib.record_layout(cw->store, layer, eid, 0u, out);
}

static inline int coli_libwex_fetch_host(coli_libwex_t *cw, int layer, const int *eids, int n,
                                         void *const *dst_slabs, const size_t *dst_caps,
                                         int scratch, int *status_out) {
    uint32_t flags = scratch ? WEX_FETCH_SCRATCH : WEX_FETCH_DEFAULT;
    if (!cw || !cw->open || !cw->store) return WEX_E_ARGS;
    return cw->lib.fetch_host(cw->store, layer, eids, n, dst_slabs, dst_caps, flags, status_out);
}

static inline int coli_libwex_fetch_device(coli_libwex_t *cw, int layer, const int *eids, int n,
                                           void *const *dst_scales, void *const *dst_weights,
                                           int expand_e8m0_f32, void *stream, int *status_out) {
    uint32_t flags = expand_e8m0_f32 ? WEX_FETCH_EXPAND_E8M0_F32 : WEX_FETCH_DEFAULT;
    if (!cw || !cw->open || !cw->store) return WEX_E_ARGS;
    return cw->lib.fetch_device(cw->store, layer, eids, n, dst_scales, dst_weights,
                                NULL, stream, flags, status_out);
}

static inline int coli_libwex_prefetch(coli_libwex_t *cw, int layer, const int *eids, int n) {
    if (!cw || !cw->open || !cw->store) return 0;
    return cw->lib.prefetch(cw->store, layer, eids, n, WEX_FETCH_DEFAULT);
}

static inline int coli_libwex_format_stats(const coli_libwex_t *cw, char *buf, size_t cap) {
    wex_info_t info;
    wex_stats_t st;
    uint64_t total;
    double l2_hit, disk_mib, dec_mib;
    if (!cw || !cw->open || !cw->store || !buf || !cap) return -1;
    if (cw->lib.info(cw->store, &info) != WEX_OK) return -1;
    cw->lib.stats(cw->store, &st);
    total    = st.records_from_l2 + st.records_from_disk;
    l2_hit   = total ? (100.0 * (double)st.records_from_l2 / (double)total) : 0.0;
    disk_mib = (double)st.disk_bytes / (1024.0 * 1024.0);
    dec_mib  = (double)st.decoded_bytes / (1024.0 * 1024.0);
    return snprintf(buf, cap, "[libwex] backend=%s io=%s l2_hit=%.1f%% disk=%.2f MiB dec=%.2f MiB",
                    info.backend_name, info.io_engine, l2_hit, disk_mib, dec_mib);
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_WEX_SHIM_H */
