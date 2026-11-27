/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "../include/wex_store.h"
#include "wex_io.h"
#include "wex_dev_ops.h"
#include "wex_cache.h"

#include "../../c/wex4.h"
#if !defined(_MSC_VER) || defined(__clang__)
#define LIBWEX_HAS_FAST_DECODE 1
#include "../../c/wex_fast_decode.h"
#else
#define LIBWEX_HAS_FAST_DECODE 0
#endif
#include "../../wnr4/wnr4.h"
#include "../../wnr4/wnr4_gg.h"
#include "../../wnr4/wnr4_i4.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

/* ---- Portable sync, thread & atomic helpers ------------------------------ */

#ifdef _WIN32
typedef SRWLOCK            wex_s_lock_t;
typedef CONDITION_VARIABLE wex_s_cond_t;
typedef HANDLE             wex_s_thread_t;
static inline void wex_s_lock_init(wex_s_lock_t *l)    { InitializeSRWLock(l); }
static inline void wex_s_lock_destroy(wex_s_lock_t *l) { (void)l; }
static inline void wex_s_lock(wex_s_lock_t *l)         { AcquireSRWLockExclusive(l); }
static inline void wex_s_unlock(wex_s_lock_t *l)       { ReleaseSRWLockExclusive(l); }
static inline void wex_s_cond_init(wex_s_cond_t *c)    { InitializeConditionVariable(c); }
static inline void wex_s_cond_destroy(wex_s_cond_t *c) { (void)c; }
static inline void wex_s_cond_wait(wex_s_cond_t *c, wex_s_lock_t *l) {
    SleepConditionVariableSRW(c, l, INFINITE, 0);
}
static inline void wex_s_cond_signal(wex_s_cond_t *c)    { WakeConditionVariable(c); }
static inline void wex_s_cond_broadcast(wex_s_cond_t *c) { WakeAllConditionVariable(c); }
#else
typedef pthread_mutex_t wex_s_lock_t;
typedef pthread_cond_t  wex_s_cond_t;
typedef pthread_t       wex_s_thread_t;
static inline void wex_s_lock_init(wex_s_lock_t *l)    { pthread_mutex_init(l, NULL); }
static inline void wex_s_lock_destroy(wex_s_lock_t *l) { pthread_mutex_destroy(l); }
static inline void wex_s_lock(wex_s_lock_t *l)         { pthread_mutex_lock(l); }
static inline void wex_s_unlock(wex_s_lock_t *l)       { pthread_mutex_unlock(l); }
static inline void wex_s_cond_init(wex_s_cond_t *c)    { pthread_cond_init(c, NULL); }
static inline void wex_s_cond_destroy(wex_s_cond_t *c) { pthread_cond_destroy(c); }
static inline void wex_s_cond_wait(wex_s_cond_t *c, wex_s_lock_t *l) {
    pthread_cond_wait(c, l);
}
static inline void wex_s_cond_signal(wex_s_cond_t *c)    { pthread_cond_signal(c); }
static inline void wex_s_cond_broadcast(wex_s_cond_t *c) { pthread_cond_broadcast(c); }
#endif

#if defined(__GNUC__) || defined(__clang__)
static inline uint8_t wex_u8_load(const uint8_t *p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void wex_u8_store(uint8_t *p, uint8_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline uint8_t wex_u8_xchg(uint8_t *p, uint8_t v) {
    return __atomic_exchange_n(p, v, __ATOMIC_ACQ_REL);
}
static inline int wex_u8_cas(uint8_t *p, uint8_t expected, uint8_t desired) {
    uint8_t e = expected;
    return __atomic_compare_exchange_n(p, &e, desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
#elif defined(_MSC_VER)
#include <intrin.h>
static inline uint8_t wex_u8_load(const uint8_t *p) {
    char v = _InterlockedOr8((volatile char *)p, 0);
    return (uint8_t)v;
}
static inline void wex_u8_store(uint8_t *p, uint8_t v) {
    _InterlockedExchange8((volatile char *)p, (char)v);
}
static inline uint8_t wex_u8_xchg(uint8_t *p, uint8_t v) {
    return (uint8_t)_InterlockedExchange8((volatile char *)p, (char)v);
}
static inline int wex_u8_cas(uint8_t *p, uint8_t expected, uint8_t desired) {
    return (uint8_t)_InterlockedCompareExchange8((volatile char *)p, (char)desired, (char)expected) == expected;
}
#else
static inline uint8_t wex_u8_load(const uint8_t *p) { return *p; }
static inline void wex_u8_store(uint8_t *p, uint8_t v) { *p = v; }
static inline uint8_t wex_u8_xchg(uint8_t *p, uint8_t v) { uint8_t o = *p; *p = v; return o; }
static inline int wex_u8_cas(uint8_t *p, uint8_t expected, uint8_t desired) {
    if (*p != expected) return 0;
    *p = desired;
    return 1;
}
#endif

/* ---- Format & store constants -------------------------------------------- */

#define LIBWEX_CODEC_WNR4     8u
#define LIBWEX_BLOB_WNM4      3u
#define LIBWEX_DT_F8_E4M3     5u
#define LIBWEX_DT_F32         13u

#define LIBWEX_STATE_UNVERIFIED 0u
#define LIBWEX_STATE_VERIFIED   1u
#define LIBWEX_STATE_REFUSED    2u

#define LIBWEX_SCRATCH_MAX_SLOTS 48u
#define LIBWEX_STACK_BATCH       64
#define LIBWEX_PF_Q_CAP          1024u
#define LIBWEX_PF_BATCH          8

static void wex_set_err(char *buf, size_t cap, const char *fmt, ...) {
    va_list ap;
    if (!buf || !cap) return;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
    buf[cap - 1u] = '\0';
}

static void wex_hex_encode(const uint8_t digest[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    unsigned int i;
    for (i = 0; i < 32u; i++) {
        out[i * 2u]      = hex[digest[i] >> 4];
        out[i * 2u + 1u] = hex[digest[i] & 0x0fu];
    }
    out[64] = '\0';
}

static int wex_hex_eq64(const char *a, const char *b) {
    unsigned int i;
    if (!a || !b) return 0;
    for (i = 0; i < 64u; i++) {
        unsigned char ca = (unsigned char)a[i];
        unsigned char cb = (unsigned char)b[i];
        if (!ca || !cb) return 0;
        if (tolower(ca) != tolower(cb)) return 0;
    }
    return a[64] == '\0' && b[64] == '\0';
}

/* ---- Dynamic device helper loader ---------------------------------------- */

static void *wex_dl_open(const char *path) {
    if (!path || !path[0]) return NULL;
#ifdef _WIN32
    if (strchr(path, '\\') || strchr(path, '/')) {
        HMODULE m = LoadLibraryExA(path, NULL,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (m) return (void *)m;
    }
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void wex_dl_close(void *h) {
    if (!h) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)h);
#else
    dlclose(h);
#endif
}

static int wex_dl_sym(void *h, const char *name, void *fn_out) {
#ifdef _WIN32
    FARPROC p = GetProcAddress((HMODULE)h, name);
    if (!p) return -1;
    memcpy(fn_out, &p, sizeof(p));
#else
    void *p = dlsym(h, name);
    if (!p) return -1;
    memcpy(fn_out, &p, sizeof(p));
#endif
    return 0;
}

static int wex_sibling_path(char *out, size_t cap, const char *dir_src, const char *filename) {
    size_t flen = strlen(filename);
    const char *sep = strrchr(dir_src, '/');
#ifdef _WIN32
    const char *bsep = strrchr(dir_src, '\\');
    if (bsep && (!sep || bsep > sep)) sep = bsep;
#endif
    if (!sep) return -1;
    if ((size_t)(sep - dir_src) + 1u + flen + 1u > cap) return -1;
    memcpy(out, dir_src, (size_t)(sep - dir_src) + 1u);
    memcpy(out + (size_t)(sep - dir_src) + 1u, filename, flen + 1u);
    return 0;
}

static int wex_self_module_dir_file(char *out, size_t cap, const char *filename) {
    char self_path[4096];
#ifdef _WIN32
    HMODULE hmod = NULL;
    DWORD n;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)(uintptr_t)&wex_abi_version, &hmod))
        return -1;
    n = GetModuleFileNameA(hmod, self_path, (DWORD)sizeof(self_path));
    if (!n || (size_t)n >= sizeof(self_path)) return -1;
    self_path[n] = '\0';
    return wex_sibling_path(out, cap, self_path, filename);
#else
    Dl_info dli;
    if (!dladdr((const void *)(uintptr_t)&wex_abi_version, &dli) || !dli.dli_fname)
        return -1;
    if (strlen(dli.dli_fname) >= sizeof(self_path)) return -1;
    strcpy(self_path, dli.dli_fname);
    return wex_sibling_path(out, cap, self_path, filename);
#endif
}

static int wex_exe_dir_file(char *out, size_t cap, const char *filename) {
    char exe_path[4096];
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, exe_path, (DWORD)sizeof(exe_path));
    if (!n || (size_t)n >= sizeof(exe_path)) return -1;
    exe_path[n] = '\0';
#elif defined(__APPLE__)
    uint32_t sz = (uint32_t)sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &sz) != 0) return -1;
    exe_path[sizeof(exe_path) - 1u] = '\0';
#else
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1u);
    if (n <= 0 || (size_t)n >= sizeof(exe_path)) return -1;
    exe_path[n] = '\0';
#endif
    return wex_sibling_path(out, cap, exe_path, filename);
}

static const wex_dev_ops_t *wex_try_dev_path(const char *path, void **handle_out) {
    void *h;
    wex_dev_get_ops_fn get_ops = NULL;
    const wex_dev_ops_t *ops;
    if (!path || !path[0]) return NULL;
    h = wex_dl_open(path);
    if (!h) return NULL;
    if (wex_dl_sym(h, WEX_DEV_INIT_SYMBOL, &get_ops) != 0 || !get_ops) {
        wex_dl_close(h);
        return NULL;
    }
    ops = get_ops();
    if (!ops || ops->abi_version != WEX_DEV_ABI_VERSION || !ops->device_count) {
        wex_dl_close(h);
        return NULL;
    }
    *handle_out = h;
    return ops;
}

static const wex_dev_ops_t *wex_load_dev_backend(const char *explicit_path,
                                                 void **handle_out) {
    char cand[4096];
    const wex_dev_ops_t *ops = NULL;
#ifdef _WIN32
    const char *libname = "libwex_cuda.dll";
#elif defined(__APPLE__)
    const char *libname = "libwex_cuda.dylib";
#else
    const char *libname = "libwex_cuda.so";
#endif
    *handle_out = NULL;

    if (explicit_path && explicit_path[0]) {
        if (strcmp(explicit_path, "0") == 0) return NULL;
        return wex_try_dev_path(explicit_path, handle_out);
    }

    {
        const char *env = getenv("WEX_CUDA_LIB");
        if (env && env[0]) {
            if (strcmp(env, "0") == 0) return NULL;
            ops = wex_try_dev_path(env, handle_out);
            if (ops) return ops;
        }
    }

    if (wex_self_module_dir_file(cand, sizeof(cand), libname) == 0) {
        ops = wex_try_dev_path(cand, handle_out);
        if (ops) return ops;
    }
    if (wex_exe_dir_file(cand, sizeof(cand), libname) == 0) {
        ops = wex_try_dev_path(cand, handle_out);
        if (ops) return ops;
    }
    return wex_try_dev_path(libname, handle_out);
}

/* ---- WEX4 + Codec-8 index validation & geometry extraction --------------- */

/* Extract legacy WexEntry and optional Wnr4Geo for an EXPERT record (MXFP4 or NVFP4). */
static int wex_store_expert_entry(const Wex4View *v, uint32_t r,
                                  WexEntry *e_out, Wnr4Geo *geo_out, uint32_t *kind_out) {
    Wex4Record rec;
    Wex4Tensor t;
    WexEntry e;
    uint32_t m, kind = 0;
    uint32_t scount[3], scols[3], soff[3], woff[3];
    uint64_t scale_sum = 0, weight_sum = 0;
    if (!v || r >= v->h.records) return -1;
    wex4_get_record(v, r, &rec);
    if (!(rec.flags & WEX4_REC_EXPERT) || rec.tensor_count != 6u) return -1;

    memset(&e, 0, sizeof(e));
    e.file_offset = rec.offset;
    e.total_bytes = rec.stored;
    e.flags       = rec.codec;

    for (m = 0; m < 6u; m++) {
        wex4_get_tensor(v, rec.first_tensor + m, &t);
        if (t.rank != 2u || t.record_count != 1u ||
            !t.shape[0] || !t.shape[1] || t.shape[0] > 65535u || t.shape[1] > 65535u)
            return -1;
        if (m < 3u) {
            if (t.dtype != WEX4_DT_U8 && t.dtype != WEX4_DT_F8_E8M0 && t.dtype != LIBWEX_DT_F8_E4M3)
                return -1;
            if (t.offset != scale_sum) return -1;
            e.scale_shapes[m][0] = (uint16_t)t.shape[0];
            e.scale_shapes[m][1] = (uint16_t)t.shape[1];
            e.scale_offsets[m]   = (uint32_t)t.offset;
            scount[m]            = (uint32_t)(t.shape[0] * t.shape[1]);
            scols[m]             = (uint32_t)t.shape[1];
            soff[m]              = (uint32_t)t.offset;
            scale_sum           += t.nbytes;
        } else {
            uint32_t mi = m - 3u;
            if (t.dtype != WEX4_DT_U8 && t.dtype != WEX4_DT_I8) return -1;
            if (t.offset != scale_sum + weight_sum) return -1;
            e.weight_shapes[mi][0] = (uint16_t)t.shape[0];
            e.weight_shapes[mi][1] = (uint16_t)t.shape[1];
            e.weight_offsets[mi]   = (uint32_t)t.offset;
            woff[mi]               = (uint32_t)t.offset;
            weight_sum            += t.nbytes;
        }
    }
    if (scale_sum + weight_sum != rec.decoded || scale_sum > UINT32_MAX || weight_sum > UINT32_MAX)
        return -1;
    e.scale_bytes  = (uint32_t)scale_sum;
    e.weight_bytes = (uint32_t)weight_sum;

    /* Determine MX (wcols == 16 * scols) vs NV (wcols == 8 * scols) consistently across all 3 matrices */
    if (e.scale_shapes[0][0] == e.weight_shapes[0][0] &&
        (uint32_t)e.scale_shapes[0][1] * 16u == (uint32_t)e.weight_shapes[0][1]) {
        kind = WNR4_KIND_MX;
        for (m = 0; m < 3u; m++) {
            if (e.scale_shapes[m][0] != e.weight_shapes[m][0] ||
                (uint32_t)e.scale_shapes[m][1] * 16u != (uint32_t)e.weight_shapes[m][1])
                return -1;
        }
    } else if (e.scale_shapes[0][0] == e.weight_shapes[0][0] &&
               (uint32_t)e.scale_shapes[0][1] * 8u == (uint32_t)e.weight_shapes[0][1]) {
        kind = WNR4_KIND_NV;
        for (m = 0; m < 3u; m++) {
            if (e.scale_shapes[m][0] != e.weight_shapes[m][0] ||
                (uint32_t)e.scale_shapes[m][1] * 8u != (uint32_t)e.weight_shapes[m][1])
                return -1;
        }
    } else {
        return -1;
    }

    if (geo_out) {
        if (wnr4_geo(geo_out, kind, scount, scols, soff, woff) != 0 || geo_out->raw != rec.decoded)
            return -1;
    }
    if (e_out) *e_out = e;
    if (kind_out) *kind_out = kind;
    return 0;
}

static int wex_store_i4_geo(const Wex4View *v, uint32_t r, uint32_t pairs,
                            Wnr4I4Geo *geo_out, wex_dev_i4_geo_t *dev_out) {
    Wex4Record rec;
    uint32_t nmat, m;
    uint32_t rows[3] = {0, 0, 0}, cols[3] = {0, 0, 0};
    uint64_t cursor = 0;
    Wnr4I4Geo geo;
    wex4_get_record(v, r, &rec);
    if (rec.tensor_count != 2u && rec.tensor_count != 4u && rec.tensor_count != 6u) return -1;
    nmat = rec.tensor_count / 2u;

    for (m = 0; m < nmat; m++) {
        Wex4Tensor tc, ts;
        uint64_t r64, c64;
        wex4_get_tensor(v, rec.first_tensor + m, &tc);
        wex4_get_tensor(v, rec.first_tensor + nmat + m, &ts);
        if (tc.record_count != 1u || ts.record_count != 1u || tc.rank != 2u || ts.rank != 2u)
            return -1;
        if (tc.dtype != WEX4_DT_U8 || ts.dtype != LIBWEX_DT_F32) return -1;
        r64 = tc.shape[0];
        c64 = tc.shape[1] * 2u;
        if (!r64 || !c64 || r64 > UINT32_MAX || c64 > UINT32_MAX || (c64 % 64u) != 0) return -1;
        if (ts.shape[0] != r64 || ts.shape[1] != c64 / 64u) return -1;
        rows[m] = (uint32_t)r64;
        cols[m] = (uint32_t)c64;
    }
    /* Verify sequential offsets: all codes first, then all scales */
    for (m = 0; m < rec.tensor_count; m++) {
        Wex4Tensor t;
        wex4_get_tensor(v, rec.first_tensor + m, &t);
        if (t.offset != cursor) return -1;
        cursor += t.nbytes;
    }
    if (cursor != rec.decoded) return -1;
    if (wnr4_i4_geo(&geo, nmat, rows, cols, pairs) != 0 || geo.raw != rec.decoded) return -1;

    if (geo_out) *geo_out = geo;
    if (dev_out) {
        memset(dev_out, 0, sizeof(*dev_out));
        dev_out->nmat  = nmat;
        dev_out->pairs = pairs;
        for (m = 0; m < nmat; m++) {
            dev_out->rows[m] = rows[m];
            dev_out->cols[m] = cols[m];
        }
    }
    return 0;
}

static int wex_store_gg_geo(const Wex4View *v, uint32_t r,
                            Wnr4GgGeo *geo_out, wex_dev_gg_geo_t *dev_out) {
    Wex4Record rec;
    uint32_t s;
    uint32_t types[WNR4_GG_MAX_SEG];
    uint32_t sizes[WNR4_GG_MAX_SEG];
    Wnr4GgGeo geo;
    wex4_get_record(v, r, &rec);
    if (!rec.tensor_count || rec.tensor_count > WNR4_GG_MAX_SEG) return -1;
    if (dev_out) {
        memset(dev_out, 0, sizeof(*dev_out));
        dev_out->nslice = rec.tensor_count;
    }
    for (s = 0; s < rec.tensor_count; s++) {
        Wex4Tensor t;
        wex4_get_tensor(v, rec.first_tensor + s, &t);
        if (t.record_count != 1u || !t.nbytes || t.nbytes > UINT32_MAX) return -1;
        types[s] = (t.dtype >= 128u) ? (t.dtype - 128u) : UINT32_MAX;
        sizes[s] = (uint32_t)t.nbytes;
        if (dev_out) {
            dev_out->ggml_type[s] = types[s];
            dev_out->offsets[s]   = t.offset;
            dev_out->nbytes[s]    = t.nbytes;
            dev_out->rows[s]      = (t.rank >= 1u) ? (uint32_t)t.shape[0] : 1u;
            dev_out->cols[s]      = (t.rank >= 2u) ? (uint32_t)t.shape[1] : 1u;
        }
    }
    if (wnr4_gg_geo(&geo, rec.tensor_count, types, sizes) != 0 || geo.raw != rec.decoded)
        return -1;
    if (geo_out) *geo_out = geo;
    return 0;
}

/* Full WEX4 index validator supporting RAW (0), WNR3 (5), WBP1 (6), AND WNR4 (8: MX/NV/GG/I4). */
static int wex_store_open_index(const Wex4Header *h, const unsigned char *index,
                                size_t index_len, Wex4View *v) {
    unsigned char digest[32];
    uint64_t pos, total = 0, pending_bytes = 0, name_at = 0, i64;
    uint32_t r, t = 0, pending = 0, maximum = 0, b = 0, i;
    int64_t last_owner = -1;
    int index_done;
    Wex4Record rec;
    Wex4Tensor ten;
    Wex4Blob blob;

    if (!h || !v || (!index && h->index_bytes) || index_len != wex4_align(h->index_bytes))
        return WEX4_E_ARG;
    wex4_sha256(index, (size_t)h->index_bytes, digest);
    if (memcmp(digest, h->index_sha, 32u)) return WEX4_E_HASH;
    for (i64 = h->index_bytes; i64 < index_len; i64++) {
        if (index[i64]) return WEX4_E_INDEX;
    }

    v->h       = *h;
    v->blobs   = index;
    v->records = v->blobs   + (size_t)h->blobs   * WEX4_BLOB_BYTES;
    v->tensors = v->records + (size_t)h->records * WEX4_RECORD_BYTES;
    v->order   = v->tensors + (size_t)h->tensors * WEX4_TENSOR_BYTES;
    v->names   = v->order   + (size_t)h->tensors * 4u;
    v->meta    = v->names   + (size_t)h->names_bytes;

    /* Tensors */
    for (i = 0; i < h->tensors; i++) {
        const unsigned char *p = v->tensors + (size_t)i * WEX4_TENSOR_BYTES;
        uint64_t elements = 1;
        unsigned int d, bits;
        if (wex_nr_u32(p) != name_at || wex_nr_u16(p + 4u) > h->names_bytes - name_at)
            return WEX4_E_NAME;
        wex4_get_tensor(v, i, &ten);
        name_at += ten.name_len;
        if (!wex4_utf8_ok(ten.name, ten.name_len)) return WEX4_E_NAME;
        if (ten.rank > 8u) return WEX4_E_TENSOR;
        for (d = 0; d < 8u; d++) {
            if (d >= ten.rank) {
                if (ten.shape[d]) return WEX4_E_TENSOR;
                continue;
            }
            if (ten.shape[d] && elements > UINT64_MAX / ten.shape[d]) return WEX4_E_TENSOR;
            elements *= ten.shape[d];
        }
        bits = ten.dtype < 23u ? wex4_dtype_bits[ten.dtype] : 0u;
        if (bits && (elements > UINT64_MAX / 64u || ten.nbytes > UINT64_MAX / 8u ||
                     elements * bits != ten.nbytes * 8u))
            return WEX4_E_TENSOR;
    }
    if (name_at != h->names_bytes) return WEX4_E_NAME;

    /* Name order strictly increasing */
    for (i = 0; i < h->tensors; i++) {
        uint32_t at = wex_nr_u32(v->order + (size_t)i * 4u);
        Wex4Tensor prev;
        int c;
        if (at >= h->tensors) return WEX4_E_NAME;
        wex4_get_tensor(v, at, &ten);
        if (i) {
            wex4_get_tensor(v, wex_nr_u32(v->order + (size_t)(i - 1u) * 4u), &prev);
            c = memcmp(prev.name, ten.name, prev.name_len < ten.name_len ? prev.name_len : ten.name_len);
            if (c > 0 || (!c && prev.name_len >= ten.name_len)) return WEX4_E_NAME;
        }
    }

    /* Blobs */
    for (i = 0; i < h->blobs; i++) {
        wex4_get_blob(v, i, &blob);
        if (!blob.bytes || blob.bytes > WEX4_MAX_DECODED || wex4_u64(blob.sha + 32u))
            return WEX4_E_BLOB;
        if (blob.kind == WEX4_BLOB_WNM2 && (blob.bytes != WEX_NR2_MODEL_BYTES || !blob.id))
            return WEX4_E_BLOB;
    }

    /* Records and tensor tiling */
    for (r = 0; r < h->records; r++) {
        uint64_t cursor = 0;
        uint32_t k;
        wex4_get_record(v, r, &rec);
        if (!rec.stored || (rec.stored & (WEX4_PAGE - 1u)) || !rec.decoded ||
            rec.decoded > WEX4_MAX_DECODED || rec.stored > wex4_align(rec.decoded) ||
            (rec.flags & ~(WEX4_REC_VALUE_EXACT | WEX4_REC_EXPERT)) ||
            wex_nr_u32(rec.sha + 32u) || rec.blob > h->blobs)
            return WEX4_E_RECORD;
        if (rec.codec == WEX4_CODEC_RAW ? (rec.stored != wex4_align(rec.decoded) || rec.blob)
                                        : (rec.stored == wex4_align(rec.decoded)))
            return WEX4_E_RECORD;
        if (rec.codec == WEX4_CODEC_WBP1 &&
            (rec.blob || rec.flags ||
             rec.stored < wex4_align(WEX4_BP_HEADER + 6u * (((uint64_t)rec.decoded + 32767u) >> 15))))
            return WEX4_E_RECORD;
        if ((rec.flags & WEX4_REC_VALUE_EXACT) &&
            ((rec.codec != WEX4_CODEC_WNR3 && rec.codec != LIBWEX_CODEC_WNR4) ||
             !(h->required & WEX4_REQ_VALUE_EXACT)))
            return WEX4_E_RECORD;
        if (rec.codec == WEX4_CODEC_WNR3) {
            if (!(rec.flags & WEX4_REC_EXPERT) || !rec.blob) return WEX4_E_RECORD;
            wex4_get_blob(v, rec.blob - 1u, &blob);
            if (blob.kind != WEX4_BLOB_WNM2) return WEX4_E_RECORD;
        }
        if (rec.codec == LIBWEX_CODEC_WNR4 && (rec.flags & WEX4_REC_EXPERT)) {
            if (!rec.blob) return WEX4_E_RECORD;
            wex4_get_blob(v, rec.blob - 1u, &blob);
            if (blob.kind != LIBWEX_BLOB_WNM4) return WEX4_E_RECORD;
        }

        total += rec.decoded;
        if (rec.decoded > maximum) maximum = rec.decoded;
        if (pending) {
            if (rec.tensor_count || rec.first_tensor != t - 1u || (rec.flags & WEX4_REC_EXPERT) ||
                rec.owner != r || rec.decoded > pending_bytes)
                return WEX4_E_RECORD;
            pending_bytes -= rec.decoded;
            if (!--pending && pending_bytes) return WEX4_E_RECORD;
            continue;
        }
        if (!rec.tensor_count || rec.first_tensor != t || rec.tensor_count > h->tensors - t)
            return WEX4_E_RECORD;
        for (k = 0; k < rec.tensor_count; k++) {
            wex4_get_tensor(v, t + k, &ten);
            if (ten.record != r || !ten.nbytes) return WEX4_E_TENSOR;
            if (ten.record_count == 1u) {
                if (ten.offset != cursor || ten.nbytes > rec.decoded - cursor) return WEX4_E_TENSOR;
                cursor += ten.nbytes;
            } else {
                if (ten.record_count < 2u || rec.tensor_count != 1u || ten.offset ||
                    (rec.flags & WEX4_REC_EXPERT) || ten.record_count > h->records - r ||
                    ten.nbytes <= rec.decoded)
                    return WEX4_E_TENSOR;
                pending = ten.record_count - 1u;
                pending_bytes = ten.nbytes - rec.decoded;
                cursor = rec.decoded;
            }
        }
        if (cursor != rec.decoded) return WEX4_E_TENSOR;
        t += rec.tensor_count;

        if (rec.flags & WEX4_REC_EXPERT) {
            WexEntry e;
            uint32_t kind = 0;
            if (rec.codec != WEX4_CODEC_RAW && rec.codec != WEX4_CODEC_WNR3 &&
                rec.codec != LIBWEX_CODEC_WNR4)
                return WEX4_E_EXPERT;
            if ((rec.owner >> 16) >= h->layers || (rec.owner & 0xffffu) >= h->experts ||
                (int64_t)rec.owner <= last_owner ||
                wex_store_expert_entry(v, r, &e, NULL, &kind) != 0)
                return WEX4_E_EXPERT;
            if (rec.codec == WEX4_CODEC_WNR3 && kind != WNR4_KIND_MX) return WEX4_E_EXPERT;
            last_owner = rec.owner;
            if (rec.codec == WEX4_CODEC_WNR3) {
                uint64_t streams = 0;
                unsigned int m;
                for (m = 0; m < 3u; m++) {
                    streams += wex_nr2_streams((uint32_t)e.scale_shapes[m][0] * e.scale_shapes[m][1],
                                               WEX_NR2_MAX_CHUNK) +
                               wex_nr2_streams((uint32_t)e.weight_shapes[m][0] * e.weight_shapes[m][1] * 2u,
                                               WEX_NR2_MAX_CHUNK);
                }
                if (rec.stored < wex4_align(WEX_NR2_HEADER_BYTES + streams * 6u))
                    return WEX4_E_EXPERT;
            }
        } else if (rec.owner != r) {
            return WEX4_E_RECORD;
        }
    }

    if (pending || total != h->total_decoded || maximum != h->max_decoded) return WEX4_E_RECORD;
    if (h->layers && last_owner < 0) return WEX4_E_EXPERT;
    for (; t < h->tensors; t++) {
        wex4_get_tensor(v, t, &ten);
        if (ten.nbytes || ten.record_count || ten.record || ten.offset) return WEX4_E_TENSOR;
    }

    /* File extent tiling */
    pos = WEX4_PAGE;
    r = 0;
    index_done = !h->index_bytes;
    for (;;) {
        uint64_t size;
        if (b < h->blobs && (wex4_get_blob(v, b, &blob), blob.offset == pos)) {
            size = wex4_align(blob.bytes);
            b++;
        } else if (r < h->records && (wex4_get_record(v, r, &rec), rec.offset == pos)) {
            size = rec.stored;
            r++;
        } else if (!index_done && h->index_offset == pos) {
            size = index_len;
            index_done = 1;
        } else {
            break;
        }
        if (size > h->file_bytes - pos) return WEX4_E_LAYOUT;
        pos += size;
    }
    if (b != h->blobs || r != h->records || !index_done || pos != h->file_bytes)
        return WEX4_E_LAYOUT;
    return WEX4_OK;
}

/* Parse optional (layer, eid) from non-EXPERT tensor names:
 *   GGUF:  blk.<L>.ffn_*#<E>
 *   HF/I4: [prefix.]layers.<L>.[mid.]experts.<E>[.suffix] */
static int wex_parse_uint_range(const uint8_t *p, const uint8_t *end,
                                uint32_t *val_out, const uint8_t **after) {
    uint64_t v = 0;
    const uint8_t *q = p;
    if (q >= end || *q < '0' || *q > '9') return 0;
    if (*q == '0' && q + 1 < end && q[1] >= '0' && q[1] <= '9') return 0;
    while (q < end && *q >= '0' && *q <= '9') {
        v = v * 10u + (uint64_t)(*q - '0');
        if (v >= 65536u) return 0;
        q++;
    }
    *val_out = (uint32_t)v;
    if (after) *after = q;
    return 1;
}

static int wex_parse_expert_name(const uint8_t *name, uint32_t len,
                                 uint32_t *layer_out, uint32_t *eid_out) {
    const uint8_t *end = name + len;
    /* Pattern 1: blk.<L>.ffn_*#<E> */
    if (len > 10u && memcmp(name, "blk.", 4) == 0) {
        const uint8_t *after = NULL;
        const uint8_t *hash = NULL;
        const uint8_t *p;
        uint32_t L = 0, E = 0;
        if (wex_parse_uint_range(name + 4, end, &L, &after) &&
            (size_t)(end - after) >= 6u && memcmp(after, ".ffn_", 5) == 0) {
            for (p = end; p > after; p--) {
                if (p[-1] == '#') { hash = p - 1; break; }
            }
            if (hash && wex_parse_uint_range(hash + 1, end, &E, &after) && after == end) {
                *layer_out = L;
                *eid_out   = E;
                return 1;
            }
        }
    }
    /* Pattern 2: [* .]layers.<L>.*.experts.<E>[.*] */
    {
        const uint8_t *p = name;
        while ((size_t)(end - p) >= 8u) {
            if ((p == name || p[-1] == '.') && memcmp(p, "layers.", 7) == 0) {
                const uint8_t *after = NULL;
                uint32_t L = 0, E = 0;
                if (wex_parse_uint_range(p + 7, end, &L, &after) &&
                    after < end && *after == '.') {
                    const uint8_t *q = after;
                    while ((size_t)(end - q) >= 10u) {
                        if (memcmp(q, ".experts.", 9) == 0) {
                            const uint8_t *after_e = NULL;
                            if (wex_parse_uint_range(q + 9, end, &E, &after_e) &&
                                (after_e == end || *after_e == '.')) {
                                *layer_out = L;
                                *eid_out   = E;
                                return 1;
                            }
                        }
                        q++;
                    }
                }
            }
            p++;
        }
    }
    return 0;
}

/* ---- Internal store structures ------------------------------------------- */

typedef enum {
    WEX_MODEL_NONE    = 0,
    WEX_MODEL_WNM2    = 1,
    WEX_MODEL_WNM4_MX = 2,
    WEX_MODEL_WNM4_NV = 3,
    WEX_MODEL_WNM4_GG = 4,
    WEX_MODEL_WNM4_I4 = 5
} wex_model_type_t;

typedef struct {
    wex_model_type_t type;
    uint32_t         id;
    uint8_t         *raw_blob;
    size_t           raw_bytes;
    WexNr2Model      wnm2;
#if LIBWEX_HAS_FAST_DECODE
    WexFastModel    *fast_wnm2;
#endif
    Wnr4Model        wnm4_mxnv;
    Wnr4GgModel      wnm4_gg;
    Wnr4I4Model      wnm4_i4;
    int              dev_registered;
} wex_model_slot_t;

struct wex_store {
    wex_io_t           *io;
    uint8_t            *index_buf;
    size_t              index_len;
    Wex4View            view;

    uint32_t            layers;
    uint32_t            experts;
    uint32_t           *expert_to_rec;      /* [layers * experts], UINT32_MAX = absent */
    uint8_t            *rec_codec_kind;     /* [records] wex_codec_kind_t */
    uint32_t            primary_codec_kind;

    uint32_t            verified_root;
    uint32_t            verify_on;
    int                 cpu_threads;
    char                root_hex[65];
    char                backend_name[32];

    uint64_t            max_record_stored_bytes;
    uint64_t            min_record_stored_bytes;
    uint64_t            max_expert_stored_bytes;
    uint64_t            total_stored_bytes;

    /* Loaded & validated model blobs (indexed by blob_idx = rec.blob - 1) */
    wex_model_slot_t   *models;
    uint32_t            num_blobs;

    /* First-read SHA-256 verification state lattice: 0=unverified, 1=verified, 2=refused */
    uint8_t            *verify_state;
    /* Tracks whether a record in L2 came from wex_prefetch */
    uint8_t            *prefetched_flag;

    /* Optional dynamically loaded GPU backend */
    void               *dev_dl_handle;
    const wex_dev_ops_t *dev_ops;
    int                 gpu_device;

    /* Pre-pinned scratch pool for disk miss reads */
    wex_s_lock_t        scratch_lock;
    uint8_t            *scratch_arena;
    size_t              scratch_slot_bytes;
    uint32_t            scratch_num_slots;
    uint64_t            scratch_free_mask;
    int                 scratch_pinned;

    /* L2 Compressed-RAM sharded LRU cache */
    wex_cache_t        *l2;

    /* Background prefetch worker (lazy-started on first wex_prefetch) */
    wex_s_lock_t        pf_lock;
    wex_s_cond_t        pf_cond;
    wex_s_thread_t      pf_thread;
    int                 pf_started;
    int                 pf_stop;
    uint32_t            pf_queue[LIBWEX_PF_Q_CAP];
    uint32_t            pf_head;
    uint32_t            pf_tail;
    uint32_t            pf_count;

    /* Aggregate telemetry */
    wex_s_lock_t        stats_lock;
    wex_stats_t         stats;
};

/* ---- Scratch pool helpers ------------------------------------------------ */

static uint8_t *wex_scratch_acquire(wex_store_t *s, size_t need_bytes, int *slot_out) {
    *slot_out = -1;
    if (s->scratch_arena && need_bytes <= s->scratch_slot_bytes) {
        wex_s_lock(&s->scratch_lock);
        if (s->scratch_free_mask) {
            uint64_t m = s->scratch_free_mask;
            uint32_t idx = 0;
            while (((m >> idx) & 1ULL) == 0ULL) idx++;
            s->scratch_free_mask &= ~(1ULL << idx);
            wex_s_unlock(&s->scratch_lock);
            *slot_out = (int)idx;
            return s->scratch_arena + (size_t)idx * s->scratch_slot_bytes;
        }
        wex_s_unlock(&s->scratch_lock);
    }
    return (uint8_t *)wex_io_alloc_aligned(need_bytes);
}

static void wex_scratch_release(wex_store_t *s, uint8_t *buf, int slot) {
    if (!buf) return;
    if (slot >= 0) {
        wex_s_lock(&s->scratch_lock);
        s->scratch_free_mask |= (1ULL << (uint32_t)slot);
        wex_s_unlock(&s->scratch_lock);
    } else {
        wex_io_free_aligned(buf);
    }
}

/* ---- Streaming SHA-256 I/O callback bridge ------------------------------- */

typedef struct {
    wex_sha_batch_t *sha;
    uint8_t         *fed_done;
} wex_io_sha_ctx_t;

static void wex_io_chunk_bridge(void *ctx, int req_idx, const uint8_t *buf,
                                uint64_t chunk_off, uint64_t chunk_len, int done) {
    wex_io_sha_ctx_t *sc = (wex_io_sha_ctx_t *)ctx;
    if (!sc || !sc->sha) return;
    if (sc->fed_done) sc->fed_done[req_idx] = 1u;
    wex_sha_batch_feed(sc->sha, req_idx, buf, chunk_off, chunk_len, done);
}

/* ---- CPU decoders -------------------------------------------------------- */

static int wex_cpu_decode_one(const wex_store_t *s, uint32_t rec_idx,
                              const Wex4Record *rec, uint32_t codec_kind,
                              const uint8_t *stored, uint8_t *dst,
                              int allow_scratch, int threads) {
    uint8_t *tmp = NULL;
    uint8_t *out = dst;
    int rc = WEX_E_UNSUPPORTED;

    if (rec->codec == WEX4_CODEC_RAW) {
        uint32_t i;
        for (i = rec->decoded; i < rec->stored; i++) {
            if (stored[i] != 0u) return WEX_E_DECODE;
        }
        memcpy(dst, stored, rec->decoded);
        return WEX_OK;
    }

    if (!allow_scratch) {
        tmp = (uint8_t *)malloc(rec->decoded);
        if (!tmp) return WEX_E_OOM;
        out = tmp;
    }

    if (rec->codec == WEX4_CODEC_WBP1) {
        rc = (wex4_bp_decode(stored, rec->stored, rec->decoded, rec->owner, out) == 0)
             ? WEX_OK : WEX_E_DECODE;
    } else if (rec->codec == WEX4_CODEC_WNR3) {
        const wex_model_slot_t *ms = (rec->blob && rec->blob <= s->num_blobs)
                                     ? &s->models[rec->blob - 1u] : NULL;
        WexEntry e;
        WexNr2Layout layout;
        WexNr2Record view;
        unsigned char *scales[3], *weights[3];
        uint32_t m;
        if (!ms || ms->type != WEX_MODEL_WNM2 || rec->stored < WEX_NR2_HEADER_BYTES ||
            wex_store_expert_entry(&s->view, rec_idx, &e, NULL, NULL) != 0) {
            rc = WEX_E_DECODE;
        } else if (wex_nr_u32(stored + 4u) != ms->id ||
                   wex_nr_u32(stored + 28u) != rec->owner ||
                   !(stored[27] & WEX_NR3_FLAG_CZ) != !(rec->flags & WEX4_REC_VALUE_EXACT)) {
            rc = WEX_E_DECODE;
        } else {
            wex_nr2_layout(&e, &layout);
            if (wex_nrx_parse(stored, rec->stored, &layout, ms->id, WEX_NR3_MAGIC, &view) != 0) {
                rc = WEX_E_DECODE;
            } else {
                for (m = 0; m < 3u; m++) {
                    scales[m]  = out + e.scale_offsets[m];
                    weights[m] = out + e.weight_offsets[m];
                }
#if LIBWEX_HAS_FAST_DECODE
                if (ms->fast_wnm2) {
                    WexFastOptions fo;
                    fo.threads = threads;
                    fo.kernel  = 1;
                    fo.lanes   = WEX_FAST_LANES;
                    fo.fuse    = 1;
                    rc = (wex_fast_decode_with(ms->fast_wnm2, &view, &layout, &ms->wnm2,
                                               scales, weights, &fo) == 0)
                         ? WEX_OK : WEX_E_DECODE;
                } else
#endif
                {
                    rc = (wex_nr2_decode(&view, &layout, &ms->wnm2, scales, weights) == 0)
                         ? WEX_OK : WEX_E_DECODE;
                }
            }
        }
    } else if (rec->codec == LIBWEX_CODEC_WNR4) {
        const wex_model_slot_t *ms = (rec->blob && rec->blob <= s->num_blobs)
                                     ? &s->models[rec->blob - 1u] : NULL;
        if (!ms || ms->type == WEX_MODEL_NONE || rec->stored < WNR4_HEADER_BYTES ||
            wnr4_u32(stored) != WNR4_MAGIC) {
            rc = WEX_E_DECODE;
        } else if (codec_kind == WEX_CODEC_WNR4_MX || codec_kind == WEX_CODEC_WNR4_NV) {
            Wnr4Geo geo;
            uint32_t kind = 0;
            if ((ms->type != WEX_MODEL_WNM4_MX && ms->type != WEX_MODEL_WNM4_NV) ||
                wex_store_expert_entry(&s->view, rec_idx, NULL, &geo, &kind) != 0 ||
                stored[12] != kind ||
                ((rec->flags & WEX4_REC_VALUE_EXACT) && !(stored[13] & WNR4_FLAG_CZ))) {
                rc = WEX_E_DECODE;
            } else {
                rc = (wnr4_decode(&ms->wnm4_mxnv, &geo, stored, rec->stored,
                                  rec->owner, out, threads) == 0u)
                     ? WEX_OK : WEX_E_DECODE;
            }
        } else if (codec_kind == WEX_CODEC_WNR4_GG) {
            Wnr4GgGeo geo;
            if (ms->type != WEX_MODEL_WNM4_GG || stored[12] != WNR4_KIND_GG ||
                wex_store_gg_geo(&s->view, rec_idx, &geo, NULL) != 0) {
                rc = WEX_E_DECODE;
            } else {
                rc = (wnr4_gg_decode(&ms->wnm4_gg, &geo, stored, rec->stored,
                                     rec->owner, out, threads) == 0u)
                     ? WEX_OK : WEX_E_DECODE;
            }
        } else if (codec_kind == WEX_CODEC_WNR4_I4) {
            Wnr4I4Geo geo;
            if (ms->type != WEX_MODEL_WNM4_I4 || stored[12] != WNR4_KIND_I4 ||
                stored[13] > 1u ||
                wex_store_i4_geo(&s->view, rec_idx, stored[13], &geo, NULL) != 0) {
                rc = WEX_E_DECODE;
            } else {
                rc = (wnr4_i4_decode(&ms->wnm4_i4, &geo, stored, rec->stored,
                                     rec->owner, out, threads) == 0u)
                     ? WEX_OK : WEX_E_DECODE;
            }
        } else {
            rc = WEX_E_UNSUPPORTED;
        }
    }

    if (tmp) {
        if (rc == WEX_OK) memcpy(dst, tmp, rec->decoded);
        free(tmp);
    }
    return rc;
}

/* ---- Background prefetch worker ------------------------------------------ */

static void wex_prefetch_run_batch(wex_store_t *s, const uint32_t *recs, int n,
                                   uint8_t **bufs, size_t *buf_caps) {
    wex_io_req_t reqs[LIBWEX_PF_BATCH];
    const uint8_t *expected_sha[LIBWEX_PF_BATCH];
    uint8_t fed_done[LIBWEX_PF_BATCH];
    int sha_ok[LIBWEX_PF_BATCH];
    Wex4Record rinfo[LIBWEX_PF_BATCH];
    int active_idx[LIBWEX_PF_BATCH];
    int m = 0, i, any_sha = 0;
    double t0, io_s = 0.0, ver_s = 0.0;
    uint64_t read_bytes = 0, refused = 0;

    for (i = 0; i < n && i < LIBWEX_PF_BATCH; i++) {
        uint32_t r = recs[i];
        uint8_t vst = wex_u8_load(&s->verify_state[r]);
        if (vst == LIBWEX_STATE_REFUSED) continue;
        if (wex_cache_contains(s->l2, r)) continue;
        wex4_get_record(&s->view, r, &rinfo[m]);
        if (buf_caps[m] < rinfo[m].stored) {
            if (bufs[m]) wex_io_free_aligned(bufs[m]);
            bufs[m] = (uint8_t *)wex_io_alloc_aligned(rinfo[m].stored);
            buf_caps[m] = bufs[m] ? rinfo[m].stored : 0u;
            if (!bufs[m]) continue;
        }
        reqs[m].file_offset   = rinfo[m].offset;
        reqs[m].length        = rinfo[m].stored;
        reqs[m].buf           = bufs[m];
        reqs[m].hint_replica  = (int)((r * UINT32_C(2654435761)) >> 16);
        reqs[m].stream_chunks = (s->verify_on && vst == LIBWEX_STATE_UNVERIFIED) ? 1 : 0;
        reqs[m].status        = 0;
        reqs[m].used_replica  = 0;
        expected_sha[m]       = reqs[m].stream_chunks ? rinfo[m].sha : NULL;
        fed_done[m]           = 0;
        sha_ok[m]             = 1;
        if (reqs[m].stream_chunks) any_sha = 1;
        active_idx[m]         = (int)r;
        m++;
    }
    if (!m) return;

    {
        wex_sha_batch_t *sha = any_sha ? wex_sha_batch_begin(m, expected_sha, 0) : NULL;
        wex_io_sha_ctx_t sctx = { sha, fed_done };
        t0 = wex_io_now_s();
        (void)wex_io_read_batch(s->io, reqs, m, sha ? wex_io_chunk_bridge : NULL, NULL,
                                sha ? &sctx : NULL);
        io_s = wex_io_now_s() - t0;
        if (sha) {
            for (i = 0; i < m; i++) {
                if (reqs[i].status == 0 && expected_sha[i] && !fed_done[i]) {
                    wex_sha_batch_feed(sha, i, reqs[i].buf, 0, reqs[i].length, 1);
                }
            }
            ver_s = wex_sha_batch_finish(sha, sha_ok);
        }
    }

    for (i = 0; i < m; i++) {
        uint32_t r = (uint32_t)active_idx[i];
        if (reqs[i].status != 0) continue;
        read_bytes += reqs[i].length;
        if (expected_sha[i]) {
            if (!sha_ok[i]) {
                wex_u8_store(&s->verify_state[r], LIBWEX_STATE_REFUSED);
                refused++;
                continue;
            }
            if (!wex_u8_cas(&s->verify_state[r], LIBWEX_STATE_UNVERIFIED, LIBWEX_STATE_VERIFIED)) {
                if (wex_u8_load(&s->verify_state[r]) == LIBWEX_STATE_REFUSED) {
                    refused++;
                    continue;
                }
            }
        }
        if (wex_cache_put(s->l2, r, reqs[i].buf, (size_t)reqs[i].length)) {
            wex_u8_store(&s->prefetched_flag[r], 1u);
        }
    }

    wex_s_lock(&s->stats_lock);
    s->stats.disk_bytes     += read_bytes;
    s->stats.refused        += refused;
    s->stats.io_seconds     += io_s;
    s->stats.verify_seconds += ver_s;
    wex_s_unlock(&s->stats_lock);
}

#ifdef _WIN32
static unsigned __stdcall wex_prefetch_worker(void *arg)
#else
static void *wex_prefetch_worker(void *arg)
#endif
{
    wex_store_t *s = (wex_store_t *)arg;
    uint8_t *bufs[LIBWEX_PF_BATCH];
    size_t buf_caps[LIBWEX_PF_BATCH];
    int i;
    memset(bufs, 0, sizeof(bufs));
    memset(buf_caps, 0, sizeof(buf_caps));

    for (;;) {
        uint32_t batch[LIBWEX_PF_BATCH];
        int cnt = 0;
        wex_s_lock(&s->pf_lock);
        while (!s->pf_stop && s->pf_count == 0u) {
            wex_s_cond_wait(&s->pf_cond, &s->pf_lock);
        }
        if (s->pf_stop) {
            wex_s_unlock(&s->pf_lock);
            break;
        }
        while (cnt < LIBWEX_PF_BATCH && s->pf_count > 0u) {
            batch[cnt++] = s->pf_queue[s->pf_head];
            s->pf_head = (s->pf_head + 1u) % LIBWEX_PF_Q_CAP;
            s->pf_count--;
        }
        wex_s_unlock(&s->pf_lock);
        if (cnt > 0) {
            wex_prefetch_run_batch(s, batch, cnt, bufs, buf_caps);
        }
    }

    for (i = 0; i < LIBWEX_PF_BATCH; i++) {
        if (bufs[i]) wex_io_free_aligned(bufs[i]);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ---- Core fetch implementation (shared by host & device) ----------------- */

typedef struct {
    uint32_t           rec_idx;
    Wex4Record         rec;
    uint32_t           codec_kind;
    int                status;        /* WEX_OK or < 0 */
    int                from_l2;
    wex_cache_handle_t l2_handle;
    uint8_t           *scratch_buf;
    int                scratch_slot;
    const uint8_t     *rec_buf;       /* points to l2 buffer or scratch_buf */
    int                decoded;       /* 1 if decoded (GPU or CPU) */
    int                need_cpu;      /* 1 if needs CPU decode / fallback */
} wex_fetch_item_t;

static int wex_resolve_record(const wex_store_t *s, int layer, int eid,
                              uint32_t flags, uint32_t *rec_out) {
    if (flags & WEX_FETCH_BY_RECORD_ID) {
        if (eid < 0 || (uint32_t)eid >= s->view.h.records) return WEX_E_NOT_FOUND;
        *rec_out = (uint32_t)eid;
        return WEX_OK;
    }
    if (!s->expert_to_rec || layer < 0 || eid < 0 ||
        (uint32_t)layer >= s->layers || (uint32_t)eid >= s->experts)
        return WEX_E_NOT_FOUND;
    {
        uint32_t r = s->expert_to_rec[(size_t)layer * s->experts + (uint32_t)eid];
        if (r == UINT32_MAX) return WEX_E_NOT_FOUND;
        *rec_out = r;
        return WEX_OK;
    }
}

static int wex_fetch_impl(wex_store_t *s, int layer, const int *eids, int n,
                          void *const *dst_slabs, const size_t *dst_caps,
                          void *const *dst_scales, void *const *dst_weights,
                          void *device_stream, uint32_t flags, int is_device,
                          int *status_out) {
    wex_fetch_item_t  items_stk[LIBWEX_STACK_BATCH];
    int               miss_idx_stk[LIBWEX_STACK_BATCH];
    wex_io_req_t      io_reqs_stk[LIBWEX_STACK_BATCH];
    const uint8_t    *sha_exp_stk[LIBWEX_STACK_BATCH];
    uint8_t           fed_stk[LIBWEX_STACK_BATCH];
    int               sha_ok_stk[LIBWEX_STACK_BATCH];

    wex_fetch_item_t *items    = items_stk;
    int              *miss_idx = miss_idx_stk;
    wex_io_req_t     *io_reqs  = io_reqs_stk;
    const uint8_t   **sha_exp  = sha_exp_stk;
    uint8_t          *fed_done = fed_stk;
    int              *sha_ok   = sha_ok_stk;
    int               heap_alloc = 0;

    int i, n_miss = 0, any_sha = 0, failed = 0;
    uint64_t st_l2 = 0, st_disk = 0, st_gpu = 0, st_cpu = 0;
    uint64_t st_pf_hits = 0, st_refused = 0, st_disk_bytes = 0, st_dec_bytes = 0;
    double   io_s = 0.0, ver_s = 0.0, dec_s = 0.0;

    if (!s || n < 0) return WEX_E_ARGS;
    if (n == 0) return 0;
    if (!eids) {
        if (status_out) for (i = 0; i < n; i++) status_out[i] = WEX_E_ARGS;
        return n;
    }
    if (!is_device && (!dst_slabs || !dst_caps)) {
        if (status_out) for (i = 0; i < n; i++) status_out[i] = WEX_E_ARGS;
        return n;
    }
    if (is_device && !s->dev_ops) {
        if (status_out) for (i = 0; i < n; i++) status_out[i] = WEX_E_DEVICE;
        return n;
    }

    if (n > LIBWEX_STACK_BATCH) {
        heap_alloc = 1;
        items    = (wex_fetch_item_t *)calloc((size_t)n, sizeof(*items));
        miss_idx = (int *)malloc((size_t)n * sizeof(*miss_idx));
        io_reqs  = (wex_io_req_t *)malloc((size_t)n * sizeof(*io_reqs));
        sha_exp  = (const uint8_t **)malloc((size_t)n * sizeof(*sha_exp));
        fed_done = (uint8_t *)malloc((size_t)n * sizeof(*fed_done));
        sha_ok   = (int *)malloc((size_t)n * sizeof(*sha_ok));
        if (!items || !miss_idx || !io_reqs || !sha_exp || !fed_done || !sha_ok) {
            free(items); free(miss_idx); free(io_reqs);
            free(sha_exp); free(fed_done); free(sha_ok);
            if (status_out) for (i = 0; i < n; i++) status_out[i] = WEX_E_OOM;
            return n;
        }
    } else {
        memset(items, 0, (size_t)n * sizeof(*items));
    }

    /* 1. Resolve records, check verification lattice & caller buffers, probe L2 cache */
    for (i = 0; i < n; i++) {
        uint32_t rec_idx = 0;
        int rc = wex_resolve_record(s, layer, eids[i], flags, &rec_idx);
        items[i].scratch_slot = -1;
        if (rc != WEX_OK) {
            items[i].status = rc;
            continue;
        }
        items[i].rec_idx    = rec_idx;
        items[i].codec_kind = s->rec_codec_kind[rec_idx];
        wex4_get_record(&s->view, rec_idx, &items[i].rec);

        if (wex_u8_load(&s->verify_state[rec_idx]) == LIBWEX_STATE_REFUSED) {
            items[i].status = WEX_E_REFUSED;
            st_refused++;
            continue;
        }

        if (!is_device) {
            if (!dst_slabs[i]) {
                items[i].status = WEX_E_ARGS;
                continue;
            }
            if (dst_caps[i] < items[i].rec.decoded) {
                items[i].status = WEX_E_CAP;
                continue;
            }
        } else {
            if (items[i].rec.flags & WEX4_REC_EXPERT) {
                if (!dst_scales || !dst_weights ||
                    !dst_scales[3 * i + 0] || !dst_scales[3 * i + 1] || !dst_scales[3 * i + 2] ||
                    !dst_weights[3 * i + 0] || !dst_weights[3 * i + 1] || !dst_weights[3 * i + 2]) {
                    items[i].status = WEX_E_ARGS;
                    continue;
                }
            } else {
                if (!dst_slabs || !dst_slabs[i]) {
                    items[i].status = WEX_E_ARGS;
                    continue;
                }
            }
        }

        /* L2 Compressed-RAM lookup */
        if (s->l2) {
            const uint8_t *cbuf = NULL;
            size_t clen = 0;
            wex_cache_handle_t ch = {0, 0, NULL};
            if (wex_cache_get(s->l2, rec_idx, &cbuf, &clen, &ch) && clen == items[i].rec.stored) {
                items[i].from_l2   = 1;
                items[i].l2_handle = ch;
                items[i].rec_buf   = cbuf;
                items[i].status    = WEX_OK;
                st_l2++;
                if (wex_u8_xchg(&s->prefetched_flag[rec_idx], 0u) == 1u) {
                    st_pf_hits++;
                }
                continue;
            } else if (ch.valid) {
                wex_cache_release(s->l2, ch);
            }
        }

        /* L2 miss: acquire aligned scratch slot for async I/O */
        items[i].scratch_buf = wex_scratch_acquire(s, items[i].rec.stored, &items[i].scratch_slot);
        if (!items[i].scratch_buf) {
            items[i].status = WEX_E_OOM;
            continue;
        }
        items[i].rec_buf = items[i].scratch_buf;
        items[i].status  = WEX_OK;

        {
            uint8_t vst = wex_u8_load(&s->verify_state[rec_idx]);
            int need_sha = (s->verify_on && vst == LIBWEX_STATE_UNVERIFIED) ? 1 : 0;
            io_reqs[n_miss].file_offset   = items[i].rec.offset;
            io_reqs[n_miss].length        = items[i].rec.stored;
            io_reqs[n_miss].buf           = items[i].scratch_buf;
            io_reqs[n_miss].hint_replica  = (int)((rec_idx * UINT32_C(2654435761)) >> 16);
            io_reqs[n_miss].stream_chunks = need_sha;
            io_reqs[n_miss].status        = 0;
            io_reqs[n_miss].used_replica  = 0;
            sha_exp[n_miss]               = need_sha ? items[i].rec.sha : NULL;
            fed_done[n_miss]              = 0;
            sha_ok[n_miss]                = 1;
            if (need_sha) any_sha = 1;
            miss_idx[n_miss] = i;
            n_miss++;
        }
    }

    /* 2. Batched async disk I/O + overlapped streaming 2 MiB SHA-256 verification */
    if (n_miss > 0) {
        wex_sha_batch_t *sha = any_sha ? wex_sha_batch_begin(n_miss, sha_exp, 0) : NULL;
        wex_io_sha_ctx_t sctx = { sha, fed_done };
        double t0 = wex_io_now_s();
        (void)wex_io_read_batch(s->io, io_reqs, n_miss,
                                sha ? wex_io_chunk_bridge : NULL, NULL,
                                sha ? &sctx : NULL);
        io_s = wex_io_now_s() - t0;

        if (sha) {
            for (i = 0; i < n_miss; i++) {
                if (io_reqs[i].status == 0 && sha_exp[i] && !fed_done[i]) {
                    wex_sha_batch_feed(sha, i, io_reqs[i].buf, 0, io_reqs[i].length, 1);
                }
            }
            ver_s = wex_sha_batch_finish(sha, sha_ok);
        }

        for (i = 0; i < n_miss; i++) {
            int idx = miss_idx[i];
            uint32_t r = items[idx].rec_idx;
            if (io_reqs[i].status != 0) {
                items[idx].status = WEX_E_IO;
                continue;
            }
            st_disk++;
            st_disk_bytes += io_reqs[i].length;
            if (sha_exp[i]) {
                if (!sha_ok[i]) {
                    wex_u8_store(&s->verify_state[r], LIBWEX_STATE_REFUSED);
                    items[idx].status = WEX_E_HASH;
                    st_refused++;
                    continue;
                }
                if (!wex_u8_cas(&s->verify_state[r], LIBWEX_STATE_UNVERIFIED, LIBWEX_STATE_VERIFIED)) {
                    if (wex_u8_load(&s->verify_state[r]) == LIBWEX_STATE_REFUSED) {
                        items[idx].status = WEX_E_REFUSED;
                        st_refused++;
                        continue;
                    }
                }
            }
        }
    }

    /* 3. Decode: GPU backend (batched by codec family) -> CPU fallback */
    {
        double t_dec0 = wex_io_now_s();
        const wex_dev_ops_t *ops = s->dev_ops;
        int scratch_dst = (flags & WEX_FETCH_SCRATCH) ? 1 : 0;
        int expand_e8m0 = (flags & WEX_FETCH_EXPAND_E8M0_F32) ? 1 : 0;

        /* 3a. GPU Expert batch (WNR3, WNR4-MX, WNR4-NV; plus RAW EXPERT on device fetch) */
        if (ops && ((is_device && ops->decode_expert_device) ||
                    (!is_device && ops->decode_expert_host))) {
            int exp_map_stk[LIBWEX_STACK_BATCH];
            WexEntry exp_ent_stk[LIBWEX_STACK_BATCH];
            const WexEntry *exp_ent_ptr_stk[LIBWEX_STACK_BATCH];
            const uint8_t *exp_rec_stk[LIBWEX_STACK_BATCH];
            uint8_t *exp_dst_stk[LIBWEX_STACK_BATCH];
            uint8_t *exp_sc_stk[LIBWEX_STACK_BATCH * 3];
            uint8_t *exp_wt_stk[LIBWEX_STACK_BATCH * 3];
            int exp_st_stk[LIBWEX_STACK_BATCH];

            int *exp_map = exp_map_stk;
            WexEntry *exp_ent = exp_ent_stk;
            const WexEntry **exp_ent_ptr = exp_ent_ptr_stk;
            const uint8_t **exp_rec = exp_rec_stk;
            uint8_t **exp_dst = exp_dst_stk;
            uint8_t **exp_sc = exp_sc_stk;
            uint8_t **exp_wt = exp_wt_stk;
            int *exp_st = exp_st_stk;
            int exp_heap = 0, cnt = 0;

            if (n > LIBWEX_STACK_BATCH) {
                exp_heap = 1;
                exp_map     = (int *)malloc((size_t)n * sizeof(*exp_map));
                exp_ent     = (WexEntry *)malloc((size_t)n * sizeof(*exp_ent));
                exp_ent_ptr = (const WexEntry **)malloc((size_t)n * sizeof(*exp_ent_ptr));
                exp_rec     = (const uint8_t **)malloc((size_t)n * sizeof(*exp_rec));
                exp_dst     = (uint8_t **)malloc((size_t)n * sizeof(*exp_dst));
                exp_sc      = (uint8_t **)malloc((size_t)n * 3u * sizeof(*exp_sc));
                exp_wt      = (uint8_t **)malloc((size_t)n * 3u * sizeof(*exp_wt));
                exp_st      = (int *)malloc((size_t)n * sizeof(*exp_st));
            }

            if (exp_map && exp_ent && exp_ent_ptr && exp_rec && exp_dst && exp_sc && exp_wt && exp_st) {
                for (i = 0; i < n; i++) {
                    uint32_t ck;
                    const wex_model_slot_t *ms = NULL;
                    if (items[i].status != WEX_OK || items[i].decoded) continue;
                    ck = items[i].codec_kind;
                    if (ck == WEX_CODEC_WNR3) {
                        if (!(ops->caps & WEX_DEV_CAP_WNR3)) continue;
                    } else if (ck == WEX_CODEC_WNR4_MX) {
                        if (!(ops->caps & WEX_DEV_CAP_WNR4_MX)) continue;
                    } else if (ck == WEX_CODEC_WNR4_NV) {
                        if (!(ops->caps & WEX_DEV_CAP_WNR4_NV)) continue;
                    } else {
                        continue;
                    }

                    if (!items[i].rec.blob || items[i].rec.blob > s->num_blobs) {
                        items[i].status = WEX_E_DECODE;
                        continue;
                    }
                    ms = &s->models[items[i].rec.blob - 1u];
                    if (!ms->dev_registered) continue;
                    /* Verify WEX4 index-to-record binding before GPU dispatch */
                    if (ck == WEX_CODEC_WNR3) {
                        if (items[i].rec.stored < WEX_NR2_HEADER_BYTES ||
                            wex_nr_u32(items[i].rec_buf + 4u) != ms->id ||
                            wex_nr_u32(items[i].rec_buf + 28u) != items[i].rec.owner ||
                            !(items[i].rec_buf[27] & WEX_NR3_FLAG_CZ) !=
                            !(items[i].rec.flags & WEX4_REC_VALUE_EXACT)) {
                            items[i].status = WEX_E_DECODE;
                            continue;
                        }
                    } else {
                        uint32_t want_k = (ck == WEX_CODEC_WNR4_NV) ? WNR4_KIND_NV : WNR4_KIND_MX;
                        if (items[i].rec.stored < WNR4_HEADER_BYTES ||
                            wnr4_u32(items[i].rec_buf) != WNR4_MAGIC ||
                            wnr4_u32(items[i].rec_buf + 4u) != ms->id ||
                            wnr4_u32(items[i].rec_buf + 8u) != items[i].rec.owner ||
                            items[i].rec_buf[12] != want_k ||
                            ((items[i].rec.flags & WEX4_REC_VALUE_EXACT) &&
                             !(items[i].rec_buf[13] & WNR4_FLAG_CZ))) {
                            items[i].status = WEX_E_DECODE;
                            continue;
                        }
                    }

                    if (wex_store_expert_entry(&s->view, items[i].rec_idx, &exp_ent[cnt], NULL, NULL) != 0) {
                        items[i].status = WEX_E_DECODE;
                        continue;
                    }
                    exp_ent[cnt].flags = ck;
                    exp_map[cnt]     = i;
                    exp_ent_ptr[cnt] = &exp_ent[cnt];
                    exp_rec[cnt]     = items[i].rec_buf;
                    if (!is_device) {
                        exp_dst[cnt] = (uint8_t *)dst_slabs[i];
                    } else {
                        exp_sc[3 * cnt + 0] = (uint8_t *)dst_scales[3 * i + 0];
                        exp_sc[3 * cnt + 1] = (uint8_t *)dst_scales[3 * i + 1];
                        exp_sc[3 * cnt + 2] = (uint8_t *)dst_scales[3 * i + 2];
                        exp_wt[3 * cnt + 0] = (uint8_t *)dst_weights[3 * i + 0];
                        exp_wt[3 * cnt + 1] = (uint8_t *)dst_weights[3 * i + 1];
                        exp_wt[3 * cnt + 2] = (uint8_t *)dst_weights[3 * i + 2];
                    }
                    exp_st[cnt] = 1;
                    cnt++;
                }

                if (cnt > 0) {
                    int k;
                    if (!is_device) {
                        int use_scratch = scratch_dst && ((ops->caps & WEX_DEV_CAP_SCRATCH_HOST) != 0u);
                        (void)ops->decode_expert_host(cnt, exp_ent_ptr, exp_rec, s->gpu_device,
                                                      exp_dst, use_scratch, exp_st);
                    } else {
                        int use_exp = expand_e8m0 && ((ops->caps & WEX_DEV_CAP_E8M0_EXPAND) != 0u);
                        (void)ops->decode_expert_device(cnt, exp_ent_ptr, exp_rec, s->gpu_device,
                                                        exp_sc, exp_wt, device_stream, use_exp, exp_st);
                    }
                    for (k = 0; k < cnt; k++) {
                        int idx = exp_map[k];
                        if (exp_st[k] == 0) {
                            items[idx].decoded = 1;
                            st_gpu++;
                            st_dec_bytes += items[idx].rec.decoded;
                        } else if (exp_st[k] < 0) {
                            items[idx].status = WEX_E_DECODE;
                        } else {
                            /* Transient CUDA failure: CPU fallback on host, WEX_E_DEVICE on device */
                            if (is_device) items[idx].status = WEX_E_DEVICE;
                            else items[idx].need_cpu = 1;
                        }
                    }
                }
            }
            if (exp_heap) {
                free(exp_map); free(exp_ent); free(exp_ent_ptr); free(exp_rec);
                free(exp_dst); free(exp_sc); free(exp_wt); free(exp_st);
            }
        }

        /* 3b. GPU WNR4-I4 batch (grouped by identical geometry; G <= 64 only on CUDA) */
        if (ops && (ops->caps & WEX_DEV_CAP_WNR4_I4) && ops->decode_i4_batch) {
            uint8_t visited_stk[LIBWEX_STACK_BATCH];
            uint8_t *visited = (n > LIBWEX_STACK_BATCH) ? (uint8_t *)calloc((size_t)n, 1) : visited_stk;
            if (visited) {
                if (n <= LIBWEX_STACK_BATCH) memset(visited, 0, (size_t)n);
                for (i = 0; i < n; i++) {
                    wex_dev_i4_geo_t base_geo;
                    const wex_model_slot_t *ms;
                    uint32_t G;
                    if (visited[i] || items[i].status != WEX_OK || items[i].decoded ||
                        items[i].codec_kind != WEX_CODEC_WNR4_I4)
                        continue;
                    if (!items[i].rec.blob || items[i].rec.blob > s->num_blobs) {
                        items[i].status = WEX_E_DECODE;
                        visited[i] = 1;
                        continue;
                    }
                    ms = &s->models[items[i].rec.blob - 1u];
                    if (ms->type != WEX_MODEL_WNM4_I4 || !ms->dev_registered) continue;
                    if (items[i].rec.stored < WNR4_HEADER_BYTES ||
                        wnr4_u32(items[i].rec_buf) != WNR4_MAGIC ||
                        items[i].rec_buf[12] != WNR4_KIND_I4 ||
                        items[i].rec_buf[13] > 1u ||
                        wnr4_u32(items[i].rec_buf + 4u) != ms->id ||
                        wnr4_u32(items[i].rec_buf + 8u) != items[i].rec.owner) {
                        items[i].status = WEX_E_DECODE;
                        visited[i] = 1;
                        continue;
                    }
                    G = (uint32_t)items[i].rec_buf[14] | ((uint32_t)items[i].rec_buf[15] << 8);
                    if (G > 64u) {
                        /* CUDA I4 kernel supports G <= 64; decode larger G on CPU */
                        visited[i] = 1;
                        if (is_device) items[i].status = WEX_E_UNSUPPORTED;
                        continue;
                    }
                    if (wex_store_i4_geo(&s->view, items[i].rec_idx, items[i].rec_buf[13],
                                         NULL, &base_geo) != 0) {
                        items[i].status = WEX_E_DECODE;
                        visited[i] = 1;
                        continue;
                    }

                    /* Collect all unvisited I4 items matching base_geo */
                    {
                        int grp_map_stk[LIBWEX_STACK_BATCH];
                        const uint8_t *grp_rec_stk[LIBWEX_STACK_BATCH];
                        size_t grp_tot_stk[LIBWEX_STACK_BATCH];
                        uint32_t grp_own_stk[LIBWEX_STACK_BATCH];
                        uint8_t *grp_dst_stk[LIBWEX_STACK_BATCH];
                        int grp_st_stk[LIBWEX_STACK_BATCH];

                        int *grp_map = grp_map_stk;
                        const uint8_t **grp_rec = grp_rec_stk;
                        size_t *grp_tot = grp_tot_stk;
                        uint32_t *grp_own = grp_own_stk;
                        uint8_t **grp_dst = grp_dst_stk;
                        int *grp_st = grp_st_stk;
                        int g_heap = 0, cnt = 0, j;

                        if (n > LIBWEX_STACK_BATCH) {
                            g_heap = 1;
                            grp_map = (int *)malloc((size_t)n * sizeof(*grp_map));
                            grp_rec = (const uint8_t **)malloc((size_t)n * sizeof(*grp_rec));
                            grp_tot = (size_t *)malloc((size_t)n * sizeof(*grp_tot));
                            grp_own = (uint32_t *)malloc((size_t)n * sizeof(*grp_own));
                            grp_dst = (uint8_t **)malloc((size_t)n * sizeof(*grp_dst));
                            grp_st  = (int *)malloc((size_t)n * sizeof(*grp_st));
                        }
                        if (grp_map && grp_rec && grp_tot && grp_own && grp_dst && grp_st) {
                            for (j = i; j < n; j++) {
                                wex_dev_i4_geo_t jgeo;
                                uint32_t jG;
                                const wex_model_slot_t *jms;
                                if (visited[j] || items[j].status != WEX_OK || items[j].decoded ||
                                    items[j].codec_kind != WEX_CODEC_WNR4_I4)
                                    continue;
                                if (!items[j].rec.blob || items[j].rec.blob > s->num_blobs) continue;
                                jms = &s->models[items[j].rec.blob - 1u];
                                if (jms->type != WEX_MODEL_WNM4_I4 || !jms->dev_registered) continue;
                                if (items[j].rec.stored < WNR4_HEADER_BYTES ||
                                    wnr4_u32(items[j].rec_buf) != WNR4_MAGIC ||
                                    items[j].rec_buf[12] != WNR4_KIND_I4 ||
                                    items[j].rec_buf[13] > 1u ||
                                    wnr4_u32(items[j].rec_buf + 4u) != jms->id ||
                                    wnr4_u32(items[j].rec_buf + 8u) != items[j].rec.owner)
                                    continue;
                                jG = (uint32_t)items[j].rec_buf[14] | ((uint32_t)items[j].rec_buf[15] << 8);
                                if (jG > 64u) continue;
                                if (wex_store_i4_geo(&s->view, items[j].rec_idx, items[j].rec_buf[13],
                                                     NULL, &jgeo) != 0)
                                    continue;
                                if (memcmp(&jgeo, &base_geo, sizeof(base_geo)) != 0) continue;

                                visited[j]   = 1;
                                grp_map[cnt] = j;
                                grp_rec[cnt] = items[j].rec_buf;
                                grp_tot[cnt] = items[j].rec.stored;
                                grp_own[cnt] = items[j].rec.owner;
                                grp_dst[cnt] = (uint8_t *)dst_slabs[j];
                                grp_st[cnt]  = 1;
                                cnt++;
                            }
                            if (cnt > 0) {
                                int k;
                                (void)ops->decode_i4_batch(cnt, &base_geo, grp_rec, grp_tot, grp_own,
                                                           s->gpu_device, grp_dst, is_device,
                                                           scratch_dst, device_stream, grp_st);
                                for (k = 0; k < cnt; k++) {
                                    int idx = grp_map[k];
                                    if (grp_st[k] == 0) {
                                        items[idx].decoded = 1;
                                        st_gpu++;
                                        st_dec_bytes += items[idx].rec.decoded;
                                    } else if (grp_st[k] < 0) {
                                        items[idx].status = WEX_E_DECODE;
                                    } else {
                                        if (is_device) items[idx].status = WEX_E_DEVICE;
                                        else items[idx].need_cpu = 1;
                                    }
                                }
                            }
                        }
                        if (g_heap) {
                            free(grp_map); free(grp_rec); free(grp_tot);
                            free(grp_own); free(grp_dst); free(grp_st);
                        }
                    }
                }
                if (n > LIBWEX_STACK_BATCH) free(visited);
            }
        }

        /* 3c. GPU WNR4-GG batch */
        if (ops && (ops->caps & WEX_DEV_CAP_WNR4_GG) && ops->decode_gg_batch) {
            int gg_map_stk[LIBWEX_STACK_BATCH];
            wex_dev_gg_geo_t gg_geo_stk[LIBWEX_STACK_BATCH];
            const uint8_t *gg_rec_stk[LIBWEX_STACK_BATCH];
            size_t gg_tot_stk[LIBWEX_STACK_BATCH];
            uint32_t gg_own_stk[LIBWEX_STACK_BATCH];
            uint8_t *gg_dst_stk[LIBWEX_STACK_BATCH];
            int gg_st_stk[LIBWEX_STACK_BATCH];

            int *gg_map = gg_map_stk;
            wex_dev_gg_geo_t *gg_geo = gg_geo_stk;
            const uint8_t **gg_rec = gg_rec_stk;
            size_t *gg_tot = gg_tot_stk;
            uint32_t *gg_own = gg_own_stk;
            uint8_t **gg_dst = gg_dst_stk;
            int *gg_st = gg_st_stk;
            int gg_heap = 0, cnt = 0;

            if (n > LIBWEX_STACK_BATCH) {
                gg_heap = 1;
                gg_map = (int *)malloc((size_t)n * sizeof(*gg_map));
                gg_geo = (wex_dev_gg_geo_t *)malloc((size_t)n * sizeof(*gg_geo));
                gg_rec = (const uint8_t **)malloc((size_t)n * sizeof(*gg_rec));
                gg_tot = (size_t *)malloc((size_t)n * sizeof(*gg_tot));
                gg_own = (uint32_t *)malloc((size_t)n * sizeof(*gg_own));
                gg_dst = (uint8_t **)malloc((size_t)n * sizeof(*gg_dst));
                gg_st  = (int *)malloc((size_t)n * sizeof(*gg_st));
            }
            if (gg_map && gg_geo && gg_rec && gg_tot && gg_own && gg_dst && gg_st) {
                for (i = 0; i < n; i++) {
                    const wex_model_slot_t *ms;
                    if (items[i].status != WEX_OK || items[i].decoded ||
                        items[i].codec_kind != WEX_CODEC_WNR4_GG)
                        continue;
                    if (!items[i].rec.blob || items[i].rec.blob > s->num_blobs) {
                        items[i].status = WEX_E_DECODE;
                        continue;
                    }
                    ms = &s->models[items[i].rec.blob - 1u];
                    if (ms->type != WEX_MODEL_WNM4_GG || !ms->dev_registered) continue;
                    if (items[i].rec.stored < WNR4_HEADER_BYTES ||
                        wnr4_u32(items[i].rec_buf) != WNR4_MAGIC ||
                        items[i].rec_buf[12] != WNR4_KIND_GG ||
                        wnr4_u32(items[i].rec_buf + 4u) != ms->id ||
                        wnr4_u32(items[i].rec_buf + 8u) != items[i].rec.owner) {
                        items[i].status = WEX_E_DECODE;
                        continue;
                    }
                    if (wex_store_gg_geo(&s->view, items[i].rec_idx, NULL, &gg_geo[cnt]) != 0) {
                        items[i].status = WEX_E_DECODE;
                        continue;
                    }
                    gg_map[cnt] = i;
                    gg_rec[cnt] = items[i].rec_buf;
                    gg_tot[cnt] = items[i].rec.stored;
                    gg_own[cnt] = items[i].rec.owner;
                    gg_dst[cnt] = (uint8_t *)dst_slabs[i];
                    gg_st[cnt]  = 1;
                    cnt++;
                }
                if (cnt > 0) {
                    int k;
                    (void)ops->decode_gg_batch(cnt, gg_geo, gg_rec, gg_tot, gg_own,
                                               s->gpu_device, gg_dst, is_device,
                                               scratch_dst, device_stream, gg_st);
                    for (k = 0; k < cnt; k++) {
                        int idx = gg_map[k];
                        if (gg_st[k] == 0) {
                            items[idx].decoded = 1;
                            st_gpu++;
                            st_dec_bytes += items[idx].rec.decoded;
                        } else if (gg_st[k] < 0) {
                            items[idx].status = WEX_E_DECODE;
                        } else {
                            if (is_device) items[idx].status = WEX_E_DEVICE;
                            else items[idx].need_cpu = 1;
                        }
                    }
                }
            }
            if (gg_heap) {
                free(gg_map); free(gg_geo); free(gg_rec); free(gg_tot);
                free(gg_own); free(gg_dst); free(gg_st);
            }
        }

        /* 3d. CPU decode for all remaining host items (or mark unsupported on device) */
        for (i = 0; i < n; i++) {
            if (items[i].status != WEX_OK || items[i].decoded) continue;
            if (is_device) {
                items[i].status = WEX_E_UNSUPPORTED;
                continue;
            }
            items[i].status = wex_cpu_decode_one(s, items[i].rec_idx, &items[i].rec,
                                                 items[i].codec_kind, items[i].rec_buf,
                                                 (uint8_t *)dst_slabs[i], scratch_dst,
                                                 s->cpu_threads);
            if (items[i].status == WEX_OK) {
                items[i].decoded = 1;
                st_cpu++;
                st_dec_bytes += items[i].rec.decoded;
            }
        }
        dec_s = wex_io_now_s() - t_dec0;
    }

    /* 4. Populate L2 cache for valid disk misses (unless BYPASS_L2), release borrows & scratch */
    for (i = 0; i < n; i++) {
        if (!items[i].from_l2 && items[i].scratch_buf && items[i].status == WEX_OK &&
            items[i].decoded && s->l2 && !(flags & WEX_FETCH_BYPASS_L2)) {
            (void)wex_cache_put(s->l2, items[i].rec_idx, items[i].scratch_buf, items[i].rec.stored);
        }
        if (items[i].from_l2) {
            wex_cache_release(s->l2, items[i].l2_handle);
        }
        if (items[i].scratch_buf) {
            wex_scratch_release(s, items[i].scratch_buf, items[i].scratch_slot);
        }
        if (status_out) status_out[i] = items[i].status;
        if (items[i].status != WEX_OK) failed++;
    }

    /* 5. Update telemetry */
    wex_s_lock(&s->stats_lock);
    s->stats.fetches           += 1u;
    s->stats.records_requested += (uint64_t)n;
    s->stats.records_from_l2   += st_l2;
    s->stats.records_from_disk += st_disk;
    s->stats.gpu_decodes       += st_gpu;
    s->stats.cpu_decodes       += st_cpu;
    s->stats.prefetch_hits     += st_pf_hits;
    s->stats.refused           += st_refused;
    s->stats.disk_bytes        += st_disk_bytes;
    s->stats.decoded_bytes     += st_dec_bytes;
    s->stats.io_seconds        += io_s;
    s->stats.decode_seconds    += dec_s;
    s->stats.verify_seconds    += ver_s;
    wex_s_unlock(&s->stats_lock);

    if (heap_alloc) {
        free(items); free(miss_idx); free(io_reqs);
        free(sha_exp); free(fed_done); free(sha_ok);
    }
    return failed;
}

/* ---- Public API implementation ------------------------------------------- */

WEX_API uint32_t wex_abi_version(void) {
    return WEX_STORE_ABI_VERSION;
}

WEX_API const char *wex_version_string(void) {
    return "libwex 1.0.0";
}

WEX_API wex_store_t *wex_open(const char *path, const wex_open_opts_t *opts,
                              char *err_buf, size_t err_cap) {
    wex_open_opts_t def;
    const wex_open_opts_t *o;
    wex_store_t *s = NULL;
    uint8_t *hdr_page = NULL;
    Wex4Header h;
    uint64_t file_size;
    size_t index_len;
    uint32_t r, b;
    uint8_t *blob_used = NULL;

    if (err_buf && err_cap) err_buf[0] = '\0';
    if (!path || !path[0]) {
        wex_set_err(err_buf, err_cap, "wex_open: path is NULL or empty");
        return NULL;
    }
    if (opts) {
        if (opts->struct_size != (uint32_t)sizeof(wex_open_opts_t)) {
            wex_set_err(err_buf, err_cap, "wex_open: opts->struct_size mismatch (%u != %u)",
                        opts->struct_size, (uint32_t)sizeof(wex_open_opts_t));
            return NULL;
        }
        o = opts;
    } else {
        wex_open_opts_init(&def);
        o = &def;
    }

    s = (wex_store_t *)calloc(1, sizeof(*s));
    if (!s) {
        wex_set_err(err_buf, err_cap, "wex_open: out of memory");
        return NULL;
    }
    wex_s_lock_init(&s->scratch_lock);
    wex_s_lock_init(&s->pf_lock);
    wex_s_cond_init(&s->pf_cond);
    wex_s_lock_init(&s->stats_lock);
    s->gpu_device = -1;
    s->cpu_threads = o->cpu_threads;

    /* 1. Open primary file + optional verified mirrors */
    s->io = wex_io_open(path, o->mirrors, o->num_mirrors, o->disk_weights,
                        o->io_direct, o->io_queue_depth, err_buf, err_cap);
    if (!s->io) {
        if (err_buf && err_cap && !err_buf[0])
            wex_set_err(err_buf, err_cap, "wex_open: failed to open container");
        goto fail;
    }
    file_size = wex_io_file_size(s->io);

    /* 2. Read & validate 4096-byte WEX4 header page */
    hdr_page = (uint8_t *)wex_io_alloc_aligned(WEX4_PAGE);
    if (!hdr_page) {
        wex_set_err(err_buf, err_cap, "wex_open: out of memory for header page");
        goto fail;
    }
    if (wex_io_read_exact(s->io, 0, hdr_page, WEX4_PAGE) != 0) {
        wex_set_err(err_buf, err_cap, "wex_open: failed to read 4096-byte header page");
        goto fail;
    }
    {
        int hrc = wex4_parse_header(hdr_page, file_size, &h);
        if (hrc != WEX4_OK) {
            wex_set_err(err_buf, err_cap, "wex_open: invalid WEX4 header (code %d)", hrc);
            goto fail;
        }
    }
    wex_io_free_aligned(hdr_page);
    hdr_page = NULL;

    wex_hex_encode(h.root, s->root_hex);
    if (o->root_hex && o->root_hex[0]) {
        if (!wex_hex_eq64(o->root_hex, s->root_hex)) {
            wex_set_err(err_buf, err_cap, "wex_open: root SHA-256 mismatch (got %s)", s->root_hex);
            goto fail;
        }
        s->verified_root = 1u;
    }

    if (o->verify == WEX_VERIFY_ON)       s->verify_on = 1u;
    else if (o->verify == WEX_VERIFY_OFF) s->verify_on = 0u;
    else                                  s->verify_on = s->verified_root ? 1u : 0u;

    /* 3. Read & validate WEX4 index (supporting RAW/WBP1/WNR3 + codec-8 WNR4 MX/NV/GG/I4) */
    index_len = (size_t)wex4_align(h.index_bytes);
    if (index_len > 0) {
        s->index_buf = (uint8_t *)wex_io_alloc_aligned(index_len);
        if (!s->index_buf) {
            wex_set_err(err_buf, err_cap, "wex_open: out of memory for index (%zu B)", index_len);
            goto fail;
        }
        if (wex_io_read_exact(s->io, h.index_offset, s->index_buf, index_len) != 0) {
            wex_set_err(err_buf, err_cap, "wex_open: failed to read WEX4 index");
            goto fail;
        }
    }
    s->index_len = index_len;
    {
        int irc = wex_store_open_index(&h, s->index_buf, index_len, &s->view);
        if (irc != WEX4_OK) {
            wex_set_err(err_buf, err_cap, "wex_open: invalid WEX4 index (code %d)", irc);
            goto fail;
        }
    }

    /* 4. Load & verify model blobs referenced by WNR3 (codec 5) and WNR4 (codec 8) records */
    s->num_blobs = h.blobs;
    if (h.blobs > 0) {
        s->models = (wex_model_slot_t *)calloc(h.blobs, sizeof(*s->models));
        blob_used = (uint8_t *)calloc(h.blobs, 1u);
        if (!s->models || !blob_used) {
            wex_set_err(err_buf, err_cap, "wex_open: out of memory for model table");
            goto fail;
        }
        for (r = 0; r < h.records; r++) {
            Wex4Record rec;
            wex4_get_record(&s->view, r, &rec);
            if ((rec.codec == WEX4_CODEC_WNR3 || rec.codec == LIBWEX_CODEC_WNR4) &&
                rec.blob > 0 && rec.blob <= h.blobs) {
                blob_used[rec.blob - 1u] = 1u;
            }
        }
        for (b = 0; b < h.blobs; b++) {
            Wex4Blob bl;
            size_t aligned_b;
            uint8_t *buf;
            unsigned char digest[32];
            uint64_t pad;
            if (!blob_used[b]) continue;
            wex4_get_blob(&s->view, b, &bl);
            if (bl.kind != WEX4_BLOB_WNM2 && bl.kind != LIBWEX_BLOB_WNM4) continue;
            aligned_b = (size_t)wex4_align(bl.bytes);
            buf = (uint8_t *)wex_io_alloc_aligned(aligned_b);
            if (!buf) {
                wex_set_err(err_buf, err_cap, "wex_open: out of memory for model blob %u", b);
                goto fail;
            }
            if (wex_io_read_exact(s->io, bl.offset, buf, aligned_b) != 0) {
                wex_io_free_aligned(buf);
                wex_set_err(err_buf, err_cap, "wex_open: failed to read model blob %u", b);
                goto fail;
            }
            wex4_sha256(buf, (size_t)bl.bytes, digest);
            if (memcmp(digest, bl.sha, 32u) != 0) {
                wex_io_free_aligned(buf);
                wex_set_err(err_buf, err_cap, "wex_open: model blob %u SHA-256 mismatch", b);
                goto fail;
            }
            for (pad = bl.bytes; pad < aligned_b; pad++) {
                if (buf[pad] != 0u) {
                    wex_io_free_aligned(buf);
                    wex_set_err(err_buf, err_cap, "wex_open: non-zero padding in model blob %u", b);
                    goto fail;
                }
            }

            if (bl.kind == WEX4_BLOB_WNM2) {
                if (wex_nr2_model_open(buf, (size_t)bl.bytes, &s->models[b].wnm2) != 0 ||
                    s->models[b].wnm2.id != bl.id) {
                    wex_io_free_aligned(buf);
                    wex_set_err(err_buf, err_cap, "wex_open: invalid WNM2 model blob %u", b);
                    goto fail;
                }
                s->models[b].type      = WEX_MODEL_WNM2;
                s->models[b].id        = s->models[b].wnm2.id;
                s->models[b].raw_blob  = buf;
                s->models[b].raw_bytes = (size_t)bl.bytes;
#if LIBWEX_HAS_FAST_DECODE
                if (wexf_cpu_simd()) {
                    s->models[b].fast_wnm2 = (WexFastModel *)malloc(sizeof(WexFastModel));
                    if (s->models[b].fast_wnm2)
                        wex_fast_model_init(s->models[b].fast_wnm2, &s->models[b].wnm2);
                }
#endif
            } else {
                /* WNM4 kind 3: distinguish MX(0), NV(1), GG(2), I4(3) */
                uint32_t kind = (bl.bytes >= 32u && wnr4_u32(buf) == WNM4_MAGIC)
                                ? wnr4_u32(buf + 8u) : UINT32_MAX;
                int ok = 0;
                if ((kind == WNR4_KIND_MX || kind == WNR4_KIND_NV) &&
                    wnr4_model_open(buf, (size_t)bl.bytes, &s->models[b].wnm4_mxnv) == 0 &&
                    s->models[b].wnm4_mxnv.id == bl.id) {
                    s->models[b].type = (kind == WNR4_KIND_MX) ? WEX_MODEL_WNM4_MX : WEX_MODEL_WNM4_NV;
                    s->models[b].id   = s->models[b].wnm4_mxnv.id;
                    ok = 1;
                } else if (kind == WNR4_KIND_GG &&
                           wnr4_gg_model_open(buf, (size_t)bl.bytes, &s->models[b].wnm4_gg) == 0 &&
                           s->models[b].wnm4_gg.id == bl.id) {
                    s->models[b].type = WEX_MODEL_WNM4_GG;
                    s->models[b].id   = s->models[b].wnm4_gg.id;
                    ok = 1;
                } else if (kind == WNR4_KIND_I4 &&
                           wnr4_i4_model_open(buf, (size_t)bl.bytes, &s->models[b].wnm4_i4) == 0 &&
                           s->models[b].wnm4_i4.id == bl.id) {
                    s->models[b].type = WEX_MODEL_WNM4_I4;
                    s->models[b].id   = s->models[b].wnm4_i4.id;
                    ok = 1;
                }
                if (ok) {
                    s->models[b].raw_blob  = buf;
                    s->models[b].raw_bytes = (size_t)bl.bytes;
                } else {
                    /* Fail closed on decode for records referencing a malformed WNM4 blob */
                    wex_io_free_aligned(buf);
                }
            }
        }
        free(blob_used);
        blob_used = NULL;
    }

    /* 5. Per-record codec kind, verification lattice, size bounds & expert grid */
    if (h.records > 0) {
        s->rec_codec_kind  = (uint8_t *)calloc(h.records, 1u);
        s->verify_state    = (uint8_t *)calloc(h.records, 1u);
        s->prefetched_flag = (uint8_t *)calloc(h.records, 1u);
        if (!s->rec_codec_kind || !s->verify_state || !s->prefetched_flag) {
            wex_set_err(err_buf, err_cap, "wex_open: out of memory for record tables");
            goto fail;
        }
    }

    s->primary_codec_kind = WEX_CODEC_RAW;
    for (r = 0; r < h.records; r++) {
        Wex4Record rec;
        uint32_t ck = WEX_CODEC_RAW;
        wex4_get_record(&s->view, r, &rec);
        s->total_stored_bytes += rec.stored;
        if (rec.stored > s->max_record_stored_bytes) s->max_record_stored_bytes = rec.stored;
        if (!s->min_record_stored_bytes || rec.stored < s->min_record_stored_bytes)
            s->min_record_stored_bytes = rec.stored;
        if ((rec.flags & WEX4_REC_EXPERT) && rec.stored > s->max_expert_stored_bytes)
            s->max_expert_stored_bytes = rec.stored;

        if (rec.codec == WEX4_CODEC_RAW)       ck = WEX_CODEC_RAW;
        else if (rec.codec == WEX4_CODEC_WNR3) ck = WEX_CODEC_WNR3;
        else if (rec.codec == WEX4_CODEC_WBP1) ck = WEX_CODEC_WBP1;
        else if (rec.codec == LIBWEX_CODEC_WNR4) {
            if (rec.blob > 0 && rec.blob <= h.blobs) {
                switch (s->models[rec.blob - 1u].type) {
                    case WEX_MODEL_WNM4_MX: ck = WEX_CODEC_WNR4_MX; break;
                    case WEX_MODEL_WNM4_NV: ck = WEX_CODEC_WNR4_NV; break;
                    case WEX_MODEL_WNM4_GG: ck = WEX_CODEC_WNR4_GG; break;
                    case WEX_MODEL_WNM4_I4: ck = WEX_CODEC_WNR4_I4; break;
                    default:                ck = WEX_CODEC_WNR4_MX; break;
                }
            } else {
                ck = WEX_CODEC_WNR4_MX;
            }
        }
        s->rec_codec_kind[r] = (uint8_t)ck;
        if (s->primary_codec_kind == WEX_CODEC_RAW && ck != WEX_CODEC_RAW)
            s->primary_codec_kind = ck;
    }

    if (h.layers > 0 && h.experts > 0) {
        size_t grid_n = (size_t)h.layers * (size_t)h.experts;
        s->layers  = h.layers;
        s->experts = h.experts;
        s->expert_to_rec = (uint32_t *)malloc(grid_n * sizeof(uint32_t));
        if (!s->expert_to_rec) {
            wex_set_err(err_buf, err_cap, "wex_open: out of memory for expert table");
            goto fail;
        }
        memset(s->expert_to_rec, 0xff, grid_n * sizeof(uint32_t));
        for (r = 0; r < h.records; r++) {
            Wex4Record rec;
            wex4_get_record(&s->view, r, &rec);
            if (rec.flags & WEX4_REC_EXPERT) {
                uint32_t l = rec.owner >> 16;
                uint32_t e = rec.owner & 0xffffu;
                s->expert_to_rec[(size_t)l * h.experts + e] = r;
            }
        }
    } else if (h.records > 0) {
        /* Discover optional (layer, eid) grid for non-EXPERT GGUF / int4-g64 containers */
        uint32_t max_l = 0, max_e = 0, found = 0;
        for (r = 0; r < h.records; r++) {
            Wex4Record rec;
            Wex4Tensor t;
            uint32_t l = 0, e = 0;
            wex4_get_record(&s->view, r, &rec);
            if (!rec.tensor_count) continue;
            wex4_get_tensor(&s->view, rec.first_tensor, &t);
            if (wex_parse_expert_name(t.name, t.name_len, &l, &e)) {
                if (!found || l > max_l) max_l = l;
                if (!found || e > max_e) max_e = e;
                found++;
                if (rec.stored > s->max_expert_stored_bytes)
                    s->max_expert_stored_bytes = rec.stored;
            }
        }
        if (found > 0 && (uint64_t)(max_l + 1u) * (uint64_t)(max_e + 1u) <= WEX4_MAX_RECORDS) {
            uint32_t nl = max_l + 1u, ne = max_e + 1u;
            size_t grid_n = (size_t)nl * (size_t)ne;
            s->expert_to_rec = (uint32_t *)malloc(grid_n * sizeof(uint32_t));
            if (s->expert_to_rec) {
                memset(s->expert_to_rec, 0xff, grid_n * sizeof(uint32_t));
                for (r = 0; r < h.records; r++) {
                    Wex4Record rec;
                    Wex4Tensor t;
                    uint32_t l = 0, e = 0;
                    wex4_get_record(&s->view, r, &rec);
                    if (!rec.tensor_count) continue;
                    wex4_get_tensor(&s->view, rec.first_tensor, &t);
                    if (wex_parse_expert_name(t.name, t.name_len, &l, &e)) {
                        if (s->expert_to_rec[(size_t)l * ne + e] == UINT32_MAX)
                            s->expert_to_rec[(size_t)l * ne + e] = r;
                    }
                }
                s->layers  = nl;
                s->experts = ne;
            }
        }
    }

    /* 6. Auto-discover optional GPU backend (seamless CPU fallback if absent/unavailable) */
    if (o->gpu_device != -2 && !(o->gpu_helper_path && strcmp(o->gpu_helper_path, "0") == 0)) {
        void *dlh = NULL;
        const wex_dev_ops_t *ops = wex_load_dev_backend(o->gpu_helper_path, &dlh);
        if (ops) {
            int ndev = ops->device_count();
            int pick = (o->gpu_device < 0) ? 0 : o->gpu_device;
            if (ndev > 0 && pick >= 0 && pick < ndev) {
                s->dev_dl_handle = dlh;
                s->dev_ops       = ops;
                s->gpu_device    = pick;
                /* Register all loaded models with the GPU backend */
                if (ops->register_model) {
                    for (b = 0; b < s->num_blobs; b++) {
                        int mkind = -1;
                        if (s->models[b].type == WEX_MODEL_WNM2)           mkind = 0;
                        else if (s->models[b].type == WEX_MODEL_WNM4_MX ||
                                 s->models[b].type == WEX_MODEL_WNM4_NV)   mkind = 1;
                        else if (s->models[b].type == WEX_MODEL_WNM4_GG)   mkind = 2;
                        else if (s->models[b].type == WEX_MODEL_WNM4_I4)   mkind = 3;
                        if (mkind >= 0 && s->models[b].raw_blob &&
                            ops->register_model(mkind, s->models[b].raw_blob,
                                                s->models[b].raw_bytes) == 0) {
                            s->models[b].dev_registered = 1;
                        }
                    }
                }
            } else {
                if (!ops->name || strcmp(ops->name, "cuda") != 0) wex_dl_close(dlh);
            }
        }
    }

    if (s->dev_ops && s->dev_ops->name && s->dev_ops->name[0]) {
        snprintf(s->backend_name, sizeof(s->backend_name), "%s", s->dev_ops->name);
    } else {
#if LIBWEX_HAS_FAST_DECODE
        const char *kname = wex_fast_kernels();
        if (strcmp(kname, "AVX2") == 0)      snprintf(s->backend_name, sizeof(s->backend_name), "cpu-avx2");
        else if (strcmp(kname, "NEON") == 0) snprintf(s->backend_name, sizeof(s->backend_name), "cpu-neon");
        else                                 snprintf(s->backend_name, sizeof(s->backend_name), "cpu-scalar");
#else
        snprintf(s->backend_name, sizeof(s->backend_name), "cpu-scalar");
#endif
    }

    /* 7. Pre-pinned scratch buffer pool for L2 misses */
    if (s->max_record_stored_bytes > 0) {
        size_t slot_sz = (size_t)wex4_align(s->max_record_stored_bytes);
        uint32_t nslots = LIBWEX_SCRATCH_MAX_SLOTS;
        if (h.records < nslots) nslots = h.records < 8u ? 8u : h.records;
        if ((uint64_t)nslots * (uint64_t)slot_sz > (UINT64_C(768) << 20)) {
            nslots = (uint32_t)((UINT64_C(768) << 20) / slot_sz);
            if (nslots < 4u) nslots = 4u;
        }
        if (nslots > 64u) nslots = 64u;
        {
            size_t total_sz = (size_t)nslots * slot_sz;
            s->scratch_arena = (uint8_t *)wex_io_alloc_aligned(total_sz);
            if (s->scratch_arena) {
                s->scratch_slot_bytes = slot_sz;
                s->scratch_num_slots  = nslots;
                s->scratch_free_mask  = (nslots == 64u) ? ~UINT64_C(0) : ((UINT64_C(1) << nslots) - 1u);
                if (s->dev_ops && (s->dev_ops->caps & WEX_DEV_CAP_PIN_HOST) &&
                    s->dev_ops->host_register) {
                    if (s->dev_ops->host_register(s->scratch_arena, total_sz) == 0)
                        s->scratch_pinned = 1;
                }
            }
        }
    }

    /* 8. Optional L2 Compressed-RAM sharded LRU cache */
    if (o->l2_cache_bytes > 0) {
        uint64_t cap = o->l2_cache_bytes;
        if (cap == UINT64_MAX) {
            uint64_t avail = wex_io_available_ram();
            uint64_t max_auto = UINT64_C(16) << 30;
            cap = avail / 4u;
            if (cap > max_auto) cap = max_auto;
            if (o->l2_pin_gpu && s->total_stored_bytes > 0 && cap > s->total_stored_bytes) {
                cap = wex4_align(s->total_stored_bytes);
            }
        }
        if (cap > 0) {
            int want_pin = (o->l2_pin_gpu && s->dev_ops &&
                            (s->dev_ops->caps & WEX_DEV_CAP_PIN_HOST) &&
                            s->dev_ops->host_register != NULL);
            size_t arena_slot = 0;
            if (want_pin) {
                uint64_t slot_ref = s->max_expert_stored_bytes
                                    ? s->max_expert_stored_bytes : s->max_record_stored_bytes;
                if (slot_ref > 0) arena_slot = (size_t)wex4_align(slot_ref);
            }
            s->l2 = wex_cache_create(cap, arena_slot,
                                     (size_t)(s->min_record_stored_bytes ? s->min_record_stored_bytes : WEX4_PAGE),
                                     want_pin ? s->dev_ops->host_register : NULL,
                                     want_pin ? s->dev_ops->host_unregister : NULL);
        }
    }

    return s;

fail:
    free(blob_used);
    if (hdr_page) wex_io_free_aligned(hdr_page);
    wex_close(s);
    return NULL;
}

WEX_API int wex_info(const wex_store_t *s, wex_info_t *out) {
    if (!s || !out) return WEX_E_ARGS;
    memset(out, 0, sizeof(*out));
    out->abi_version              = WEX_STORE_ABI_VERSION;
    out->layers                   = s->layers;
    out->experts                  = s->experts;
    out->num_records              = s->view.h.records;
    out->num_tensors              = s->view.h.tensors;
    out->codec_kind               = s->primary_codec_kind;
    out->value_exact              = (s->view.h.required & WEX4_REQ_VALUE_EXACT) ? 1u : 0u;
    out->verified_root            = s->verified_root;
    out->verify_on                = s->verify_on;
    out->num_replicas             = (uint32_t)wex_io_replica_count(s->io);
    out->file_bytes               = s->view.h.file_bytes;
    out->total_decoded_bytes      = s->view.h.total_decoded;
    out->max_record_decoded_bytes = s->view.h.max_decoded;
    out->max_record_stored_bytes  = s->max_record_stored_bytes;
    out->l2_capacity_bytes        = s->l2 ? wex_cache_capacity(s->l2) : 0u;
    memcpy(out->root_hex, s->root_hex, sizeof(out->root_hex));
    snprintf(out->backend_name, sizeof(out->backend_name), "%s", s->backend_name);
    snprintf(out->io_engine, sizeof(out->io_engine), "%s", wex_io_engine_name(s->io));
    return WEX_OK;
}

WEX_API int wex_has_expert(const wex_store_t *s, int layer, int eid) {
    if (!s || !s->expert_to_rec || layer < 0 || eid < 0 ||
        (uint32_t)layer >= s->layers || (uint32_t)eid >= s->experts)
        return 0;
    return s->expert_to_rec[(size_t)layer * s->experts + (uint32_t)eid] != UINT32_MAX ? 1 : 0;
}

WEX_API int wex_record_layout(const wex_store_t *s, int layer, int eid,
                              uint32_t flags, wex_layout_t *out) {
    uint32_t rec_idx = 0;
    Wex4Record rec;
    uint32_t k, lim;
    int rc;
    if (!s || !out) return WEX_E_ARGS;
    rc = wex_resolve_record(s, layer, eid, flags, &rec_idx);
    if (rc != WEX_OK) return rc;

    memset(out, 0, sizeof(*out));
    wex4_get_record(&s->view, rec_idx, &rec);
    out->record_index  = rec_idx;
    out->codec_kind    = s->rec_codec_kind[rec_idx];
    out->flags         = rec.flags;
    out->tensor_count  = rec.tensor_count;
    out->file_offset   = rec.offset;
    out->stored_bytes  = rec.stored;
    out->decoded_bytes = rec.decoded;

    lim = rec.tensor_count < WEX_MAX_RECORD_TENSORS ? rec.tensor_count : WEX_MAX_RECORD_TENSORS;
    for (k = 0; k < lim; k++) {
        Wex4Tensor t;
        wex4_get_tensor(&s->view, rec.first_tensor + k, &t);
        out->tensor_offsets[k] = t.offset;
        out->tensor_bytes[k]   = t.nbytes;
    }

    if ((rec.flags & WEX4_REC_EXPERT) && rec.tensor_count == 6u) {
        WexEntry e;
        if (wex_store_expert_entry(&s->view, rec_idx, &e, NULL, NULL) == 0) {
            uint32_t m;
            for (m = 0; m < 3u; m++) {
                out->scale_offsets[m]    = e.scale_offsets[m];
                out->weight_offsets[m]   = e.weight_offsets[m];
                out->scale_shapes[m][0]  = e.scale_shapes[m][0];
                out->scale_shapes[m][1]  = e.scale_shapes[m][1];
                out->weight_shapes[m][0] = e.weight_shapes[m][0];
                out->weight_shapes[m][1] = e.weight_shapes[m][1];
            }
        }
    }
    return WEX_OK;
}

WEX_API int wex_fetch_host(wex_store_t *s, int layer, const int *eids, int n,
                           void *const *dst_slabs, const size_t *dst_caps,
                           uint32_t flags, int *status_out) {
    return wex_fetch_impl(s, layer, eids, n, dst_slabs, dst_caps,
                          NULL, NULL, NULL, flags, 0, status_out);
}

WEX_API int wex_fetch_device(wex_store_t *s, int layer, const int *eids, int n,
                             void *const *dst_scales, void *const *dst_weights,
                             void *const *dst_slabs, void *device_stream,
                             uint32_t flags, int *status_out) {
    return wex_fetch_impl(s, layer, eids, n, dst_slabs, NULL,
                          dst_scales, dst_weights, device_stream, flags, 1, status_out);
}

WEX_API int wex_prefetch(wex_store_t *s, int layer, const int *eids, int n,
                         uint32_t flags) {
    int i, queued = 0;
    if (!s || !s->l2 || !eids || n <= 0) return 0;

    wex_s_lock(&s->pf_lock);
    if (!s->pf_started && !s->pf_stop) {
#ifdef _WIN32
        uintptr_t th = _beginthreadex(NULL, 0, wex_prefetch_worker, s, 0, NULL);
        if (th != 0) {
            s->pf_thread  = (HANDLE)th;
            s->pf_started = 1;
        }
#else
        if (pthread_create(&s->pf_thread, NULL, wex_prefetch_worker, s) == 0) {
            s->pf_started = 1;
        }
#endif
    }
    if (!s->pf_started) {
        wex_s_unlock(&s->pf_lock);
        return 0;
    }

    for (i = 0; i < n; i++) {
        uint32_t rec_idx = 0;
        if (s->pf_count >= LIBWEX_PF_Q_CAP) break;
        if (wex_resolve_record(s, layer, eids[i], flags, &rec_idx) != WEX_OK) continue;
        if (wex_u8_load(&s->verify_state[rec_idx]) == LIBWEX_STATE_REFUSED) continue;
        if (wex_cache_contains(s->l2, rec_idx)) continue;
        s->pf_queue[s->pf_tail] = rec_idx;
        s->pf_tail = (s->pf_tail + 1u) % LIBWEX_PF_Q_CAP;
        s->pf_count++;
        queued++;
    }
    if (queued > 0) wex_s_cond_signal(&s->pf_cond);
    wex_s_unlock(&s->pf_lock);
    return queued;
}

WEX_API void wex_stats(const wex_store_t *s, wex_stats_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!s) return;
    wex_s_lock((wex_s_lock_t *)&s->stats_lock);
    *out = s->stats;
    wex_s_unlock((wex_s_lock_t *)&s->stats_lock);
    if (s->l2) {
        wex_cache_stats(s->l2, &out->l2_used_bytes, &out->l2_entries);
    }
}

WEX_API void wex_close(wex_store_t *s) {
    uint32_t b;
    if (!s) return;

    /* 1. Stop & join prefetch worker before freeing cache or I/O */
    if (s->pf_started) {
        wex_s_lock(&s->pf_lock);
        s->pf_stop = 1;
        wex_s_cond_broadcast(&s->pf_cond);
        wex_s_unlock(&s->pf_lock);
#ifdef _WIN32
        WaitForSingleObject(s->pf_thread, INFINITE);
        CloseHandle(s->pf_thread);
#else
        pthread_join(s->pf_thread, NULL);
#endif
        s->pf_started = 0;
    }

    /* 2. Destroy L2 cache (unpins GPU arena if pinned) */
    if (s->l2) {
        wex_cache_destroy(s->l2);
        s->l2 = NULL;
    }

    /* 3. Unpin & free scratch buffer pool */
    if (s->scratch_arena) {
        if (s->scratch_pinned && s->dev_ops && s->dev_ops->host_unregister) {
            (void)s->dev_ops->host_unregister(s->scratch_arena);
        }
        wex_io_free_aligned(s->scratch_arena);
        s->scratch_arena = NULL;
    }

    /* 4. Free loaded models */
    if (s->models) {
        for (b = 0; b < s->num_blobs; b++) {
#if LIBWEX_HAS_FAST_DECODE
            free(s->models[b].fast_wnm2);
#endif
            wnr4_model_free(&s->models[b].wnm4_mxnv);
            wnr4_gg_model_free(&s->models[b].wnm4_gg);
            wnr4_i4_model_free(&s->models[b].wnm4_i4);
            if (s->models[b].raw_blob) wex_io_free_aligned(s->models[b].raw_blob);
        }
        free(s->models);
    }

    /* 5. Retain CUDA helper handle for process lifetime to avoid CUDA runtime unload issues;
     *    unload non-CUDA mock helpers cleanly. */
    if (s->dev_dl_handle && (!s->dev_ops || !s->dev_ops->name || strcmp(s->dev_ops->name, "cuda") != 0)) {
        wex_dl_close(s->dev_dl_handle);
    }

    if (s->io) wex_io_close(s->io);
    if (s->index_buf) wex_io_free_aligned(s->index_buf);
    free(s->expert_to_rec);
    free(s->rec_codec_kind);
    free(s->verify_state);
    free(s->prefetched_flag);

    wex_s_lock_destroy(&s->scratch_lock);
    wex_s_lock_destroy(&s->pf_lock);
    wex_s_cond_destroy(&s->pf_cond);
    wex_s_lock_destroy(&s->stats_lock);
    free(s);
}
