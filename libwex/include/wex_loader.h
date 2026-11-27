/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_LOADER_H
#define LIBWEX_WEX_LOADER_H

/* Header-only runtime dynamic loader for libwex (C99 / C++11, zero build deps).
 * Inference engines copy ONLY wex_store.h + wex_loader.h and call wex_lib_open(). */

#include "wex_store.h"

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
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wex_lib {
    void *handle;
    uint32_t     (*abi_version)(void);
    const char  *(*version_string)(void);
    wex_store_t *(*open)(const char *path, const wex_open_opts_t *opts,
                         char *err_buf, size_t err_cap);
    int          (*info)(const wex_store_t *s, wex_info_t *out);
    int          (*has_expert)(const wex_store_t *s, int layer, int eid);
    int          (*record_layout)(const wex_store_t *s, int layer, int eid,
                                  uint32_t flags, wex_layout_t *out);
    int          (*fetch_host)(wex_store_t *s, int layer, const int *eids, int n,
                               void *const *dst_slabs, const size_t *dst_caps,
                               uint32_t flags, int *status_out);
    int          (*fetch_device)(wex_store_t *s, int layer, const int *eids, int n,
                                 void *const *dst_scales, void *const *dst_weights,
                                 void *const *dst_slabs, void *device_stream,
                                 uint32_t flags, int *status_out);
    int          (*prefetch)(wex_store_t *s, int layer, const int *eids, int n,
                             uint32_t flags);
    void         (*stats)(const wex_store_t *s, wex_stats_t *out);
    void         (*close)(wex_store_t *s);
} wex_lib_t;

static inline void *wex_loader_dlopen_(const char *path) {
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

static inline void wex_loader_dlclose_(void *h) {
    if (!h) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)h);
#else
    dlclose(h);
#endif
}

static inline int wex_loader_sym_(void *h, const char *name, void *fn_out) {
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

static inline int wex_loader_exe_sibling_(char *buf, size_t cap, const char *filename) {
    size_t flen = strlen(filename);
    char *sep;
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, buf, (DWORD)cap);
    if (!n || (size_t)n >= cap) return -1;
    buf[n] = '\0';
#elif defined(__APPLE__)
    uint32_t sz = (uint32_t)cap;
    if (_NSGetExecutablePath(buf, &sz) != 0) return -1;
    buf[cap - 1u] = '\0';
#else
    ssize_t n = readlink("/proc/self/exe", buf, cap - 1u);
    if (n <= 0 || (size_t)n >= cap) return -1;
    buf[n] = '\0';
#endif
    sep = strrchr(buf, '/');
#ifdef _WIN32
    {
        char *bsep = strrchr(buf, '\\');
        if (bsep && (!sep || bsep > sep)) sep = bsep;
    }
#endif
    if (!sep) return -1;
    if ((size_t)(sep - buf) + 1u + flen + 1u > cap) return -1;
    memcpy(sep + 1, filename, flen + 1u);
    return 0;
}

static inline void wex_lib_close(wex_lib_t *lib) {
    if (!lib) return;
    if (lib->handle) wex_loader_dlclose_(lib->handle);
    memset(lib, 0, sizeof(*lib));
}

static inline int wex_lib_open(wex_lib_t *lib, const char *path_or_null) {
    void *h = NULL;
    if (!lib) return WEX_E_ARGS;
    memset(lib, 0, sizeof(*lib));

    if (path_or_null && path_or_null[0]) {
        h = wex_loader_dlopen_(path_or_null);
    } else {
        const char *env = getenv("WEX_LIB_PATH");
        char sibling[4096];
#ifdef _WIN32
        const char *libname = "libwex.dll";
#elif defined(__APPLE__)
        const char *libname = "libwex.dylib";
#else
        const char *libname = "libwex.so";
#endif
        if (env && env[0]) h = wex_loader_dlopen_(env);
        if (!h && wex_loader_exe_sibling_(sibling, sizeof(sibling), libname) == 0)
            h = wex_loader_dlopen_(sibling);
        if (!h) h = wex_loader_dlopen_(libname);
#if !defined(_WIN32) && !defined(__APPLE__)
        if (!h) h = wex_loader_dlopen_("libwex.so.1");
#endif
    }
    if (!h) return WEX_E_NOT_FOUND;

    lib->handle = h;
    if (wex_loader_sym_(h, "wex_abi_version",    &lib->abi_version)    ||
        wex_loader_sym_(h, "wex_version_string", &lib->version_string) ||
        wex_loader_sym_(h, "wex_open",           &lib->open)           ||
        wex_loader_sym_(h, "wex_info",           &lib->info)           ||
        wex_loader_sym_(h, "wex_has_expert",     &lib->has_expert)     ||
        wex_loader_sym_(h, "wex_record_layout",  &lib->record_layout)  ||
        wex_loader_sym_(h, "wex_fetch_host",     &lib->fetch_host)     ||
        wex_loader_sym_(h, "wex_fetch_device",   &lib->fetch_device)   ||
        wex_loader_sym_(h, "wex_prefetch",       &lib->prefetch)       ||
        wex_loader_sym_(h, "wex_stats",          &lib->stats)          ||
        wex_loader_sym_(h, "wex_close",          &lib->close)) {
        wex_lib_close(lib);
        return WEX_E_UNSUPPORTED;
    }

    if (lib->abi_version() != WEX_STORE_ABI_VERSION) {
        wex_lib_close(lib);
        return WEX_E_UNSUPPORTED;
    }
    return WEX_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_WEX_LOADER_H */
