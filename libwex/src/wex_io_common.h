/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_IO_COMMON_H
#define LIBWEX_WEX_IO_COMMON_H

/* Internal helpers shared by platform async I/O backends (wex_io_win.c,
 * wex_io_linux.c, wex_io_posix.c) and streaming verifier (wex_verify.c). */

#include "wex_io.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#endif

/* ---- Error formatting ---------------------------------------------------- */
static inline void wex_io_set_err(char *err_buf, size_t err_cap, const char *msg) {
    if (err_buf && err_cap > 0) {
        snprintf(err_buf, err_cap, "%s", msg ? msg : "I/O error");
    }
}

/* ---- Portable 4096-byte aligned allocation ------------------------------- */
static inline void *wex_io_common_alloc_aligned(size_t bytes) {
    size_t req = bytes ? bytes : (size_t)WEX_IO_ALIGN;
    if (req > SIZE_MAX - ((size_t)WEX_IO_ALIGN - 1u)) return NULL;
    size_t aligned = (req + ((size_t)WEX_IO_ALIGN - 1u)) & ~((size_t)WEX_IO_ALIGN - 1u);
#ifdef _WIN32
    return _aligned_malloc(aligned, WEX_IO_ALIGN);
#else
    void *p = NULL;
    if (posix_memalign(&p, WEX_IO_ALIGN, aligned) != 0) return NULL;
    return p;
#endif
}

static inline void wex_io_common_free_aligned(void *p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ---- Monotonic clock in seconds ------------------------------------------ */
static inline double wex_io_common_now_s(void) {
#ifdef _WIN32
    static volatile LONGLONG s_freq = 0;
    LONGLONG f = s_freq;
    if (f <= 0) {
        LARGE_INTEGER li;
        if (QueryPerformanceFrequency(&li) && li.QuadPart > 0) {
            f = li.QuadPart;
        } else {
            f = 1;
        }
        InterlockedCompareExchange64(&s_freq, f, 0);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ---- Available system RAM in bytes (0 if unknown) ------------------------ */
static inline uint64_t wex_io_common_available_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX st;
    memset(&st, 0, sizeof(st));
    st.dwLength = (DWORD)sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (uint64_t)st.ullAvailPhys;
    }
    return 0;
#elif defined(__linux__)
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        uint64_t avail_kb = 0, free_kb = 0, cached_kb = 0, buffers_kb = 0;
        int got_avail = 0;
        while (fgets(line, (int)sizeof(line), fp)) {
            unsigned long long val = 0;
            if (sscanf(line, "MemAvailable: %llu kB", &val) == 1) {
                avail_kb = (uint64_t)val;
                got_avail = 1;
                break;
            } else if (sscanf(line, "MemFree: %llu kB", &val) == 1) {
                free_kb = (uint64_t)val;
            } else if (sscanf(line, "Cached: %llu kB", &val) == 1) {
                cached_kb = (uint64_t)val;
            } else if (sscanf(line, "Buffers: %llu kB", &val) == 1) {
                buffers_kb = (uint64_t)val;
            }
        }
        fclose(fp);
        if (got_avail) return avail_kb * UINT64_C(1024);
        if (free_kb || cached_kb || buffers_kb) {
            return (free_kb + cached_kb + buffers_kb) * UINT64_C(1024);
        }
    }
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) return (uint64_t)pages * (uint64_t)psize;
    return 0;
#elif defined(__APPLE__)
    /* Query free + inactive pages via Mach VM statistics; fall back to sysctl HW_MEMSIZE. */
    vm_size_t page_size = 0;
    mach_port_t host = mach_host_self();
    if (host_page_size(host, &page_size) == KERN_SUCCESS && page_size > 0) {
        vm_statistics64_data_t vm;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) == KERN_SUCCESS) {
            uint64_t avail_pages = (uint64_t)vm.free_count + (uint64_t)vm.inactive_count;
            if (avail_pages > 0) return avail_pages * (uint64_t)page_size;
        }
    }
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t total = 0;
    size_t len = sizeof(total);
    if (sysctl(mib, 2, &total, &len, NULL, 0) == 0 && total > 0) {
        return total;
    }
    return 0;
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#ifdef HW_PHYSMEM64
    {
        uint64_t mem = 0;
        size_t len = sizeof(mem);
        int mib[2] = { CTL_HW, HW_PHYSMEM64 };
        if (sysctl(mib, 2, &mem, &len, NULL, 0) == 0 && mem > 0) return mem;
    }
#elif defined(HW_USERMEM)
    {
        unsigned long umem = 0;
        size_t ulen = sizeof(umem);
        int mib[2] = { CTL_HW, HW_USERMEM };
        if (sysctl(mib, 2, &umem, &ulen, NULL, 0) == 0 && umem > 0) return (uint64_t)umem;
    }
#endif
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) return (uint64_t)pages * (uint64_t)psize;
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) return (uint64_t)pages * (uint64_t)psize;
    return 0;
#endif
}

/* ---- Default adaptive queue depth ---------------------------------------- */
static inline int wex_io_default_qd(int requested_qd, int nrep, int max_qd) {
    int qd = requested_qd;
    if (qd <= 0) {
        if (nrep <= 1) {
            qd = 6;
        } else {
            qd = 8 * nrep;
            if (qd > 24) qd = 24;
        }
    }
    if (qd < 1) qd = 1;
    if (qd > max_qd) qd = max_qd;
    return qd;
}

/* ---- Shared mirror identity validator ------------------------------------ */
typedef int (*wex_io_read_page0_fn)(void *ctx, uint8_t *page_out, size_t page_len,
                                    uint64_t *size_out);

/* Verifies a mirror has identical file_size and identical 4096-byte header page
 * (or min(file_size, 4096) bytes for smaller files) to primary.
 * Returns 1 if identical, 0 if mismatched or unreadable. */
static inline int wex_io_mirror_matches(uint64_t primary_size,
                                        const uint8_t *primary_hdr,
                                        size_t hdr_len,
                                        wex_io_read_page0_fn read_fn,
                                        void *mirror_ctx) {
    uint64_t msize = 0;
    uint8_t *mhdr = (uint8_t *)wex_io_common_alloc_aligned(WEX_IO_ALIGN);
    if (!mhdr) return 0;
    memset(mhdr, 0, WEX_IO_ALIGN);
    int ok = 0;
    if (read_fn(mirror_ctx, mhdr, hdr_len, &msize) == 0 &&
        msize == primary_size &&
        (hdr_len == 0 || memcmp(primary_hdr, mhdr, hdr_len) == 0)) {
        ok = 1;
    }
    wex_io_common_free_aligned(mhdr);
    return ok;
}

/* ---- Weighted active-queue + deficit replica picker ---------------------- */
typedef struct {
    int      nrep;
    int      has_explicit_weights;
    uint32_t weights[WEX_IO_MAX_REPLICAS];
    uint64_t active_pages[WEX_IO_MAX_REPLICAS];
    uint64_t issued_pages[WEX_IO_MAX_REPLICAS];
    uint32_t rr_cursor;
#ifdef _WIN32
    volatile LONG lock;
#else
    volatile int  lock;
#endif
} wex_io_picker_t;

static inline void wex_io_picker_lock(wex_io_picker_t *pk) {
#ifdef _WIN32
    while (InterlockedExchange(&pk->lock, 1) != 0) {
        YieldProcessor();
    }
#else
    while (__atomic_exchange_n(&pk->lock, 1, __ATOMIC_ACQUIRE) != 0) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
#endif
    }
#endif
}

static inline void wex_io_picker_unlock(wex_io_picker_t *pk) {
#ifdef _WIN32
    InterlockedExchange(&pk->lock, 0);
#else
    __atomic_store_n(&pk->lock, 0, __ATOMIC_RELEASE);
#endif
}

static inline void wex_io_picker_init(wex_io_picker_t *pk, int nrep,
                                      const uint32_t *weights) {
    memset(pk, 0, sizeof(*pk));
    if (nrep < 1) nrep = 1;
    if (nrep > WEX_IO_MAX_REPLICAS) nrep = WEX_IO_MAX_REPLICAS;
    pk->nrep = nrep;
    pk->has_explicit_weights = (weights != NULL) ? 1 : 0;
    for (int i = 0; i < nrep; i++) {
        uint32_t w = (weights && weights[i] > 0) ? weights[i] : 1u;
        pk->weights[i] = w;
    }
}

static inline uint64_t wex_io_bytes_to_pages(uint64_t bytes) {
    uint64_t pages = (bytes + (WEX_IO_ALIGN - 1u)) / WEX_IO_ALIGN;
    return pages ? pages : 1u;
}

/* Pick replica index in [0, pk->nrep).
 * Single-NVMe fast path (nrep == 1): immediate return 0 with ZERO atomics.
 * Multi-NVMe (nrep >= 2): combines active in-flight page load + weighted
 * virtual-time deficit so concurrent batches saturate all drives in proportion
 * to their weights and sequential traffic tracks explicit weights closely. */
static inline int wex_io_pick_rep(wex_io_picker_t *pk, int hint_replica, uint64_t bytes) {
    if (pk->nrep <= 1) return 0;

    uint64_t req_pages = wex_io_bytes_to_pages(bytes);
    wex_io_picker_lock(pk);

    double min_vt = (double)pk->issued_pages[0] / (double)pk->weights[0];
    for (int r = 1; r < pk->nrep; r++) {
        double vt = (double)pk->issued_pages[r] / (double)pk->weights[r];
        if (vt < min_vt) min_vt = vt;
    }

    int best = 0;
    double best_score = 0.0;
    int valid_hint = (hint_replica >= 0 && hint_replica < pk->nrep) ? hint_replica : -1;

    for (int k = 0; k < pk->nrep; k++) {
        int r = (int)((pk->rr_cursor + (uint32_t)k) % (uint32_t)pk->nrep);
        double w = (double)pk->weights[r];
        double vt_excess = ((double)pk->issued_pages[r] / w) - min_vt;
        double score = ((double)(pk->active_pages[r] + req_pages) / w) + vt_excess;
        if (k == 0 || score < best_score - 1e-9) {
            best = r;
            best_score = score;
        } else if (score <= best_score + 1e-9) {
            /* Deterministic tie-breaking: prefer valid hint_replica, else lower deficit */
            if (r == valid_hint && best != valid_hint) {
                best = r;
                best_score = score;
            } else if (best != valid_hint) {
                double best_vt = (double)pk->issued_pages[best] / (double)pk->weights[best];
                double r_vt = (double)pk->issued_pages[r] / w;
                if (r_vt < best_vt - 1e-9) {
                    best = r;
                    best_score = score;
                }
            }
        }
    }

    pk->active_pages[best] += req_pages;
    pk->issued_pages[best] += req_pages;
    pk->rr_cursor = (uint32_t)((best + 1) % pk->nrep);

    wex_io_picker_unlock(pk);
    return best;
}

/* Release active-queue accounting when a request completes on `rep`.
 * Single-NVMe fast path (nrep == 1): immediate return with ZERO atomics. */
static inline void wex_io_unpick_rep(wex_io_picker_t *pk, int rep, uint64_t bytes) {
    if (pk->nrep <= 1) return;
    if (rep < 0 || rep >= pk->nrep) return;

    uint64_t req_pages = wex_io_bytes_to_pages(bytes);
    wex_io_picker_lock(pk);
    if (pk->active_pages[rep] >= req_pages) {
        pk->active_pages[rep] -= req_pages;
    } else {
        pk->active_pages[rep] = 0;
    }
    wex_io_picker_unlock(pk);
}

/* Instantiated exactly once by the active platform backend translation unit. */
#ifdef WEX_IO_DEFINE_COMMON_FUNCS
void *wex_io_alloc_aligned(size_t bytes) {
    return wex_io_common_alloc_aligned(bytes);
}

void wex_io_free_aligned(void *p) {
    wex_io_common_free_aligned(p);
}

uint64_t wex_io_available_ram(void) {
    return wex_io_common_available_ram();
}

double wex_io_now_s(void) {
    return wex_io_common_now_s();
}
#endif

#endif /* LIBWEX_WEX_IO_COMMON_H */
