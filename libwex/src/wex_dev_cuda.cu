/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Unified CUDA device backend for libwex (libwex_cuda.dll / libwex_cuda.so).
 * Implements libwex/src/wex_dev_ops.h across all four GPU codecs:
 *   - WNR2 / WNR3 (c/wex_cuda.cuh)
 *   - WNR4-MX & WNR4-NV (wnr4/wnr4_cuda.cuh)
 *   - WNR4-I4 (wnr4/wnr4_i4_cuda.cuh)
 *   - WNR4-GG (wnr4/wnr4_gg_cuda.cuh)
 *
 * All device and pinned-host workspaces live in process-wide pools keyed by
 * call concurrency rather than thread-local storage, so thread churn never
 * leaks VRAM or page-locked memory. Model tables are uploaded lazily per
 * device on first use. */

#include "wex_dev_ops.h"
#include "../../c/wex_cuda.cuh"
#include "../../wnr4/wnr4_cuda.cuh"
#include "../../wnr4/wnr4_i4_cuda.cuh"
#include "../../wnr4/wnr4_gg_cuda.cuh"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

#ifndef WEX_DEV_EXPORT
#ifdef _WIN32
#define WEX_DEV_EXPORT __declspec(dllexport)
#else
#define WEX_DEV_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef WEX_ENTRY_WNR4
#define WEX_ENTRY_WNR4 UINT32_C(8)
#endif
#ifndef WEX_ENTRY_WNR4_NV
#define WEX_ENTRY_WNR4_NV UINT32_C(9)
#endif

#define WEX_DEV_MAX_DEVICES 64
#define WEX_NR3_BATCH_LANES 8

namespace {

/* ---- Device guard & allocation helpers ---------------------------------- */

struct DeviceGuard {
    int saved = -1;
    bool switched = false;
    DeviceGuard() {
        if (cudaGetDevice(&saved) != cudaSuccess) {
            (void)cudaGetLastError();
            saved = -1;
        }
    }
    bool set(int dev) {
        if (dev < 0 || dev >= WEX_DEV_MAX_DEVICES) return false;
        if (saved == dev) return true;
        if (cudaSetDevice(dev) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        switched = true;
        return true;
    }
    ~DeviceGuard() {
        if (switched && saved >= 0) {
            (void)cudaSetDevice(saved);
            (void)cudaGetLastError();
        }
    }
};

template<typename T>
static bool grow_dev(T *&p, size_t &cap, size_t n) {
    if (n <= cap) return true;
    if (p) cudaFree(p);
    p = nullptr;
    cap = 0;
    if (!n) return true;
    if (cudaMalloc((void **)&p, n * sizeof(T)) != cudaSuccess) {
        (void)cudaGetLastError();
        p = nullptr;
        return false;
    }
    cap = n;
    return true;
}

template<typename T>
static bool grow_host(T *&p, size_t &cap, size_t n) {
    if (n <= cap) return true;
    if (p) cudaFreeHost(p);
    p = nullptr;
    cap = 0;
    if (!n) return true;
    if (cudaMallocHost((void **)&p, n * sizeof(T)) != cudaSuccess) {
        (void)cudaGetLastError();
        p = nullptr;
        return false;
    }
    cap = n;
    return true;
}

static inline size_t align_up(size_t v, size_t a) {
    return (v + (a - 1u)) & ~(a - 1u);
}

/* ---- Process-wide workspace pool ---------------------------------------- */

template<class T>
struct WorkspacePool {
    std::mutex mu;
    std::vector<T *> idle;

    T *take(int target_dev) {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!idle.empty()) {
                size_t pick = idle.size() - 1u;
                for (size_t i = idle.size(); i-- > 0;) {
                    if (idle[i]->device_id() == target_dev) {
                        pick = i;
                        break;
                    }
                }
                T *w = idle[pick];
                idle.erase(idle.begin() + (std::ptrdiff_t)pick);
                return w;
            }
        }
        return new (std::nothrow) T();
    }

    void give(T *w) {
        if (!w) return;
        std::lock_guard<std::mutex> lock(mu);
        try {
            idle.push_back(w);
        } catch (...) {
            delete w;
        }
    }
};

template<class T>
struct WorkspaceLease {
    WorkspacePool<T> &pool;
    T *w;
    WorkspaceLease(WorkspacePool<T> &p, int dev) : pool(p), w(p.take(dev)) {}
    ~WorkspaceLease() { pool.give(w); }
    WorkspaceLease(const WorkspaceLease &) = delete;
    WorkspaceLease &operator=(const WorkspaceLease &) = delete;
};

/* ---- In-VRAM E8M0 -> IEEE-754 float32 expansion kernels ----------------- */

struct WexE8m0Region {
    const unsigned char *src;
    uint32_t *dst;
    uint32_t count;
    uint32_t rec_idx;
};

static __device__ __forceinline__ uint32_t wex_e8m0_to_f32_bits(uint32_t e) {
    /* Bit-exact with ldexpf(1.0f, (int)e - 127) for all e in 0..255:
     *   e == 0      -> 2^-127 = subnormal 0x00400000
     *   1..254      -> 2^(e-127) = e << 23
     *   e == 255    -> +Inf = 0x7f800000 = 255 << 23 */
    return e ? (e << 23) : 0x00400000u;
}

static __global__ void wex_e8m0_to_f32_k(const WexE8m0Region *regions, const uint32_t *errors) {
    const WexE8m0Region reg = regions[blockIdx.y];
    if (errors && errors[reg.rec_idx]) return;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const unsigned char *src = reg.src;
    uint32_t *dst = reg.dst;
    const uint32_t n = reg.count;
    if (!(((uintptr_t)src | (uintptr_t)dst) & 15u)) {
        const uint32_t n4 = n >> 2;
        const uint32_t *src32 = (const uint32_t *)src;
        uint4 *dst4 = (uint4 *)dst;
        for (uint32_t i = tid; i < n4; i += stride) {
            const uint32_t p = src32[i];
            dst4[i] = make_uint4(wex_e8m0_to_f32_bits(p & 0xffu),
                                 wex_e8m0_to_f32_bits((p >> 8) & 0xffu),
                                 wex_e8m0_to_f32_bits((p >> 16) & 0xffu),
                                 wex_e8m0_to_f32_bits(p >> 24));
        }
        for (uint32_t i = (n4 << 2) + tid; i < n; i += stride)
            dst[i] = wex_e8m0_to_f32_bits((uint32_t)src[i]);
    } else {
        for (uint32_t i = tid; i < n; i += stride)
            dst[i] = wex_e8m0_to_f32_bits((uint32_t)src[i]);
    }
}

static cudaError_t launch_e8m0_expand(const WexE8m0Region *regions_dev, uint32_t nregions,
                                      uint32_t max_count, const uint32_t *errors_dev,
                                      cudaStream_t stream) {
    if (!nregions || !max_count) return cudaSuccess;
    uint32_t gx = (max_count / 4u + 255u) / 256u;
    if (gx < 1u) gx = 1u;
    if (gx > 256u) gx = 256u;
    uint32_t off = 0u;
    while (off < nregions) {
        const uint32_t chunk = (nregions - off) < 65535u ? (nregions - off) : 65535u;
        wex_e8m0_to_f32_k<<<dim3(gx, chunk), 256u, 0, stream>>>(regions_dev + off, errors_dev);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        off += chunk;
    }
    return cudaSuccess;
}

/* Gated WNR3 commit kernel that expands regions 0..2 (E8M0 scales) -> float32
 * and copies regions 3..5 (packed FP4 weights) only when *error == 0. */
static __global__ void wex_nr3_commit_e8m0_k(const unsigned char *raw, WexCudaCommit commit,
                                             const uint32_t *error) {
    if (*(const volatile uint32_t *)error) return;
    const uint32_t region = blockIdx.y;
    const unsigned char *from = raw + commit.source[region];
    unsigned char *to = commit.destination[region];
    const size_t bytes = commit.bytes[region];
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (region < 3u) {
        uint32_t *dst32 = (uint32_t *)to;
        if (!(((uintptr_t)from | (uintptr_t)dst32) & 15u)) {
            const size_t n4 = bytes >> 2;
            const uint32_t *src32 = (const uint32_t *)from;
            uint4 *dst4 = (uint4 *)dst32;
            for (size_t i = tid; i < n4; i += stride) {
                const uint32_t p = src32[i];
                dst4[i] = make_uint4(wex_e8m0_to_f32_bits(p & 0xffu),
                                     wex_e8m0_to_f32_bits((p >> 8) & 0xffu),
                                     wex_e8m0_to_f32_bits((p >> 16) & 0xffu),
                                     wex_e8m0_to_f32_bits(p >> 24));
            }
            for (size_t i = (n4 << 2) + tid; i < bytes; i += stride)
                dst32[i] = wex_e8m0_to_f32_bits((uint32_t)from[i]);
        } else {
            for (size_t i = tid; i < bytes; i += stride)
                dst32[i] = wex_e8m0_to_f32_bits((uint32_t)from[i]);
        }
    } else {
        if (!(((uintptr_t)from | (uintptr_t)to | bytes) & 15u)) {
            for (size_t i = tid; i < bytes / 16u; i += stride)
                ((uint4 *)to)[i] = ((const uint4 *)from)[i];
        } else {
            for (size_t i = tid; i < bytes; i += stride)
                to[i] = from[i];
        }
    }
}

/* ---- Unified process-wide model registries (lazy per-device upload) ----- */

struct Wnm2Entry {
    uint32_t id = 0;
    unsigned char blob[WEX_NR2_MODEL_BYTES];
    unsigned char *dev[WEX_DEV_MAX_DEVICES] = {};
};

struct Wnm4MxEntry {
    Wnr4Model host = {};
    unsigned char *dev[WEX_DEV_MAX_DEVICES] = {};
    Wnr4CudaTables tab[WEX_DEV_MAX_DEVICES] = {};
};

struct Wnm4GgEntry {
    Wnr4GgModel host = {};
    unsigned char *dev[WEX_DEV_MAX_DEVICES] = {};
};

struct Wnm4I4Entry {
    Wnr4I4Model host = {};
    unsigned char *dev[WEX_DEV_MAX_DEVICES] = {};
};

static std::mutex g_models_mu;
static std::vector<Wnm2Entry *> *g_wnm2_models = nullptr;
static std::vector<Wnm4MxEntry *> *g_wnm4_mx_models = nullptr;
static std::vector<Wnm4GgEntry *> *g_wnm4_gg_models = nullptr;
static std::vector<Wnm4I4Entry *> *g_wnm4_i4_models = nullptr;

template<typename T>
static std::vector<T *> &model_list(std::vector<T *> *&ptr) {
    if (!ptr) ptr = new std::vector<T *>();
    return *ptr;
}

static int reg_wnm2(const uint8_t *blob, size_t bytes) {
    WexNr2Model probe;
    if (wex_nr2_model_open(blob, bytes, &probe)) return -1;
    std::lock_guard<std::mutex> lock(g_models_mu);
    auto &list = model_list(g_wnm2_models);
    for (Wnm2Entry *e : list) {
        if (e->id == probe.id)
            return std::memcmp(e->blob, blob, WEX_NR2_MODEL_BYTES) ? -1 : 0;
    }
    Wnm2Entry *e = new (std::nothrow) Wnm2Entry();
    if (!e) return 1;
    e->id = probe.id;
    std::memcpy(e->blob, blob, WEX_NR2_MODEL_BYTES);
    try {
        list.push_back(e);
    } catch (...) {
        delete e;
        return 1;
    }
    return 0;
}

static int reg_wnm4_mx(const uint8_t *blob, size_t bytes) {
    Wnm4MxEntry *e = new (std::nothrow) Wnm4MxEntry();
    if (!e) return 1;
    if (wnr4_model_open(blob, bytes, &e->host)) {
        delete e;
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_models_mu);
    auto &list = model_list(g_wnm4_mx_models);
    for (Wnm4MxEntry *k : list) {
        if (k->host.id == e->host.id) {
            const bool same = k->host.kind == e->host.kind &&
                              k->host.pb == e->host.pb &&
                              k->host.wctx == e->host.wctx &&
                              k->host.sctx == e->host.sctx &&
                              std::memcmp(k->host.wcdf, e->host.wcdf, 3u * (size_t)k->host.wctx * 17u * 2u) == 0 &&
                              std::memcmp(k->host.scdf, e->host.scdf, 3u * (size_t)k->host.sctx * 257u * 2u) == 0;
            wnr4_model_free(&e->host);
            delete e;
            return same ? 0 : -1;
        }
    }
    try {
        list.push_back(e);
    } catch (...) {
        wnr4_model_free(&e->host);
        delete e;
        return 1;
    }
    return 0;
}

static int reg_wnm4_gg(const uint8_t *blob, size_t bytes) {
    Wnm4GgEntry *e = new (std::nothrow) Wnm4GgEntry();
    if (!e) return 1;
    if (wnr4_gg_model_open(blob, bytes, &e->host)) {
        delete e;
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_models_mu);
    auto &list = model_list(g_wnm4_gg_models);
    for (Wnm4GgEntry *k : list) {
        if (k->host.id == e->host.id) {
            const bool same = std::memcmp(k->host.map, e->host.map, sizeof(k->host.map)) == 0 &&
                              std::memcmp(k->host.hi, e->host.hi, WNR4_GG_SUBTYPES * 257u * 2u) == 0 &&
                              std::memcmp(k->host.lo, e->host.lo, WNR4_GG_SUBTYPES * WNR4_GG_CLASSES * 257u * 2u) == 0 &&
                              std::memcmp(k->host.nib, e->host.nib, WNR4_GG_SUBTYPES * WNR4_GG_CLASSES * 17u * 2u) == 0;
            wnr4_gg_model_free(&e->host);
            delete e;
            return same ? 0 : -1;
        }
    }
    try {
        list.push_back(e);
    } catch (...) {
        wnr4_gg_model_free(&e->host);
        delete e;
        return 1;
    }
    return 0;
}

static int reg_wnm4_i4(const uint8_t *blob, size_t bytes) {
    Wnm4I4Entry *e = new (std::nothrow) Wnm4I4Entry();
    if (!e) return 1;
    if (wnr4_i4_model_open(blob, bytes, &e->host)) {
        delete e;
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_models_mu);
    auto &list = model_list(g_wnm4_i4_models);
    for (Wnm4I4Entry *k : list) {
        if (k->host.id == e->host.id) {
            const bool same = std::memcmp(k->host.w, e->host.w, 3u * WNR4_I4_WROWS * 17u * 2u) == 0 &&
                              std::memcmp(k->host.s, e->host.s, 3u * WNR4_I4_SROWS * 257u * 2u) == 0;
            wnr4_i4_model_free(&e->host);
            delete e;
            return same ? 0 : -1;
        }
    }
    try {
        list.push_back(e);
    } catch (...) {
        wnr4_i4_model_free(&e->host);
        delete e;
        return 1;
    }
    return 0;
}

static bool wnm2_is_registered(uint32_t id) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm2_models) return false;
    for (Wnm2Entry *e : *g_wnm2_models) if (e->id == id) return true;
    return false;
}

static int wnm2_get_tables(uint32_t id, int device, cudaStream_t stream, WexCudaNr2Tables *tables) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm2_models || device < 0 || device >= WEX_DEV_MAX_DEVICES) return 1;
    Wnm2Entry *m = nullptr;
    for (Wnm2Entry *e : *g_wnm2_models) if (e->id == id) { m = e; break; }
    if (!m) return 1;
    const int buckets = (WEX_CUDA_NR2_LOOKUP == 0);
    if (!m->dev[device]) {
        unsigned char *base = nullptr;
        if (cudaMalloc((void **)&base, wex_cuda_nr2_table_bytes(buckets)) != cudaSuccess) {
            (void)cudaGetLastError();
            return 1;
        }
        WexCudaNr2Tables built;
        if (cudaMemcpyAsync(base, m->blob, WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            wex_cuda_build_nr2_tables(base, buckets, stream, &built) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaStreamSynchronize(stream);
            cudaFree(base);
            (void)cudaGetLastError();
            return 1;
        }
        m->dev[device] = base;
    }
    wex_cuda_nr2_table_view(m->dev[device], buckets, tables);
    return 0;
}

static bool wnm4_mx_is_registered(uint32_t id, uint32_t *kind_out) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm4_mx_models) return false;
    for (Wnm4MxEntry *e : *g_wnm4_mx_models) {
        if (e->host.id == id) {
            if (kind_out) *kind_out = e->host.kind;
            return true;
        }
    }
    return false;
}

static int wnm4_mx_get(uint32_t id, int device, const Wnr4Model **host_out, Wnr4CudaTables *tab_out) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm4_mx_models || device < 0 || device >= WEX_DEV_MAX_DEVICES) return 1;
    Wnm4MxEntry *m = nullptr;
    for (Wnm4MxEntry *e : *g_wnm4_mx_models) if (e->host.id == id) { m = e; break; }
    if (!m) return 1;
    if (!m->dev[device]) {
        std::vector<unsigned char> img;
        try {
            img.resize(wnr4_cuda_table_bytes(&m->host));
        } catch (...) {
            return 1;
        }
        wnr4_cuda_table_image(&m->host, img.data());
        unsigned char *d = nullptr;
        if (cudaMalloc((void **)&d, img.size()) != cudaSuccess ||
            cudaMemcpy(d, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            if (d) cudaFree(d);
            (void)cudaGetLastError();
            return 1;
        }
        m->dev[device] = d;
        wnr4_cuda_tables_view(&m->host, d, &m->tab[device]);
    }
    *host_out = &m->host;
    *tab_out = m->tab[device];
    return 0;
}

static int wnm4_gg_get(uint32_t id, int device, const Wnr4GgModel **host_out, const unsigned char **dev_out) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm4_gg_models || device < 0 || device >= WEX_DEV_MAX_DEVICES) return 1;
    Wnm4GgEntry *m = nullptr;
    for (Wnm4GgEntry *e : *g_wnm4_gg_models) if (e->host.id == id) { m = e; break; }
    if (!m) return 1;
    if (!m->dev[device]) {
        std::vector<unsigned char> img;
        try {
            img.resize(wnr4_gg_cuda_table_bytes());
        } catch (...) {
            return 1;
        }
        wnr4_gg_cuda_table_image(&m->host, img.data());
        unsigned char *d = nullptr;
        if (cudaMalloc((void **)&d, img.size()) != cudaSuccess ||
            cudaMemcpy(d, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            if (d) cudaFree(d);
            (void)cudaGetLastError();
            return 1;
        }
        m->dev[device] = d;
    }
    *host_out = &m->host;
    *dev_out = m->dev[device];
    return 0;
}

static int wnm4_i4_get(uint32_t id, int device, const Wnr4I4Model **host_out, const unsigned char **dev_out) {
    std::lock_guard<std::mutex> lock(g_models_mu);
    if (!g_wnm4_i4_models || device < 0 || device >= WEX_DEV_MAX_DEVICES) return 1;
    Wnm4I4Entry *m = nullptr;
    for (Wnm4I4Entry *e : *g_wnm4_i4_models) if (e->host.id == id) { m = e; break; }
    if (!m) return 1;
    if (!m->dev[device]) {
        std::vector<unsigned char> img;
        try {
            img.resize(wnr4_i4_cuda_table_bytes());
        } catch (...) {
            return 1;
        }
        wnr4_i4_cuda_table_image(&m->host, img.data());
        unsigned char *d = nullptr;
        if (cudaMalloc((void **)&d, img.size()) != cudaSuccess ||
            cudaMemcpy(d, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            if (d) cudaFree(d);
            (void)cudaGetLastError();
            return 1;
        }
        m->dev[device] = d;
    }
    *host_out = &m->host;
    *dev_out = m->dev[device];
    return 0;
}

/* ---- WNR2/WNR3 pooled batch workspace & decode -------------------------- */

struct Nr3Lane {
    unsigned char *device_record = nullptr, *device_raw = nullptr;
    uint32_t *device_offsets = nullptr, *device_error = nullptr;
    cudaStream_t stream = nullptr, aux = nullptr;
    cudaEvent_t head = nullptr, data = nullptr, consumed = nullptr;
    size_t capacity = 0, tiles = 0;
    int device = -1;

    ~Nr3Lane() { clear(); }

    void clear() {
        if (aux) { (void)cudaStreamSynchronize(aux); cudaStreamDestroy(aux); }
        if (stream) { (void)cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        if (head) cudaEventDestroy(head);
        if (data) cudaEventDestroy(data);
        if (consumed) cudaEventDestroy(consumed);
        if (device_record) cudaFree(device_record);
        if (device_raw) cudaFree(device_raw);
        if (device_offsets) cudaFree(device_offsets);
        if (device_error) cudaFree(device_error);
        device_record = device_raw = nullptr;
        device_offsets = device_error = nullptr;
        stream = aux = nullptr;
        head = data = consumed = nullptr;
        capacity = tiles = 0;
        device = -1;
        (void)cudaGetLastError();
    }

    bool ensure(int current, size_t bytes, size_t stream_offs) {
        if (device == current && capacity >= bytes && tiles >= stream_offs && aux) return true;
        if (device >= 0 && device != current) {
            (void)cudaSetDevice(device);
            clear();
            (void)cudaSetDevice(current);
        } else {
            clear();
        }
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            free_bytes < bytes * 2u + stream_offs * 4u + (8u << 20)) {
            (void)cudaGetLastError();
            return false;
        }
#define NR3_ALLOC_OK(call) do { if ((call) != cudaSuccess) { clear(); return false; } } while (0)
        NR3_ALLOC_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        NR3_ALLOC_OK(cudaStreamCreateWithFlags(&aux, cudaStreamNonBlocking));
        NR3_ALLOC_OK(cudaEventCreateWithFlags(&head, cudaEventDisableTiming));
        NR3_ALLOC_OK(cudaEventCreateWithFlags(&data, cudaEventDisableTiming));
        NR3_ALLOC_OK(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
        NR3_ALLOC_OK(cudaMalloc((void **)&device_record, bytes));
        NR3_ALLOC_OK(cudaMalloc((void **)&device_raw, bytes));
        NR3_ALLOC_OK(cudaMalloc((void **)&device_offsets, stream_offs * 4u));
        NR3_ALLOC_OK(cudaMalloc((void **)&device_error, 4u));
#undef NR3_ALLOC_OK
        capacity = bytes;
        tiles = stream_offs;
        device = current;
        return true;
    }
};

struct Nr3BatchSet {
    Nr3Lane lanes[WEX_NR3_BATCH_LANES];
    uint32_t *host_offsets = nullptr, *host_errors = nullptr;
    size_t offset_cap = 0, error_cap = 0;
    cudaEvent_t e_in = nullptr;
    int e_in_dev = -1;

    int device_id() const { return lanes[0].device; }

    ~Nr3BatchSet() {
        if (e_in && e_in_dev >= 0) {
            int cur = -1;
            (void)cudaGetDevice(&cur);
            if (cudaSetDevice(e_in_dev) == cudaSuccess) cudaEventDestroy(e_in);
            if (cur >= 0) (void)cudaSetDevice(cur);
        }
        if (host_offsets) cudaFreeHost(host_offsets);
        if (host_errors) cudaFreeHost(host_errors);
    }

    bool ensure_entry_event(int dev) {
        if (e_in && e_in_dev == dev) return true;
        if (e_in) {
            if (e_in_dev >= 0 && e_in_dev != dev) {
                (void)cudaSetDevice(e_in_dev);
                cudaEventDestroy(e_in);
                (void)cudaSetDevice(dev);
            } else {
                cudaEventDestroy(e_in);
            }
            e_in = nullptr;
            e_in_dev = -1;
        }
        if (cudaEventCreateWithFlags(&e_in, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            e_in = nullptr;
            return false;
        }
        e_in_dev = dev;
        return true;
    }
};

static WorkspacePool<Nr3BatchSet> &nr3_pool() {
    static auto *p = new WorkspacePool<Nr3BatchSet>();
    return *p;
}

static cudaError_t nr3_copy_back(const WexEntry *entry, const WexCudaNr2Plan &plan,
                                 const unsigned char *device_raw, unsigned char *dst,
                                 cudaStream_t work) {
    if (plan.weight_base == entry->scale_bytes)
        return cudaMemcpyAsync(dst, device_raw, (size_t)entry->scale_bytes + entry->weight_bytes,
                               cudaMemcpyDeviceToHost, work);
    cudaError_t e = cudaMemcpyAsync(dst, device_raw, entry->scale_bytes, cudaMemcpyDeviceToHost, work);
    return e != cudaSuccess ? e
                            : cudaMemcpyAsync(dst + entry->scale_bytes, device_raw + plan.weight_base,
                                              entry->weight_bytes, cudaMemcpyDeviceToHost, work);
}

static void nr3_report_err(uint32_t error) {
    static std::atomic<int> reported{0};
    if ((error & (WEX_CUDA_ERR_HASH | WEX_CUDA_ERR_CEIL | WEX_CUDA_ERR_CZ)) && !reported.exchange(1))
        std::fprintf(stderr, "[libwex-cuda] WNR3 record failed integrity check (error 0x%x); rejected\n", error);
}

static void decode_nr3_subset(const std::vector<int> &indices,
                              const WexEntry *const *entries,
                              const uint8_t *const *records,
                              int device,
                              uint8_t *const *dst_scale,
                              uint8_t *const *dst_weight,
                              uint8_t *const *dst_host,
                              int scratch_dst,
                              void *caller_stream,
                              int expand_e8m0_f32,
                              int *status) {
    const size_t cnt = indices.size();
    if (!cnt) return;
    const int PENDING = 2;
    std::vector<WexNr2Record> views(cnt);
    std::vector<WexCudaNr2Plan> plans(cnt);
    std::vector<size_t> offset_at(cnt, 0);
    size_t offsets = 0, max_aligned = 0, max_streams = 0;
    int pending = 0;

    for (size_t k = 0; k < cnt; k++) {
        const int orig = indices[k];
        status[orig] = -1;
        const WexEntry *entry = entries[orig];
        uint64_t raw = 0, aligned = 0;
        if (!entry || !records[orig] || !wex_entry_nr_model(entry) ||
            wex_entry_raw_sizes(entry, &raw, &aligned) || entry->total_bytes >= aligned)
            continue;
        bool targets = dst_host ? (dst_host[orig] != nullptr) : true;
        if (!dst_host) {
            for (int m = 0; m < 3; m++) {
                if (!dst_scale[3 * orig + m] || !dst_weight[3 * orig + m]) targets = false;
                if (expand_e8m0_f32 && ((uintptr_t)dst_scale[3 * orig + m] & 3u)) targets = false;
            }
        }
        if (!targets) continue;
        if (wex_cuda_plan_nr2(entry, records[orig], entry->total_bytes, &views[k], &plans[k]))
            continue;
        if (!wnm2_is_registered(plans[k].model_id)) {
            status[orig] = 1;
            continue;
        }
        status[orig] = PENDING;
        pending++;
        offset_at[k] = offsets;
        offsets += (size_t)plans[k].streams + 1u;
        if ((size_t)aligned > max_aligned) max_aligned = (size_t)aligned;
        if ((size_t)plans[k].streams + 1u > max_streams) max_streams = (size_t)plans[k].streams + 1u;
    }

    auto finish_pending = [&](int val) {
        for (size_t k = 0; k < cnt; k++) {
            if (status[indices[k]] == PENDING) status[indices[k]] = val;
        }
    };
    if (!pending) return;

    DeviceGuard guard;
    if (!guard.set(device)) {
        finish_pending(1);
        return;
    }
    WorkspaceLease<Nr3BatchSet> lease(nr3_pool(), device);
    if (!lease.w) {
        finish_pending(1);
        return;
    }
    Nr3BatchSet &set = *lease.w;
    const int lanes = pending < WEX_NR3_BATCH_LANES ? pending : WEX_NR3_BATCH_LANES;
    if (!grow_host(set.host_offsets, set.offset_cap, offsets) ||
        !grow_host(set.host_errors, set.error_cap, cnt)) {
        finish_pending(1);
        return;
    }
    for (int l = 0; l < lanes; l++) {
        if (!set.lanes[l].ensure(device, max_aligned, max_streams)) {
            finish_pending(1);
            return;
        }
    }

    const bool use_entry_ev = (!dst_host && caller_stream != nullptr);
    if (use_entry_ev) {
        if (!set.ensure_entry_event(device) ||
            cudaEventRecord(set.e_in, (cudaStream_t)caller_stream) != cudaSuccess) {
            (void)cudaGetLastError();
            finish_pending(1);
            return;
        }
    }

    auto drain = [&](int count) -> bool {
        bool clean = true;
        for (int l = 0; l < count; l++) {
            if (cudaStreamSynchronize(set.lanes[l].aux) != cudaSuccess) clean = false;
            if (cudaStreamSynchronize(set.lanes[l].stream) != cudaSuccess) clean = false;
        }
        if (!clean) (void)cudaGetLastError();
        return clean;
    };

    auto queue_one = [&](size_t k, Nr3Lane &w) -> int {
        const int orig = indices[k];
        const WexEntry *entry = entries[orig];
        const WexCudaNr2Plan &plan = plans[k];
        WexCudaNr2Tables tables;
        if (wnm2_get_tables(plan.model_id, device, w.stream, &tables)) {
            status[orig] = 1;
            return 0;
        }
        uint32_t *hoff = set.host_offsets + offset_at[k];
        wex_cuda_nr2_offsets(&views[k], &plan, hoff);
        WexCudaCommit commit = {};
        if (!dst_host) {
            for (int m = 0; m < 3; m++) {
                commit.destination[m] = dst_scale[3 * orig + m];
                commit.source[m] = plan.scale_offset[m];
                commit.bytes[m] = (uint32_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
                commit.destination[3 + m] = dst_weight[3 * orig + m];
                commit.source[3 + m] = plan.weight_offset[m];
                commit.bytes[3 + m] = (uint32_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
            }
        }
        cudaStream_t copy = w.stream, work = w.aux;
        const unsigned char *record = records[orig];
        if (cudaStreamWaitEvent(copy, w.consumed, 0) != cudaSuccess ||
            cudaMemcpyAsync(w.device_offsets, hoff, ((size_t)plan.streams + 1u) * 4u,
                            cudaMemcpyHostToDevice, copy) != cudaSuccess ||
            cudaMemcpyAsync(w.device_record, record, plan.weight_payload,
                            cudaMemcpyHostToDevice, copy) != cudaSuccess ||
            cudaEventRecord(w.head, copy) != cudaSuccess ||
            cudaMemcpyAsync(w.device_record + plan.weight_payload, record + plan.weight_payload,
                            entry->total_bytes - plan.weight_payload,
                            cudaMemcpyHostToDevice, copy) != cudaSuccess ||
            cudaEventRecord(w.data, copy) != cudaSuccess ||
            cudaStreamWaitEvent(work, w.head, 0) != cudaSuccess ||
            cudaMemsetAsync(w.device_error, 0, 4u, work) != cudaSuccess ||
            wex_cuda_launch_nr2_scales(&plan, &tables, w.device_record, w.device_offsets,
                                       w.device_raw, w.device_error, work) != cudaSuccess ||
            cudaStreamWaitEvent(work, w.data, 0) != cudaSuccess ||
            wex_cuda_launch_nr2_weights<WEX_CUDA_NR2_LOOKUP>(&plan, &tables, w.device_record,
                                                             w.device_offsets, w.device_raw,
                                                             w.device_error, work) != cudaSuccess ||
            cudaEventRecord(w.consumed, work) != cudaSuccess) {
            (void)cudaGetLastError();
            return -1;
        }
        if (!dst_host) {
            if (use_entry_ev && cudaStreamWaitEvent(work, set.e_in, 0) != cudaSuccess) {
                (void)cudaGetLastError();
                return -1;
            }
            if (expand_e8m0_f32) {
                wex_nr3_commit_e8m0_k<<<dim3(64u, 6u), 256u, 0, work>>>(w.device_raw, commit, w.device_error);
                if (cudaGetLastError() != cudaSuccess) {
                    (void)cudaGetLastError();
                    return -1;
                }
            } else {
                if (wex_cuda_launch_commit(w.device_raw, &commit, w.device_error, work) != cudaSuccess) {
                    (void)cudaGetLastError();
                    return -1;
                }
            }
        } else if (scratch_dst) {
            if (nr3_copy_back(entry, plan, w.device_raw, dst_host[orig], work) != cudaSuccess) {
                (void)cudaGetLastError();
                return -1;
            }
        }
        if (cudaMemcpyAsync(set.host_errors + k, w.device_error, 4u,
                            cudaMemcpyDeviceToHost, work) != cudaSuccess) {
            (void)cudaGetLastError();
            return -1;
        }
        return 0;
    };

    size_t broken_at = cnt;
    bool synced = true;
    if (!dst_host || scratch_dst) {
        int slot_idx = 0;
        for (size_t k = 0; k < cnt && broken_at == cnt; k++) {
            if (status[indices[k]] == PENDING && queue_one(k, set.lanes[slot_idx++ % lanes]))
                broken_at = k;
        }
        synced = drain(lanes);
        for (size_t k = 0; k < cnt; k++) {
            const int orig = indices[k];
            if (status[orig] == PENDING) {
                status[orig] = (k >= broken_at || !synced) ? 1 : (set.host_errors[k] ? -1 : 0);
                if (status[orig] == -1) nr3_report_err(set.host_errors[k]);
            }
        }
        return;
    }

    /* Strict host mode (scratch_dst == 0): untouched on error. */
    size_t round[WEX_NR3_BATCH_LANES];
    for (size_t k = 0; k < cnt && broken_at == cnt && synced;) {
        int used = 0;
        for (; k < cnt && used < lanes; k++) {
            if (status[indices[k]] != PENDING) continue;
            if (queue_one(k, set.lanes[used])) { broken_at = k; break; }
            if (status[indices[k]] == PENDING) round[used++] = k;
        }
        if (!(synced = drain(used))) break;
        bool any_copy = false;
        for (int r = 0; r < used; r++) {
            const size_t rk = round[r];
            const int orig = indices[rk];
            if (set.host_errors[rk]) {
                status[orig] = -1;
                nr3_report_err(set.host_errors[rk]);
                continue;
            }
            if (nr3_copy_back(entries[orig], plans[rk], set.lanes[r].device_raw,
                              dst_host[orig], set.lanes[r].aux) != cudaSuccess) {
                (void)cudaGetLastError();
                status[orig] = 1;
                continue;
            }
            status[orig] = -2;
            any_copy = true;
        }
        const bool copied = any_copy ? drain(used) : true;
        for (int r = 0; r < used; r++) {
            const int orig = indices[round[r]];
            if (status[orig] == -2) status[orig] = copied ? 0 : 1;
        }
        synced = copied;
    }
    if (broken_at < cnt) (void)drain(lanes);
    finish_pending(1);
}

/* ---- WNR4-MX/NV pooled 2-lane workspace & decode ------------------------ */

static int wnr4_entry_to_geo(const WexEntry *entry, Wnr4Geo *g, uint64_t *raw_out, uint64_t *aligned_out) {
    if (!entry) return -1;
    uint32_t kind = 0u;
    if (entry->flags == WEX_ENTRY_WNR4_NV) {
        kind = WNR4_KIND_NV;
    } else if (entry->flags == WEX_ENTRY_WNR4) {
        const uint32_t sc0 = entry->scale_shapes[0][1];
        const uint32_t wc0 = entry->weight_shapes[0][1];
        if (!sc0 || !wc0) return -1;
        if (wc0 == sc0 * 16u) kind = WNR4_KIND_MX;
        else if (wc0 == sc0 * 8u) kind = WNR4_KIND_NV;
        else return -1;
    } else {
        return -1;
    }

    const uint32_t bytes_per_scale = (kind == WNR4_KIND_NV) ? 8u : 16u;
    uint32_t scount[3], scols[3], soff[3], woff[3];
    uint64_t scales = 0, weights = 0;
    for (uint32_t m = 0; m < 3u; m++) {
        const uint32_t sr = entry->scale_shapes[m][0];
        const uint32_t sc = entry->scale_shapes[m][1];
        const uint32_t wr = entry->weight_shapes[m][0];
        const uint32_t wc = entry->weight_shapes[m][1];
        if (!sr || !sc || !wr || !wc || sr != wr || sc * bytes_per_scale != wc ||
            entry->scale_offsets[m] != scales)
            return -1;
        scount[m] = sr * sc;
        scols[m] = sc;
        soff[m] = entry->scale_offsets[m];
        woff[m] = entry->weight_offsets[m];
        scales += (uint64_t)sr * sc;
    }
    if (scales != entry->scale_bytes) return -1;
    for (uint32_t m = 0; m < 3u; m++) {
        if (entry->weight_offsets[m] != scales + weights) return -1;
        weights += (uint64_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
    }
    if (weights != entry->weight_bytes || scales + weights > UINT32_MAX) return -1;
    const uint64_t raw = scales + weights;
    uint64_t aligned = 0;
    if (wex_aligned_size(raw, &aligned) || aligned > UINT32_MAX || aligned > SIZE_MAX) return -1;
    if (wnr4_geo(g, kind, scount, scols, soff, woff) || g->raw != raw) return -1;
    if (entry->total_bytes < WNR4_HEADER_BYTES || (entry->total_bytes & 4095u)) return -1;
    if (raw_out) *raw_out = raw;
    if (aligned_out) *aligned_out = aligned;
    return 0;
}

struct Wnr4MxLane {
    unsigned char *drec = nullptr; size_t drec_cap = 0;
    unsigned char *draw = nullptr; size_t draw_cap = 0;
    unsigned char *dscale_tmp = nullptr; size_t dscale_tmp_cap = 0;
    Wnr4CudaJob *djob = nullptr, *hjob = nullptr; size_t djob_cap = 0, hjob_cap = 0;
    Wnr4CudaRec *dinfo = nullptr, *hinfo = nullptr; size_t dinfo_cap = 0, hinfo_cap = 0;
    uint32_t *derr = nullptr, *herr = nullptr; size_t derr_cap = 0, herr_cap = 0;
    WexE8m0Region *de8 = nullptr, *he8 = nullptr; size_t de8_cap = 0, he8_cap = 0;
    cudaStream_t stream = nullptr;
    bool err_dirty = true;
    int device = -1;
    std::vector<Wnr4CudaJob> by_m[3];
    std::vector<Wnr4CudaJob> mine[3];

    ~Wnr4MxLane() { clear(); }

    void clear() {
        if (stream) { (void)cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        if (drec) cudaFree(drec);
        if (draw) cudaFree(draw);
        if (dscale_tmp) cudaFree(dscale_tmp);
        if (djob) cudaFree(djob);
        if (dinfo) cudaFree(dinfo);
        if (derr) cudaFree(derr);
        if (de8) cudaFree(de8);
        if (hjob) cudaFreeHost(hjob);
        if (hinfo) cudaFreeHost(hinfo);
        if (herr) cudaFreeHost(herr);
        if (he8) cudaFreeHost(he8);
        drec = draw = dscale_tmp = nullptr;
        djob = hjob = nullptr;
        dinfo = hinfo = nullptr;
        derr = herr = nullptr;
        de8 = he8 = nullptr;
        stream = nullptr;
        drec_cap = draw_cap = dscale_tmp_cap = 0;
        djob_cap = hjob_cap = dinfo_cap = hinfo_cap = 0;
        derr_cap = herr_cap = de8_cap = he8_cap = 0;
        err_dirty = true;
        device = -1;
        (void)cudaGetLastError();
    }

    bool bind_device(int dev) {
        if (device == dev && stream) return true;
        if (device >= 0 && device != dev) {
            (void)cudaSetDevice(device);
            clear();
            (void)cudaSetDevice(dev);
        } else if (!stream) {
            clear();
        }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            (void)cudaGetLastError();
            stream = nullptr;
            return false;
        }
        device = dev;
        err_dirty = true;
        return true;
    }

    bool grow_err(size_t n) {
        const size_t want = n < 8u ? 8u : n;
        const size_t prev = derr_cap;
        if (!grow_dev(derr, derr_cap, want) || !grow_host(herr, herr_cap, want)) return false;
        if (derr_cap != prev) err_dirty = true;
        return true;
    }

    cudaError_t clear_err_if_dirty(cudaStream_t st) {
        if (!err_dirty) return cudaSuccess;
        cudaError_t e = cudaMemsetAsync(derr, 0, derr_cap * 4u, st);
        if (e == cudaSuccess) err_dirty = false;
        return e;
    }
};

struct Wnr4MxSet {
    Wnr4MxLane lanes[2];
    cudaEvent_t e_in = nullptr;
    int e_in_dev = -1;

    int device_id() const { return lanes[0].device; }

    ~Wnr4MxSet() {
        if (e_in && e_in_dev >= 0) {
            int cur = -1;
            (void)cudaGetDevice(&cur);
            if (cudaSetDevice(e_in_dev) == cudaSuccess) cudaEventDestroy(e_in);
            if (cur >= 0) (void)cudaSetDevice(cur);
        }
    }

    bool ensure_entry_event(int dev) {
        if (e_in && e_in_dev == dev) return true;
        if (e_in) {
            if (e_in_dev >= 0 && e_in_dev != dev) {
                (void)cudaSetDevice(e_in_dev);
                cudaEventDestroy(e_in);
                (void)cudaSetDevice(dev);
            } else {
                cudaEventDestroy(e_in);
            }
            e_in = nullptr;
            e_in_dev = -1;
        }
        if (cudaEventCreateWithFlags(&e_in, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            e_in = nullptr;
            return false;
        }
        e_in_dev = dev;
        return true;
    }
};

static WorkspacePool<Wnr4MxSet> &wnr4_mx_pool() {
    static auto *p = new WorkspacePool<Wnr4MxSet>();
    return *p;
}

struct Wnr4MxItem {
    int orig;
    uint32_t model_id;
    uint32_t owner;
    uint32_t weight_base;
    Wnr4Geo geo;
};

struct Wnr4MxLaneRun {
    std::vector<size_t> item_idx;      /* index into group_items for each queued record r */
    std::vector<size_t> draw_off;      /* offset into lane.draw for host mode */
    cudaStream_t st = nullptr;
    bool launched = false;
    bool broken = false;
};

static cudaError_t wnr4_mx_copy_back_host(const WexEntry *entry, uint32_t weight_base,
                                          const unsigned char *draw_rec, unsigned char *dst_slab,
                                          cudaStream_t st) {
    if (weight_base == entry->scale_bytes) {
        return cudaMemcpyAsync(dst_slab, draw_rec,
                               (size_t)entry->scale_bytes + entry->weight_bytes,
                               cudaMemcpyDeviceToHost, st);
    }
    cudaError_t e = cudaMemcpyAsync(dst_slab, draw_rec, entry->scale_bytes,
                                    cudaMemcpyDeviceToHost, st);
    if (e != cudaSuccess) return e;
    return cudaMemcpyAsync(dst_slab + entry->scale_bytes, draw_rec + weight_base,
                           entry->weight_bytes, cudaMemcpyDeviceToHost, st);
}

/* Enqueues one lane's slice [begin, end) of `items` (all sharing `model` & `tab`).
 * Does NOT synchronize `st` so two lanes can overlap breadth-first! */
static bool wnr4_mx_enqueue_lane(Wnr4MxLane &lane,
                                 cudaStream_t st,
                                 cudaEvent_t wait_ev,
                                 const Wnr4Model *model,
                                 const Wnr4CudaTables &tab,
                                 const std::vector<Wnr4MxItem> &items,
                                 size_t begin,
                                 size_t end,
                                 const WexEntry *const *entries,
                                 const uint8_t *const *records,
                                 uint8_t *const *dst_scale,
                                 uint8_t *const *dst_weight,
                                 uint8_t *const *dst_host,
                                 int scratch_dst,
                                 int expand_e8m0_f32,
                                 int *status,
                                 Wnr4MxLaneRun &run) {
    run.item_idx.clear();
    run.draw_off.clear();
    run.st = st;
    run.launched = false;
    run.broken = false;

    const size_t slice_n = end - begin;
    if (!slice_n) return true;

    /* Pre-compute byte budgets for 8-record workspace pre-sizing (4 per lane x 2 lanes). */
    size_t need_rec = 0, max_rec = 0;
    size_t need_raw = 0, max_raw = 0;
    size_t need_stmp = 0, max_stmp = 0;
    std::vector<size_t> rec_off(slice_n), raw_off(slice_n), stmp_off(slice_n);
    for (size_t s = 0; s < slice_n; s++) {
        const Wnr4MxItem &it = items[begin + s];
        const WexEntry *e = entries[it.orig];
        const size_t rb = align_up((size_t)e->total_bytes, 256u);
        rec_off[s] = need_rec;
        need_rec += rb;
        if (rb > max_rec) max_rec = rb;

        if (dst_host) {
            const size_t sb = align_up((size_t)it.weight_base + (size_t)e->weight_bytes, 256u);
            raw_off[s] = need_raw;
            need_raw += sb;
            if (sb > max_raw) max_raw = sb;
        } else if (expand_e8m0_f32) {
            size_t st_rec = 0;
            for (int m = 0; m < 3; m++) {
                st_rec += align_up((size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1], 16u);
            }
            st_rec = align_up(st_rec, 256u);
            stmp_off[s] = need_stmp;
            need_stmp += st_rec;
            if (st_rec > max_stmp) max_stmp = st_rec;
        }
    }

    const size_t pre_rec = lane.drec_cap ? need_rec
                                         : (need_rec > (4u * max_rec) ? need_rec
                                                                      : (4u * max_rec < (48u << 20) ? 4u * max_rec : (need_rec > (48u << 20) ? need_rec : (48u << 20))));
    const size_t want_info = slice_n < 8u ? 8u : slice_n;
    if ((!grow_dev(lane.drec, lane.drec_cap, pre_rec) && !(pre_rec > need_rec && grow_dev(lane.drec, lane.drec_cap, need_rec))) ||
        !grow_dev(lane.dinfo, lane.dinfo_cap, want_info) ||
        !grow_host(lane.hinfo, lane.hinfo_cap, want_info) ||
        !lane.grow_err(want_info)) {
        return false;
    }
    if (dst_host) {
        const size_t pre_raw = lane.draw_cap ? need_raw
                                             : (need_raw > (4u * max_raw) ? need_raw
                                                                          : (4u * max_raw < (48u << 20) ? 4u * max_raw : (need_raw > (48u << 20) ? need_raw : (48u << 20))));
        if (!grow_dev(lane.draw, lane.draw_cap, pre_raw) && !(pre_raw > need_raw && grow_dev(lane.draw, lane.draw_cap, need_raw)))
            return false;
    } else if (expand_e8m0_f32) {
        const size_t pre_stmp = lane.dscale_tmp_cap ? need_stmp
                                                    : (need_stmp > (4u * max_stmp) ? need_stmp
                                                                                   : (4u * max_stmp < (8u << 20) ? 4u * max_stmp : (need_stmp > (8u << 20) ? need_stmp : (8u << 20))));
        const size_t want_e8 = (want_info * 3u) < 24u ? 24u : (want_info * 3u);
        if ((!grow_dev(lane.dscale_tmp, lane.dscale_tmp_cap, pre_stmp) && !(pre_stmp > need_stmp && grow_dev(lane.dscale_tmp, lane.dscale_tmp_cap, need_stmp))) ||
            !grow_dev(lane.de8, lane.de8_cap, want_e8) ||
            !grow_host(lane.he8, lane.he8_cap, want_e8)) {
            return false;
        }
    }

    if (wait_ev && cudaStreamWaitEvent(st, wait_ev, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }

    for (int k = 0; k < 3; k++) lane.by_m[k].clear();
    uint32_t ne8 = 0u, max_e8_count = 0u;
    cudaError_t e = lane.clear_err_if_dirty(st);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        lane.err_dirty = true;
        return false;
    }

    /* Plan each record and immediately issue its async H2D so DMA overlaps CPU planning. */
    for (size_t s = 0; s < slice_n; s++) {
        const size_t idx = begin + s;
        const Wnr4MxItem &it = items[idx];
        const WexEntry *entry = entries[it.orig];
        unsigned char *ds[3], *dw[3];
        if (dst_host) {
            unsigned char *base = lane.draw + raw_off[s];
            for (int m = 0; m < 3; m++) {
                ds[m] = base + entry->scale_offsets[m];
                dw[m] = base + it.weight_base + (entry->weight_offsets[m] - entry->scale_bytes);
            }
        } else if (expand_e8m0_f32) {
            unsigned char *sbase = lane.dscale_tmp + stmp_off[s];
            size_t soff_m = 0;
            for (int m = 0; m < 3; m++) {
                ds[m] = sbase + soff_m;
                dw[m] = dst_weight[3 * it.orig + m];
                soff_m += align_up((size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1], 16u);
            }
        } else {
            for (int m = 0; m < 3; m++) {
                ds[m] = dst_scale[3 * it.orig + m];
                dw[m] = dst_weight[3 * it.orig + m];
            }
        }

        for (int k = 0; k < 3; k++) lane.mine[k].clear();
        const uint32_t local_r = (uint32_t)run.item_idx.size();
        if (wnr4_cuda_plan(&it.geo, model, records[it.orig], (size_t)entry->total_bytes,
                           it.owner, lane.drec + rec_off[s], local_r,
                           ds, dw, &lane.hinfo[local_r], lane.mine)) {
            status[it.orig] = -1;
            continue;
        }
        /* Valid record: immediately start async H2D overlap! */
        if (e == cudaSuccess) {
            e = cudaMemcpyAsync(lane.drec + rec_off[s], records[it.orig],
                                (size_t)entry->total_bytes, cudaMemcpyHostToDevice, st);
        }
        for (int k = 0; k < 3; k++) {
            lane.by_m[k].insert(lane.by_m[k].end(), lane.mine[k].begin(), lane.mine[k].end());
        }
        if (!dst_host && expand_e8m0_f32) {
            for (int m = 0; m < 3; m++) {
                const uint32_t sc = (uint32_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
                lane.he8[ne8].src = ds[m];
                lane.he8[ne8].dst = (uint32_t *)dst_scale[3 * it.orig + m];
                lane.he8[ne8].count = sc;
                lane.he8[ne8].rec_idx = local_r;
                if (sc > max_e8_count) max_e8_count = sc;
                ne8++;
            }
        }
        run.item_idx.push_back(idx);
        if (dst_host) run.draw_off.push_back(raw_off[s]);
    }

    const size_t nvalid = run.item_idx.size();
    if (!nvalid) {
        if (e != cudaSuccess) { (void)cudaGetLastError(); lane.err_dirty = true; }
        return true;
    }

    const size_t njobs_cap = wnr4_cuda_jobs_cap(lane.by_m, nullptr);
    const size_t want_jobs = (njobs_cap + 1u) < 8192u ? 8192u : (njobs_cap + 1u);
    if (!grow_dev(lane.djob, lane.djob_cap, want_jobs) ||
        !grow_host(lane.hjob, lane.hjob_cap, want_jobs)) {
        (void)cudaStreamSynchronize(st);
        (void)cudaGetLastError();
        run.broken = true;
        return false;
    }
    size_t njobs = 0;
    const uint32_t W = njobs_cap ? wnr4_cuda_jobs_into(lane.by_m, lane.hjob, &njobs) : 4u;

    if (e == cudaSuccess)
        e = cudaMemcpyAsync(lane.dinfo, lane.hinfo, sizeof(Wnr4CudaRec) * nvalid, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess && njobs)
        e = cudaMemcpyAsync(lane.djob, lane.hjob, sizeof(Wnr4CudaJob) * njobs, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess && njobs)
        e = wnr4_cuda_launch_w(tab, lane.djob, (uint32_t)njobs, lane.dinfo, lane.derr, W, st);
    if (e == cudaSuccess && !dst_host && expand_e8m0_f32 && ne8) {
        e = cudaMemcpyAsync(lane.de8, lane.he8, sizeof(WexE8m0Region) * ne8, cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess)
            e = launch_e8m0_expand(lane.de8, ne8, max_e8_count, lane.derr, st);
    }
    if (e == cudaSuccess && dst_host && scratch_dst) {
        for (size_t r = 0; r < nvalid && e == cudaSuccess; r++) {
            const Wnr4MxItem &it = items[run.item_idx[r]];
            e = wnr4_mx_copy_back_host(entries[it.orig], it.weight_base,
                                       lane.draw + run.draw_off[r], dst_host[it.orig], st);
        }
    }
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(lane.herr, lane.derr, 4u * nvalid, cudaMemcpyDeviceToHost, st);

    run.launched = true;
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        lane.err_dirty = true;
        run.broken = true;
    }
    return true;
}

static void decode_wnr4_mx_subset(const std::vector<int> &indices,
                                  const WexEntry *const *entries,
                                  const uint8_t *const *records,
                                  int device,
                                  uint8_t *const *dst_scale,
                                  uint8_t *const *dst_weight,
                                  uint8_t *const *dst_host,
                                  int scratch_dst,
                                  void *caller_stream,
                                  int expand_e8m0_f32,
                                  int *status) {
    if (indices.empty()) return;

    std::vector<Wnr4MxItem> valid_items;
    valid_items.reserve(indices.size());

    for (int orig : indices) {
        status[orig] = -1;
        const WexEntry *entry = entries[orig];
        const uint8_t *rec = records[orig];
        Wnr4MxItem it = {};
        it.orig = orig;
        uint64_t raw = 0, aligned = 0;
        if (!entry || !rec || wnr4_entry_to_geo(entry, &it.geo, &raw, &aligned)) continue;
        if (wnr4_u32(rec) != WNR4_MAGIC) continue;
        it.model_id = wnr4_u32(rec + 4u);
        it.owner = wnr4_u32(rec + 8u);
        if (!it.model_id) continue;
        if (it.geo.kind == WNR4_KIND_NV && !dst_host && expand_e8m0_f32) continue;

        bool targets_ok = dst_host ? (dst_host[orig] != nullptr) : true;
        if (!dst_host) {
            for (int m = 0; m < 3; m++) {
                const uint8_t *ds = dst_scale[3 * orig + m];
                const uint8_t *dw = dst_weight[3 * orig + m];
                if (!ds || !dw || ((uintptr_t)dw & 15u)) targets_ok = false;
                if (expand_e8m0_f32 && ((uintptr_t)ds & 3u)) targets_ok = false;
            }
        }
        if (!targets_ok) continue;

        uint32_t reg_kind = 0u;
        if (!wnm4_mx_is_registered(it.model_id, &reg_kind)) {
            status[orig] = 1;
            continue;
        }
        if (reg_kind != it.geo.kind) {
            status[orig] = -1;
            continue;
        }
        it.weight_base = (entry->scale_bytes + 15u) & ~15u;
        valid_items.push_back(it);
    }

    if (valid_items.empty()) return;

    DeviceGuard guard;
    if (!guard.set(device)) {
        for (const auto &it : valid_items) status[it.orig] = 1;
        return;
    }

    WorkspaceLease<Wnr4MxSet> lease(wnr4_mx_pool(), device);
    if (!lease.w) {
        for (const auto &it : valid_items) status[it.orig] = 1;
        return;
    }
    Wnr4MxSet &set = *lease.w;

    /* Group by model_id (in practice all items share one model_id -> 1 iteration). */
    std::vector<uint8_t> done_flag(valid_items.size(), 0u);
    for (size_t g0 = 0; g0 < valid_items.size(); g0++) {
        if (done_flag[g0]) continue;
        const uint32_t mid = valid_items[g0].model_id;
        std::vector<Wnr4MxItem> group;
        for (size_t k = g0; k < valid_items.size(); k++) {
            if (!done_flag[k] && valid_items[k].model_id == mid) {
                done_flag[k] = 1u;
                group.push_back(valid_items[k]);
            }
        }

        const Wnr4Model *model = nullptr;
        Wnr4CudaTables tab = {};
        if (wnm4_mx_get(mid, device, &model, &tab)) {
            for (const auto &it : group) status[it.orig] = 1;
            continue;
        }

        const size_t gn = group.size();
        const bool two_lane = (gn >= 4u);

        if (!two_lane) {
            if (!set.lanes[0].bind_device(device)) {
                for (const auto &it : group) status[it.orig] = 1;
                continue;
            }
            cudaStream_t st = (!dst_host && caller_stream) ? (cudaStream_t)caller_stream : set.lanes[0].stream;
            Wnr4MxLaneRun run0;
            if (!wnr4_mx_enqueue_lane(set.lanes[0], st, nullptr, model, tab, group, 0u, gn,
                                      entries, records, dst_scale, dst_weight, dst_host,
                                      scratch_dst, expand_e8m0_f32, status, run0)) {
                for (const auto &it : group) if (status[it.orig] != -1) status[it.orig] = 1;
                continue;
            }
            if (!run0.launched) continue;

            const bool ok0 = !run0.broken && (cudaStreamSynchronize(st) == cudaSuccess);
            if (!ok0) {
                (void)cudaGetLastError();
                (void)cudaStreamSynchronize(st);
                set.lanes[0].err_dirty = true;
                for (size_t r = 0; r < run0.item_idx.size(); r++)
                    status[group[run0.item_idx[r]].orig] = 1;
                continue;
            }

            if (!dst_host || scratch_dst) {
                for (size_t r = 0; r < run0.item_idx.size(); r++) {
                    const uint32_t err = set.lanes[0].herr[r];
                    if (err) set.lanes[0].err_dirty = true;
                    status[group[run0.item_idx[r]].orig] = err ? -1 : 0;
                }
            } else {
                /* Strict host mode (scratch_dst == 0): D2H only clean records! */
                bool any_clean = false;
                for (size_t r = 0; r < run0.item_idx.size(); r++) {
                    const Wnr4MxItem &it = group[run0.item_idx[r]];
                    if (set.lanes[0].herr[r]) {
                        set.lanes[0].err_dirty = true;
                        status[it.orig] = -1;
                        continue;
                    }
                    if (wnr4_mx_copy_back_host(entries[it.orig], it.weight_base,
                                               set.lanes[0].draw + run0.draw_off[r],
                                               dst_host[it.orig], st) != cudaSuccess) {
                        (void)cudaGetLastError();
                        status[it.orig] = 1;
                        continue;
                    }
                    status[it.orig] = -2;
                    any_clean = true;
                }
                const bool copied = any_clean ? (cudaStreamSynchronize(st) == cudaSuccess) : true;
                if (!copied) (void)cudaGetLastError();
                for (size_t r = 0; r < run0.item_idx.size(); r++) {
                    const int orig = group[run0.item_idx[r]].orig;
                    if (status[orig] == -2) status[orig] = copied ? 0 : 1;
                }
            }
            continue;
        }

        /* 2-lane breadth-first overlap for valid_count >= 4. */
        if (!set.lanes[0].bind_device(device) || !set.lanes[1].bind_device(device)) {
            for (const auto &it : group) status[it.orig] = 1;
            continue;
        }
        cudaEvent_t wait_ev = nullptr;
        if (!dst_host && caller_stream) {
            if (!set.ensure_entry_event(device) ||
                cudaEventRecord(set.e_in, (cudaStream_t)caller_stream) != cudaSuccess) {
                (void)cudaGetLastError();
                for (const auto &it : group) status[it.orig] = 1;
                continue;
            }
            wait_ev = set.e_in;
        }

        const size_t n0 = gn / 2u;
        Wnr4MxLaneRun run[2];
        const bool enq0 = wnr4_mx_enqueue_lane(set.lanes[0], set.lanes[0].stream, wait_ev,
                                               model, tab, group, 0u, n0,
                                               entries, records, dst_scale, dst_weight, dst_host,
                                               scratch_dst, expand_e8m0_f32, status, run[0]);
        const bool enq1 = wnr4_mx_enqueue_lane(set.lanes[1], set.lanes[1].stream, wait_ev,
                                               model, tab, group, n0, gn,
                                               entries, records, dst_scale, dst_weight, dst_host,
                                               scratch_dst, expand_e8m0_f32, status, run[1]);

        if (!enq0) {
            for (size_t k = 0; k < n0; k++) if (status[group[k].orig] != -1) status[group[k].orig] = 1;
        }
        if (!enq1) {
            for (size_t k = n0; k < gn; k++) if (status[group[k].orig] != -1) status[group[k].orig] = 1;
        }

        if (!dst_host || scratch_dst) {
            for (int l = 0; l < 2; l++) {
                if (!run[l].launched) continue;
                const bool okl = !run[l].broken && (cudaStreamSynchronize(run[l].st) == cudaSuccess);
                if (!okl) {
                    (void)cudaGetLastError();
                    (void)cudaStreamSynchronize(run[l].st);
                    set.lanes[l].err_dirty = true;
                    for (size_t r = 0; r < run[l].item_idx.size(); r++)
                        status[group[run[l].item_idx[r]].orig] = 1;
                    continue;
                }
                for (size_t r = 0; r < run[l].item_idx.size(); r++) {
                    const uint32_t err = set.lanes[l].herr[r];
                    if (err) set.lanes[l].err_dirty = true;
                    status[group[run[l].item_idx[r]].orig] = err ? -1 : 0;
                }
            }
        } else {
            /* Strict host mode: sync lane 0 and kick off its clean D2H while lane 1 finishes decode! */
            bool any_copy[2] = {false, false};
            for (int l = 0; l < 2; l++) {
                if (!run[l].launched) continue;
                const bool okl = !run[l].broken && (cudaStreamSynchronize(run[l].st) == cudaSuccess);
                if (!okl) {
                    (void)cudaGetLastError();
                    (void)cudaStreamSynchronize(run[l].st);
                    set.lanes[l].err_dirty = true;
                    for (size_t r = 0; r < run[l].item_idx.size(); r++)
                        status[group[run[l].item_idx[r]].orig] = 1;
                    continue;
                }
                for (size_t r = 0; r < run[l].item_idx.size(); r++) {
                    const Wnr4MxItem &it = group[run[l].item_idx[r]];
                    if (set.lanes[l].herr[r]) {
                        set.lanes[l].err_dirty = true;
                        status[it.orig] = -1;
                        continue;
                    }
                    if (wnr4_mx_copy_back_host(entries[it.orig], it.weight_base,
                                               set.lanes[l].draw + run[l].draw_off[r],
                                               dst_host[it.orig], run[l].st) != cudaSuccess) {
                        (void)cudaGetLastError();
                        status[it.orig] = 1;
                        continue;
                    }
                    status[it.orig] = -2;
                    any_copy[l] = true;
                }
            }
            for (int l = 0; l < 2; l++) {
                if (!any_copy[l]) continue;
                const bool copied = (cudaStreamSynchronize(run[l].st) == cudaSuccess);
                if (!copied) (void)cudaGetLastError();
                for (size_t r = 0; r < run[l].item_idx.size(); r++) {
                    const int orig = group[run[l].item_idx[r]].orig;
                    if (status[orig] == -2) status[orig] = copied ? 0 : 1;
                }
            }
        }
    }
}

/* ---- WNR4-I4 pooled workspace & decode ---------------------------------- */

struct Wnr4I4Slot {
    unsigned char *drec = nullptr; size_t drec_cap = 0;
    unsigned char *draw = nullptr; size_t draw_cap = 0;
    uint32_t *derr = nullptr, *herr = nullptr; size_t derr_cap = 0, herr_cap = 0;
    Wnr4I4CudaRec *direc = nullptr, *hirec = nullptr; size_t direc_cap = 0, hirec_cap = 0;
    Wnr4I4CudaJob *dijob = nullptr, *hijob = nullptr; size_t dijob_cap = 0, hijob_cap = 0;
    cudaStream_t own = nullptr;
    bool err_dirty = true;
    int device = -1;
    std::vector<Wnr4I4CudaRec> irecs;
    std::vector<Wnr4I4CudaJob> by_m[3], jobs;

    int device_id() const { return device; }

    ~Wnr4I4Slot() { clear(); }

    void clear() {
        if (own) { (void)cudaStreamSynchronize(own); cudaStreamDestroy(own); }
        if (drec) cudaFree(drec);
        if (draw) cudaFree(draw);
        if (derr) cudaFree(derr);
        if (direc) cudaFree(direc);
        if (dijob) cudaFree(dijob);
        if (herr) cudaFreeHost(herr);
        if (hirec) cudaFreeHost(hirec);
        if (hijob) cudaFreeHost(hijob);
        drec = draw = nullptr;
        derr = herr = nullptr;
        direc = hirec = nullptr;
        dijob = hijob = nullptr;
        own = nullptr;
        drec_cap = draw_cap = derr_cap = herr_cap = 0;
        direc_cap = hirec_cap = dijob_cap = hijob_cap = 0;
        err_dirty = true;
        device = -1;
        (void)cudaGetLastError();
    }

    bool bind_device(int dev) {
        if (device == dev && own) return true;
        if (device >= 0 && device != dev) {
            (void)cudaSetDevice(device);
            clear();
            (void)cudaSetDevice(dev);
        } else if (!own) {
            clear();
        }
        if (cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking) != cudaSuccess) {
            (void)cudaGetLastError();
            own = nullptr;
            return false;
        }
        device = dev;
        err_dirty = true;
        return true;
    }

    bool grow_err(size_t n) {
        const size_t want = n < 8u ? 8u : n;
        const size_t prev = derr_cap;
        if (!grow_dev(derr, derr_cap, want) || !grow_host(herr, herr_cap, want)) return false;
        if (derr_cap != prev) err_dirty = true;
        return true;
    }

    cudaError_t clear_err_if_dirty(cudaStream_t st) {
        if (!err_dirty) return cudaSuccess;
        cudaError_t e = cudaMemsetAsync(derr, 0, derr_cap * 4u, st);
        if (e == cudaSuccess) err_dirty = false;
        return e;
    }
};

static WorkspacePool<Wnr4I4Slot> &wnr4_i4_pool() {
    static auto *p = new WorkspacePool<Wnr4I4Slot>();
    return *p;
}

static cudaError_t copy_records_coalesced(unsigned char *drec,
                                          const std::vector<int> &origs,
                                          const std::vector<size_t> &at,
                                          const uint8_t *const *records,
                                          const size_t *totals,
                                          cudaStream_t st) {
    cudaError_t e = cudaSuccess;
    const size_t m = origs.size();
    for (size_t i = 0; i < m && e == cudaSuccess;) {
        const uint8_t *src0 = records[origs[i]];
        const size_t dst0 = at[i];
        size_t len = totals[origs[i]];
        size_t j = i + 1u;
        while (j < m && records[origs[j]] == src0 + len && at[j] == dst0 + len) {
            len += totals[origs[j]];
            j++;
        }
        if (len) e = cudaMemcpyAsync(drec + dst0, src0, len, cudaMemcpyHostToDevice, st);
        i = j;
    }
    return e;
}

/* ---- WNR4-GG pooled workspace & decode ---------------------------------- */

struct Wnr4GgSlot {
    unsigned char *drec = nullptr; size_t drec_cap = 0;
    unsigned char *draw = nullptr; size_t draw_cap = 0;
    uint32_t *derr = nullptr, *herr = nullptr; size_t derr_cap = 0, herr_cap = 0;
    Wnr4GgCudaJob *dgjob = nullptr, *hgjob = nullptr; size_t dgjob_cap = 0, hgjob_cap = 0;
    Wnr4GgCudaCopy *dcopy = nullptr, *hcopy = nullptr; size_t dcopy_cap = 0, hcopy_cap = 0;
    cudaStream_t own = nullptr;
    bool err_dirty = true;
    int device = -1;
    std::vector<Wnr4GgCudaJob> by_sub[2], jobs;
    std::vector<Wnr4GgCudaCopy> copies;

    int device_id() const { return device; }

    ~Wnr4GgSlot() { clear(); }

    void clear() {
        if (own) { (void)cudaStreamSynchronize(own); cudaStreamDestroy(own); }
        if (drec) cudaFree(drec);
        if (draw) cudaFree(draw);
        if (derr) cudaFree(derr);
        if (dgjob) cudaFree(dgjob);
        if (dcopy) cudaFree(dcopy);
        if (herr) cudaFreeHost(herr);
        if (hgjob) cudaFreeHost(hgjob);
        if (hcopy) cudaFreeHost(hcopy);
        drec = draw = nullptr;
        derr = herr = nullptr;
        dgjob = hgjob = nullptr;
        dcopy = hcopy = nullptr;
        own = nullptr;
        drec_cap = draw_cap = derr_cap = herr_cap = 0;
        dgjob_cap = hgjob_cap = dcopy_cap = hcopy_cap = 0;
        err_dirty = true;
        device = -1;
        (void)cudaGetLastError();
    }

    bool bind_device(int dev) {
        if (device == dev && own) return true;
        if (device >= 0 && device != dev) {
            (void)cudaSetDevice(device);
            clear();
            (void)cudaSetDevice(dev);
        } else if (!own) {
            clear();
        }
        if (cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking) != cudaSuccess) {
            (void)cudaGetLastError();
            own = nullptr;
            return false;
        }
        device = dev;
        err_dirty = true;
        return true;
    }

    bool grow_err(size_t n) {
        const size_t want = n < 8u ? 8u : n;
        const size_t prev = derr_cap;
        if (!grow_dev(derr, derr_cap, want) || !grow_host(herr, herr_cap, want)) return false;
        if (derr_cap != prev) err_dirty = true;
        return true;
    }

    cudaError_t clear_err_if_dirty(cudaStream_t st) {
        if (!err_dirty) return cudaSuccess;
        cudaError_t e = cudaMemsetAsync(derr, 0, derr_cap * 4u, st);
        if (e == cudaSuccess) err_dirty = false;
        return e;
    }
};

static WorkspacePool<Wnr4GgSlot> &wnr4_gg_pool() {
    static auto *p = new WorkspacePool<Wnr4GgSlot>();
    return *p;
}

/* ---- vtable entry point implementations --------------------------------- */

static int cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return count > 0 ? count : 0;
}

static int cuda_host_register(void *ptr, size_t bytes) {
    if (!ptr || !bytes) return -1;
    if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        return 1;
    }
    return 0;
}

static int cuda_host_unregister(void *ptr) {
    if (!ptr) return -1;
    if (cudaHostUnregister(ptr) != cudaSuccess) {
        (void)cudaGetLastError();
        return 1;
    }
    return 0;
}

static int cuda_register_model(int kind, const uint8_t *blob, size_t bytes) {
    if (!blob || !bytes) return -1;
    switch (kind) {
        case 0: return reg_wnm2(blob, bytes);
        case 1: return reg_wnm4_mx(blob, bytes);
        case 2: return reg_wnm4_gg(blob, bytes);
        case 3: return reg_wnm4_i4(blob, bytes);
        default: return -1;
    }
}

static int cuda_decode_expert_common(int n, const WexEntry *const *entries,
                                     const uint8_t *const *records, int device,
                                     uint8_t *const *dst_scale, uint8_t *const *dst_weight,
                                     uint8_t *const *dst_host, int scratch_dst,
                                     void *stream, int expand_e8m0_f32, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    if (!entries || !records || !(dst_host || (dst_scale && dst_weight))) {
        for (int i = 0; i < n; i++) status[i] = -1;
        return n;
    }

    std::vector<int> nr3_idx;
    std::vector<int> wnr4_idx;
    for (int i = 0; i < n; i++) {
        status[i] = -1;
        const WexEntry *e = entries[i];
        if (!e) continue;
        if (wex_entry_nr_model(e)) {
            nr3_idx.push_back(i);
        } else if (e->flags == WEX_ENTRY_WNR4 || e->flags == WEX_ENTRY_WNR4_NV) {
            wnr4_idx.push_back(i);
        }
    }

    if (!nr3_idx.empty()) {
        decode_nr3_subset(nr3_idx, entries, records, device,
                          dst_scale, dst_weight, dst_host, scratch_dst,
                          stream, expand_e8m0_f32, status);
    }
    if (!wnr4_idx.empty()) {
        decode_wnr4_mx_subset(wnr4_idx, entries, records, device,
                              dst_scale, dst_weight, dst_host, scratch_dst,
                              stream, expand_e8m0_f32, status);
    }

    int bad = 0;
    for (int i = 0; i < n; i++) bad += (status[i] != 0);
    return bad;
}

static int cuda_decode_expert_host(int n, const WexEntry *const *entries,
                                   const uint8_t *const *records, int device,
                                   uint8_t *const *dst_slabs, int scratch_dst, int *status) {
    return cuda_decode_expert_common(n, entries, records, device,
                                     nullptr, nullptr, dst_slabs, scratch_dst,
                                     nullptr, 0, status);
}

static int cuda_decode_expert_device(int n, const WexEntry *const *entries,
                                     const uint8_t *const *records, int device,
                                     uint8_t *const *dst_scale, uint8_t *const *dst_weight,
                                     void *stream, int expand_e8m0_f32, int *status) {
    return cuda_decode_expert_common(n, entries, records, device,
                                     dst_scale, dst_weight, nullptr, 0,
                                     stream, expand_e8m0_f32, status);
}

static int cuda_decode_i4_batch(int n, const wex_dev_i4_geo_t *geo,
                                const uint8_t *const *records, const size_t *totals,
                                const uint32_t *owners, int device,
                                uint8_t *const *dst, int dst_is_device,
                                int scratch_dst, void *stream, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    Wnr4I4Geo g;
    if (!geo || !records || !totals || !owners || !dst ||
        wnr4_i4_geo(&g, geo->nmat, geo->rows, geo->cols, geo->pairs)) {
        for (int i = 0; i < n; i++) status[i] = -1;
        return n;
    }

    struct I4Parsed {
        int orig;
        Wnr4I4Record r;
    };
    std::vector<I4Parsed> valid;
    valid.reserve((size_t)n);

    for (int i = 0; i < n; i++) {
        status[i] = -1;
        if (!records[i] || !dst[i]) continue;
        if (dst_is_device && ((uintptr_t)dst[i] & 3u)) continue;
        Wnr4I4Record parsed;
        if (wnr4_i4_parse(records[i], totals[i], &g, &parsed) ||
            parsed.owner != owners[i] || parsed.G > WNR4_I4_CUDA_MAX_G) {
            continue;
        }
        valid.push_back({i, parsed});
    }

    auto count_bad = [&]() -> int {
        int bad = 0;
        for (int i = 0; i < n; i++) bad += (status[i] != 0);
        return bad;
    };
    if (valid.empty()) return count_bad();

    DeviceGuard guard;
    if (!guard.set(device)) {
        for (const auto &v : valid) status[v.orig] = 1;
        return count_bad();
    }
    WorkspaceLease<Wnr4I4Slot> lease(wnr4_i4_pool(), device);
    if (!lease.w || !lease.w->bind_device(device)) {
        for (const auto &v : valid) status[v.orig] = 1;
        return count_bad();
    }
    Wnr4I4Slot &s = *lease.w;
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;
    const size_t slab_stride = align_up((size_t)g.raw, 256u);

    std::vector<uint8_t> seen(valid.size(), 0u);
    for (size_t g0 = 0; g0 < valid.size(); g0++) {
        if (seen[g0]) continue;
        const uint32_t mid = valid[g0].r.model_id;
        std::vector<I4Parsed> grp;
        for (size_t k = g0; k < valid.size(); k++) {
            if (!seen[k] && valid[k].r.model_id == mid) {
                seen[k] = 1u;
                grp.push_back(valid[k]);
            }
        }

        const Wnr4I4Model *model = nullptr;
        const unsigned char *mdev = nullptr;
        if (wnm4_i4_get(mid, device, &model, &mdev)) {
            for (const auto &v : grp) status[v.orig] = 1;
            continue;
        }

        const size_t nvalid = grp.size();
        size_t all_rec = 0;
        std::vector<int> origs(nvalid);
        std::vector<size_t> at(nvalid);
        for (size_t r = 0; r < nvalid; r++) {
            origs[r] = grp[r].orig;
            at[r] = all_rec;
            all_rec += align_up(totals[grp[r].orig], 256u);
        }
        if (!grow_dev(s.drec, s.drec_cap, all_rec) ||
            !s.grow_err(nvalid) ||
            (!dst_is_device && !grow_dev(s.draw, s.draw_cap, slab_stride * nvalid))) {
            for (const auto &v : grp) status[v.orig] = 1;
            continue;
        }

        bool uniform_G = true;
        const uint32_t batch_G = grp[0].r.G;
        const uint32_t ntiles_rec = grp[0].r.ntiles;
        s.irecs.clear();
        s.irecs.reserve(nvalid);
        for (size_t r = 0; r < nvalid; r++) {
            const Wnr4I4Record &pr = grp[r].r;
            if (pr.G != batch_G) uniform_G = false;
            Wnr4I4CudaRec rec = {};
            rec.rec = s.drec + at[r];
            rec.dst = dst_is_device ? dst[grp[r].orig] : (s.draw + slab_stride * r);
            rec.owner = pr.owner;
            rec.rec_idx = (uint32_t)r;
            rec.bases_at = (uint32_t)(pr.bases - records[grp[r].orig]);
            rec.data_at = (uint32_t)(pr.data - records[grp[r].orig]);
            rec.payload = pr.payload;
            for (uint32_t k = 0; k < 3u; k++) rec.anchor[k] = (uint16_t)pr.anchor[k];
            s.irecs.push_back(rec);
        }

        cudaError_t e = cudaSuccess;
        std::vector<uint8_t> planned_ok(nvalid, 1u);
        if (uniform_G) {
            const uint32_t W = wnr4_i4_cuda_choose_warps((size_t)ntiles_rec * nvalid);
            if (W != 1u && W != 2u && W != 4u && W != 8u && W != 16u) {
                for (const auto &v : grp) status[v.orig] = 1;
                continue;
            }
            Wnr4I4CudaBatchGeo bg = {};
            bg.nmat = g.nmat;
            bg.G = batch_G;
            bg.pairs = g.pairs;
            bg.nrecs = (uint32_t)nvalid;
            for (uint32_t k = 0; k < 3u; k++) {
                bg.cells[k] = g.cells[k];
                bg.gc[k] = g.gc[k];
                bg.nbc[k] = g.nbc[k];
                bg.bbase[k] = g.bbase[k];
                bg.coff[k] = g.coff[k];
                bg.soff[k] = g.soff[k];
            }
            for (uint32_t k = 0; k < 4u; k++) bg.tile_base[k] = grp[0].r.tile_base[k];
            uint32_t blk = 0u;
            for (uint32_t k = 0; k < 3u; k++) {
                bg.mat_blocks[k] = blk;
                if (k < g.nmat) {
                    const uint32_t mt = (bg.tile_base[k + 1u] - bg.tile_base[k]) * (uint32_t)nvalid;
                    blk += (mt + W - 1u) / W;
                }
            }
            bg.mat_blocks[3] = blk;

            if (nvalid > WNR4_I4_CUDA_INLINE_RECS) {
                if (!grow_dev(s.direc, s.direc_cap, nvalid) || !grow_host(s.hirec, s.hirec_cap, nvalid)) {
                    for (const auto &v : grp) status[v.orig] = 1;
                    continue;
                }
                std::memcpy(s.hirec, s.irecs.data(), sizeof(Wnr4I4CudaRec) * nvalid);
            }
            e = copy_records_coalesced(s.drec, origs, at, records, totals, st);
            if (e == cudaSuccess && nvalid > WNR4_I4_CUDA_INLINE_RECS)
                e = cudaMemcpyAsync(s.direc, s.hirec, sizeof(Wnr4I4CudaRec) * nvalid, cudaMemcpyHostToDevice, st);
            if (e == cudaSuccess) e = s.clear_err_if_dirty(st);
            if (e == cudaSuccess) {
                if (nvalid <= WNR4_I4_CUDA_INLINE_RECS) {
                    Wnr4I4InlineRecs irecs = {};
                    std::memcpy(irecs.r, s.irecs.data(), sizeof(Wnr4I4CudaRec) * nvalid);
                    e = wnr4_i4_cuda_launch_recs(mdev, model->id, bg, irecs, W, s.derr, st);
                } else {
                    e = wnr4_i4_cuda_launch_recs_ptr(mdev, model->id, bg, s.direc, W, s.derr, st);
                }
            }
        } else {
            /* Mixed-G fallback: per-tile job planning. */
            for (uint32_t k = 0; k < 3u; k++) s.by_m[k].clear();
            s.jobs.clear();
            bool any_planned = false;
            for (size_t r = 0; r < nvalid; r++) {
                unsigned char *dst_dev = dst_is_device ? dst[grp[r].orig] : (s.draw + slab_stride * r);
                if (wnr4_i4_cuda_plan(&g, model, records[grp[r].orig], totals[grp[r].orig],
                                      owners[grp[r].orig], s.drec + at[r], (uint32_t)r, dst_dev, s.by_m)) {
                    planned_ok[r] = 0u;
                    status[grp[r].orig] = -1;
                } else {
                    any_planned = true;
                }
            }
            if (!any_planned) continue;
            const uint32_t W = wnr4_i4_cuda_jobs(s.by_m, s.jobs);
            if (!grow_dev(s.dijob, s.dijob_cap, s.jobs.size() + 1u) ||
                !grow_host(s.hijob, s.hijob_cap, s.jobs.size() + 1u)) {
                for (size_t r = 0; r < nvalid; r++) if (planned_ok[r]) status[grp[r].orig] = 1;
                continue;
            }
            if (!s.jobs.empty()) std::memcpy(s.hijob, s.jobs.data(), sizeof(Wnr4I4CudaJob) * s.jobs.size());
            e = copy_records_coalesced(s.drec, origs, at, records, totals, st);
            if (e == cudaSuccess && !s.jobs.empty())
                e = cudaMemcpyAsync(s.dijob, s.hijob, sizeof(Wnr4I4CudaJob) * s.jobs.size(), cudaMemcpyHostToDevice, st);
            if (e == cudaSuccess) e = s.clear_err_if_dirty(st);
            if (e == cudaSuccess && !s.jobs.empty())
                e = wnr4_i4_cuda_launch(mdev, model->id, s.dijob, (uint32_t)s.jobs.size(), W, s.derr, st);
        }

        if (e == cudaSuccess && !dst_is_device && scratch_dst) {
            for (size_t r = 0; r < nvalid && e == cudaSuccess; r++) {
                if (!planned_ok[r]) continue;
                e = cudaMemcpyAsync(dst[grp[r].orig], s.draw + slab_stride * r,
                                    (size_t)g.raw, cudaMemcpyDeviceToHost, st);
            }
        }
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s.herr, s.derr, 4u * nvalid, cudaMemcpyDeviceToHost, st);
        const bool ok = (e == cudaSuccess) && (cudaStreamSynchronize(st) == cudaSuccess);
        if (!ok) {
            (void)cudaGetLastError();
            (void)cudaStreamSynchronize(st);
            s.err_dirty = true;
            for (size_t r = 0; r < nvalid; r++) {
                if (planned_ok[r]) status[grp[r].orig] = 1;
            }
            continue;
        }

        if (dst_is_device || scratch_dst) {
            for (size_t r = 0; r < nvalid; r++) {
                if (!planned_ok[r]) continue;
                const uint32_t err = s.herr[r];
                if (err) s.err_dirty = true;
                status[grp[r].orig] = err ? -1 : 0;
            }
        } else {
            /* Strict host mode: copy back ONLY clean records. */
            bool any_clean = false;
            for (size_t r = 0; r < nvalid; r++) {
                if (!planned_ok[r]) continue;
                const int orig = grp[r].orig;
                if (s.herr[r]) {
                    s.err_dirty = true;
                    status[orig] = -1;
                    continue;
                }
                if (cudaMemcpyAsync(dst[orig], s.draw + slab_stride * r,
                                    (size_t)g.raw, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                    (void)cudaGetLastError();
                    status[orig] = 1;
                    continue;
                }
                status[orig] = -2;
                any_clean = true;
            }
            const bool copied = any_clean ? (cudaStreamSynchronize(st) == cudaSuccess) : true;
            if (!copied) (void)cudaGetLastError();
            for (size_t r = 0; r < nvalid; r++) {
                if (!planned_ok[r]) continue;
                const int orig = grp[r].orig;
                if (status[orig] == -2) status[orig] = copied ? 0 : 1;
            }
        }
    }

    return count_bad();
}

static int cuda_decode_gg_batch(int n, const wex_dev_gg_geo_t *geos,
                                const uint8_t *const *records, const size_t *totals,
                                const uint32_t *owners, int device,
                                uint8_t *const *dst, int dst_is_device,
                                int scratch_dst, void *stream, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    if (!geos || !records || !totals || !owners || !dst) {
        for (int i = 0; i < n; i++) status[i] = -1;
        return n;
    }

    struct GgItem {
        int orig;
        uint32_t model_id;
        Wnr4GgGeo geo;
    };
    std::vector<GgItem> valid;
    valid.reserve((size_t)n);

    for (int i = 0; i < n; i++) {
        status[i] = -1;
        if (!records[i] || !dst[i] || totals[i] < WNR4_HEADER_BYTES) continue;
        const wex_dev_gg_geo_t &dg = geos[i];
        if (!dg.nslice || dg.nslice > WNR4_GG_MAX_SEG) continue;
        uint32_t bytes32[WNR4_GG_MAX_SEG] = {};
        bool size_ok = true;
        for (uint32_t s = 0; s < dg.nslice; s++) {
            if (!dg.nbytes[s] || dg.nbytes[s] > UINT32_MAX) { size_ok = false; break; }
            bytes32[s] = (uint32_t)dg.nbytes[s];
        }
        if (!size_ok) continue;
        Wnr4GgGeo gg;
        if (wnr4_gg_geo(&gg, dg.nslice, dg.ggml_type, bytes32)) continue;
        /* Verify explicit offsets if populated by caller. */
        if (dg.offsets[0] != 0u) continue;
        bool off_ok = true;
        for (uint32_t s = 1u; s < dg.nslice; s++) {
            if (dg.offsets[s] != 0u && dg.offsets[s] != gg.off[s]) { off_ok = false; break; }
        }
        if (!off_ok) continue;

        if (wnr4_u32(records[i]) != WNR4_MAGIC) continue;
        const uint32_t mid = wnr4_u32(records[i] + 4u);
        if (!mid) continue;
        valid.push_back({i, mid, gg});
    }

    auto count_bad = [&]() -> int {
        int bad = 0;
        for (int i = 0; i < n; i++) bad += (status[i] != 0);
        return bad;
    };
    if (valid.empty()) return count_bad();

    DeviceGuard guard;
    if (!guard.set(device)) {
        for (const auto &v : valid) status[v.orig] = 1;
        return count_bad();
    }
    WorkspaceLease<Wnr4GgSlot> lease(wnr4_gg_pool(), device);
    if (!lease.w || !lease.w->bind_device(device)) {
        for (const auto &v : valid) status[v.orig] = 1;
        return count_bad();
    }
    Wnr4GgSlot &s = *lease.w;
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;

    std::vector<uint8_t> seen(valid.size(), 0u);
    for (size_t g0 = 0; g0 < valid.size(); g0++) {
        if (seen[g0]) continue;
        const uint32_t mid = valid[g0].model_id;
        std::vector<GgItem> grp;
        for (size_t k = g0; k < valid.size(); k++) {
            if (!seen[k] && valid[k].model_id == mid) {
                seen[k] = 1u;
                grp.push_back(valid[k]);
            }
        }

        const Wnr4GgModel *model = nullptr;
        const unsigned char *mdev = nullptr;
        if (wnm4_gg_get(mid, device, &model, &mdev)) {
            for (const auto &v : grp) status[v.orig] = 1;
            continue;
        }

        const size_t gn = grp.size();
        size_t all_rec = 0, all_raw = 0;
        std::vector<size_t> rec_at(gn), raw_at(gn);
        for (size_t k = 0; k < gn; k++) {
            rec_at[k] = all_rec;
            all_rec += align_up(totals[grp[k].orig], 256u);
            if (!dst_is_device) {
                raw_at[k] = all_raw;
                all_raw += align_up((size_t)grp[k].geo.raw, 256u);
            }
        }
        if (!grow_dev(s.drec, s.drec_cap, all_rec) ||
            !s.grow_err(gn) ||
            (!dst_is_device && !grow_dev(s.draw, s.draw_cap, all_raw))) {
            for (const auto &v : grp) status[v.orig] = 1;
            continue;
        }

        s.by_sub[0].clear();
        s.by_sub[1].clear();
        s.jobs.clear();
        s.copies.clear();

        std::vector<size_t> planned_idx;
        std::vector<int> planned_origs;
        std::vector<size_t> planned_rec_at;
        planned_idx.reserve(gn);
        planned_origs.reserve(gn);
        planned_rec_at.reserve(gn);

        for (size_t k = 0; k < gn; k++) {
            const int orig = grp[k].orig;
            const uint32_t local_r = (uint32_t)planned_idx.size();
            unsigned char *dst_dev = dst_is_device ? dst[orig] : (s.draw + raw_at[k]);
            if (wnr4_gg_cuda_plan(&grp[k].geo, model, records[orig], totals[orig], owners[orig],
                                  s.drec + rec_at[k], local_r, dst_dev, s.by_sub, s.copies)) {
                status[orig] = -1;
                continue;
            }
            planned_idx.push_back(k);
            planned_origs.push_back(orig);
            planned_rec_at.push_back(rec_at[k]);
        }

        const size_t nvalid = planned_idx.size();
        if (!nvalid) continue;

        const uint32_t W = wnr4_gg_cuda_jobs(s.by_sub, s.jobs);
        if (!grow_dev(s.dgjob, s.dgjob_cap, s.jobs.size() + 1u) ||
            !grow_host(s.hgjob, s.hgjob_cap, s.jobs.size() + 1u) ||
            !grow_dev(s.dcopy, s.dcopy_cap, s.copies.size() + 1u) ||
            !grow_host(s.hcopy, s.hcopy_cap, s.copies.size() + 1u)) {
            for (size_t r = 0; r < nvalid; r++) status[grp[planned_idx[r]].orig] = 1;
            continue;
        }
        if (!s.jobs.empty()) std::memcpy(s.hgjob, s.jobs.data(), sizeof(Wnr4GgCudaJob) * s.jobs.size());
        if (!s.copies.empty()) std::memcpy(s.hcopy, s.copies.data(), sizeof(Wnr4GgCudaCopy) * s.copies.size());

        size_t max_copy = 0;
        for (const auto &c : s.copies) if (c.bytes > max_copy) max_copy = c.bytes;

        cudaError_t e = copy_records_coalesced(s.drec, planned_origs, planned_rec_at, records, totals, st);
        if (e == cudaSuccess && !s.jobs.empty())
            e = cudaMemcpyAsync(s.dgjob, s.hgjob, sizeof(Wnr4GgCudaJob) * s.jobs.size(), cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess && !s.copies.empty())
            e = cudaMemcpyAsync(s.dcopy, s.hcopy, sizeof(Wnr4GgCudaCopy) * s.copies.size(), cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess) e = s.clear_err_if_dirty(st);
        if (e == cudaSuccess && !s.jobs.empty())
            e = wnr4_gg_cuda_launch(mdev, model->id, s.dgjob, (uint32_t)s.jobs.size(), W, s.derr, st);
        if (e == cudaSuccess && !s.copies.empty()) {
            size_t off = 0;
            while (off < s.copies.size() && e == cudaSuccess) {
                const uint32_t chunk = (s.copies.size() - off) < 65535u ? (uint32_t)(s.copies.size() - off) : 65535u;
                e = wnr4_gg_cuda_copy(s.dcopy + off, chunk, max_copy, st);
                off += chunk;
            }
        }
        if (e == cudaSuccess && !dst_is_device && scratch_dst) {
            for (size_t r = 0; r < nvalid && e == cudaSuccess; r++) {
                const size_t k = planned_idx[r];
                e = cudaMemcpyAsync(dst[grp[k].orig], s.draw + raw_at[k],
                                    (size_t)grp[k].geo.raw, cudaMemcpyDeviceToHost, st);
            }
        }
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s.herr, s.derr, 4u * nvalid, cudaMemcpyDeviceToHost, st);

        const bool ok = (e == cudaSuccess) && (cudaStreamSynchronize(st) == cudaSuccess);
        if (!ok) {
            (void)cudaGetLastError();
            (void)cudaStreamSynchronize(st);
            s.err_dirty = true;
            for (size_t r = 0; r < nvalid; r++) status[grp[planned_idx[r]].orig] = 1;
            continue;
        }

        if (dst_is_device || scratch_dst) {
            for (size_t r = 0; r < nvalid; r++) {
                const uint32_t err = s.herr[r];
                if (err) s.err_dirty = true;
                status[grp[planned_idx[r]].orig] = err ? -1 : 0;
            }
        } else {
            /* Strict host mode: copy back ONLY clean records. */
            bool any_clean = false;
            for (size_t r = 0; r < nvalid; r++) {
                const size_t k = planned_idx[r];
                const int orig = grp[k].orig;
                if (s.herr[r]) {
                    s.err_dirty = true;
                    status[orig] = -1;
                    continue;
                }
                if (cudaMemcpyAsync(dst[orig], s.draw + raw_at[k],
                                    (size_t)grp[k].geo.raw, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                    (void)cudaGetLastError();
                    status[orig] = 1;
                    continue;
                }
                status[orig] = -2;
                any_clean = true;
            }
            const bool copied = any_clean ? (cudaStreamSynchronize(st) == cudaSuccess) : true;
            if (!copied) (void)cudaGetLastError();
            for (size_t r = 0; r < nvalid; r++) {
                const int orig = grp[planned_idx[r]].orig;
                if (status[orig] == -2) status[orig] = copied ? 0 : 1;
            }
        }
    }

    return count_bad();
}

static const wex_dev_ops_t g_cuda_ops = {
    WEX_DEV_ABI_VERSION,
    WEX_DEV_CAP_PIN_HOST |
        WEX_DEV_CAP_WNR3 |
        WEX_DEV_CAP_WNR4_MX |
        WEX_DEV_CAP_WNR4_NV |
        WEX_DEV_CAP_WNR4_I4 |
        WEX_DEV_CAP_WNR4_GG |
        WEX_DEV_CAP_DIRECT_DEV |
        WEX_DEV_CAP_SCRATCH_HOST |
        WEX_DEV_CAP_E8M0_EXPAND,
    "cuda",
    cuda_device_count,
    cuda_host_register,
    cuda_host_unregister,
    cuda_register_model,
    cuda_decode_expert_host,
    cuda_decode_expert_device,
    cuda_decode_i4_batch,
    cuda_decode_gg_batch
};

} /* namespace */

extern "C" WEX_DEV_EXPORT const wex_dev_ops_t *wex_dev_get_ops(void) {
    return &g_cuda_ops;
}
