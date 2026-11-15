/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_CUDA_HOST_H
#define WEX_FUZZ_CUDA_HOST_H
/* The CUDA constructs that c/wex_cuda.cuh and c/tools/wex_cuda_bridge.cu use, emulated on the CPU, so that the
 * WNR3 kernels and the bridge build as host C++ (g++ or clang++) and run without a GPU or the CUDA toolkit.
 * It is included through the host copies that wex4/fuzz/cuda_host.py writes; the header comment of that script
 * lists the four constructs it rewrites. What this file does, construct by construct:
 *
 * Qualifiers. __global__, __device__ and __host__ are empty, __forceinline__ is inline, __launch_bounds__ is
 * dropped. The CUDA source is compiled once for every device build the caller wants: the caller defines
 * __CUDA_ARCH__ (and WEX_CUDA_NR2_LOOKUP) before it includes the copies, as nvcc's device pass would.
 *
 * Grid. A launch is queued on its stream (below) and runs when the host waits for it. The blocks run one after
 * the other, in a random order drawn from wex_shim_seed. The threads of a block run at the same time, one host
 * thread (pthreads) each, and __syncthreads() is a barrier of those threads; a thread that returned no longer
 * counts (what the hardware does since Volta). For each kernel the first block it ever runs decides: when no
 * thread of that block reached a __syncthreads(), the kernel's other blocks, in that launch and later ones, run
 * thread by thread on the calling thread, and a __syncthreads() there stops the program. threadIdx, blockIdx,
 * blockDim and gridDim are thread-local.
 * The launch is checked as the CUDA runtime checks it (block of 1 to 1024 threads, grid not empty, dynamic
 * shared memory above 48 KiB only with cudaFuncSetAttribute and never above the device's limit), and every
 * pointer argument must be null or point into memory from cudaMalloc.
 *
 * Shared memory. Each __shared__ array and the dynamic shared memory of a launch are allocated for each block
 * and freed after it, filled with 0xA5 first, so that a decode that reads a byte the block did not write gets
 * a wrong byte instead of the right one by chance. All threads of a block get the same memory. The static arrays
 * of a block may use at most 48 KiB, static and dynamic together at most the device's limit. The fill is the
 * same under MemorySanitizer: the kernels read shared and device memory that nothing wrote when a record is
 * malformed (a stream that runs past its last word reads stale ring slots, c/wex_cuda.cuh says so where it opens
 * a stream; kernel B reads the scales that kernel A left unwritten after an error), the error word catches
 * those reads, and MemorySanitizer would report them all.
 *
 * Intrinsics. __ldg is a plain load; __byte_perm, __funnelshift_l, __popc, __vmaxu2, __vminu2, min, max,
 * atomicOr, atomicExch and make_uint4 do what the CUDA documentation says. The inline PTX of wex_cuda.cuh:
 *   prmt.b32 (default mode, the selector's bit 3 replicates the sign of the byte) is computed exactly;
 *   cp.async.ca.shared.global of 4 bytes is queued in the thread's current group, and its destination word is
 *   overwritten with 0x9E3779B9 at once, so that a read before its wait sees a wrong word instead of the old
 *   one; cp.async.commit_group closes the group, cp.async.wait_group N copies every group but the N newest. A
 *   copy still queued when the thread returns is made then. __cvta_generic_to_shared returns a handle that
 *   only the emulated cp.async takes, and the destination must lie in the block's shared memory.
 * Warp-level operations (shuffles, votes, warp-synchronous code) are not emulated: wex_cuda.cuh has none, and
 * a kernel that relied on threads of a warp running in lockstep would not get it here.
 *
 * Runtime. One device, number 0; its compute capability and shared-memory limit are set by the caller
 * (wex_shim_set_device). cudaMalloc and cudaMallocHost return host memory (256-byte aligned), device memory
 * filled with 0xA5. Copies, memsets, kernels and event records are queued per stream and run only when the
 * host waits: cudaStreamSynchronize, cudaDeviceSynchronize, and cudaFree and cudaFreeHost, which wait for the
 * whole device as the CUDA runtime does. While waiting, the next operation is drawn at random among the streams
 * whose first operation may run (in stream order; cudaStreamWaitEvent waits for the last cudaEventRecord of
 * that event before it), so work of other streams runs before or after, as on a GPU. Each operation runs whole:
 * a kernel never runs while a copy of another stream is half done. A copy reads its source when it runs, not
 * when it is queued. cudaMemcpyAsync checks its kind against the memory it is given. A destroyed stream is
 * freed once its queued work has run. Not emulated: an operation that overlaps another one, the legacy default
 * stream's implicit synchronization (stream 0 is an ordinary stream here; the bridge uses non-blocking streams
 * only), several devices, peer access, timing, memory that is mapped or managed, and the GPU's memory model
 * beyond the order of the operations.
 *
 * What a run proves is the decoding logic and the order of the host's calls, not the GPU, its compiler or its
 * timing. */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#undef min
#undef max

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#ifndef _WIN32
#define __declspec(x)
#endif

/* ---- types ---------------------------------------------------------------- */
struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int a = 1u, unsigned int b = 1u, unsigned int c = 1u) : x(a), y(b), z(c) {}
};
struct alignas(16) uint4 { unsigned int x, y, z, w; };
static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    uint4 r;
    r.x = x; r.y = y; r.z = z; r.w = w;
    return r;
}

enum cudaError {
    cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidDevice = 101, cudaErrorInvalidResourceHandle = 400
};
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76 };
#define cudaStreamNonBlocking 0x01u
#define cudaEventDisableTiming 0x02u
#define cudaHostRegisterPortable 0x01u
typedef struct WexShimStream *cudaStream_t;
typedef struct WexShimEvent *cudaEvent_t;

/* ---- state ------------------------------------------------------------------ */
__attribute__((noreturn)) static void wex_shim_fail(const char *what) {
    fprintf(stderr, "ORACLE FAILED: CUDA emulation: %s\n", what);
    abort();
}

struct WexShimDevice { int major, minor; size_t shared_optin; };
static WexShimDevice wex_shim_device = {8, 6, 101376u};   /* compute 8.6: 99 KiB of shared memory per block */
static inline void wex_shim_set_device(int major, int minor, size_t shared_optin) {
    wex_shim_device.major = major; wex_shim_device.minor = minor; wex_shim_device.shared_optin = shared_optin;
}
#define WEX_SHIM_STATIC_SHARED 49152u

static uint64_t wex_shim_rng_state = UINT64_C(0x9e3779b97f4a7c15);
static inline void wex_shim_seed(uint64_t seed) { wex_shim_rng_state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15); }
static inline uint32_t wex_shim_rng(void) {
    uint64_t x = wex_shim_rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    wex_shim_rng_state = x;
    return (uint32_t)(x >> 32);
}

/* what the emulation ran, for the harness's summary */
struct WexShimStats {
    unsigned long long launches, blocks, threads, barriers, copies, async_copies, pooled_blocks;
};
static WexShimStats wex_shim_stats;

static cudaError_t wex_shim_last_error = cudaSuccess;
static inline cudaError_t wex_shim_error(cudaError_t e) {
    if (e != cudaSuccess) wex_shim_last_error = e;
    return e;
}
static inline cudaError_t cudaGetLastError(void) {
    cudaError_t e = wex_shim_last_error;
    wex_shim_last_error = cudaSuccess;
    return e;
}
static inline cudaError_t cudaPeekAtLastError(void) { return wex_shim_last_error; }
static inline const char *cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "emulated CUDA error"; }

/* ---- memory ------------------------------------------------------------------- */
struct WexShimBlock { unsigned char *p; size_t n; int device; };
static WexShimBlock *wex_shim_memory;
static size_t wex_shim_memory_count, wex_shim_memory_cap;

static void *wex_shim_aligned(size_t n) {
    void *p = NULL;
    if (!n) n = 1u;
#ifdef _WIN32
    p = _aligned_malloc(n, 256u);
#else
    if (posix_memalign(&p, 256u, n)) p = NULL;
#endif
    return p;
}
static void wex_shim_aligned_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static const WexShimBlock *wex_shim_find(const void *p) {
    const unsigned char *c = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < wex_shim_memory_count; i++)
        if (c >= wex_shim_memory[i].p && c < wex_shim_memory[i].p + wex_shim_memory[i].n) return &wex_shim_memory[i];
    return NULL;
}
/* p .. p + n inside one allocation of cudaMalloc */
static inline int wex_shim_is_device(const void *p, size_t n) {
    const WexShimBlock *b = wex_shim_find(p);
    return b && b->device && n <= (size_t)(b->p + b->n - (const unsigned char *)p);
}
static inline int wex_shim_on_device(const void *p) {
    const WexShimBlock *b = wex_shim_find(p);
    return b && b->device;
}
static cudaError_t wex_shim_alloc(void **out, size_t n, int device) {
    unsigned char *p;
    if (!out) return wex_shim_error(cudaErrorInvalidValue);
    p = (unsigned char *)wex_shim_aligned(n);
    if (!p) { *out = NULL; return wex_shim_error(cudaErrorMemoryAllocation); }
    if (device) memset(p, 0xa5, n ? n : 1u);
    if (wex_shim_memory_count == wex_shim_memory_cap) {
        size_t cap = wex_shim_memory_cap ? wex_shim_memory_cap * 2u : 64u;
        WexShimBlock *grown = (WexShimBlock *)realloc(wex_shim_memory, cap * sizeof(*grown));
        if (!grown) abort();
        wex_shim_memory = grown; wex_shim_memory_cap = cap;
    }
    wex_shim_memory[wex_shim_memory_count].p = p;
    wex_shim_memory[wex_shim_memory_count].n = n ? n : 1u;
    wex_shim_memory[wex_shim_memory_count].device = device;
    wex_shim_memory_count++;
    *out = p;
    return cudaSuccess;
}
static void wex_shim_drain(struct WexShimStream *target);
static cudaError_t wex_shim_release(void *p, int device) {
    size_t i;
    if (!p) return cudaSuccess;
    wex_shim_drain(NULL);   /* cudaFree and cudaFreeHost wait for the device */
    for (i = 0; i < wex_shim_memory_count; i++)
        if (wex_shim_memory[i].p == p) {
            if (wex_shim_memory[i].device != device) wex_shim_fail("memory freed with the wrong function (cudaFree / cudaFreeHost)");
            wex_shim_aligned_free(p);
            wex_shim_memory[i] = wex_shim_memory[--wex_shim_memory_count];
            return cudaSuccess;
        }
    wex_shim_fail("a pointer freed that no cudaMalloc or cudaMallocHost returned");
    return cudaErrorInvalidValue;
}
static inline cudaError_t cudaMalloc(void **p, size_t n) { return wex_shim_alloc(p, n, 1); }
static inline cudaError_t cudaFree(void *p) { return wex_shim_release(p, 1); }
static inline cudaError_t cudaMallocHost(void **p, size_t n) { return wex_shim_alloc(p, n, 0); }
static inline cudaError_t cudaFreeHost(void *p) { return wex_shim_release(p, 0); }
static inline cudaError_t cudaHostRegister(void *p, size_t n, unsigned int flags) {
    (void)flags;
    return p && n ? cudaSuccess : wex_shim_error(cudaErrorInvalidValue);
}
static inline cudaError_t cudaHostUnregister(void *p) { return p ? cudaSuccess : wex_shim_error(cudaErrorInvalidValue); }
static inline cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    *free_bytes = (size_t)1 << 34; *total_bytes = (size_t)1 << 34;   /* 16 GiB, none of it used */
    return cudaSuccess;
}

/* ---- device ------------------------------------------------------------------- */
static inline cudaError_t cudaGetDevice(int *device) { *device = 0; return cudaSuccess; }
static inline cudaError_t cudaSetDevice(int device) { return device == 0 ? cudaSuccess : wex_shim_error(cudaErrorInvalidDevice); }
static inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    if (device != 0) return wex_shim_error(cudaErrorInvalidDevice);
    if (attr == cudaDevAttrComputeCapabilityMajor) { *value = wex_shim_device.major; return cudaSuccess; }
    if (attr == cudaDevAttrComputeCapabilityMinor) { *value = wex_shim_device.minor; return cudaSuccess; }
    return wex_shim_error(cudaErrorInvalidValue);
}

/* the dynamic shared memory a kernel may launch with (cudaFuncSetAttribute), by kernel */
struct WexShimFuncLimit { uintptr_t kernel; size_t bytes; };
static WexShimFuncLimit wex_shim_limits[64];
static unsigned wex_shim_limit_count;
template<class F> static inline uintptr_t wex_shim_kernel_id(F *kernel) {
    return reinterpret_cast<uintptr_t>(reinterpret_cast<void (*)()>(kernel));
}
template<class F> static inline cudaError_t cudaFuncSetAttribute(F *kernel, cudaFuncAttribute attr, int value) {
    uintptr_t id = wex_shim_kernel_id(kernel);
    unsigned i;
    if (attr != cudaFuncAttributeMaxDynamicSharedMemorySize || value < 0 ||
        (size_t)value > wex_shim_device.shared_optin) return wex_shim_error(cudaErrorInvalidValue);
    for (i = 0; i < wex_shim_limit_count && wex_shim_limits[i].kernel != id; i++) {}
    if (i == wex_shim_limit_count) {
        if (i == 64u) wex_shim_fail("more than 64 kernels with a shared-memory attribute");
        wex_shim_limit_count++;
    }
    wex_shim_limits[i].kernel = id; wex_shim_limits[i].bytes = (size_t)value;
    return cudaSuccess;
}
static inline size_t wex_shim_dynamic_limit(uintptr_t id) {
    unsigned i;
    for (i = 0; i < wex_shim_limit_count; i++) if (wex_shim_limits[i].kernel == id) return wex_shim_limits[i].bytes;
    return WEX_SHIM_STATIC_SHARED;
}

/* ---- intrinsics --------------------------------------------------------------- */
template<class T> static inline T __ldg(const T *p) { return *p; }
static inline uint32_t wex_shim_select(uint32_t a, uint32_t b, uint32_t selector) {
    return (uint32_t)(((uint64_t)b << 32 | a) >> ((selector & 7u) * 8u)) & 0xffu;
}
/* CUDA __byte_perm: byte n of the result is byte (s >> 4n) & 7 of {y, x}; bit 3 of each selector is ignored */
static inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    uint32_t r = 0, n;
    for (n = 0; n < 4u; n++) r |= wex_shim_select(x, y, s >> (4u * n)) << (8u * n);
    return r;
}
/* PTX prmt.b32 d, a, b, c in its default mode: as __byte_perm, and a selector with bit 3 set gives the byte's
 * sign bit replicated over 8 bits */
static inline uint32_t wex_shim_prmt(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r = 0, n;
    for (n = 0; n < 4u; n++) {
        uint32_t s = (c >> (4u * n)) & 15u, v = wex_shim_select(a, b, s);
        if (s & 8u) v = v & 0x80u ? 0xffu : 0u;
        r |= v << (8u * n);
    }
    return r;
}
/* the upper 32 bits of hi:lo shifted left by shift & 31 */
static inline uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t shift) {
    return (uint32_t)((((uint64_t)hi << 32) | lo) << (shift & 31u) >> 32);
}
static inline int __popc(uint32_t x) { return __builtin_popcount(x); }
static inline uint32_t __vmaxu2(uint32_t a, uint32_t b) {
    uint32_t lo = (a & 0xffffu) > (b & 0xffffu) ? a & 0xffffu : b & 0xffffu;
    uint32_t hi = (a >> 16) > (b >> 16) ? a >> 16 : b >> 16;
    return hi << 16 | lo;
}
static inline uint32_t __vminu2(uint32_t a, uint32_t b) {
    uint32_t lo = (a & 0xffffu) < (b & 0xffffu) ? a & 0xffffu : b & 0xffffu;
    uint32_t hi = (a >> 16) < (b >> 16) ? a >> 16 : b >> 16;
    return hi << 16 | lo;
}
/* the overloads of CUDA's math API: a mixed pair is compared as unsigned */
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
static inline unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
static inline unsigned int min(int a, unsigned int b) { return min((unsigned int)a, b); }
static inline unsigned int min(unsigned int a, int b) { return min(a, (unsigned int)b); }
static inline unsigned int max(int a, unsigned int b) { return max((unsigned int)a, b); }
static inline unsigned int max(unsigned int a, int b) { return max(a, (unsigned int)b); }
static inline uint32_t atomicOr(uint32_t *address, uint32_t value) {
    return __atomic_fetch_or(address, value, __ATOMIC_SEQ_CST);
}
static inline uint32_t atomicExch(uint32_t *address, uint32_t value) {
    return __atomic_exchange_n(address, value, __ATOMIC_SEQ_CST);
}

/* ---- the block being run ------------------------------------------------------- */
static thread_local uint3 threadIdx;
static thread_local uint3 blockIdx;
static thread_local dim3 blockDim;
static thread_local dim3 gridDim;

struct WexShimShared { int key; size_t bytes; unsigned char *p; };
struct WexShimBlockState {
    pthread_mutex_t lock;
    pthread_cond_t done, barrier;
    /* the job: one block */
    unsigned long long job;
    unsigned threads, remaining, alive, arrived;
    unsigned long long barrier_generation, barriers;
    void (*body)(void *);
    void *context;
    int pooled;   /* 1: one host thread per CUDA thread; 0: thread by thread on the caller, no barrier allowed */
    /* its shared memory */
    WexShimShared shared[8];
    unsigned shared_count;
    size_t static_bytes, dynamic_bytes;
    unsigned char *dynamic;
    unsigned workers;
};
static WexShimBlockState wex_shim_block = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
                                           0, 0, 0, 0, 0, 0, 0, NULL, NULL, 0, {}, 0, 0, 0, NULL, 0};

static unsigned char *wex_shim_shared(size_t bytes, int key) {
    WexShimBlockState *b = &wex_shim_block;
    unsigned char *p = NULL;
    unsigned i;
    pthread_mutex_lock(&b->lock);
    for (i = 0; i < b->shared_count; i++)
        if (b->shared[i].key == key) {
            if (b->shared[i].bytes != bytes) wex_shim_fail("one __shared__ declaration with two sizes in a block");
            p = b->shared[i].p;
        }
    if (!p) {
        if (b->shared_count == 8u) wex_shim_fail("more than 8 __shared__ arrays in a block");
        b->static_bytes += bytes;
        if (b->static_bytes > WEX_SHIM_STATIC_SHARED) wex_shim_fail("more than 48 KiB of static shared memory in a block");
        if (b->static_bytes + b->dynamic_bytes > wex_shim_device.shared_optin)
            wex_shim_fail("static and dynamic shared memory of a block above the device's limit");
        p = (unsigned char *)wex_shim_aligned(bytes);
        if (!p) abort();
        memset(p, 0xa5, bytes);
        b->shared[b->shared_count].key = key; b->shared[b->shared_count].bytes = bytes; b->shared[b->shared_count].p = p;
        b->shared_count++;
    }
    pthread_mutex_unlock(&b->lock);
    return p;
}
static inline unsigned char *wex_shim_dynamic_shared(void) { return wex_shim_block.dynamic; }

/* ---- cp.async, per CUDA thread ------------------------------------------------- */
#define WEX_SHIM_ASYNC_MAX 256u
#define WEX_SHIM_ASYNC_POISON UINT32_C(0x9e3779b9)
struct WexShimAsync {
    uint32_t *destination[WEX_SHIM_ASYNC_MAX];
    const uint32_t *source[WEX_SHIM_ASYNC_MAX];
    unsigned long long group[WEX_SHIM_ASYNC_MAX];
    unsigned count;
    unsigned long long open_group;   /* the group new copies join; the committed ones are below it */
    void *handles[64];
    unsigned next_handle;
};
static thread_local WexShimAsync wex_shim_async;

static int wex_shim_in_shared(const void *p, size_t n) {
    WexShimBlockState *b = &wex_shim_block;
    const unsigned char *c = (const unsigned char *)p;
    unsigned i;
    int found = 0;
    pthread_mutex_lock(&b->lock);
    if (b->dynamic && c >= b->dynamic && n <= b->dynamic_bytes && (size_t)(c - b->dynamic) <= b->dynamic_bytes - n) found = 1;
    for (i = 0; i < b->shared_count && !found; i++)
        if (c >= b->shared[i].p && n <= b->shared[i].bytes && (size_t)(c - b->shared[i].p) <= b->shared[i].bytes - n) found = 1;
    pthread_mutex_unlock(&b->lock);
    return found;
}
static inline size_t __cvta_generic_to_shared(const void *p) {
    WexShimAsync *a = &wex_shim_async;
    unsigned slot = a->next_handle++ % 64u;
    a->handles[slot] = const_cast<void *>(p);
    return 0x40000000u | slot;
}
static inline void wex_shim_cp_async_ca4(unsigned handle, const uint32_t *source, int predicate) {
    WexShimAsync *a = &wex_shim_async;
    uint32_t *destination;
    if ((handle & ~63u) != 0x40000000u) wex_shim_fail("cp.async with a shared address that __cvta_generic_to_shared did not give");
    if (!predicate) return;
    destination = (uint32_t *)a->handles[handle & 63u];
    if (((uintptr_t)destination | (uintptr_t)source) & 3u) wex_shim_fail("cp.async of 4 bytes at an address that is not 4-byte aligned");
    if (!wex_shim_in_shared(destination, 4u)) wex_shim_fail("cp.async to an address outside the block's shared memory");
    if (a->count == WEX_SHIM_ASYNC_MAX) wex_shim_fail("more than 256 cp.async copies in flight in one thread");
    a->destination[a->count] = destination; a->source[a->count] = source; a->group[a->count] = a->open_group;
    a->count++;
    *destination = WEX_SHIM_ASYNC_POISON;   /* the word is not there before its wait */
    __atomic_fetch_add(&wex_shim_stats.async_copies, 1ull, __ATOMIC_RELAXED);
}
/* make every copy whose group is below `below` */
static void wex_shim_async_complete(unsigned long long below) {
    WexShimAsync *a = &wex_shim_async;
    unsigned i, kept = 0;
    for (i = 0; i < a->count; i++) {
        if (a->group[i] < below) { *a->destination[i] = *a->source[i]; continue; }
        a->destination[kept] = a->destination[i]; a->source[kept] = a->source[i]; a->group[kept] = a->group[i];
        kept++;
    }
    a->count = kept;
}
static inline void wex_shim_cp_async_commit_wait(unsigned long long newest_pending) {
    WexShimAsync *a = &wex_shim_async;
    a->open_group++;   /* commit_group */
    if (a->open_group >= newest_pending) wex_shim_async_complete(a->open_group - newest_pending);   /* wait_group N */
}

/* ---- running a block -------------------------------------------------------------- */
static inline void __syncthreads(void) {
    WexShimBlockState *b = &wex_shim_block;
    unsigned long long generation;
    if (!b->pooled) wex_shim_fail("__syncthreads() in a block run thread by thread (no thread of the first block this kernel ran reached one)");
    pthread_mutex_lock(&b->lock);
    generation = b->barrier_generation;
    b->barriers++;
    if (++b->arrived == b->alive) {
        b->arrived = 0;
        b->barrier_generation++;
        pthread_cond_broadcast(&b->barrier);
    } else {
        while (b->barrier_generation == generation) pthread_cond_wait(&b->barrier, &b->lock);
    }
    pthread_mutex_unlock(&b->lock);
}
static void wex_shim_set_thread(unsigned index) {
    threadIdx.x = index % blockDim.x;
    threadIdx.y = index / blockDim.x % blockDim.y;
    threadIdx.z = index / (blockDim.x * blockDim.y);
    wex_shim_async.count = 0; wex_shim_async.open_group = 0; wex_shim_async.next_handle = 0;
}
struct WexShimJob { uint3 block_idx; dim3 grid, block; };
static WexShimJob wex_shim_job;

/* one host thread per CUDA thread of a block; worker i is woken only for blocks of more than i threads */
#define WEX_SHIM_MAX_THREADS 1024u
struct WexShimWorker { pthread_cond_t wake; unsigned long long job; };
static WexShimWorker wex_shim_worker_state[WEX_SHIM_MAX_THREADS];

static void *wex_shim_worker(void *arg) {
    WexShimBlockState *b = &wex_shim_block;
    unsigned index = (unsigned)(uintptr_t)arg;
    WexShimWorker *w = &wex_shim_worker_state[index];
    unsigned long long seen = 0;
    for (;;) {
        pthread_mutex_lock(&b->lock);
        while (w->job == seen) pthread_cond_wait(&w->wake, &b->lock);
        seen = w->job;
        blockIdx = wex_shim_job.block_idx; blockDim = wex_shim_job.block; gridDim = wex_shim_job.grid;
        pthread_mutex_unlock(&b->lock);
        wex_shim_set_thread(index);
        b->body(b->context);
        wex_shim_async_complete(~0ull);
        pthread_mutex_lock(&b->lock);
        b->alive--;   /* a thread that returned no longer holds up a barrier */
        if (b->arrived && b->arrived == b->alive) {
            b->arrived = 0;
            b->barrier_generation++;
            pthread_cond_broadcast(&b->barrier);
        }
        if (!--b->remaining) pthread_cond_signal(&b->done);
        pthread_mutex_unlock(&b->lock);
    }
    return NULL;
}
static void wex_shim_workers(unsigned n) {
    WexShimBlockState *b = &wex_shim_block;
    while (b->workers < n) {
        pthread_t t;
        pthread_attr_t attr;
        if (pthread_cond_init(&wex_shim_worker_state[b->workers].wake, NULL)) wex_shim_fail("pthread_cond_init failed");
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_attr_setstacksize(&attr, (size_t)1 << 21);
        if (pthread_create(&t, &attr, wex_shim_worker, (void *)(uintptr_t)b->workers)) wex_shim_fail("pthread_create failed");
        pthread_attr_destroy(&attr);
        b->workers++;
    }
}
/* one block of a launch; returns how many barriers its threads passed */
static unsigned long long wex_shim_run_block(void (*body)(void *), void *context, dim3 grid, dim3 block, uint3 index,
                                             size_t dynamic_bytes, int pooled) {
    WexShimBlockState *b = &wex_shim_block;
    unsigned threads = block.x * block.y * block.z, i;
    unsigned long long barriers;
    b->shared_count = 0; b->static_bytes = 0;
    b->dynamic_bytes = dynamic_bytes;
    b->dynamic = (unsigned char *)wex_shim_aligned(dynamic_bytes);
    if (!b->dynamic) abort();
    if (dynamic_bytes) memset(b->dynamic, 0xa5, dynamic_bytes);
    b->barriers = 0;
    b->body = body; b->context = context; b->pooled = pooled;
    wex_shim_job.block_idx = index; wex_shim_job.grid = grid; wex_shim_job.block = block;
    if (pooled) {
        wex_shim_workers(threads);
        pthread_mutex_lock(&b->lock);
        b->threads = threads; b->remaining = threads; b->alive = threads; b->arrived = 0;
        b->job++;
        for (i = 0; i < threads; i++) {
            wex_shim_worker_state[i].job = b->job;
            pthread_cond_signal(&wex_shim_worker_state[i].wake);
        }
        while (b->remaining) pthread_cond_wait(&b->done, &b->lock);
        pthread_mutex_unlock(&b->lock);
        wex_shim_stats.pooled_blocks++;
    } else {
        blockIdx = index; blockDim = block; gridDim = grid;
        for (i = 0; i < threads; i++) {
            wex_shim_set_thread(i);
            body(context);
            wex_shim_async_complete(~0ull);
        }
    }
    barriers = b->barriers;
    for (i = 0; i < b->shared_count; i++) wex_shim_aligned_free(b->shared[i].p);
    b->shared_count = 0;
    wex_shim_aligned_free(b->dynamic);
    b->dynamic = NULL;
    wex_shim_stats.blocks++;
    wex_shim_stats.threads += threads;
    wex_shim_stats.barriers += barriers;
    return barriers;
}
/* per kernel: 0 not run yet, 1 its first block reached a __syncthreads(), 2 it reached none */
struct WexShimKernelKind { uintptr_t kernel; int kind; };
static WexShimKernelKind wex_shim_kinds[64];
static unsigned wex_shim_kind_count;
static void wex_shim_run_grid(uintptr_t kernel, void (*body)(void *), void *context, dim3 grid, dim3 block,
                              size_t dynamic_bytes) {
    unsigned long long blocks = (unsigned long long)grid.x * grid.y * grid.z, i;
    unsigned long long *order = (unsigned long long *)malloc((size_t)blocks * sizeof(*order));
    unsigned k;
    int pooled;
    if (!order) abort();
    for (k = 0; k < wex_shim_kind_count && wex_shim_kinds[k].kernel != kernel; k++) {}
    if (k == wex_shim_kind_count) {
        if (k == 64u) wex_shim_fail("more than 64 kernels");
        wex_shim_kinds[k].kernel = kernel; wex_shim_kinds[k].kind = 0;
        wex_shim_kind_count++;
    }
    pooled = wex_shim_kinds[k].kind != 2;
    for (i = 0; i < blocks; i++) order[i] = i;
    for (i = blocks; i > 1u; i--) {   /* the blocks in a random order */
        unsigned long long j = wex_shim_rng() % i, t = order[i - 1u];
        order[i - 1u] = order[j]; order[j] = t;
    }
    for (i = 0; i < blocks; i++) {
        uint3 index;
        index.x = (unsigned)(order[i] % grid.x);
        index.y = (unsigned)(order[i] / grid.x % grid.y);
        index.z = (unsigned)(order[i] / ((unsigned long long)grid.x * grid.y));
        unsigned long long barriers = wex_shim_run_block(body, context, grid, block, index, dynamic_bytes, pooled);
        if (!wex_shim_kinds[k].kind) {   /* the first block this kernel ever ran decides */
            wex_shim_kinds[k].kind = barriers ? 1 : 2;
            pooled = barriers != 0;
        }
    }
    free(order);
    wex_shim_stats.launches++;
}

/* ---- streams and events ---------------------------------------------------------------- */
struct WexShimEvent { unsigned long long recorded, completed; };
struct WexShimOp {
    WexShimOp *next;
    WexShimOp() : next(NULL) {}
    virtual ~WexShimOp() {}
    virtual int ready() const { return 1; }
    virtual void run() = 0;
};
struct WexShimStream { WexShimOp *head, *tail; WexShimStream *all; int destroyed; };
static WexShimStream wex_shim_default_stream = {NULL, NULL, NULL, 0};
static WexShimStream *wex_shim_streams = &wex_shim_default_stream;

static inline WexShimStream *wex_shim_stream(cudaStream_t s) { return s ? s : &wex_shim_default_stream; }
/* a destroyed stream leaves the list and is freed once its queue is empty (its work still runs, as in CUDA) */
static void wex_shim_reap(WexShimStream *q) {
    WexShimStream **at;
    if (!q->destroyed || q->head || q == &wex_shim_default_stream) return;
    for (at = &wex_shim_streams; *at && *at != q; at = &(*at)->all) {}
    if (*at) *at = q->all;
    free(q);
}
static void wex_shim_enqueue(cudaStream_t s, WexShimOp *op) {
    WexShimStream *q = wex_shim_stream(s);
    if (q->destroyed) wex_shim_fail("work queued on a destroyed stream");
    if (q->tail) q->tail->next = op; else q->head = op;
    q->tail = op;
}
/* Run queued work until `target` (NULL: every stream) is empty, drawing the next operation at random among
 * the streams whose first operation may run. */
static void wex_shim_drain(WexShimStream *target) {
    for (;;) {
        WexShimStream *q, *pick = NULL;
        unsigned runnable = 0, pending = 0;
        for (q = wex_shim_streams; q; q = q->all) {
            if (!q->head) continue;
            pending++;
            if (q->head->ready() && !(wex_shim_rng() % ++runnable)) pick = q;
        }
        if (target ? !target->head : !pending) return;
        if (!pick) wex_shim_fail("the queued work waits for an event that no queued work records");
        WexShimOp *op = pick->head;
        pick->head = op->next;
        if (!pick->head) pick->tail = NULL;
        op->run();
        delete op;
        wex_shim_reap(pick);
    }
}
static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int flags) {
    WexShimStream *q = (WexShimStream *)calloc(1u, sizeof(*q));
    (void)flags;
    if (!s || !q) { free(q); return wex_shim_error(cudaErrorInvalidValue); }
    q->all = wex_shim_streams;
    wex_shim_streams = q;
    *s = q;
    return cudaSuccess;
}
static inline cudaError_t cudaStreamCreate(cudaStream_t *s) { return cudaStreamCreateWithFlags(s, 0u); }
static inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (!s || s->destroyed) return wex_shim_error(cudaErrorInvalidResourceHandle);
    s->destroyed = 1;   /* its queued work still runs, as in CUDA */
    wex_shim_reap(s);
    return cudaSuccess;
}
static inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    if (s && s->destroyed && s->head) wex_shim_fail("cudaStreamSynchronize on a destroyed stream");
    wex_shim_drain(wex_shim_stream(s));
    return cudaSuccess;
}
static inline cudaError_t cudaDeviceSynchronize(void) { wex_shim_drain(NULL); return cudaSuccess; }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int flags) {
    (void)flags;
    if (!e) return wex_shim_error(cudaErrorInvalidValue);
    *e = (WexShimEvent *)calloc(1u, sizeof(WexShimEvent));   /* never freed: queued work may refer to it */
    return *e ? cudaSuccess : wex_shim_error(cudaErrorMemoryAllocation);
}
static inline cudaError_t cudaEventDestroy(cudaEvent_t e) { return e ? cudaSuccess : wex_shim_error(cudaErrorInvalidResourceHandle); }
struct WexShimRecord : WexShimOp {
    WexShimEvent *event; unsigned long long generation;
    void run() { if (event->completed < generation) event->completed = generation; }
};
struct WexShimWait : WexShimOp {
    WexShimEvent *event; unsigned long long generation;
    int ready() const { return event->completed >= generation; }
    void run() {}
};
static inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = 0) {
    WexShimRecord *op;
    if (!e) return wex_shim_error(cudaErrorInvalidResourceHandle);
    op = new WexShimRecord();
    op->event = e; op->generation = ++e->recorded;
    wex_shim_enqueue(s, op);
    return cudaSuccess;
}
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags) {
    WexShimWait *op;
    (void)flags;
    if (!e) return wex_shim_error(cudaErrorInvalidResourceHandle);
    op = new WexShimWait();
    op->event = e; op->generation = e->recorded;   /* the last record queued before this call */
    wex_shim_enqueue(s, op);
    return cudaSuccess;
}
struct WexShimCopy : WexShimOp {
    void *to; const void *from; size_t bytes; int value;
    void run() {
        if (from) memmove(to, from, bytes); else memset(to, value, bytes);
        wex_shim_stats.copies++;
    }
};
static inline cudaError_t cudaMemcpyAsync(void *to, const void *from, size_t bytes, cudaMemcpyKind kind, cudaStream_t s = 0) {
    WexShimCopy *op;
    int to_device, from_device;
    if (!bytes) return cudaSuccess;
    if (!to || !from) return wex_shim_error(cudaErrorInvalidValue);
    to_device = wex_shim_is_device(to, bytes); from_device = wex_shim_is_device(from, bytes);
    if ((kind == cudaMemcpyHostToDevice && (!to_device || wex_shim_on_device(from))) ||
        (kind == cudaMemcpyDeviceToHost && (!from_device || wex_shim_on_device(to))) ||
        (kind == cudaMemcpyDeviceToDevice && (!to_device || !from_device)) ||
        (kind == cudaMemcpyHostToHost && (wex_shim_on_device(to) || wex_shim_on_device(from))))
        wex_shim_fail("cudaMemcpyAsync whose kind does not match its memory, or a device range outside its allocation");
    op = new WexShimCopy();
    op->to = to; op->from = from; op->bytes = bytes; op->value = 0;
    wex_shim_enqueue(s, op);
    return cudaSuccess;
}
static inline cudaError_t cudaMemsetAsync(void *to, int value, size_t bytes, cudaStream_t s = 0) {
    WexShimCopy *op;
    if (!bytes) return cudaSuccess;
    if (!wex_shim_is_device(to, bytes)) wex_shim_fail("cudaMemsetAsync on memory that is not device memory");
    op = new WexShimCopy();
    op->to = to; op->from = NULL; op->bytes = bytes; op->value = value;
    wex_shim_enqueue(s, op);
    return cudaSuccess;
}

/* ---- kernel launches (K<<<G, B, S, Q>>>(ARGS), rewritten by cuda_host.py) -------------------- */
template<class T> static inline void wex_shim_check_arg(T *p) {
    if (p && !wex_shim_on_device(p)) wex_shim_fail("a kernel argument points to memory that cudaMalloc did not return");
}
template<class T> static inline void wex_shim_check_arg(const T &) {}
template<class F> struct WexShimKernel : WexShimOp {
    F body; uintptr_t kernel; dim3 grid, block; size_t dynamic_bytes;
    WexShimKernel(const F &f, uintptr_t k, dim3 g, dim3 b, size_t s) : body(f), kernel(k), grid(g), block(b), dynamic_bytes(s) {}
    static void call(void *self) { static_cast<WexShimKernel *>(self)->body(); }
    void run() { wex_shim_run_grid(kernel, &WexShimKernel::call, this, grid, block, dynamic_bytes); }
};
template<class... P> struct WexShimLaunch {
    void (*kernel)(P...);
    dim3 grid, block;
    size_t dynamic_bytes;
    cudaStream_t stream;
    void operator()(P... args) const {
        unsigned long long threads = (unsigned long long)block.x * block.y * block.z;
        void (*k)(P...) = kernel;
        int checked[] = {0, (wex_shim_check_arg(args), 0)...};
        (void)checked;
        if (!grid.x || !grid.y || !grid.z || grid.y > 65535u || grid.z > 65535u || !threads || threads > 1024u ||
            block.z > 64u) { wex_shim_error(cudaErrorInvalidConfiguration); return; }
        if (dynamic_bytes > wex_shim_dynamic_limit(wex_shim_kernel_id(kernel))) { wex_shim_error(cudaErrorInvalidValue); return; }
        auto body = [=]() { k(args...); };
        wex_shim_enqueue(stream, new WexShimKernel<decltype(body)>(body, wex_shim_kernel_id(kernel), grid, block, dynamic_bytes));
    }
};
template<class... P> static inline WexShimLaunch<P...> wex_shim_launch(void (*kernel)(P...), dim3 grid, dim3 block,
                                                                       size_t dynamic_bytes = 0, cudaStream_t stream = 0) {
    WexShimLaunch<P...> l;
    l.kernel = kernel; l.grid = grid; l.block = block; l.dynamic_bytes = dynamic_bytes; l.stream = stream;
    return l;
}

#endif /* WEX_FUZZ_CUDA_HOST_H */
