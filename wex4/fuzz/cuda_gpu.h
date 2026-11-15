/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_CUDA_GPU_H
#define WEX_FUZZ_CUDA_GPU_H
/* The GPU mode of fuzz_wnr3_cuda.cpp (built with -DWEX_FUZZ_GPU; wex4/fuzz/run.sh --gpu): besides the emulated
 * decodes, every record goes through the kernels of c/wex_cuda.cuh on a real NVIDIA GPU, without the CUDA
 * toolkit. The kernels come as PTX that clang wrote from the unchanged source (cuda_device.cu and cuda_device.h:
 * sm86.ptx, built for __CUDA_ARCH__ 860, and sm75.ptx, built for 750, in the folder WEX_FUZZ_GPU_PTX, default
 * wex4/fuzz/out/wex_cuda_ptx); the NVIDIA driver compiles them for the GPU when they are loaded. The driver
 * library (nvcuda.dll, libcuda.so.1) is opened at run time and called through the part of its C API declared
 * below.
 *
 * For each record whose plan the host accepts and whose model is registered, on each of the two modules:
 *   - the kernels alone, with lookup 0 and with lookup 1, on one stream: the scale kernel, then the weight
 *     kernel, into a buffer filled with 0xA5 first; the error word and the buffer come back;
 *   - the order of the bridge's wex_cuda_nr2_decode with its commit (c/tools/wex_cuda_bridge.cu): on a copy
 *     stream the offsets, the head of the record (up to the weight payload), an event, the rest, an event; on a
 *     work stream a wait for the head, the error word cleared, the scale kernel, a wait for the whole record,
 *     the weight kernel with the lookup of that build's bridge (1 for sm86, 0 for sm75), the commit kernel into
 *     six destination regions filled with 0xC3, the error word back. The record and the offsets are in
 *     page-locked memory, so the copy of the weight payload can run while the scale kernel runs, as in the
 *     bridge; this workspace is kept from record to record, as the bridge keeps its own.
 * The scale kernel gets the dynamic shared memory that wex_cuda_nr2_scale_smem gives it on this device. What
 * must hold is what fuzz_wnr3_cuda.cpp asks of the emulated kernels: the error word is 0 exactly when the CPU
 * reference accepts the record, it holds no bit but 1, 2, 4 and 8 (8 only under the CZ flag, 4 only under a CEIL
 * flag), the bytes are the CPU's when it is 0, and the six regions are untouched when it is not.
 *
 * The host side (plan, offsets, table layout) is the host build of the same source, namespace sm86 of the
 * harness. At start a kernel of cuda_device.cu reports the sizes and offsets of the structures passed by value,
 * which must equal the host's, and the __CUDA_ARCH__ it was built for. A CUDA call that fails stops the harness
 * with an ORACLE FAILED line: a kernel that reads or writes outside its memory is reported by the next call that
 * waits for it.
 *
 * What it does not run: nvcc (the production helper is built by nvcc; here clang's NVPTX back end and the
 * driver's JIT compile the same source), the bridge's own code and the CUDA runtime it calls (the sequence above
 * is the harness's copy of it), a GPU below compute capability 8.0 (sm75.ptx runs on the GPU there is, through
 * the driver's JIT), several GPUs, and any GPU but device 0. */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

/* ---- the CUDA driver API, as far as it is used --------------------------------------------------- */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUdeviceptr;
#ifdef _WIN32
#define WEX_GPU_API __stdcall
#else
#define WEX_GPU_API
#endif
#define WEX_GPU_FUNCTIONS(X) \
    X(cuInit, (unsigned int)) \
    X(cuDriverGetVersion, (int *)) \
    X(cuDeviceGet, (CUdevice *, int)) \
    X(cuDeviceGetName, (char *, int, CUdevice)) \
    X(cuDeviceGetAttribute, (int *, int, CUdevice)) \
    X(cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice)) \
    X(cuCtxSetCurrent, (CUcontext)) \
    X(cuCtxSynchronize, (void)) \
    X(cuModuleLoadDataEx, (CUmodule *, const void *, unsigned int, int *, void **)) \
    X(cuModuleGetFunction, (CUfunction *, CUmodule, const char *)) \
    X(cuFuncSetAttribute, (CUfunction, int, int)) \
    X(cuMemAlloc_v2, (CUdeviceptr *, size_t)) \
    X(cuMemFree_v2, (CUdeviceptr)) \
    X(cuMemAllocHost_v2, (void **, size_t)) \
    X(cuMemFreeHost, (void *)) \
    X(cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t)) \
    X(cuMemcpyDtoH_v2, (void *, CUdeviceptr, size_t)) \
    X(cuMemcpyHtoDAsync_v2, (CUdeviceptr, const void *, size_t, CUstream)) \
    X(cuMemcpyDtoHAsync_v2, (void *, CUdeviceptr, size_t, CUstream)) \
    X(cuMemsetD8Async, (CUdeviceptr, unsigned char, size_t, CUstream)) \
    X(cuStreamCreate, (CUstream *, unsigned int)) \
    X(cuStreamSynchronize, (CUstream)) \
    X(cuStreamWaitEvent, (CUstream, CUevent, unsigned int)) \
    X(cuEventCreate, (CUevent *, unsigned int)) \
    X(cuEventRecord, (CUevent, CUstream)) \
    X(cuLaunchKernel, (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, \
                       unsigned int, unsigned int, CUstream, void **, void **)) \
    X(cuGetErrorName, (CUresult, const char **))
#define WEX_GPU_DECLARE(name, args) typedef CUresult (WEX_GPU_API *name##_fn) args; static name##_fn gpu_##name;
WEX_GPU_FUNCTIONS(WEX_GPU_DECLARE)
#undef WEX_GPU_DECLARE
#define WEX_GPU_ATTR_CC_MAJOR 75                 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
#define WEX_GPU_ATTR_CC_MINOR 76                 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */
#define WEX_GPU_ATTR_SHARED_OPTIN 97             /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN */
#define WEX_GPU_FUNC_MAX_DYNAMIC_SHARED 8        /* CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES */
#define WEX_GPU_JIT_ERROR_LOG 5                  /* CU_JIT_ERROR_LOG_BUFFER */
#define WEX_GPU_JIT_ERROR_LOG_BYTES 6            /* CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES */
#define WEX_GPU_STREAM_NON_BLOCKING 1u           /* CU_STREAM_NON_BLOCKING */
#define WEX_GPU_EVENT_DISABLE_TIMING 2u          /* CU_EVENT_DISABLE_TIMING */

static void gpu_check(CUresult r, const char *what) {
    const char *name = "?";
    if (!r) return;
    if (gpu_cuGetErrorName) gpu_cuGetErrorName(r, &name);
    fprintf(stderr, "ORACLE FAILED: GPU: %s returned %d (%s)\n", what, r, name);
    abort();
}
#define GPU_OK(call) gpu_check(gpu_##call, #call)

/* ---- the two modules ------------------------------------------------------------------------------ */
struct GpuModule {
    const char *name;                 /* sm86 / sm75: the PTX file */
    int arch, bridge_lookup;          /* its __CUDA_ARCH__, the lookup of the bridge built for it */
    CUmodule module;
    CUfunction build_rows, build_scales, build_buckets, scales, weights[2], commit, layout;
    uint32_t ids[16];                 /* the models whose tables are built on it */
    CUdeviceptr tables[16];
    unsigned count;
    CUstream copy, work;
    CUevent head, data;
    /* the kernels alone */
    CUdeviceptr d_record, d_offsets, d_raw, d_error;
    size_t d_record_cap, d_offsets_cap, d_raw_cap;
    /* the bridge order: its workspace, kept from record to record, and the six regions */
    CUdeviceptr b_record, b_offsets, b_raw, b_error, b_dst;
    unsigned char *b_host_record;
    uint32_t *b_host_offsets, *b_host_error;
    size_t b_record_cap, b_offsets_cap, b_raw_cap, b_dst_cap, b_host_record_cap, b_host_offsets_cap;
};
static GpuModule gpu_modules[2];
static char gpu_device_name[256];
static int gpu_major, gpu_minor, gpu_driver;
static unsigned gpu_scale_smem;
static unsigned long long gpu_records, gpu_decodes, gpu_accepted, gpu_refused, gpu_error_bits[4];

static void gpu_grow(CUdeviceptr *p, size_t *cap, size_t want) {
    if (want <= *cap) return;
    if (*p) GPU_OK(cuMemFree_v2(*p));
    *p = 0;
    GPU_OK(cuMemAlloc_v2(p, want));
    *cap = want;
}
static void gpu_grow_host(void **p, size_t *cap, size_t want) {
    if (want <= *cap) return;
    if (*p) GPU_OK(cuMemFreeHost(*p));
    *p = NULL;
    GPU_OK(cuMemAllocHost_v2(p, want));
    *cap = want;
}
static CUdeviceptr gpu_ptr(const void *p) { return (CUdeviceptr)(uintptr_t)p; }

/* the mangled name of the one .entry of the PTX whose name holds `part` */
static void gpu_entry(const char *ptx, const char *part, char *out, size_t cap) {
    const char *p = ptx;
    size_t n = strlen(part);
    int found = 0;
    while ((p = strstr(p, ".entry ")) != NULL) {
        const char *name = p + 7, *end = strchr(name, '('), *q;
        if (!end) break;
        for (q = name; q + n <= end; q++)
            if (!memcmp(q, part, n)) break;
        if (q + n <= end && (size_t)(end - name) < cap) {
            if (found++) { fprintf(stderr, "GPU: two kernels of the PTX match %s\n", part); exit(2); }
            memcpy(out, name, (size_t)(end - name));
            out[end - name] = 0;
        }
        p = end;
    }
    if (!found) { fprintf(stderr, "GPU: no kernel of the PTX matches %s\n", part); exit(2); }
}

static void gpu_init(void) {
    const char *dir = getenv("WEX_FUZZ_GPU_PTX");
    CUdevice device;
    CUcontext context;
    unsigned b;
    int optin = 0;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
#define WEX_GPU_LOAD(name, args) gpu_##name = lib ? (name##_fn)(void *)GetProcAddress(lib, #name) : NULL; \
    if (!gpu_##name) { fprintf(stderr, "GPU: nvcuda.dll %s\n", lib ? "has no " #name : "cannot be loaded"); exit(2); }
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#define WEX_GPU_LOAD(name, args) gpu_##name = lib ? (name##_fn)dlsym(lib, #name) : NULL; \
    if (!gpu_##name) { fprintf(stderr, "GPU: libcuda.so.1 %s\n", lib ? "has no " #name : "cannot be loaded"); exit(2); }
#endif
    WEX_GPU_FUNCTIONS(WEX_GPU_LOAD)
#undef WEX_GPU_LOAD
    if (!dir || !dir[0]) dir = "wex4/fuzz/out/wex_cuda_ptx";
    GPU_OK(cuInit(0));
    GPU_OK(cuDriverGetVersion(&gpu_driver));
    GPU_OK(cuDeviceGet(&device, 0));
    GPU_OK(cuDeviceGetName(gpu_device_name, (int)sizeof gpu_device_name - 1, device));
    GPU_OK(cuDeviceGetAttribute(&gpu_major, WEX_GPU_ATTR_CC_MAJOR, device));
    GPU_OK(cuDeviceGetAttribute(&gpu_minor, WEX_GPU_ATTR_CC_MINOR, device));
    GPU_OK(cuDeviceGetAttribute(&optin, WEX_GPU_ATTR_SHARED_OPTIN, device));
    GPU_OK(cuDevicePrimaryCtxRetain(&context, device));   /* the context the CUDA runtime would use */
    GPU_OK(cuCtxSetCurrent(context));
    gpu_scale_smem = gpu_major < 8 ? WEX_CUDA_NR2_SCALE_SMEM_GLOBAL : WEX_CUDA_NR2_SCALE_SMEM;   /* as wex_cuda_nr2_scale_smem */
    if ((int)gpu_scale_smem > optin) {
        fprintf(stderr, "GPU: %d bytes of shared memory per block, the scale kernel needs %u\n", optin, gpu_scale_smem);
        exit(2);
    }
    for (b = 0; b < 2u; b++) {
        GpuModule &g = gpu_modules[b];
        char path[4096], name[512], log[8192] = "";
        int options[2] = {WEX_GPU_JIT_ERROR_LOG, WEX_GPU_JIT_ERROR_LOG_BYTES};
        void *values[2] = {log, (void *)(uintptr_t)sizeof log};
        size_t size = 0;
        unsigned char *ptx;
        CUdeviceptr out;
        uint32_t layout[8];
        CUresult r;
        g.name = b ? "sm75" : "sm86";
        g.arch = b ? 750 : 860;
        g.bridge_lookup = b ? 0 : 1;
        snprintf(path, sizeof path, "%s/%s.ptx", dir, g.name);
        ptx = fuzz_read_file(path, &size);
        if (!ptx) { fprintf(stderr, "GPU: cannot read %s (wex4/fuzz/run.sh --gpu writes it)\n", path); exit(2); }
        ptx = (unsigned char *)realloc(ptx, size + 1u);
        if (!ptx) abort();
        ptx[size] = 0;
        r = gpu_cuModuleLoadDataEx(&g.module, ptx, 2u, options, values);
        if (r) fprintf(stderr, "GPU: the driver does not load %s: %s\n", path, log);
        gpu_check(r, "cuModuleLoadDataEx");
#define WEX_GPU_KERNEL(field, part) gpu_entry((const char *)ptx, part, name, sizeof name); \
        GPU_OK(cuModuleGetFunction(&g.field, g.module, name))
        WEX_GPU_KERNEL(build_rows, "wex_cuda_nr2_build_rowsP");
        WEX_GPU_KERNEL(build_scales, "wex_cuda_nr2_build_scalesP");
        WEX_GPU_KERNEL(build_buckets, "wex_cuda_nr2_build_bucketsP");
        WEX_GPU_KERNEL(scales, "wex_cuda_nr2_scalesP");
        WEX_GPU_KERNEL(weights[0], "wex_cuda_nr2_weightsILi0E");
        WEX_GPU_KERNEL(weights[1], "wex_cuda_nr2_weightsILi1E");
        WEX_GPU_KERNEL(commit, "wex_cuda_commit_regionsP");
        WEX_GPU_KERNEL(layout, "wex_gpu_layoutP");
#undef WEX_GPU_KERNEL
        free(ptx);
        GPU_OK(cuFuncSetAttribute(g.scales, WEX_GPU_FUNC_MAX_DYNAMIC_SHARED, (int)gpu_scale_smem));
        GPU_OK(cuStreamCreate(&g.copy, WEX_GPU_STREAM_NON_BLOCKING));
        GPU_OK(cuStreamCreate(&g.work, WEX_GPU_STREAM_NON_BLOCKING));
        GPU_OK(cuEventCreate(&g.head, WEX_GPU_EVENT_DISABLE_TIMING));
        GPU_OK(cuEventCreate(&g.data, WEX_GPU_EVENT_DISABLE_TIMING));
        GPU_OK(cuMemAlloc_v2(&g.d_error, 4u));
        GPU_OK(cuMemAlloc_v2(&g.b_error, 4u));
        GPU_OK(cuMemAllocHost_v2((void **)&g.b_host_error, 4u));
        GPU_OK(cuMemAlloc_v2(&out, sizeof layout));
        {
            void *args[] = {&out};
            GPU_OK(cuLaunchKernel(g.layout, 1, 1, 1, 1, 1, 1, 0, g.work, args, NULL));
        }
        GPU_OK(cuMemcpyDtoHAsync_v2(layout, out, sizeof layout, g.work));
        GPU_OK(cuStreamSynchronize(g.work));
        GPU_OK(cuMemFree_v2(out));
        if (layout[0] != sizeof(sm86::WexCudaNr2Plan) || layout[1] != offsetof(sm86::WexCudaNr2Plan, model_id) ||
            layout[2] != offsetof(sm86::WexCudaNr2Plan, word24) || layout[3] != sizeof(sm86::WexCudaNr2Tables) ||
            layout[4] != offsetof(sm86::WexCudaNr2Tables, scale_symbol) || layout[5] != sizeof(sm86::WexCudaCommit) ||
            layout[6] != offsetof(sm86::WexCudaCommit, bytes) || (int)layout[7] != g.arch) {
            fprintf(stderr, "GPU: %s.ptx: the device sees the structures passed by value differently from the host, "
                    "or was built for __CUDA_ARCH__ %u\n", g.name, layout[7]);
            exit(2);
        }
    }
    fprintf(stderr, "GPU: %s, compute capability %d.%d, driver API %d; the kernels of c/wex_cuda.cuh from %s/sm86.ptx "
            "and sm75.ptx\n", gpu_device_name, gpu_major, gpu_minor, gpu_driver, dir);
}

/* the tables of a model on module g, built by its kernels as wex_cuda_build_nr2_tables builds them (with buckets) */
static void gpu_tables(GpuModule &g, const Model *model, sm86::WexCudaNr2Tables *t) {
    unsigned i;
    for (i = 0; i < g.count && g.ids[i] != model->id; i++) {}
    if (i == g.count) {
        CUdeviceptr base, rows, direct, symbol, buckets;
        if (i == 16u) abort();
        GPU_OK(cuMemAlloc_v2(&base, sm86::wex_cuda_nr2_table_bytes(1)));
        GPU_OK(cuMemcpyHtoDAsync_v2(base, model->blob, WEX_NR2_MODEL_BYTES, g.work));
        sm86::wex_cuda_nr2_table_view((unsigned char *)(uintptr_t)base, 1, t);
        rows = gpu_ptr(t->weight_rows); direct = gpu_ptr(t->scale_direct);
        symbol = gpu_ptr(t->scale_symbol); buckets = gpu_ptr(t->weight_buckets);
        {
            void *a_rows[] = {&base, &rows}, *a_scales[] = {&base, &direct, &symbol}, *a_buckets[] = {&base, &buckets};
            GPU_OK(cuLaunchKernel(g.build_rows, (WEX_CUDA_NR2_ROWS * 8u + 255u) / 256u, 1, 1, 256, 1, 1, 0, g.work,
                                  a_rows, NULL));
            GPU_OK(cuLaunchKernel(g.build_scales, 3u * WEX_NR2_TOTAL / 256u, 1, 1, 256, 1, 1, 0, g.work, a_scales, NULL));
            GPU_OK(cuLaunchKernel(g.build_buckets, (WEX_CUDA_NR2_ROWS << WEX_CUDA_NR2_BUCKET_BITS) / 256u, 1, 1, 256, 1, 1,
                                  0, g.work, a_buckets, NULL));
        }
        GPU_OK(cuStreamSynchronize(g.work));
        g.ids[i] = model->id;
        g.tables[i] = base;
        g.count++;
    }
    sm86::wex_cuda_nr2_table_view((unsigned char *)(uintptr_t)g.tables[i], 1, t);
}

static void gpu_verdict(uint32_t error, int cpu, const unsigned char *record, const char *what) {
    char message[256];
    gpu_decodes++;
    if (error) gpu_refused++; else gpu_accepted++;
    snprintf(message, sizeof message, "GPU, %s: the error word and the CPU reference give different verdicts", what);
    FUZZ_REQUIRE(!error == !cpu, message);
    FUZZ_REQUIRE(!(error & ~15u), "GPU: an error bit outside 1 | 2 | 4 | 8");
    FUZZ_REQUIRE(!(error & WEX_CUDA_ERR_CZ) || (record[27] & WEX_NR3_FLAG_CZ), "GPU: error bit 8 for a record without the CZ flag");
    FUZZ_REQUIRE(!(error & WEX_CUDA_ERR_CEIL) || (record[27] & WEX_NR3_FLAG_CEIL_MASK),
                 "GPU: error bit 4 for a record without a CEIL flag");
}

/* the kernels alone on module g: the error word; the decoded buffer (plan layout) in `out` */
static uint32_t gpu_direct(GpuModule &g, int lookup, const sm86::WexCudaNr2Plan &plan, const sm86::WexCudaNr2Tables &tables,
                           const unsigned char *record, size_t total, const uint32_t *offsets, size_t aligned,
                           unsigned char *out) {
    uint32_t error = 0;
    size_t offset_bytes = ((size_t)plan.streams + 1u) * 4u;
    gpu_grow(&g.d_record, &g.d_record_cap, total);
    gpu_grow(&g.d_offsets, &g.d_offsets_cap, offset_bytes);
    gpu_grow(&g.d_raw, &g.d_raw_cap, aligned);
    GPU_OK(cuMemsetD8Async(g.d_raw, 0xa5, aligned, g.work));
    GPU_OK(cuMemcpyHtoDAsync_v2(g.d_record, record, total, g.work));
    GPU_OK(cuMemcpyHtoDAsync_v2(g.d_offsets, offsets, offset_bytes, g.work));
    GPU_OK(cuMemsetD8Async(g.d_error, 0, 4u, g.work));
    {
        void *args[] = {&g.d_record, &g.d_offsets, (void *)&plan, (void *)&tables, &g.d_raw, &g.d_error};
        GPU_OK(cuLaunchKernel(g.scales, plan.block_base[0][3], 1, 1, WEX_CUDA_NR2_THREADS, 1, 1, gpu_scale_smem, g.work,
                              args, NULL));
        GPU_OK(cuLaunchKernel(g.weights[lookup], plan.block_base[1][3], 1, 1, WEX_CUDA_NR2_THREADS, 1, 1, 0, g.work,
                              args, NULL));
    }
    GPU_OK(cuMemcpyDtoHAsync_v2(&error, g.d_error, 4u, g.work));
    GPU_OK(cuMemcpyDtoHAsync_v2(out, g.d_raw, aligned, g.work));
    GPU_OK(cuStreamSynchronize(g.work));
    return error;
}

/* the order of the bridge's wex_cuda_nr2_decode with its commit, on module g: the error word; the six regions
 * (scales of matrix 0..2, weights of matrix 0..2, each at a 256-byte boundary of `dst`) in `out` */
static uint32_t gpu_bridge_order(GpuModule &g, const WexEntry *entry, const sm86::WexCudaNr2Plan &plan,
                                 const sm86::WexCudaNr2Tables &tables, const unsigned char *record, const uint32_t *offsets,
                                 size_t aligned, const size_t region_at[7], unsigned char *out) {
    size_t total = entry->total_bytes, offset_bytes = ((size_t)plan.streams + 1u) * 4u, head = plan.weight_payload;
    sm86::WexCudaCommit commit;
    unsigned m;
    gpu_grow(&g.b_record, &g.b_record_cap, total);
    gpu_grow(&g.b_offsets, &g.b_offsets_cap, offset_bytes);
    gpu_grow(&g.b_raw, &g.b_raw_cap, aligned);
    gpu_grow(&g.b_dst, &g.b_dst_cap, region_at[6]);
    gpu_grow_host((void **)&g.b_host_record, &g.b_host_record_cap, total);
    gpu_grow_host((void **)&g.b_host_offsets, &g.b_host_offsets_cap, offset_bytes);
    memcpy(g.b_host_record, record, total);
    memcpy(g.b_host_offsets, offsets, offset_bytes);
    for (m = 0; m < 3u; m++) {
        commit.destination[m] = (unsigned char *)(uintptr_t)(g.b_dst + region_at[m]);
        commit.source[m] = plan.scale_offset[m];
        commit.bytes[m] = (uint32_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
        commit.destination[3u + m] = (unsigned char *)(uintptr_t)(g.b_dst + region_at[3u + m]);
        commit.source[3u + m] = plan.weight_offset[m];
        commit.bytes[3u + m] = (uint32_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
    }
    GPU_OK(cuMemsetD8Async(g.b_dst, 0xc3, region_at[6], g.work));   /* the caller's buffers, before the call */
    GPU_OK(cuStreamSynchronize(g.work));
    GPU_OK(cuMemcpyHtoDAsync_v2(g.b_offsets, g.b_host_offsets, offset_bytes, g.copy));
    GPU_OK(cuMemcpyHtoDAsync_v2(g.b_record, g.b_host_record, head, g.copy));
    GPU_OK(cuEventRecord(g.head, g.copy));
    GPU_OK(cuMemcpyHtoDAsync_v2(g.b_record + head, g.b_host_record + head, total - head, g.copy));
    GPU_OK(cuEventRecord(g.data, g.copy));
    GPU_OK(cuStreamWaitEvent(g.work, g.head, 0));
    GPU_OK(cuMemsetD8Async(g.b_error, 0, 4u, g.work));
    {
        void *args[] = {&g.b_record, &g.b_offsets, (void *)&plan, (void *)&tables, &g.b_raw, &g.b_error};
        void *commit_args[] = {&g.b_raw, &commit, &g.b_error};
        GPU_OK(cuLaunchKernel(g.scales, plan.block_base[0][3], 1, 1, WEX_CUDA_NR2_THREADS, 1, 1, gpu_scale_smem, g.work,
                              args, NULL));
        GPU_OK(cuStreamWaitEvent(g.work, g.data, 0));
        GPU_OK(cuLaunchKernel(g.weights[g.bridge_lookup], plan.block_base[1][3], 1, 1, WEX_CUDA_NR2_THREADS, 1, 1, 0,
                              g.work, args, NULL));
        GPU_OK(cuLaunchKernel(g.commit, 64u, 6u, 1, 256, 1, 1, 0, g.work, commit_args, NULL));   /* wex_cuda_launch_commit */
    }
    GPU_OK(cuMemcpyDtoHAsync_v2(g.b_host_error, g.b_error, 4u, g.work));
    GPU_OK(cuStreamSynchronize(g.work));
    GPU_OK(cuStreamSynchronize(g.copy));
    GPU_OK(cuMemcpyDtoHAsync_v2(out, g.b_dst, region_at[6], g.work));
    GPU_OK(cuStreamSynchronize(g.work));
    return *g.b_host_error;
}

/* one record on the GPU; `expected` is the CPU's slab when cpu == 0 */
static void gpu_compare(const WexEntry *entry, const unsigned char *record, const Model *model, int cpu,
                        const unsigned char *expected) {
    WexNr2Record view;
    sm86::WexCudaNr2Plan plan;
    sm86::WexCudaNr2Tables tables;
    uint64_t raw, aligned;
    size_t region_at[7], bytes[6], at = 0;
    uint32_t *offsets;
    unsigned char *out, *regions;
    unsigned b, m;
    int lookup;
    if (!model || sm86::wex_cuda_plan_nr2(entry, record, entry->total_bytes, &view, &plan)) return;
    if (wex_entry_raw_sizes(entry, &raw, &aligned)) abort();
    for (m = 0; m < 3u; m++) {
        bytes[m] = (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
        bytes[3u + m] = (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
    }
    for (m = 0; m < 6u; m++) { region_at[m] = at; at += (bytes[m] + 255u) & ~(size_t)255u; }
    region_at[6] = at;
    offsets = (uint32_t *)malloc(((size_t)plan.streams + 1u) * 4u);
    out = (unsigned char *)malloc((size_t)aligned);
    regions = (unsigned char *)malloc(at);
    if (!offsets || !out || !regions) abort();
    sm86::wex_cuda_nr2_offsets(&view, &plan, offsets);
    gpu_records++;
    for (b = 0; b < 2u; b++) {
        GpuModule &g = gpu_modules[b];
        char what[64];
        gpu_tables(g, model, &tables);
        for (lookup = 0; lookup < 2; lookup++) {
            uint32_t error = gpu_direct(g, lookup, plan, tables, record, entry->total_bytes, offsets, (size_t)aligned, out);
            snprintf(what, sizeof what, "%s, the kernels alone with lookup %d", g.name, lookup);
            gpu_verdict(error, cpu, record, what);
            if (!b && lookup == 1) {
                unsigned bit;
                for (bit = 0; bit < 4u; bit++) gpu_error_bits[bit] += (error >> bit) & 1u;
            }
            if (error) continue;
            for (m = 0; m < 3u; m++)
                FUZZ_REQUIRE(!memcmp(out + plan.scale_offset[m], expected + entry->scale_offsets[m], bytes[m]) &&
                             !memcmp(out + plan.weight_offset[m], expected + entry->weight_offsets[m], bytes[3u + m]),
                             "GPU: the kernels alone decode other bytes than the CPU");
        }
        {
            uint32_t error = gpu_bridge_order(g, entry, plan, tables, record, offsets, (size_t)aligned, region_at, regions);
            snprintf(what, sizeof what, "%s, the bridge's order with lookup %d and the commit", g.name, g.bridge_lookup);
            gpu_verdict(error, cpu, record, what);
            for (m = 0; m < 6u; m++) {
                const unsigned char *want = expected + (m < 3u ? entry->scale_offsets[m] : entry->weight_offsets[m - 3u]);
                if (!error) FUZZ_REQUIRE(!memcmp(regions + region_at[m], want, bytes[m]),
                                         "GPU: the bridge's order decodes other bytes than the CPU");
                else FUZZ_REQUIRE(all_bytes(regions + region_at[m], bytes[m], 0xc3),
                                  "GPU: the commit kernel wrote the destination of a record the kernels refused");
            }
        }
    }
    free(regions);
    free(out);
    free(offsets);
}

static void gpu_report(void) {
    fprintf(stderr, "fuzz_wnr3_cuda GPU: %s (compute capability %d.%d, driver API %d): %llu WNR3 records, each through 2 "
            "PTX modules x (the kernels alone with 2 lookups + the bridge's order with the commit) (%llu GPU decodes): "
            "%llu decodes accepted with the CPU's bytes, %llu refused with the CPU (error bits of sm86 with lookup 1: "
            "1: %llu, 2: %llu, 4: %llu, 8: %llu)\n", gpu_device_name, gpu_major, gpu_minor, gpu_driver, gpu_records,
            gpu_decodes, gpu_accepted, gpu_refused, gpu_error_bits[0], gpu_error_bits[1], gpu_error_bits[2],
            gpu_error_bits[3]);
}

#endif /* WEX_FUZZ_CUDA_GPU_H */
