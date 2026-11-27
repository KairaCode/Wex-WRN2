/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the CUDA WNR3 decoder, run on the CPU: the kernels of c/wex_cuda.cuh and the host code of
 * c/tools/wex_cuda_bridge.cu, built as host C++ through cuda_host.py (which rewrites the launches, the
 * __shared__ declarations and the inline PTX) and cuda_host.h (which emulates the rest; its header comment says
 * how). No GPU and no CUDA toolkit are needed. What a run proves is the decoding logic and the order of the
 * bridge's CUDA calls, not the GPU: not its compiler, not its memory model, not its speed.
 *
 * The CUDA source is built twice, as nvcc's device pass would build it for two targets:
 *   sm86: __CUDA_ARCH__ 860 (the scale kernel's direct table in shared memory, the payload ring filled by
 *         cp.async), the weight lookup of the bridge's default (WEX_CUDA_NR2_LOOKUP 1, the CDF rows in shared
 *         memory), on an emulated device of compute capability 8.6 with 99 KiB of shared memory per block;
 *   sm75: __CUDA_ARCH__ 750 (the direct table in global memory, the ring filled by plain loads), the bucket
 *         lookup (WEX_CUDA_NR2_LOOKUP 0), on a device of compute capability 7.5 with 64 KiB.
 *
 * The input is either the stored bytes of one WNR3 record, decoded with the model and the layout of the first
 * WNR3 record of WEX_FUZZ_BASE (default wex4/vectors/valid/tiny2-cz-upgraded.wex4, as fuzz_wnr3_record.c), or a
 * whole WEX4 file whose header page and index open, of which every WNR3 record is decoded with its own entry and
 * the model of its blob (the record's SHA-256 is not checked: a record that fails it is still decoded). The
 * models that occur are registered with the CPU side and with both builds of the bridge, as the engine registers
 * the model of a container; a record is decoded with the model its bytes 4..7 name.
 *
 * Every record goes through, on each build: the bridge's wex_cuda_decode_to_host, wex_cuda_decode_to_device,
 * wex_cuda_decode_batch_to_device and wex_cuda_decode_batch_to_host (both batches with a second, valid record
 * next to it, in either order; the batch to host decodes each record over itself, the record in the tail of
 * its slab, as c/qwen38_wex.h calls it), and the kernels without the bridge (plan, offsets, tables,
 * wex_cuda_launch_nr2_lookup) with lookup 0 and 1, which give the device's error word. The reference is
 * wex_decode_nr2_record of c/wex_format.h (the CPU decoder the engine falls back to). What must hold, besides
 * the sanitizers:
 *   - the CPU accepts the record exactly when every GPU path returns 0, and then every GPU path gives its bytes;
 *   - a GPU path that does not return 0 returns -1, or 1 when the record parses and names a model that is not
 *     registered (what the bridge reports as "helper unavailable");
 *   - on failure the host slab of wex_cuda_decode_to_host, the device buffers of wex_cuda_decode_to_device and
 *     of wex_cuda_decode_batch_to_device and the host slab of wex_cuda_decode_batch_to_host (the record in its
 *     tail included) are untouched (WNR3-SPEC 4.7; the batch to host since 2026-11-17, when it went to rounds
 *     that copy back only the records that decoded clean); the valid record of a batch decodes whatever its
 *     neighbour is;
 *   - the error word has no bit outside 1 | 2 | 4 | 8, bit 8 only for a record with the CZ flag and bit 4 only
 *     for a record with a CEIL flag.
 * At exit the harness prints what it ran: inputs, records, verdicts, the error bits seen and the size of the
 * emulated work.
 *
 * Built with -DWEX_FUZZ_GPU (run.sh --gpu), it also decodes every record with the same kernels on an NVIDIA GPU,
 * from PTX that clang writes without the CUDA toolkit, and asks the same of them (cuda_gpu.h). */
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed);
extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
#include "wex4.h"
#define FUZZ_MUTATOR_MAX_LEN 1048576u
#ifndef FUZZ_MUTATOR_ROUNDS
#define FUZZ_MUTATOR_ROUNDS 4   /* run.sh --replay: 4 inputs of the mutator from each of the 119 seeds */
#endif
#include "fuzz_common.h"
#include "wnr3_mutator.h"
#include "cuda_host.h"

/* ---- the two builds of the CUDA source ------------------------------------------------------- */
namespace sm86 {
#define __CUDA_ARCH__ 860
#define WEX_CUDA_NR2_LOOKUP 1
#define wex_cuda_register_model sm86_register_model
#define wex_cuda_helper_abi sm86_helper_abi
#define wex_cuda_helper_features sm86_helper_features
#define wex_cuda_decode_to_host sm86_decode_to_host
#define wex_cuda_decode_to_device sm86_decode_to_device
#define wex_cuda_decode_batch_to_device sm86_decode_batch_to_device
#define wex_cuda_decode_batch_to_host sm86_decode_batch_to_host
#define wex_cuda_decode_batch_to_host_scratch sm86_decode_batch_to_host_scratch
#define wex_cuda_host_register sm86_host_register
#define wex_cuda_host_unregister sm86_host_unregister
#include "wex_cuda_host/tools/wex_cuda_bridge.cu"
}
#undef __CUDA_ARCH__
#undef WEX_CUDA_NR2_LOOKUP
#undef WEX_CUDA_NR2_WAIT
#undef WEX_CUDA_NR2_DIRECT
#undef COLI_WEX_CUDA_CUH
#undef wex_cuda_register_model
#undef wex_cuda_helper_abi
#undef wex_cuda_helper_features
#undef wex_cuda_decode_to_host
#undef wex_cuda_decode_to_device
#undef wex_cuda_decode_batch_to_device
#undef wex_cuda_decode_batch_to_host
#undef wex_cuda_decode_batch_to_host_scratch
#undef wex_cuda_host_register
#undef wex_cuda_host_unregister
namespace sm75 {
#define __CUDA_ARCH__ 750
#define WEX_CUDA_NR2_LOOKUP 0
#define wex_cuda_register_model sm75_register_model
#define wex_cuda_helper_abi sm75_helper_abi
#define wex_cuda_helper_features sm75_helper_features
#define wex_cuda_decode_to_host sm75_decode_to_host
#define wex_cuda_decode_to_device sm75_decode_to_device
#define wex_cuda_decode_batch_to_device sm75_decode_batch_to_device
#define wex_cuda_decode_batch_to_host sm75_decode_batch_to_host
#define wex_cuda_decode_batch_to_host_scratch sm75_decode_batch_to_host_scratch
#define wex_cuda_host_register sm75_host_register
#define wex_cuda_host_unregister sm75_host_unregister
#include "wex_cuda_host/tools/wex_cuda_bridge.cu"
}
#undef __CUDA_ARCH__

/* ---- the kernels without the bridge: the error word ------------------------------------------ */
template<class Plan, class Tables> struct Direct {
    int (*plan)(const WexEntry *, const unsigned char *, size_t, WexNr2Record *, Plan *);
    void (*offsets)(const WexNr2Record *, const Plan *, uint32_t *);
    size_t (*table_bytes)(int);
    cudaError_t (*build)(unsigned char *, int, cudaStream_t, Tables *);
    cudaError_t (*launch[2])(const Plan *, const Tables *, const unsigned char *, const uint32_t *, unsigned char *,
                             uint32_t *, cudaStream_t);
    uint32_t ids[16];
    Tables tables[16];
    unsigned count;
};
static Direct<sm86::WexCudaNr2Plan, sm86::WexCudaNr2Tables> direct86 = {
    sm86::wex_cuda_plan_nr2, sm86::wex_cuda_nr2_offsets, sm86::wex_cuda_nr2_table_bytes, sm86::wex_cuda_build_nr2_tables,
    {sm86::wex_cuda_launch_nr2_lookup<0>, sm86::wex_cuda_launch_nr2_lookup<1>}, {0}, {}, 0};
static Direct<sm75::WexCudaNr2Plan, sm75::WexCudaNr2Tables> direct75 = {
    sm75::wex_cuda_plan_nr2, sm75::wex_cuda_nr2_offsets, sm75::wex_cuda_nr2_table_bytes, sm75::wex_cuda_build_nr2_tables,
    {sm75::wex_cuda_launch_nr2_lookup<0>, sm75::wex_cuda_launch_nr2_lookup<1>}, {0}, {}, 0};

#define CUDA_OK(call) do { if ((call) != cudaSuccess) { fprintf(stderr, "%s failed\n", #call); abort(); } } while (0)

/* -1: the plan refuses the record; 1: its model has no tables here; 0: decoded, *error holds the error word */
template<class Plan, class Tables>
static int direct_decode(Direct<Plan, Tables> &d, int lookup, const WexEntry *entry, const unsigned char *record,
                         const unsigned char *blob, uint32_t *error) {
    WexNr2Record view;
    Plan plan;
    unsigned char *device_record, *raw, *base;
    uint32_t *offsets, *device_offsets, *device_error, *host_error;
    uint64_t raw_bytes, aligned;
    cudaStream_t stream;
    unsigned i;
    if (d.plan(entry, record, entry->total_bytes, &view, &plan)) return -1;
    if (!blob) return 1;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (i = 0; i < d.count && d.ids[i] != plan.model_id; i++) {}
    if (i == d.count) {   /* the tables of this model, with buckets (lookup 0 needs them, lookup 1 ignores them) */
        if (i == 16u) abort();
        CUDA_OK(cudaMalloc((void **)&base, d.table_bytes(1)));
        CUDA_OK(cudaMemcpyAsync(base, blob, WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice, stream));
        CUDA_OK(d.build(base, 1, stream, &d.tables[i]));
        CUDA_OK(cudaStreamSynchronize(stream));
        d.ids[i] = plan.model_id;
        d.count++;
    }
    if (wex_entry_raw_sizes(entry, &raw_bytes, &aligned)) abort();
    CUDA_OK(cudaMalloc((void **)&device_record, entry->total_bytes));
    CUDA_OK(cudaMalloc((void **)&raw, (size_t)aligned));
    CUDA_OK(cudaMalloc((void **)&device_offsets, ((size_t)plan.streams + 1u) * 4u));
    CUDA_OK(cudaMalloc((void **)&device_error, 4u));
    CUDA_OK(cudaMallocHost((void **)&offsets, ((size_t)plan.streams + 1u) * 4u));
    CUDA_OK(cudaMallocHost((void **)&host_error, 4u));
    d.offsets(&view, &plan, offsets);
    CUDA_OK(cudaMemcpyAsync(device_record, record, entry->total_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(device_offsets, offsets, ((size_t)plan.streams + 1u) * 4u, cudaMemcpyHostToDevice, stream));
    CUDA_OK(d.launch[lookup](&plan, &d.tables[i], device_record, device_offsets, raw, device_error, stream));
    CUDA_OK(cudaMemcpyAsync(host_error, device_error, 4u, cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    *error = *host_error;
    CUDA_OK(cudaFreeHost(host_error)); CUDA_OK(cudaFreeHost(offsets));
    CUDA_OK(cudaFree(device_error)); CUDA_OK(cudaFree(device_offsets)); CUDA_OK(cudaFree(raw)); CUDA_OK(cudaFree(device_record));
    CUDA_OK(cudaStreamDestroy(stream));
    return 0;
}

/* ---- the bridges --------------------------------------------------------------------------------- */
typedef int (*RegisterFn)(const unsigned char *, size_t);
typedef int (*HostFn)(const WexEntry *, unsigned char *, size_t);
typedef int (*DeviceFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*BatchFn)(int, const WexEntry *const *, const unsigned char *const *, int, unsigned char *const *,
                       unsigned char *const *, int *);
typedef int (*BatchHostFn)(int, const WexEntry *const *, const unsigned char *const *, int, unsigned char *const *, int *);
struct Build {
    const char *name;
    int major, minor;
    size_t shared;
    RegisterFn reg;
    HostFn to_host;
    DeviceFn to_device;
    BatchFn batch;
    BatchHostFn batch_host;
    BatchHostFn batch_host_scratch;
    unsigned (*features)(void);
};
static const Build builds[2] = {
    {"sm86", 8, 6, 101376u, sm86::sm86_register_model, sm86::sm86_decode_to_host, sm86::sm86_decode_to_device,
     sm86::sm86_decode_batch_to_device, sm86::sm86_decode_batch_to_host, sm86::sm86_decode_batch_to_host_scratch,
     sm86::sm86_helper_features},
    {"sm75", 7, 5, 65536u, sm75::sm75_register_model, sm75::sm75_decode_to_host, sm75::sm75_decode_to_device,
     sm75::sm75_decode_batch_to_device, sm75::sm75_decode_batch_to_host, sm75::sm75_decode_batch_to_host_scratch,
     sm75::sm75_helper_features},
};

/* ---- the models: registered on the CPU side and with both bridges ------------------------------ */
struct Model { uint32_t id; unsigned char *blob; WexNr2Model view; };
static Model models[16];
static unsigned model_count;
static const Model *model_named(uint32_t id) {
    unsigned i;
    for (i = 0; i < model_count; i++) if (models[i].id == id) return &models[i];
    return NULL;
}
/* register a blob that wex_nr2_model_open accepts; NULL when it does not or the registries are full */
static const Model *model_register(const unsigned char *blob, size_t bytes) {
    WexNr2Model probe;
    Model *m;
    unsigned b;
    if (wex_nr2_model_open(blob, bytes, &probe)) return NULL;
    if (model_named(probe.id)) return model_named(probe.id);
    if (model_count == 16u) return NULL;
    for (b = 0; b < 2u; b++) {
        wex_shim_set_device(builds[b].major, builds[b].minor, builds[b].shared);
        FUZZ_REQUIRE(builds[b].reg(blob, bytes) == 0, "the bridge refuses a model blob that wex_nr2_model_open accepts");
    }
    m = &models[model_count++];
    m->blob = (unsigned char *)malloc(WEX_NR2_MODEL_BYTES);
    if (!m->blob) abort();
    memcpy(m->blob, blob, WEX_NR2_MODEL_BYTES);
    m->view.blob = m->blob; m->view.id = probe.id; m->id = probe.id;
    return m;
}

/* ---- the base record (as fuzz_wnr3_record.c) --------------------------------------------------- */
static unsigned char *base_file, *base_stored, *base_expected;
static Wex4Record base_record;
static WexNr2Layout base_layout;
static WexEntry base_entry;
static const Model *base_model;
static size_t record_cap;

static unsigned long long count_inputs, count_files, count_records, count_accepted, count_host_refused,
    count_device_refused, count_unregistered, count_skipped, count_error_bits[4], count_calls, count_selftest;
#ifdef WEX_FUZZ_GPU
static void gpu_init(void);
static void gpu_report(void);
#endif

static void fuzz_report(void) {
    fprintf(stderr, "fuzz_wnr3_cuda: %llu inputs (%llu WEX4 files), %llu WNR3 records (%llu of them synthetic or crafted, "
            "at start), each through 2 builds x (5 bridge "
            "calls + 2 kernel lookups) (%llu GPU calls): %llu accepted by the CPU and every GPU path, %llu refused "
            "by the host checks, %llu refused on the device (error bits 1: %llu, 2: %llu, 4: %llu, 8: %llu), %llu "
            "naming a model that is not registered, %llu skipped (layout or size); %llu%s different decoded "
            "records; emulated: %llu launches, %llu blocks (%llu with a host thread per CUDA thread), %llu threads, "
            "%llu barriers, %llu cp.async copies, %llu queued copies (structured mutations %s)\n",
            count_inputs, count_files, count_records, count_selftest, count_calls, count_accepted, count_host_refused,
            count_device_refused, count_error_bits[0], count_error_bits[1], count_error_bits[2], count_error_bits[3],
            count_unregistered, count_skipped, fuzz_distinct, fuzz_distinct >= FUZZ_SEEN_MAX ? " or more" : "",
            wex_shim_stats.launches, wex_shim_stats.blocks, wex_shim_stats.pooled_blocks, wex_shim_stats.threads,
            wex_shim_stats.barriers, wex_shim_stats.async_copies, wex_shim_stats.copies, fuzz_structured() ? "on" : "off");
#ifdef WEX_FUZZ_GPU
    gpu_report();
#endif
}

static void selftest(void);
static int cpu_decode(const WexEntry *entry, const unsigned char *record, const Model *model, unsigned char *slab,
                      size_t aligned) {
    size_t at = wex_record_read_offset(entry);
    memset(slab, 0x5a, aligned);
    if (at && entry->total_bytes <= aligned - at) memcpy(slab + at, record, entry->total_bytes);
    return model ? wex_decode_nr2_record(entry, slab, aligned, &model->view) : -1;
}

static void fuzz_setup(void) {
    const char *path = getenv("WEX_FUZZ_BASE");
    Wex4Header h;
    Wex4View v;
    Wex4Blob blob;
    size_t size = 0;
    uint64_t raw, aligned;
    uint32_t r;
#ifdef WEX_FUZZ_GPU
    gpu_init();
#endif
    if (!path || !path[0]) path = "wex4/vectors/valid/tiny2-cz-upgraded.wex4";
    base_file = fuzz_read_file(path, &size);
    if (!base_file || size < WEX4_PAGE || wex4_parse_header(base_file, size, &h) ||
        wex4_open(&h, base_file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v)) {
        fprintf(stderr, "%s: not a WEX4 file that opens (set WEX_FUZZ_BASE, or run from the repository root)\n", path);
        exit(2);
    }
    for (r = 0; r < h.records; r++) {
        wex4_get_record(&v, r, &base_record);
        if (base_record.codec == WEX4_CODEC_WNR3) break;
    }
    if (r == h.records) { fprintf(stderr, "%s: no WNR3 record\n", path); exit(2); }
    wex4_get_blob(&v, base_record.blob - 1u, &blob);
    base_model = model_register(base_file + blob.offset, (size_t)blob.bytes);
    if (!base_model || base_model->id != blob.id || wex4_expert_entry(&v, r, &base_entry) ||
        wex_entry_raw_sizes(&base_entry, &raw, &aligned)) {
        fprintf(stderr, "%s: the model blob or the expert entry of record %u is not valid\n", path, r);
        exit(2);
    }
    wex_nr2_layout(&base_entry, &base_layout);
    base_stored = base_file + base_record.offset;
    record_cap = (size_t)wex4_align(base_record.decoded);
    base_expected = (unsigned char *)malloc((size_t)aligned);
    if (!base_expected) abort();
    if (cpu_decode(&base_entry, base_stored, base_model, base_expected, (size_t)aligned)) {
        fprintf(stderr, "%s: record %u does not decode\n", path, r);
        exit(2);
    }
    atexit(fuzz_report);
    if (!getenv("WEX_FUZZ_NO_SELFTEST")) selftest();
}

/* ---- one record through every path -------------------------------------------------------------- */
#define FUZZ_CUDA_MAX_SLAB (16u << 20)

static int all_bytes(const unsigned char *p, size_t n, unsigned char value) {
    size_t i;
    for (i = 0; i < n; i++) if (p[i] != value) return 0;
    return 1;
}

#ifdef WEX_FUZZ_GPU
#include "cuda_gpu.h"   /* the same kernels on an NVIDIA GPU, as PTX that clang wrote (run.sh --gpu) */
#endif

/* the CPU reference's verdict (0 or -1), or 1 when the record was skipped */
static int compare(const WexEntry *entry, const unsigned char *stored) {
    uint64_t raw, aligned;
    WexNr2Layout layout;
    WexNr2Record view;
    const Model *model;
    unsigned char *record, *expected, *slab, *scale_dst[3], *weight_dst[3];
    unsigned char *batch_scale[6], *batch_weight[6], *host_dst[2], *host_before[2];
    size_t host_bytes[2];
    size_t matrix_bytes[6];
    int cpu, parses, expect_fail, b, i, input_first = (int)(count_records & 1u);
    unsigned m;
    if (wex_entry_raw_sizes(entry, &raw, &aligned) || aligned > FUZZ_CUDA_MAX_SLAB || !entry->total_bytes) {
        count_skipped++;
        return 1;
    }
    count_records++;
    record = (unsigned char *)malloc(entry->total_bytes);   /* exactly the record: a read past it is reported */
    expected = (unsigned char *)malloc((size_t)aligned);
    slab = (unsigned char *)malloc((size_t)aligned);
    if (!record || !expected || !slab) abort();
    memcpy(record, stored, entry->total_bytes);
    /* the bridge parses with any model id and then asks its registry; the engine passes the model the record names */
    wex_nr2_layout(entry, &layout);
    parses = wex_entry_nr_model(entry) && entry->total_bytes < aligned &&
             !wex_nrx_parse(record, entry->total_bytes, &layout, 0, wex_entry_nr_magic(entry), &view);
    model = parses ? model_named(view.model_id) : NULL;
    cpu = cpu_decode(entry, record, model, expected, (size_t)aligned);
    /* the status every bridge call must return */
    expect_fail = cpu ? (parses && !model ? 1 : -1) : 0;
    if (parses && !model) count_unregistered++;
    else if (!cpu) { count_accepted++; fuzz_note_distinct(expected, (size_t)raw); }
    else if (!parses) count_host_refused++;
    else count_device_refused++;
    for (m = 0; m < 3u; m++) {
        matrix_bytes[m] = (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
        matrix_bytes[3u + m] = (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
    }
    for (b = 0; b < 2; b++) {
        const Build &g = builds[b];
        const WexEntry *entries[2];
        const unsigned char *records[2], *host_records[2];
        int status[2], bad, k, base_at = input_first ? 1 : 0, input_at = input_first ? 0 : 1;
        uint32_t error[2] = {0, 0};
        wex_shim_set_device(g.major, g.minor, g.shared);
        FUZZ_REQUIRE(g.features() == (63u | 256u), "the bridge's feature bits are not 319");
        /* to host: the record at the end of the slab, the slab untouched on failure */
        cpu_decode(entry, record, NULL, slab, (size_t)aligned);
        i = g.to_host(entry, slab, (size_t)aligned);
        count_calls++;
        FUZZ_REQUIRE(i == expect_fail, "wex_cuda_decode_to_host and the CPU reference give different verdicts");
        if (!i) FUZZ_REQUIRE(!memcmp(slab, expected, (size_t)raw), "wex_cuda_decode_to_host decodes other bytes than the CPU");
        else {
            unsigned char *before = (unsigned char *)malloc((size_t)aligned);
            if (!before) abort();
            cpu_decode(entry, record, NULL, before, (size_t)aligned);
            FUZZ_REQUIRE(!memcmp(slab, before, (size_t)aligned), "wex_cuda_decode_to_host changed the slab of a record it refused");
            free(before);
        }
        /* to device: six buffers of the caller, untouched on failure */
        for (m = 0; m < 3u; m++) {
            CUDA_OK(cudaMalloc((void **)&scale_dst[m], matrix_bytes[m]));
            CUDA_OK(cudaMalloc((void **)&weight_dst[m], matrix_bytes[3u + m]));
            memset(scale_dst[m], 0xc3, matrix_bytes[m]); memset(weight_dst[m], 0xc3, matrix_bytes[3u + m]);
        }
        i = g.to_device(entry, record, 0, scale_dst, weight_dst);
        count_calls++;
        FUZZ_REQUIRE(i == expect_fail, "wex_cuda_decode_to_device and the CPU reference give different verdicts");
        for (m = 0; m < 3u; m++) {
            if (!i) {
                FUZZ_REQUIRE(!memcmp(scale_dst[m], expected + entry->scale_offsets[m], matrix_bytes[m]) &&
                             !memcmp(weight_dst[m], expected + entry->weight_offsets[m], matrix_bytes[3u + m]),
                             "wex_cuda_decode_to_device decodes other bytes than the CPU");
            } else {
                FUZZ_REQUIRE(all_bytes(scale_dst[m], matrix_bytes[m], 0xc3) && all_bytes(weight_dst[m], matrix_bytes[3u + m], 0xc3),
                             "wex_cuda_decode_to_device wrote the destination of a record it refused");
            }
            CUDA_OK(cudaFree(scale_dst[m])); CUDA_OK(cudaFree(weight_dst[m]));
        }
        /* the two batches: the record and the valid base record, in either order */
        entries[input_at] = entry; records[input_at] = record;
        entries[base_at] = &base_entry; records[base_at] = base_stored;
        for (k = 0; k < 2; k++)
            for (m = 0; m < 3u; m++) {
                const WexEntry *e = entries[k];
                size_t sb = (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
                size_t wb = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
                CUDA_OK(cudaMalloc((void **)&batch_scale[3 * k + m], sb));
                CUDA_OK(cudaMalloc((void **)&batch_weight[3 * k + m], wb));
                memset(batch_scale[3 * k + m], 0xc3, sb); memset(batch_weight[3 * k + m], 0xc3, wb);
            }
        bad = g.batch(2, entries, records, 0, batch_scale, batch_weight, status);
        count_calls++;
        FUZZ_REQUIRE(status[base_at] == 0, "wex_cuda_decode_batch_to_device refused the valid record next to another one");
        FUZZ_REQUIRE(status[input_at] == expect_fail, "wex_cuda_decode_batch_to_device and the CPU reference give different verdicts");
        FUZZ_REQUIRE(bad == (status[0] != 0) + (status[1] != 0), "wex_cuda_decode_batch_to_device returns a wrong count");
        for (k = 0; k < 2; k++)
            for (m = 0; m < 3u; m++) {
                const WexEntry *e = entries[k];
                const unsigned char *want = k == base_at ? base_expected : expected;
                size_t sb = (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
                size_t wb = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
                if (!status[k]) {
                    FUZZ_REQUIRE(!memcmp(batch_scale[3 * k + m], want + e->scale_offsets[m], sb) &&
                                 !memcmp(batch_weight[3 * k + m], want + e->weight_offsets[m], wb),
                                 "wex_cuda_decode_batch_to_device decodes other bytes than the CPU");
                } else {
                    FUZZ_REQUIRE(all_bytes(batch_scale[3 * k + m], sb, 0xc3) && all_bytes(batch_weight[3 * k + m], wb, 0xc3),
                                 "wex_cuda_decode_batch_to_device wrote the destination of a record it refused");
                }
                CUDA_OK(cudaFree(batch_scale[3 * k + m])); CUDA_OK(cudaFree(batch_weight[3 * k + m]));
            }
        /* to host as c/qwen38_wex.h calls it: each record in the tail of its own slab, decoded over itself */
        for (k = 0; k < 2; k++) {
            uint64_t r2, a2;
            size_t at = wex_record_read_offset(entries[k]);
            if (wex_entry_raw_sizes(entries[k], &r2, &a2)) abort();
            host_dst[k] = (unsigned char *)malloc((size_t)a2);
            if (!host_dst[k]) abort();
            memset(host_dst[k], 0xc3, (size_t)a2);
            host_records[k] = records[k];
            if (at) {   /* at + total_bytes is the slab's size: a read past the record is a read past the slab */
                memcpy(host_dst[k] + at, records[k], entries[k]->total_bytes);
                host_records[k] = host_dst[k] + at;
            }
            host_bytes[k] = (size_t)a2;   /* the slab as the call finds it, record in its tail included */
            host_before[k] = (unsigned char *)malloc(host_bytes[k]);
            if (!host_before[k]) abort();
            memcpy(host_before[k], host_dst[k], host_bytes[k]);
        }
        bad = g.batch_host(2, entries, host_records, 0, host_dst, status);
        count_calls++;
        FUZZ_REQUIRE(status[base_at] == 0 && !memcmp(host_dst[base_at], base_expected, (size_t)base_record.decoded),
                     "wex_cuda_decode_batch_to_host did not decode the valid record next to another one");
        FUZZ_REQUIRE(status[input_at] == expect_fail, "wex_cuda_decode_batch_to_host and the CPU reference give different verdicts");
        FUZZ_REQUIRE(bad == (status[0] != 0) + (status[1] != 0), "wex_cuda_decode_batch_to_host returns a wrong count");
        if (!status[input_at]) FUZZ_REQUIRE(!memcmp(host_dst[input_at], expected, (size_t)raw),
                                            "wex_cuda_decode_batch_to_host decodes other bytes than the CPU");
        for (k = 0; k < 2; k++)
            if (status[k]) FUZZ_REQUIRE(!memcmp(host_dst[k], host_before[k], host_bytes[k]),
                                        "wex_cuda_decode_batch_to_host wrote the destination of a record it refused");
        /* to host scratch (feature 256): non-overlapping caller scratch buffers, single sync */
        for (k = 0; k < 2; k++) memset(host_dst[k], 0xc3, host_bytes[k]);
        bad = g.batch_host_scratch(2, entries, records, 0, host_dst, status);
        count_calls++;
        FUZZ_REQUIRE(status[base_at] == 0 && !memcmp(host_dst[base_at], base_expected, (size_t)base_record.decoded),
                     "wex_cuda_decode_batch_to_host_scratch did not decode the valid record next to another one");
        FUZZ_REQUIRE(status[input_at] == expect_fail,
                     "wex_cuda_decode_batch_to_host_scratch and the CPU reference give different verdicts");
        FUZZ_REQUIRE(bad == (status[0] != 0) + (status[1] != 0),
                     "wex_cuda_decode_batch_to_host_scratch returns a wrong count");
        if (!status[input_at]) FUZZ_REQUIRE(!memcmp(host_dst[input_at], expected, (size_t)raw),
                                            "wex_cuda_decode_batch_to_host_scratch decodes other bytes than the CPU");
        free(host_dst[0]); free(host_dst[1]); free(host_before[0]); free(host_before[1]);
        /* the kernels alone, both lookups: the error word */
        for (k = 0; k < 2; k++) {
            int d = b ? direct_decode(direct75, k, entry, record, model ? model->blob : NULL, &error[k])
                      : direct_decode(direct86, k, entry, record, model ? model->blob : NULL, &error[k]);
            count_calls++;
            FUZZ_REQUIRE(d == (parses ? (model ? 0 : 1) : -1), "the plan of wex_cuda.cuh and the record checks of the CPU disagree");
            if (d) continue;
            FUZZ_REQUIRE(!error[k] == !cpu, "the kernels' error word and the CPU reference give different verdicts");
            FUZZ_REQUIRE(!(error[k] & ~15u), "an error bit outside 1 | 2 | 4 | 8");
            FUZZ_REQUIRE(!(error[k] & WEX_CUDA_ERR_CZ) || (record[27] & WEX_NR3_FLAG_CZ), "error bit 8 for a record without the CZ flag");
            FUZZ_REQUIRE(!(error[k] & WEX_CUDA_ERR_CEIL) || (record[27] & WEX_NR3_FLAG_CEIL_MASK),
                         "error bit 4 for a record without a CEIL flag");
            if (!b && k == 1) {
                unsigned bit;
                for (bit = 0; bit < 4u; bit++) count_error_bits[bit] += (error[k] >> bit) & 1u;
            }
        }
    }
#ifdef WEX_FUZZ_GPU
    gpu_compare(entry, record, model, cpu, expected);
#endif
    free(slab);
    free(expected);
    free(record);
    return cpu;
}

/* ---- synthetic and crafted records: what no vector has --------------------------------------------- *
 * Run once at start. A synthetic expert (the block kinds of the generator of c/tests/test_wex_nr3_gpu.cu, with
 * smaller magnitudes and fewer sign bits so that it compresses, plus a block of maximum 5 in matrix 1, so that
 * matrices 0 and 2 are in ceil form and matrix 1 is not) of 497, 315 and 495
 * scales (none a multiple of 4: the last scale stream of each matrix ends in a hash tail), a model trained on it,
 * and records without and with canonical zero at three chunk pairs. Then records that only one check refuses:
 * a stream with one byte after its last symbol, a stream whose initial state is below L or at 256 L and above
 * although its symbols, its end and its hash are right (each for the scale and for the weight kernel), a
 * canonical-zero record that decodes a -0, and a block in ceil form at coded scale 0. Last, a record at the
 * highest cost per symbol (selftest_cost). Each must decode to the source (or Z(source)), or be refused by the
 * CPU reference, before it goes through compare(). */
static uint32_t synth_rng(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static void synth_fill(const WexEntry *e, unsigned char *raw, uint32_t seed) {
    static const unsigned char general[16] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 6};
    static const unsigned char even[8] = {0, 0, 2, 2, 4, 5, 6, 7};
    uint32_t s = seed, m, b, k;
    for (m = 0; m < 3u; m++) {
        uint32_t blocks = (uint32_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
        unsigned char *sc = raw + e->scale_offsets[m], *w = raw + e->weight_offsets[m];
        for (b = 0; b < blocks; b++) {
            uint32_t kind = synth_rng(&s) % 8u;
            unsigned char *q = w + (size_t)b * 16u;
            sc[b] = (unsigned char)(118u + synth_rng(&s) % 5u);
            if (kind == 4u && b % 37u == 0) sc[b] = (unsigned char)(b & 1u ? 255u : 254u);
            if (kind == 6u && b % 41u == 0) sc[b] = 0;   /* scale 0 on a block that is not even is legal */
            for (k = 0; k < 32u; k++) {
                uint32_t r = synth_rng(&s);
                uint32_t a = kind < 4u ? general[r & 15u] : kind < 6u ? even[r & 7u] : kind == 6u ? (r & 7u) % 5u : 0u;
                uint32_t x = a | ((r >> 8) & 3u ? 0u : 8u);   /* a sign bit in one nibble in four */
                if (k < 2u && kind < 6u) x = (x & 8u) | 7u;
                if (m == 1u && b == 3u) x = (x & 8u) | (k ? (r & 7u) % 6u : 5u);   /* the block of maximum 5 */
                if (k & 1u) q[k / 2u] = (unsigned char)(q[k / 2u] | x << 4); else q[k / 2u] = (unsigned char)x;
            }
        }
    }
}
/* the record with stream g replaced by `bytes` (n of them), zero-padded; its size in *total */
static unsigned char *with_stream(const unsigned char *record, size_t total, const WexNr2Layout *layout, uint32_t g,
                                  const unsigned char *bytes, uint32_t n, size_t *out_total) {
    WexNr2Record v;
    uint32_t s, at, old, used, sp, wp;
    unsigned char *out;
    if (wex_nr3_parse(record, total, layout, 0, &v)) abort();
    old = wex_nr_u16(v.lengths + (size_t)g * 2u);
    for (s = 0, at = v.metadata_bytes; s < g; s++) at += wex_nr_u16(v.lengths + (size_t)s * 2u);
    used = v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
    *out_total = ((size_t)used - old + n + 4095u) & ~(size_t)4095u;
    out = (unsigned char *)calloc(*out_total, 1u);
    if (!out) abort();
    memcpy(out, record, at);
    memcpy(out + at, bytes, n);
    memcpy(out + at + n, record + at + old, used - at - old);
    out[WEX_NR2_HEADER_BYTES + g * 2u] = (unsigned char)n; out[WEX_NR2_HEADER_BYTES + g * 2u + 1u] = (unsigned char)(n >> 8);
    sp = v.scale_payload_bytes; wp = v.weight_payload_bytes;
    if (g < v.scale_streams) sp = sp - old + n; else wp = wp - old + n;
    memcpy(out + 8, &sp, 4); memcpy(out + 12, &wp, 4);   /* little-endian hosts only, as the decoders */
    return out;
}
/* stream g of the record re-opened below L: its first symbol coded from the state x1 >> 8, the byte x1 & 255
 * put back in front of its other bytes. A decoder without the check of the initial state decodes the same
 * symbols to the same end and the same final state; NULL when this stream does not allow it (x1 < 2^15). */
static unsigned char *below_l(const unsigned char *record, size_t total, const WexNr2Layout *layout, uint32_t g,
                              const unsigned char *cdf, size_t *out_total) {
    WexNr2Record v;
    uint32_t s, at, n, x, slot, lo = 0, hi, start, frequency, x1, y, opened;
    unsigned char *bytes, *out;
    if (wex_nr3_parse(record, total, layout, 0, &v)) abort();
    for (s = 0, at = v.metadata_bytes; s < g; s++) at += wex_nr_u16(v.lengths + (size_t)s * 2u);
    n = wex_nr_u16(v.lengths + (size_t)g * 2u);
    x = (uint32_t)record[at] << 24 | (uint32_t)record[at + 1] << 16 | (uint32_t)record[at + 2] << 8 | record[at + 3];
    slot = x & (WEX_NR2_TOTAL - 1u);
    hi = g < v.scale_streams ? 256u : 16u;
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
    start = wex_nr_u16(cdf + lo * 2u); frequency = wex_nr_u16(cdf + lo * 2u + 2u) - start;
    x1 = frequency * (x >> WEX_NR2_PRECISION) + slot - start;
    if (x1 < (1u << 15) || n + 1u > 65535u) return NULL;
    y = x1 >> 8;
    opened = ((y / frequency) << WEX_NR2_PRECISION) + y % frequency + start;   /* decodes to symbol lo and state y */
    FUZZ_REQUIRE(opened < WEX_NR_L, "the self-test could not open a stream below L");
    bytes = (unsigned char *)malloc(n + 1u);
    if (!bytes) abort();
    bytes[0] = (unsigned char)(opened >> 24); bytes[1] = (unsigned char)(opened >> 16);
    bytes[2] = (unsigned char)(opened >> 8); bytes[3] = (unsigned char)opened;
    bytes[4] = (unsigned char)x1;
    memcpy(bytes + 5, record + at + 4, n - 4u);
    out = with_stream(record, total, layout, g, bytes, n + 1u, out_total);
    free(bytes);
    return out;
}
/* stream g of the record re-opened at or above 256 L: x1, the state after its first symbol and the bytes that
 * symbol renormalized, is coded back with no byte out, so the 4 bytes of the new initial state stand for the old
 * ones and those renormalization bytes. A decoder without the upper bound of the initial state decodes the same
 * symbols to the same end and the same final state; NULL when x1 does not allow it (x1 / f must be in
 * [2^17, 2^18), f the first symbol's frequency). */
static unsigned char *above_256l(const unsigned char *record, size_t total, const WexNr2Layout *layout, uint32_t g,
                                 const unsigned char *cdf, size_t *out_total) {
    WexNr2Record v;
    uint32_t s, at, n, x, slot, lo = 0, hi, start, frequency, x1, used = 4, opened;
    unsigned char *bytes, *out;
    if (wex_nr3_parse(record, total, layout, 0, &v)) abort();
    for (s = 0, at = v.metadata_bytes; s < g; s++) at += wex_nr_u16(v.lengths + (size_t)s * 2u);
    n = wex_nr_u16(v.lengths + (size_t)g * 2u);
    x = (uint32_t)record[at] << 24 | (uint32_t)record[at + 1] << 16 | (uint32_t)record[at + 2] << 8 | record[at + 3];
    slot = x & (WEX_NR2_TOTAL - 1u);
    hi = g < v.scale_streams ? 256u : 16u;
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
    start = wex_nr_u16(cdf + lo * 2u); frequency = wex_nr_u16(cdf + lo * 2u + 2u) - start;
    x1 = frequency * (x >> WEX_NR2_PRECISION) + slot - start;
    while (x1 < WEX_NR_L && used < n) x1 = x1 << 8 | record[at + used++];
    if (x1 < WEX_NR_L || x1 / frequency < (1u << 17) || x1 / frequency >= (1u << 18)) return NULL;
    opened = ((x1 / frequency) << WEX_NR2_PRECISION) + x1 % frequency + start;   /* decodes to symbol lo, state x1 */
    FUZZ_REQUIRE(opened >= (WEX_NR_L << 8), "the self-test could not open a stream at 256 L or above");
    bytes = (unsigned char *)malloc(n);
    if (!bytes) abort();
    bytes[0] = (unsigned char)(opened >> 24); bytes[1] = (unsigned char)(opened >> 16);
    bytes[2] = (unsigned char)(opened >> 8); bytes[3] = (unsigned char)opened;
    memcpy(bytes + 4, record + at + used, n - used);
    out = with_stream(record, total, layout, g, bytes, n - (used - 4u), out_total);
    free(bytes);
    return out;
}
/* the CDF of the first symbol of stream g: the scale CDF of its matrix, or the weight CDF of the context of a
 * block start (previous and maximum "none") */
static const unsigned char *first_cdf(const WexNr2Model *model, const WexNr2Record *v, const WexNr2Layout *layout,
                                      const unsigned char *const scales[3], uint32_t g) {
    uint32_t m, k = g < v->scale_streams ? 0u : 1u, s = k ? g - v->scale_streams : g;
    for (m = 0; m < 3u; m++) {
        uint32_t count = k ? layout->symbol_count[m] : layout->scale_count[m];
        uint32_t chunk = k ? v->weight_chunk : v->scale_chunk, streams = wex_nr2_streams(count, chunk);
        if (s < streams) {
            if (!k) return wex_nr2_scale_cdf(model, m);
            return wex_nr2_weight_cdf(model, m, wex_nr2_srel(scales[m][s * chunk / 32u], v->median[m]) * 81u + 80u);
        }
        s -= streams;
    }
    abort();
}

static void selftest_one(WexEntry entry, const unsigned char *record, size_t total, const unsigned char *want,
                         const char *what) {
    unsigned char *copy = (unsigned char *)malloc(total);
    int cpu;
    if (!copy) abort();
    memcpy(copy, record, total);
    entry.total_bytes = (uint32_t)total;
    cpu = compare(&entry, copy);
    count_selftest++;
    if (want) FUZZ_REQUIRE(cpu == 0, what);
    else FUZZ_REQUIRE(cpu == -1, what);
    if (want) {
        uint64_t raw, aligned;
        unsigned char *slab;
        if (wex_entry_raw_sizes(&entry, &raw, &aligned)) abort();
        slab = (unsigned char *)malloc((size_t)aligned);
        if (!slab) abort();
        cpu_decode(&entry, copy, model_named(wex_nr_u32(copy + 4)), slab, (size_t)aligned);
        FUZZ_REQUIRE(!memcmp(slab, want, (size_t)raw), "a synthetic record of the self-test does not decode to its source");
        free(slab);
    }
    free(copy);
}

/* The payload ring of the two kernels at its highest load: a model in which, in every context, weight symbol 0
 * has 16,369 of the 16,384 slots and the other 15 symbols one slot each, and scale 120 all but 255 slots, so that
 * every other symbol costs 14 bits (1 or 2 bytes of renormalization each). The first weight stream of matrix 0
 * (16 blocks at chunk 512) holds only such symbols, 56 bytes per block, and so do the first 128 scales of matrix
 * 0 (4 groups of 32); the rest is weight 0 and scale 120. Each 32 symbols of those streams take 14 of the 16
 * words a refill of the ring adds. */
static void selftest_cost(const WexEntry &e, const WexNr2Layout &layout, size_t raw, size_t aligned) {
    uint64_t *wc = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), *sc = (uint64_t *)calloc(3u * 256u, 8u);
    unsigned char blob[WEX_NR2_MODEL_BYTES], *data = (unsigned char *)calloc(raw, 1u), *out, *scales[3], *weights[3], anchor[3];
    size_t cap = aligned * 2u + 65536u, written = 0;
    const Model *model;
    uint32_t i, m;
    out = (unsigned char *)calloc(cap, 1u);
    if (!wc || !sc || !data || !out) abort();
    for (i = 0; i < 3u * WEX_NR2_CONTEXTS; i++) wc[(size_t)i * 16u] = 1u << 20;
    for (m = 0; m < 3u; m++) sc[m * 256u + 120u] = 1u << 20;
    wex_nr2_model_build(wc, sc, blob);
    model = model_register(blob, sizeof blob);
    FUZZ_REQUIRE(model != NULL, "the model of the cost self-test does not register");
    for (m = 0; m < 3u; m++) {
        scales[m] = data + e.scale_offsets[m];
        weights[m] = data + e.weight_offsets[m];
        memset(scales[m], 120, layout.scale_count[m]);
    }
    for (i = 0; i < 128u; i++) scales[0][i] = (unsigned char)(i * 37u + 1u == 120u ? 121u : i * 37u + 1u);
    for (i = 0; i < 512u; i++) weights[0][i / 2u] |= (unsigned char)((1u + i % 15u) << ((i & 1u) * 4u));
    for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[m], layout.scale_count[m]);
    FUZZ_REQUIRE(!wex_nrx_encode(&model->view, &layout, (const unsigned char *const *)scales,
                                 (const unsigned char *const *)weights, 512u, 256u, anchor, WEX_NR3_MAGIC, 0u, 0x00010002u,
                                 out, cap, &written), "the record of the cost self-test does not encode");
    written = (written + 4095u) & ~(size_t)4095u;
    FUZZ_REQUIRE(written < aligned, "the record of the cost self-test is not smaller than its slab");
    selftest_one(e, out, written, data, "the record at 14 bits per symbol is refused by the CPU");
    free(out); free(data); free(wc); free(sc);
}

static void selftest(void) {
    static const uint16_t rows[3] = {7, 5, 9}, scale_cols[3] = {71, 63, 55};   /* 497, 315 and 495 scales */
    static const uint32_t chunks[3][2] = {{512u, 256u}, {1024u, 512u}, {16384u, 16384u}};
    WexEntry e;
    WexNr2Layout layout;
    uint64_t raw, aligned;
    unsigned char *src, *coded[2], *scales[2][3], *weights[2][3], *expected[2], *out, blob[WEX_NR2_MODEL_BYTES];
    uint64_t *wc = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof(uint64_t)), *sc = (uint64_t *)calloc(3u * 256u, 8u);
    uint32_t flags[2], m, cz, c, at = 0;
    const Model *model;
    size_t cap, written;
    memset(&e, 0, sizeof e);
    for (m = 0; m < 3u; m++) {
        e.scale_shapes[m][0] = e.weight_shapes[m][0] = rows[m];
        e.scale_shapes[m][1] = scale_cols[m]; e.weight_shapes[m][1] = (uint16_t)(scale_cols[m] * 16u);
        e.scale_offsets[m] = at; at += (uint32_t)rows[m] * scale_cols[m];
    }
    e.scale_bytes = at;
    for (m = 0; m < 3u; m++) { e.weight_offsets[m] = at; at += (uint32_t)rows[m] * scale_cols[m] * 16u; }
    e.weight_bytes = at - e.scale_bytes;
    e.flags = WEX_ENTRY_NEURAL_RANS3;
    if (!wc || !sc || wex_entry_raw_sizes(&e, &raw, &aligned)) abort();
    wex_nr2_layout(&e, &layout);
    src = (unsigned char *)malloc((size_t)raw);
    cap = (size_t)aligned * 2u + 65536u;
    out = (unsigned char *)malloc(cap);
    if (!src || !out) abort();
    synth_fill(&e, src, 4242u);
    for (cz = 0; cz < 2u; cz++) {   /* the coded data: canonical zero (cz = 1), then the ceil form */
        coded[cz] = (unsigned char *)malloc((size_t)raw);
        expected[cz] = (unsigned char *)malloc((size_t)raw);
        if (!coded[cz] || !expected[cz]) abort();
        memcpy(coded[cz], src, (size_t)raw);
        memcpy(expected[cz], src, (size_t)raw);
        if (cz) {
            wex_nr3_canonical_zero(coded[cz] + e.scale_bytes, e.weight_bytes);
            wex_nr3_canonical_zero(expected[cz] + e.scale_bytes, e.weight_bytes);
        }
        for (m = 0; m < 3u; m++) { scales[cz][m] = coded[cz] + e.scale_offsets[m]; weights[cz][m] = coded[cz] + e.weight_offsets[m]; }
        flags[cz] = wex_nr3_ceil_forward(&layout, scales[cz], weights[cz], NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
        FUZZ_REQUIRE((flags[cz] & WEX_NR3_FLAG_CEIL_MASK) == 5u, "the self-test expert is not in ceil form in matrices 0 and 2 only");
        wex_nr3_count(&layout, (const unsigned char *const *)scales[cz], (const unsigned char *const *)weights[cz], wc, sc);
    }
    wex_nr2_model_build(wc, sc, blob);
    model = model_register(blob, sizeof blob);
    FUZZ_REQUIRE(model != NULL, "the model of the self-test does not register");
    for (cz = 0; cz < 2u; cz++)
        for (c = 0; c < 3u; c++) {
            memset(out, 0, cap);
            FUZZ_REQUIRE(!wex_nr3_encode(&model->view, &layout, (const unsigned char *const *)scales[cz],
                                         (const unsigned char *const *)weights[cz], chunks[c][0], chunks[c][1], flags[cz],
                                         0x00010002u, out, cap, &written), "the self-test record does not encode");
            written = (written + 4095u) & ~(size_t)4095u;
            FUZZ_REQUIRE(written + 4096u < aligned, "the self-test record is not a page smaller than its slab");
            selftest_one(e, out, written, expected[cz], "a synthetic record that the CPU reference refuses");
            if (cz && c == 0) {   /* the records that only one check refuses, from the CZ record of chunks 512:256 */
                WexNr2Record v;
                unsigned char *bad, extra[65536];
                size_t total, k;
                uint32_t g, n, first_weight;
                unsigned char anchor[3];
                if (wex_nr3_parse(out, written, &layout, model->id, &v)) abort();
                first_weight = v.scale_streams;
                for (g = 0; g < 2u; g++) {   /* one byte after the last symbol: scale stream 0, weight stream 0 */
                    uint32_t stream = g ? first_weight : 0u, s, pos = v.metadata_bytes;
                    for (s = 0; s < stream; s++) pos += wex_nr_u16(v.lengths + (size_t)s * 2u);
                    n = wex_nr_u16(v.lengths + (size_t)stream * 2u);
                    memcpy(extra, out + pos, n);
                    extra[n] = 0x5a;
                    bad = with_stream(out, written, &layout, stream, extra, n + 1u, &total);
                    selftest_one(e, bad, total, NULL, "a stream with a byte after its end is not refused by the CPU");
                    free(bad);
                }
                for (g = 0; g < 2u; g++) {   /* opened below L: scale stream 0, then the first weight stream */
                    const unsigned char *cdf = first_cdf(&model->view, &v, &layout, scales[cz], g ? first_weight : 0u);
                    bad = below_l(out, written, &layout, g ? first_weight : 0u, cdf, &total);
                    FUZZ_REQUIRE(bad != NULL, "the self-test record has no stream that can be opened below L");
                    selftest_one(e, bad, total, NULL, "a stream opened below L is not refused by the CPU");
                    free(bad);
                }
                for (g = 0; g < 2u; g++) {   /* opened at 256 L or above: the first scale and weight stream that allow it */
                    uint32_t stream = g ? first_weight : 0u, end = g ? v.scale_streams + v.weight_streams : v.scale_streams;
                    for (bad = NULL; !bad && stream < end; stream++)
                        bad = above_256l(out, written, &layout, stream,
                                         first_cdf(&model->view, &v, &layout, scales[cz], stream), &total);
                    FUZZ_REQUIRE(bad != NULL, "the self-test record has no stream that can be opened at 256 L");
                    selftest_one(e, bad, total, NULL, "a stream opened at 256 L or above is not refused by the CPU");
                    free(bad);
                }
                /* the CZ flag on data that holds -0 (the non-CZ coded data, which has -0 in its all-zero blocks) */
                for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[0][m], layout.scale_count[m]);
                memset(out, 0, cap);
                FUZZ_REQUIRE(!wex_nrx_encode(&model->view, &layout, (const unsigned char *const *)scales[0],
                                             (const unsigned char *const *)weights[0], 512u, 256u, anchor, WEX_NR3_MAGIC,
                                             flags[0] | WEX_NR3_FLAG_CZ, 0x00010002u, out, cap, &total),
                             "the self-test record with -0 does not encode");
                total = (total + 4095u) & ~(size_t)4095u;
                if (total < aligned) selftest_one(e, out, total, NULL, "a canonical-zero record with -0 is not refused by the CPU");
                /* a block in ceil form (coded maximum 5) of matrix 2 at coded scale 0 */
                for (k = 0; k < layout.scale_count[2] && wex_nr3_block_max(weights[1][2] + k * 16u, NULL) != 5u; k++) {}
                FUZZ_REQUIRE(k < layout.scale_count[2], "the self-test expert has no block in ceil form in matrix 2");
                scales[1][2][k] = 0;
                for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[1][m], layout.scale_count[m]);
                memset(out, 0, cap);
                FUZZ_REQUIRE(!wex_nrx_encode(&model->view, &layout, (const unsigned char *const *)scales[1],
                                             (const unsigned char *const *)weights[1], 512u, 256u, anchor, WEX_NR3_MAGIC,
                                             flags[1], 0x00010002u, out, cap, &total),
                             "the self-test record with a ceil block at scale 0 does not encode");
                total = (total + 4095u) & ~(size_t)4095u;
                if (total < aligned) selftest_one(e, out, total, NULL, "a ceil block at coded scale 0 is not refused by the CPU");
                scales[1][2][k] = (unsigned char)(src[e.scale_offsets[2] + k] + 1u);   /* back to the coded scale */
            }
        }
    free(out); free(src); free(wc); free(sc);
    for (cz = 0; cz < 2u; cz++) { free(coded[cz]); free(expected[cz]); }
    selftest_cost(e, layout, (size_t)raw, (size_t)aligned);
}

/* ---- the input ------------------------------------------------------------------------------------- */
static int open_wex4(const unsigned char *data, size_t size, Wex4Header *h, Wex4View *v) {
    return size >= WEX4_PAGE && wex_nr_u32(data) == WEX4_MAGIC && !wex4_parse_header(data, size, h) &&
           !wex4_open(h, data + h->index_offset, (size_t)wex4_align(h->index_bytes), v);
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed) {
    unsigned char *work;
    size_t capacity, stored, n;
    Wex4Header h;
    Wex4View v;
    if (!base_file) fuzz_setup();
    if (!fuzz_structured()) return LLVMFuzzerMutate(data, size, max_size);
    if (open_wex4(data, size, &h, &v)) {   /* a WEX4 file: one of its WNR3 records re-encoded in place, 1 call in 2 */
        if (seed % 2u == 0 && fuzz_wnr3_mutate_wex4(data, data, size, seed)) return size;
        return LLVMFuzzerMutate(data, size, max_size);
    }
    capacity = max_size & ~(size_t)4095u;   /* a record, as fuzz_wnr3_record.c: 2 calls in 3 */
    if (capacity > record_cap) capacity = record_cap;
    if (!(seed % 3u) || capacity < 4096u) return LLVMFuzzerMutate(data, size, max_size);
    work = (unsigned char *)calloc(capacity, 1u);
    if (!work) abort();
    stored = (size + 4095u) & ~(size_t)4095u;
    if (stored < 4096u) stored = 4096u;
    if (stored > capacity) stored = capacity;
    memcpy(work, data, size < stored ? size : stored);
    n = fuzz_wnr3_mutate(work, stored, capacity, &base_layout, &base_model->view, seed);
    if (!n && base_record.stored <= capacity) {
        memset(work, 0, capacity);
        memcpy(work, base_stored, (size_t)base_record.stored);
        n = fuzz_wnr3_mutate(work, (size_t)base_record.stored, capacity, &base_layout, &base_model->view, seed);
    }
    if (n) memcpy(data, work, n);
    free(work);
    return n ? n : LLVMFuzzerMutate(data, size, max_size);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    Wex4Header h;
    Wex4View v;
    uint64_t seed = UINT64_C(14695981039346656037);
    size_t i;
    if (!base_file) fuzz_setup();
    count_inputs++;
    for (i = 0; i < size; i++) { seed ^= data[i]; seed *= UINT64_C(1099511628211); }
    wex_shim_seed(seed);   /* the block order and the stream interleaving follow from the input */
    if (open_wex4(data, size, &h, &v)) {
        uint32_t r;
        count_files++;
        for (r = 0; r < h.records; r++) {
            Wex4Record rec;
            Wex4Blob blob;
            WexEntry entry;
            wex4_get_record(&v, r, &rec);
            if (rec.codec != WEX4_CODEC_WNR3) continue;
            wex4_get_blob(&v, rec.blob - 1u, &blob);
            if (blob.kind == WEX4_BLOB_WNM2) model_register(data + blob.offset, (size_t)blob.bytes);
            if (wex4_expert_entry(&v, r, &entry)) { count_skipped++; continue; }
            compare(&entry, data + rec.offset);
        }
        return 0;
    }
    {   /* one record against the base layout, zero-padded to 4096 bytes and cut at the base's decoded size */
        size_t stored_bytes = (size + 4095u) & ~(size_t)4095u;
        unsigned char *stored;
        WexEntry entry = base_entry;
        if (stored_bytes < 4096u) stored_bytes = 4096u;
        if (stored_bytes > record_cap) stored_bytes = record_cap;
        stored = (unsigned char *)calloc(stored_bytes, 1u);
        if (!stored) abort();
        memcpy(stored, data, size < stored_bytes ? size : stored_bytes);
        entry.total_bytes = (uint32_t)stored_bytes;
        compare(&entry, stored);
        free(stored);
    }
    return 0;
}
