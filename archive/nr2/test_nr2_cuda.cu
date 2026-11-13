/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* kaira_wex_cuda.dll feature test: WNR2 GPU decoding (WEX_CUDA_FEATURE_NR2).
 *
 * Like test_decode_to_device.cu, this executable links its own CUDA runtime
 * and loads the helper (which has another) with LoadLibrary: destination
 * buffers come from this runtime and are written by the helper's.
 *
 *  a. 12 real experts from a WEX1 container across layers, plus 3 synthetic
 *     experts (odd shapes, partial streams, scale_bytes % 16 != 0, wide scale
 *     spread, extra noise) sampled from a model of the real ones; the model
 *     is built from all their counts and records are encoded with the CPU
 *     reference encoder at weight_chunk 4096 and 8192 (scale_chunk 2048);
 *  b. the model is registered; wex_cuda_decode_to_host and _to_device must
 *     reproduce the raw slab byte for byte (and agree with wex_decode_nr2_record);
 *  c. malformed records return -1 with the destination untouched (0xA5 / the
 *     host slab unchanged), unregistered model ids return 1, and a fuzz pass
 *     checks that the helper and the CPU reference agree on every mutation;
 *  d. WNR1 regression: records of a WNR1 container still decode byte-exactly;
 *  e. timing: median of 20 wex_cuda_decode_to_device calls with page-locked
 *     staging, WNR2 at 4096 / 8192 against WNR1, and kernel-only times (this
 *     runtime, CUDA events) for both WNR2 lookups and for WNR1.
 *
 *  f. optional (--wnr2 container.wex, repeatable): records of real WNR2
 *     containers with their own model (header-page descriptor), helper vs
 *     CPU reference, and how many equal the WEX1 raw slab.
 *
 * usage: test_nr2_cuda helper.dll v1.wex v3-neural.wex [--baseline old_helper.dll]
 *        [--wnr2 container.wex ...] [--wnr2-records N] [--json out.json] [--dump DIR]
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "../../c/wex_cuda.cuh"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)

typedef unsigned int (*AbiFn)(void);
typedef int (*HostFn)(const WexEntry *, unsigned char *, size_t);
typedef int (*DeviceFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*RegisterFn)(void *, size_t);
typedef int (*UnregisterFn)(void *);
typedef int (*ModelFn)(const unsigned char *, size_t);

struct Helper { HMODULE module; AbiFn abi, features; HostFn to_host; DeviceFn to_device; RegisterFn pin; UnregisterFn unpin; ModelFn model; };

static Helper load_helper(const char *path) {
    Helper h;
    memset(&h, 0, sizeof(h));
    h.module = LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    CHECK(h.module);
    h.abi = (AbiFn)GetProcAddress(h.module, "wex_cuda_helper_abi");
    h.features = (AbiFn)GetProcAddress(h.module, "wex_cuda_helper_features");
    h.to_host = (HostFn)GetProcAddress(h.module, "wex_cuda_decode_to_host");
    h.to_device = (DeviceFn)GetProcAddress(h.module, "wex_cuda_decode_to_device");
    h.pin = (RegisterFn)GetProcAddress(h.module, "wex_cuda_host_register");
    h.unpin = (UnregisterFn)GetProcAddress(h.module, "wex_cuda_host_unregister");
    h.model = (ModelFn)GetProcAddress(h.module, "wex_cuda_register_model");
    CHECK(h.abi && h.abi() == 1u && h.features && h.to_host && h.to_device && h.pin && h.unpin);
    return h;
}

static double now() { LARGE_INTEGER n, f; QueryPerformanceCounter(&n); QueryPerformanceFrequency(&f); return (double)n.QuadPart / f.QuadPart; }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
static size_t align4k(size_t v) { return (v + 4095u) & ~(size_t)4095u; }

/* ---- containers (header + table only; records read on demand) ---------- */
struct Container { FILE *file; WexHeader h; std::vector<WexEntry> entries; uint64_t bytes; };

static Container open_container(const char *path) {
    Container c;
    c.file = fopen(path, "rb"); CHECK(c.file);
    CHECK(!_fseeki64(c.file, 0, SEEK_END)); c.bytes = (uint64_t)_ftelli64(c.file);
    CHECK(!_fseeki64(c.file, 0, SEEK_SET));
    CHECK(fread(&c.h, 1, sizeof(c.h), c.file) == sizeof(c.h));
    size_t table_bytes = 0, aligned = 0;
    CHECK(!wex_validate_header(&c.h, c.bytes, 0, 0, &table_bytes, &aligned));
    c.entries.resize(table_bytes / sizeof(WexEntry));
    CHECK(!_fseeki64(c.file, (long long)c.h.table_offset, SEEK_SET));
    CHECK(fread(c.entries.data(), 1, table_bytes, c.file) == table_bytes);
    CHECK(!wex_validate_table(&c.h, c.entries.data(), c.bytes));
    return c;
}
static std::vector<unsigned char> read_record(Container &c, const WexEntry &e) {
    std::vector<unsigned char> out(e.total_bytes);
    CHECK(!_fseeki64(c.file, (long long)e.file_offset, SEEK_SET));
    CHECK(fread(out.data(), 1, out.size(), c.file) == out.size());
    return out;
}

/* ---- experts --------------------------------------------------------- */
struct Expert {
    std::string name;
    WexEntry entry;                    /* canonical layout, flags 0 */
    std::vector<unsigned char> raw;    /* scale_bytes + weight_bytes */
    size_t aligned;
};

static void views(const Expert &x, const unsigned char *scales[3], const unsigned char *weights[3]) {
    for (int m = 0; m < 3; m++) { scales[m] = x.raw.data() + x.entry.scale_offsets[m]; weights[m] = x.raw.data() + x.entry.weight_offsets[m]; }
}

static WexEntry make_entry(const uint32_t rows[3], const uint32_t scale_columns[3]) {
    WexEntry e;
    memset(&e, 0, sizeof(e));
    uint32_t scales = 0, weights = 0;
    for (int m = 0; m < 3; m++) {
        e.scale_shapes[m][0] = (uint16_t)rows[m]; e.scale_shapes[m][1] = (uint16_t)scale_columns[m];
        e.weight_shapes[m][0] = (uint16_t)rows[m]; e.weight_shapes[m][1] = (uint16_t)(scale_columns[m] * 16u);
        e.scale_offsets[m] = scales; scales += rows[m] * scale_columns[m];
    }
    e.scale_bytes = scales;
    for (int m = 0; m < 3; m++) { e.weight_offsets[m] = scales + weights; weights += rows[m] * scale_columns[m] * 16u; }
    e.weight_bytes = weights;
    uint64_t raw, aligned;
    CHECK(!wex_entry_raw_sizes(&e, &raw, &aligned));
    e.total_bytes = (uint32_t)aligned;
    return e;
}

static uint32_t xs(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

static uint32_t sample_cdf(const unsigned char *cdf, uint32_t symbols, uint32_t slot) {
    uint32_t lo = 0, hi = symbols;
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
    return lo;
}

/* Synthetic expert drawn from a model of the real experts (same context rule),
 * so the final model compresses it too. wide: every 7th scale is pushed 6
 * away from its value (both srel clamps); noise: that share of weights (in
 * 1/16) is uniform instead (more 2-byte renormalizations). */
static Expert synthetic(const char *name, const uint32_t rows[3], const uint32_t sc[3], uint32_t seed,
                        const WexNr2Model &source, int wide, uint32_t noise) {
    Expert x;
    x.name = name;
    x.entry = make_entry(rows, sc);
    x.aligned = x.entry.total_bytes;
    x.raw.assign((size_t)x.entry.scale_bytes + x.entry.weight_bytes, 0);
    WexNr2Layout layout;
    wex_nr2_layout(&x.entry, &layout);
    uint32_t s = seed;
    for (uint32_t m = 0; m < 3u; m++) {
        unsigned char *scales = x.raw.data() + x.entry.scale_offsets[m], *weights = x.raw.data() + x.entry.weight_offsets[m];
        for (uint32_t i = 0; i < layout.scale_count[m]; i++) {
            int v = (int)sample_cdf(wex_nr2_scale_cdf(&source, m), 256u, xs(s) & (WEX_NR2_TOTAL - 1u));
            if (wide && i % 7u == 3u) v += (i / 7u) % 2u ? 6 : -6;
            scales[i] = (unsigned char)std::min(255, std::max(0, v));
        }
        unsigned char median = wex_nr2_median(scales, layout.scale_count[m]);
        uint32_t srel = 0, previous = 8u, maximum = 8u;
        for (uint32_t i = 0; i < layout.symbol_count[m]; i++) {
            if (!(i % 32u)) { srel = wex_nr2_srel(scales[i / 32u], median); previous = 8u; maximum = 8u; }
            uint32_t r = xs(s), symbol;
            if ((r >> 28) < noise) symbol = (r >> 14) & 15u;
            else symbol = sample_cdf(wex_nr2_weight_cdf(&source, m, wex_nr2_context(srel, previous, maximum)), 16u, r & (WEX_NR2_TOTAL - 1u));
            weights[i / 2u] |= (unsigned char)(symbol << ((i & 1u) * 4u));
            previous = symbol & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
    }
    return x;
}

/* ---- records --------------------------------------------------------- */
struct Record { WexEntry entry; std::vector<unsigned char> bytes; size_t written; const Expert *expert; uint32_t chunk; };

static Record encode(const Expert &x, const WexNr2Model &model, uint32_t weight_chunk, uint32_t scale_chunk) {
    Record r;
    WexNr2Layout layout;
    const unsigned char *scales[3], *weights[3];
    wex_nr2_layout(&x.entry, &layout);
    views(x, scales, weights);
    size_t capacity = x.raw.size() + x.raw.size() / 4u + (1u << 20);
    r.bytes.assign(capacity, 0);
    CHECK(!wex_nr2_encode(&model, &layout, scales, weights, weight_chunk, scale_chunk, r.bytes.data(), capacity, &r.written));
    r.bytes.resize(align4k(r.written));
    r.entry = x.entry;
    r.entry.flags = WEX_ENTRY_NEURAL_RANS2;
    r.entry.total_bytes = (uint32_t)r.bytes.size();
    if (r.bytes.size() >= x.aligned)
        fprintf(stderr, "%s chunk %u: record %zu (written %zu) >= raw %zu\n", x.name.c_str(), weight_chunk,
                r.bytes.size(), r.written, x.aligned);
    CHECK(r.bytes.size() < x.aligned);   /* a WNR2 record is smaller than the raw slab */
    r.expert = &x;
    r.chunk = weight_chunk;
    return r;
}

static int cpu_decode(const WexEntry &e, const std::vector<unsigned char> &record, const WexNr2Model &model,
                      size_t aligned, std::vector<unsigned char> &slab) {
    size_t at = wex_record_read_offset(&e);
    slab.assign(aligned, 0x5a);
    if (!at || at + record.size() > aligned) return -1;
    memcpy(slab.data() + at, record.data(), record.size());
    return wex_decode_nr2_record(&e, slab.data(), aligned, &model);
}

/* ---- device buffers of this runtime --------------------------------- */
struct Device { unsigned char *scale[3], *weight[3]; size_t scale_bytes[3], weight_bytes[3]; };

static Device device_buffers(const WexEntry &e) {
    Device d;
    for (int m = 0; m < 3; m++) {
        d.scale_bytes[m] = (size_t)e.scale_shapes[m][0] * e.scale_shapes[m][1];
        d.weight_bytes[m] = (size_t)e.weight_shapes[m][0] * e.weight_shapes[m][1];
        CUDA(cudaMalloc((void **)&d.scale[m], d.scale_bytes[m]));
        CUDA(cudaMalloc((void **)&d.weight[m], d.weight_bytes[m]));
    }
    return d;
}
static void fill(const Device &d, int value) {
    for (int m = 0; m < 3; m++) { CUDA(cudaMemset(d.scale[m], value, d.scale_bytes[m])); CUDA(cudaMemset(d.weight[m], value, d.weight_bytes[m])); }
    CUDA(cudaDeviceSynchronize());
}
static bool matches(const Device &d, const WexEntry &e, const unsigned char *raw) {
    std::vector<unsigned char> back;
    for (int m = 0; m < 3; m++) {
        back.resize(d.scale_bytes[m]); CUDA(cudaMemcpy(back.data(), d.scale[m], back.size(), cudaMemcpyDeviceToHost));
        if (memcmp(back.data(), raw + e.scale_offsets[m], back.size())) return false;
        back.resize(d.weight_bytes[m]); CUDA(cudaMemcpy(back.data(), d.weight[m], back.size(), cudaMemcpyDeviceToHost));
        if (memcmp(back.data(), raw + e.weight_offsets[m], back.size())) return false;
    }
    return true;
}
static bool all_bytes(const Device &d, int value) {
    std::vector<unsigned char> back;
    for (int m = 0; m < 3; m++) for (int k = 0; k < 2; k++) {
        size_t n = k ? d.weight_bytes[m] : d.scale_bytes[m];
        back.resize(n); CUDA(cudaMemcpy(back.data(), k ? d.weight[m] : d.scale[m], n, cudaMemcpyDeviceToHost));
        for (unsigned char b : back) if (b != (unsigned char)value) return false;
    }
    return true;
}
static void release(Device &d) { for (int m = 0; m < 3; m++) { cudaFree(d.scale[m]); cudaFree(d.weight[m]); } }

/* Both helper paths on one record. expect 0: byte-exact against `raw`;
 * otherwise that return code with the destination untouched. */
static void helper_check(const Helper &h, const WexEntry &e, const std::vector<unsigned char> &record,
                         size_t aligned, const unsigned char *raw, int expect, const char *what) {
    size_t at = wex_record_read_offset(&e);
    std::vector<unsigned char> slab(aligned, 0xa5), before;
    if (at && at + record.size() <= aligned) memcpy(slab.data() + at, record.data(), record.size());
    else memcpy(slab.data(), record.data(), std::min(record.size(), aligned));
    before = slab;
    int rc = h.to_host(&e, slab.data(), aligned);
    if (rc != expect) { fprintf(stderr, "FAIL %s: decode_to_host returned %d, expected %d\n", what, rc, expect); exit(1); }
    if (expect) CHECK(slab == before);
    else CHECK(!memcmp(slab.data(), raw, (size_t)e.scale_bytes + e.weight_bytes));
    Device d = device_buffers(e);
    fill(d, 0xa5);
    rc = h.to_device(&e, record.data(), 0, d.scale, d.weight);
    if (rc != expect) { fprintf(stderr, "FAIL %s: decode_to_device returned %d, expected %d\n", what, rc, expect); exit(1); }
    if (expect) CHECK(all_bytes(d, 0xa5));
    else CHECK(matches(d, e, raw));
    release(d);
}

static void put_u32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }
static uint32_t get_u16(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static void put_u16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

/* ---- kernel-only timing in this runtime ------------------------------ */
struct KernelTimes { double h2d, scales, weights[2], total[2]; };

static KernelTimes kernel_times(const Record &r, const WexCudaNr2Tables &tables, int iterations) {
    KernelTimes t;
    memset(&t, 0, sizeof(t));
    WexNr2Record view;
    WexCudaNr2Plan plan;
    CHECK(!wex_cuda_plan_nr2(&r.entry, r.bytes.data(), r.bytes.size(), &view, &plan));
    std::vector<uint32_t> offsets(plan.streams + 1u);
    wex_cuda_nr2_offsets(&view, &plan, offsets.data());
    unsigned char *host = NULL, *record = NULL, *raw = NULL;
    uint32_t *device_offsets = NULL, *error = NULL;
    size_t aligned = r.expert->aligned;
    CUDA(cudaMallocHost((void **)&host, r.bytes.size()));
    memcpy(host, r.bytes.data(), r.bytes.size());
    CUDA(cudaMalloc((void **)&record, aligned));
    CUDA(cudaMalloc((void **)&raw, aligned));
    CUDA(cudaMalloc((void **)&device_offsets, offsets.size() * 4u));
    CUDA(cudaMalloc((void **)&error, 4u));
    CUDA(cudaMemcpy(device_offsets, offsets.data(), offsets.size() * 4u, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaEvent_t a, b, c;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b)); CUDA(cudaEventCreate(&c));
    std::vector<double> h2d, sc, w[2], total[2];
    for (int k = 0; k < iterations + 3; k++) {
        CUDA(cudaEventRecord(a, stream));
        CUDA(cudaMemcpyAsync(record, host, r.bytes.size(), cudaMemcpyHostToDevice, stream));
        CUDA(cudaEventRecord(b, stream));
        CUDA(cudaEventSynchronize(b));
        float ms; CUDA(cudaEventElapsedTime(&ms, a, b));
        if (k >= 3) h2d.push_back(ms * 1e3);
    }
    for (int lookup = 0; lookup < 2; lookup++) {
        for (int k = 0; k < iterations + 3; k++) {
            CUDA(cudaMemsetAsync(raw, 0x5a, aligned, stream));
            CUDA(cudaEventRecord(a, stream));
            CUDA(lookup ? wex_cuda_launch_nr2_lookup<1>(&plan, &tables, record, device_offsets, raw, error, stream)
                        : wex_cuda_launch_nr2_lookup<0>(&plan, &tables, record, device_offsets, raw, error, stream));
            CUDA(cudaEventRecord(b, stream));
            CUDA(cudaEventSynchronize(b));
            float ms; CUDA(cudaEventElapsedTime(&ms, a, b));
            if (k >= 3) total[lookup].push_back(ms * 1e3);
            if (k == 0) {   /* both lookups are bit-exact too */
                uint32_t flag = 1;
                std::vector<unsigned char> back(aligned);
                CUDA(cudaMemcpy(&flag, error, 4u, cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(back.data(), raw, aligned, cudaMemcpyDeviceToHost));
                CHECK(!flag);
                CHECK(plan.weight_base == r.entry.scale_bytes);
                CHECK(!memcmp(back.data(), r.expert->raw.data(), r.expert->raw.size()));
            }
        }
        for (int k = 0; k < iterations + 3; k++) {   /* kernel B alone (scales stay decoded) */
            CUDA(cudaEventRecord(a, stream));
            CUDA(lookup ? wex_cuda_launch_nr2_weights<1>(&plan, &tables, record, device_offsets, raw, error, stream)
                        : wex_cuda_launch_nr2_weights<0>(&plan, &tables, record, device_offsets, raw, error, stream));
            CUDA(cudaEventRecord(b, stream));
            CUDA(cudaEventSynchronize(b));
            float ms; CUDA(cudaEventElapsedTime(&ms, a, b));
            if (k >= 3) w[lookup].push_back(ms * 1e3);
        }
    }
    for (int k = 0; k < iterations + 3; k++) {   /* kernel A alone */
        CUDA(cudaEventRecord(a, stream));
        CUDA(wex_cuda_launch_nr2_scales(&plan, &tables, record, device_offsets, raw, error, stream));
        CUDA(cudaEventRecord(c, stream));
        CUDA(cudaEventSynchronize(c));
        float ms; CUDA(cudaEventElapsedTime(&ms, a, c));
        if (k >= 3) sc.push_back(ms * 1e3);
    }
    t.h2d = median(h2d); t.scales = median(sc);
    for (int l = 0; l < 2; l++) { t.weights[l] = median(w[l]); t.total[l] = median(total[l]); }
    cudaEventDestroy(a); cudaEventDestroy(b); cudaEventDestroy(c); cudaStreamDestroy(stream);
    cudaFreeHost(host); cudaFree(record); cudaFree(raw); cudaFree(device_offsets); cudaFree(error);
    return t;
}

static double wnr1_kernel_time(const WexEntry &e, const std::vector<unsigned char> &record, const unsigned char *raw_expected,
                               size_t aligned, int iterations) {
    WexNeuralRans view;
    std::vector<uint32_t> tiles(((size_t)e.scale_bytes + 255u) / 256u);
    uint32_t metadata = 0;
    CHECK(!wex_cuda_prepare_neural_rans(&e, record.data(), record.size(), tiles.data(), tiles.size(), &metadata));
    CHECK(!wex_nr_parse(record.data(), e.total_bytes, e.weight_bytes, &view));
    unsigned char *d_record, *raw; uint32_t *d_tiles, *lut, *error;
    size_t lut_bytes = (size_t)view.classes * (view.scale_only ? 1u : 17u) * WEX_CUDA_NR_LUT_SLOTS * 4u;
    CUDA(cudaMalloc((void **)&d_record, record.size())); CUDA(cudaMalloc((void **)&raw, aligned));
    CUDA(cudaMalloc((void **)&d_tiles, tiles.size() * 4u)); CUDA(cudaMalloc((void **)&lut, lut_bytes)); CUDA(cudaMalloc((void **)&error, 4u));
    CUDA(cudaMemcpy(d_record, record.data(), record.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_tiles, tiles.data(), tiles.size() * 4u, cudaMemcpyHostToDevice));
    cudaStream_t stream; cudaEvent_t a, b;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b));
    std::vector<double> times;
    for (int k = 0; k < iterations + 3; k++) {
        CUDA(cudaEventRecord(a, stream));
        CUDA(wex_cuda_launch_neural_rans(&e, &view, d_record, d_tiles, raw, lut, error, stream));
        CUDA(cudaEventRecord(b, stream));
        CUDA(cudaEventSynchronize(b));
        float ms; CUDA(cudaEventElapsedTime(&ms, a, b));
        if (k >= 3) times.push_back(ms * 1e3);
    }
    uint32_t flag = 1;
    std::vector<unsigned char> back((size_t)e.scale_bytes + e.weight_bytes);
    CUDA(cudaMemcpy(&flag, error, 4u, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(back.data(), raw, back.size(), cudaMemcpyDeviceToHost));
    CHECK(!flag && !memcmp(back.data(), raw_expected, back.size()));
    cudaEventDestroy(a); cudaEventDestroy(b); cudaStreamDestroy(stream);
    cudaFree(d_record); cudaFree(raw); cudaFree(d_tiles); cudaFree(lut); cudaFree(error);
    return median(times);
}

/* The GPU drops its PCIe link to a low generation when idle (the disk reads of
 * the checks above leave it idle) and raises it only under sustained traffic:
 * ~0.4 s of pinned uploads first, so timings measure the decode, not the link. */
static void warm_link() {
    unsigned char *host = NULL, *device = NULL;
    const size_t bytes = 64u << 20;
    CUDA(cudaMallocHost((void **)&host, bytes));
    CUDA(cudaMalloc((void **)&device, bytes));
    double t0 = now();
    while (now() - t0 < 0.4) CUDA(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    cudaFree(device); cudaFreeHost(host);
}

/* Median of `iterations` helper decode_to_device calls from page-locked staging. */
static double device_call_time(const Helper &h, const WexEntry &e, const std::vector<unsigned char> &record,
                               const unsigned char *raw, size_t aligned, int iterations) {
    warm_link();
    unsigned char *staging = (unsigned char *)_aligned_malloc(aligned, 4096);
    CHECK(staging);
    memcpy(staging, record.data(), record.size());
    CHECK(!h.pin(staging, aligned));
    Device d = device_buffers(e);
    fill(d, 0xa5);
    std::vector<double> times;
    for (int k = 0; k < iterations + 3; k++) {
        double t0 = now();
        CHECK(!h.to_device(&e, staging, 0, d.scale, d.weight));
        if (k >= 3) times.push_back((now() - t0) * 1e6);
    }
    CHECK(matches(d, e, raw));
    CHECK(!h.unpin(staging));
    release(d);
    _aligned_free(staging);
    return median(times);
}

int main(int argc, char **argv) {
    const char *baseline_path = NULL, *json_path = NULL, *dump_dir = NULL;
    std::vector<const char *> positional, containers;
    int container_records = 16;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--baseline") && a + 1 < argc) baseline_path = argv[++a];
        else if (!strcmp(argv[a], "--wnr2") && a + 1 < argc) containers.push_back(argv[++a]);
        else if (!strcmp(argv[a], "--wnr2-records") && a + 1 < argc) container_records = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--json") && a + 1 < argc) json_path = argv[++a];
        else if (!strcmp(argv[a], "--dump") && a + 1 < argc) dump_dir = argv[++a];
        else positional.push_back(argv[a]);
    }
    if (positional.size() != 3) {
        fprintf(stderr, "usage: test_nr2_cuda helper.dll v1.wex v3-neural.wex [--baseline old_helper.dll] "
                        "[--wnr2 container.wex ...] [--wnr2-records N] [--json out.json] [--dump DIR]\n");
        return 2;
    }
    double t_start = now();
    Helper helper = load_helper(positional[0]);
    CHECK(helper.features() == 7u && helper.model);
    CUDA(cudaSetDevice(0));
    CUDA(cudaFree(0));   /* this runtime's context first, as the backend does */

    /* a. experts: 12 real ones across layers, plus synthetic shapes */
    Container v1 = open_container(positional[1]);
    CHECK(v1.h.magic == WEX_MAGIC_V1);
    std::vector<Expert> experts;
    const uint32_t experts_per_layer = v1.h.num_experts;
    static const int layers[12] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 42};
    std::vector<size_t> real_index;
    for (int l : layers) {
        CHECK((uint32_t)l < v1.h.num_layers);
        for (uint32_t x = 0; x < experts_per_layer; x++) {
            size_t i = (size_t)l * experts_per_layer + x;
            const WexEntry &e = v1.entries[i];
            if (!e.total_bytes) continue;
            Expert ex;
            char name[64]; snprintf(name, sizeof(name), "real L%d E%u (#%zu)", l, x, i);
            ex.name = name;
            ex.entry = e;
            uint64_t raw, aligned;
            CHECK(!e.flags && !wex_entry_raw_sizes(&e, &raw, &aligned) && e.total_bytes == aligned);
            ex.aligned = (size_t)aligned;
            ex.raw = read_record(v1, e);
            ex.raw.resize((size_t)raw);
            experts.push_back(ex);
            real_index.push_back(i);
            break;
        }
    }
    CHECK(experts.size() == 12u);
    double t_read = now();
    std::vector<uint64_t> wcounts(3u * WEX_NR2_CONTEXTS * 16u, 0), scounts(3u * 256u, 0);
    {   /* synthetic experts sampled from a model of the real ones */
        std::vector<unsigned char> real_blob(WEX_NR2_MODEL_BYTES);
        WexNr2Model real_model;
        for (const Expert &x : experts) {
            WexNr2Layout layout;
            const unsigned char *scales[3], *weights[3];
            wex_nr2_layout(&x.entry, &layout);
            views(x, scales, weights);
            wex_nr2_count(&layout, scales, weights, wcounts.data(), scounts.data());
        }
        wex_nr2_model_build(wcounts.data(), scounts.data(), real_blob.data());
        CHECK(!wex_nr2_model_open(real_blob.data(), real_blob.size(), &real_model));
        std::fill(wcounts.begin(), wcounts.end(), 0); std::fill(scounts.begin(), scounts.end(), 0);
        static const uint32_t r0[3] = {37, 50, 37}, c0[3] = {72, 9, 72};      /* partial streams, scale_bytes % 16 = 2 */
        static const uint32_t r1[3] = {512, 1024, 512}, c1[3] = {64, 32, 64};
        static const uint32_t r2[3] = {256, 512, 256}, c2[3] = {32, 16, 32};
        experts.push_back(synthetic("synthetic odd 37x1152/50x144", r0, c0, 0x1234567u, real_model, 1, 0u));
        experts.push_back(synthetic("synthetic mid 512x1024", r1, c1, 0x9e3779b9u, real_model, 1, 1u));
        experts.push_back(synthetic("synthetic noisy 256x512", r2, c2, 0x2545f491u, real_model, 0, 3u));
    }
    const size_t real_count = 12, odd = 12, mid = 13;
    CHECK(experts[odd].entry.scale_bytes % 16u != 0);

    /* model from all their counts; a second one from the synthetic experts only */
    std::vector<uint64_t> wcounts2(wcounts.size(), 0), scounts2(scounts.size(), 0);
    for (size_t i = 0; i < experts.size(); i++) {
        WexNr2Layout layout;
        const unsigned char *scales[3], *weights[3];
        wex_nr2_layout(&experts[i].entry, &layout);
        views(experts[i], scales, weights);
        wex_nr2_count(&layout, scales, weights, wcounts.data(), scounts.data());
        if (i >= real_count) wex_nr2_count(&layout, scales, weights, wcounts2.data(), scounts2.data());
    }
    std::vector<unsigned char> blob(WEX_NR2_MODEL_BYTES), blob2(WEX_NR2_MODEL_BYTES);
    WexNr2Model model, model2;
    wex_nr2_model_build(wcounts.data(), scounts.data(), blob.data());
    wex_nr2_model_build(wcounts2.data(), scounts2.data(), blob2.data());
    CHECK(!wex_nr2_model_open(blob.data(), blob.size(), &model));
    CHECK(!wex_nr2_model_open(blob2.data(), blob2.size(), &model2));
    CHECK(model.id != model2.id);

    /* b. registration */
    CHECK(helper.model(blob.data(), blob.size() - 2u) == -1);
    {
        std::vector<unsigned char> broken = blob;
        broken[34] = broken[35] = 0;   /* a zero-width symbol */
        CHECK(helper.model(broken.data(), broken.size()) == -1);
        broken = blob; broken[0] ^= 1u;
        CHECK(helper.model(broken.data(), broken.size()) == -1);
        CHECK(helper.model(NULL, blob.size()) == -1);
    }
    CHECK(helper.model(blob.data(), blob.size()) == 0);
    CHECK(helper.model(blob.data(), blob.size()) == 0);   /* again: a no-op */

    /* encode everything at both chunk sizes */
    std::vector<Record> records;
    static const uint32_t chunks[2] = {4096u, 8192u};
    size_t raw_total = 0, record_total[2] = {0, 0};
    for (size_t i = 0; i < experts.size(); i++)
        for (int c = 0; c < 2; c++) {
            records.push_back(encode(experts[i], model, chunks[c], 2048u));
            if (i < real_count) record_total[c] += records.back().bytes.size();
        }
    for (size_t i = 0; i < real_count; i++) raw_total += experts[i].raw.size();
    double t_encode = now();

    /* b. every record: CPU reference, then both helper paths, byte-exact */
    int exact = 0;
    for (const Record &r : records) {
        std::vector<unsigned char> slab;
        CHECK(!cpu_decode(r.entry, r.bytes, model, r.expert->aligned, slab));
        CHECK(!memcmp(slab.data(), r.expert->raw.data(), r.expert->raw.size()));
        std::string what = r.expert->name + " chunk " + std::to_string(r.chunk);
        helper_check(helper, r.entry, r.bytes, r.expert->aligned, r.expert->raw.data(), 0, what.c_str());
        exact++;
    }
    double t_exact = now();

    /* c. malformed records: -1, destination untouched; CPU must refuse as well */
    int refused = 0;
    auto reject = [&](const Record &good, const char *name, int expect,
                      const std::function<void(std::vector<unsigned char> &, WexEntry &, const WexNr2Record &)> &mutate) {
        WexNr2Layout layout;
        WexNr2Record view;
        wex_nr2_layout(&good.entry, &layout);
        CHECK(!wex_nr2_parse(good.bytes.data(), good.bytes.size(), &layout, model.id, &view));
        std::vector<unsigned char> bad = good.bytes, slab;
        WexEntry entry = good.entry;
        mutate(bad, entry, view);
        if (expect == -1) CHECK(cpu_decode(entry, bad, model, good.expert->aligned, slab) != 0);
        std::string what = std::string(name) + " / " + good.expert->name + " chunk " + std::to_string(good.chunk);
        helper_check(helper, entry, bad, good.expert->aligned, NULL, expect, what.c_str());
        refused++;
    };
    const Record *targets[3] = {&records[0], &records[odd * 2], &records[mid * 2 + 1]};
    for (const Record *target : targets) {
        const Record &g = *target;
        reject(g, "bad magic", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { b[0] ^= 1u; });
        reject(g, "model id 0", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { put_u32(b.data() + 4, 0); });
        reject(g, "weight chunk 3x", -1, [&](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { put_u32(b.data() + 16, g.chunk * 3u); });
        reject(g, "scale chunk too large", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { put_u32(b.data() + 20, 32768u); });
        reject(g, "reserved byte", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { b[27] = 1u; });
        reject(g, "weight payload size", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) { put_u32(b.data() + 12, v.weight_payload_bytes + 1u); });
        reject(g, "stream shorter than 4", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { b[32] = 3u; b[33] = 0; });
        reject(g, "lengths do not add up", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { b[32] ^= 1u; });
        reject(g, "truncated (total - 4096)", -1, [](std::vector<unsigned char> &b, WexEntry &e, const WexNr2Record &) {
            b.resize(b.size() - 4096u); e.total_bytes -= 4096u; });
        reject(g, "truncated (cut payload)", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            /* claim less payload, keep the lengths: parse sees a mismatch */
            put_u32(b.data() + 12, v.weight_payload_bytes - 64u); });
        if (g.written < g.bytes.size())
            reject(g, "non-zero padding", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { b.back() = 1u; });
        /* device-side checks: the record parses, a stream fails in a kernel */
        reject(g, "scale state below L", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            memset(b.data() + v.metadata_bytes, 0, 4u); });
        reject(g, "scale state >= 256 L", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            b[v.metadata_bytes] = 0x80u; });
        reject(g, "weight state below L", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            memset(b.data() + v.metadata_bytes + v.scale_payload_bytes, 0, 4u); });
        reject(g, "weight state >= 256 L", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            b[v.metadata_bytes + v.scale_payload_bytes] = 0xffu; });
        reject(g, "last weight stream state >= 256 L", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            uint32_t at = v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
            at -= get_u16(v.lengths + 2u * (v.scale_streams + v.weight_streams - 1u));
            b[at] = 0xffu; });
        reject(g, "weight lengths +1/-1 (stream ends early / reads past end)", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            unsigned char *len = b.data() + 32u + 2u * (v.scale_streams + 1u);
            put_u16(len, get_u16(len) + 1u); put_u16(len + 2, get_u16(len + 2) - 1u); });
        reject(g, "weight lengths -1/+1", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            unsigned char *len = b.data() + 32u + 2u * v.scale_streams;
            put_u16(len, get_u16(len) - 1u); put_u16(len + 2, get_u16(len + 2) + 1u); });
        reject(g, "scale lengths +1/-1", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            if (v.scale_streams < 2u) { b[0] ^= 1u; return; }
            unsigned char *len = b.data() + 32u;
            put_u16(len, get_u16(len) + 1u); put_u16(len + 2, get_u16(len + 2) - 1u); });
        reject(g, "corrupted weight payload (64 zero bytes)", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            memset(b.data() + v.metadata_bytes + v.scale_payload_bytes + 4u, 0, 64u); });
        reject(g, "corrupted scale payload (16 x 0xff)", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            memset(b.data() + v.metadata_bytes + 4u, 0xff, 16u); });
        reject(g, "last weight byte flipped", -1, [](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &v) {
            b[v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes - 1u] ^= 0x5au; });
        /* model resolution */
        reject(g, "unregistered model id", 1, [&](std::vector<unsigned char> &b, WexEntry &, const WexNr2Record &) { put_u32(b.data() + 4, model.id ^ 0x5a5a5a5au); });
    }
    /* a record of a second, not yet registered model: 1, then 0 once registered */
    {
        Record other = encode(experts[mid], model2, 4096u, 2048u);
        helper_check(helper, other.entry, other.bytes, other.expert->aligned, NULL, 1, "model2 before registration");
        CHECK(helper.model(blob2.data(), blob2.size()) == 0);
        helper_check(helper, other.entry, other.bytes, other.expert->aligned, other.expert->raw.data(), 0, "model2 registered");
        refused++;
    }

    /* fuzz: random mutations; the helper must agree with the CPU reference */
    int fuzz = 0, fuzz_cpu_ok = 0;
    {
        uint32_t s = 20261020u;
        for (int round = 0; round < 2; round++) {
            const Record &g = records[mid * 2 + round];
            for (int k = 0; k < 150; k++, fuzz++) {
                std::vector<unsigned char> bad = g.bytes, slab;
                uint32_t n = 1u + (xs(s) % 4u == 0 ? xs(s) % 64u : 0u);
                uint32_t at = xs(s) % (uint32_t)g.written;
                if (at >= 4u && at < 8u) at = 8u;   /* model ids: covered above */
                for (uint32_t i = 0; i < n && at + i < g.written; i++) bad[at + i] ^= (unsigned char)(1u + xs(s) % 255u);
                int cpu = cpu_decode(g.entry, bad, model, g.expert->aligned, slab);
                if (!cpu) {
                    fuzz_cpu_ok++;
                    helper_check(helper, g.entry, bad, g.expert->aligned, slab.data(), 0, "fuzz (CPU accepted)");
                } else {
                    helper_check(helper, g.entry, bad, g.expert->aligned, NULL, -1, "fuzz (CPU refused)");
                }
            }
        }
    }
    double t_reject = now();

    /* d. WNR1 regression */
    Container v3 = open_container(positional[2]);
    CHECK(v3.entries.size() == v1.entries.size());
    std::vector<size_t> wnr1;
    for (size_t i : {real_index[0], real_index.back()}) if (v3.entries[i].flags == WEX_ENTRY_NEURAL_RANS) wnr1.push_back(i);
    for (size_t i = 0; wnr1.size() < 2u && i < v3.entries.size(); i++)
        if (v3.entries[i].flags == WEX_ENTRY_NEURAL_RANS && std::find(wnr1.begin(), wnr1.end(), i) == wnr1.end()) wnr1.push_back(i);
    CHECK(wnr1.size() == 2u);
    std::vector<unsigned char> wnr1_record[2], wnr1_raw[2];
    for (int k = 0; k < 2; k++) {
        size_t i = wnr1[k];
        const WexEntry &e = v3.entries[i];
        uint64_t raw, aligned;
        CHECK(!wex_entry_raw_sizes(&e, &raw, &aligned));
        wnr1_record[k] = read_record(v3, e);
        wnr1_raw[k] = read_record(v1, v1.entries[i]);
        wnr1_raw[k].resize((size_t)raw);
        std::vector<unsigned char> slab(aligned, 0);
        memcpy(slab.data() + wex_record_read_offset(&e), wnr1_record[k].data(), e.total_bytes);
        CHECK(!wex_decode_record(&e, slab.data(), (size_t)aligned));   /* CPU reference */
        CHECK(!memcmp(slab.data(), wnr1_raw[k].data(), (size_t)raw));
        helper_check(helper, e, wnr1_record[k], (size_t)aligned, wnr1_raw[k].data(), 0, "WNR1 regression");
    }

    /* e. timing on the first real expert (also the first WNR1 record when present) */
    const int iterations = 20;
    const Record &r4 = records[0], &r8 = records[1];
    CHECK(r4.expert == &experts[0] && r8.expert == &experts[0]);
    const WexEntry &w1e = v3.entries[wnr1[0]];
    if (dump_dir) {   /* inputs for kernel experiments: entry + bytes per file */
        auto dump = [&](const char *name, const WexEntry &e, const unsigned char *p, size_t n) {
            std::string path = std::string(dump_dir) + "\\" + name;
            FILE *f = fopen(path.c_str(), "wb"); CHECK(f);
            CHECK(fwrite(&e, sizeof(e), 1, f) == 1 && fwrite(p, 1, n, f) == n); fclose(f);
        };
        WexEntry none; memset(&none, 0, sizeof(none));
        dump("model.bin", none, blob.data(), blob.size());
        dump("raw.bin", experts[0].entry, experts[0].raw.data(), experts[0].raw.size());
        dump("wnr2_4096.bin", r4.entry, r4.bytes.data(), r4.bytes.size());
        dump("wnr2_8192.bin", r8.entry, r8.bytes.data(), r8.bytes.size());
        dump("wnr1.bin", w1e, wnr1_record[0].data(), wnr1_record[0].size());
    }
    bool same_expert = wnr1[0] == real_index[0];
    double call_nr2[2], call_nr1, call_nr1_baseline = 0, call_nr2_8192_again;
    call_nr2[0] = device_call_time(helper, r4.entry, r4.bytes, experts[0].raw.data(), experts[0].aligned, iterations);
    call_nr2[1] = device_call_time(helper, r8.entry, r8.bytes, experts[0].raw.data(), experts[0].aligned, iterations);
    call_nr1 = device_call_time(helper, w1e, wnr1_record[0], wnr1_raw[0].data(), experts[0].aligned, iterations);
    call_nr2_8192_again = device_call_time(helper, r8.entry, r8.bytes, experts[0].raw.data(), experts[0].aligned, iterations);
    if (baseline_path) {
        Helper old = load_helper(baseline_path);
        call_nr1_baseline = device_call_time(old, w1e, wnr1_record[0], wnr1_raw[0].data(), experts[0].aligned, iterations);
    }
    /* the CPU reference decoders, for scale (median of 3) */
    double cpu_us[3];
    for (int r = 0; r < 3; r++) {
        const WexEntry &e = r < 2 ? (r ? r8.entry : r4.entry) : w1e;
        const std::vector<unsigned char> &bytes = r < 2 ? (r ? r8.bytes : r4.bytes) : wnr1_record[0];
        std::vector<unsigned char> slab(experts[0].aligned);
        std::vector<double> t;
        for (int k = 0; k < 3; k++) {
            memcpy(slab.data() + wex_record_read_offset(&e), bytes.data(), bytes.size());
            double t0 = now();
            CHECK(!(r < 2 ? wex_decode_nr2_record(&e, slab.data(), slab.size(), &model) : wex_decode_record(&e, slab.data(), slab.size())));
            t.push_back((now() - t0) * 1e6);
        }
        cpu_us[r] = median(t);
    }
    /* kernel-only, this runtime: tables with buckets so both lookups run */
    unsigned char *table_base = NULL;
    WexCudaNr2Tables tables;
    CUDA(cudaMalloc((void **)&table_base, wex_cuda_nr2_table_bytes(1)));
    CUDA(cudaMemcpy(table_base, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    CUDA(wex_cuda_build_nr2_tables(table_base, 1, 0, &tables));
    CUDA(cudaDeviceSynchronize());
    KernelTimes k4 = kernel_times(r4, tables, iterations), k8 = kernel_times(r8, tables, iterations);
    double k_nr1 = wnr1_kernel_time(w1e, wnr1_record[0], wnr1_raw[0].data(), experts[0].aligned, iterations);
    cudaFree(table_base);
    double t_end = now();

    /* optional: records of real WNR2 containers (their own model, from the
     * header-page descriptor), helper vs CPU reference, and vs the WEX1 raw
     * slab when every record matches it byte for byte */
    std::string container_json;
    for (const char *path : containers) {
        Container c = open_container(path);
        CHECK(c.entries.size() == v1.entries.size() && (c.h.flags & WEX_HEADER_NEURAL_RANS2));
        unsigned char page[WEX_ALIGN];
        uint64_t at; uint32_t bytes, id;
        CHECK(!_fseeki64(c.file, 0, SEEK_SET) && fread(page, 1, sizeof(page), c.file) == sizeof(page));
        CHECK(!wex_nr2_model_location(&c.h, page, &at, &bytes, &id));
        std::vector<unsigned char> cblob(bytes);
        CHECK(!_fseeki64(c.file, (long long)at, SEEK_SET) && fread(cblob.data(), 1, bytes, c.file) == bytes);
        WexNr2Model cmodel;
        CHECK(!wex_nr2_model_open(cblob.data(), cblob.size(), &cmodel) && cmodel.id == id);
        CHECK(helper.model(cblob.data(), cblob.size()) == 0);
        std::vector<size_t> present;
        for (size_t i = 0; i < c.entries.size(); i++) if (c.entries[i].total_bytes) present.push_back(i);
        int checked = 0, raw_equal = 0;
        double call_us = 0;
        for (int k = 0; k < container_records && !present.empty(); k++) {
            size_t i = present[(size_t)k * present.size() / (size_t)container_records];
            const WexEntry &e = c.entries[i];
            CHECK(e.flags == WEX_ENTRY_NEURAL_RANS2);
            uint64_t raw, aligned;
            CHECK(!wex_entry_raw_sizes(&e, &raw, &aligned));
            std::vector<unsigned char> record = read_record(c, e), slab, original = read_record(v1, v1.entries[i]);
            CHECK(!cpu_decode(e, record, cmodel, (size_t)aligned, slab));
            std::string what = std::string("container ") + path + " #" + std::to_string(i);
            helper_check(helper, e, record, (size_t)aligned, slab.data(), 0, what.c_str());
            raw_equal += !memcmp(slab.data(), original.data(), (size_t)raw);
            if (!k) call_us = device_call_time(helper, e, record, slab.data(), (size_t)aligned, 20);
            checked++;
        }
        char line[512];
        snprintf(line, sizeof(line), "%s{\"file\":\"%s\",\"model_id\":\"%08x\",\"records_exact_vs_cpu\":%d,\"records_equal_to_wex1_raw\":%d,"
                 "\"decode_to_device_pinned_median_us\":%.1f}", container_json.empty() ? "" : ",", path, id, checked, raw_equal, call_us);
        for (char *p = line; *p; p++) if (*p == '\\') *p = '/';
        container_json += line;
        fclose(c.file);
    }


    char json[8192];
    snprintf(json, sizeof(json),
        "{\"helper_features\":%u,\"real_experts\":%zu,\"synthetic_experts\":%zu,\"records_exact\":%d,"
        "\"model_id\":\"%08x\",\"model2_id\":\"%08x\","
        "\"real_raw_bytes\":%zu,\"real_record_bytes\":{\"4096\":%zu,\"8192\":%zu},"
        "\"malformed_checks\":%d,\"fuzz_cases\":%d,\"fuzz_cpu_accepted\":%d,"
        "\"wnr1_regression_records\":[%zu,%zu],"
        "\"timing_expert\":\"%s\",\"timing_wnr1_same_expert\":%s,"
        "\"record_bytes\":{\"wnr2_4096\":%zu,\"wnr2_8192\":%zu,\"wnr1\":%u},"
        "\"decode_to_device_pinned_median_us\":{\"wnr2_4096\":%.1f,\"wnr2_8192\":%.1f,\"wnr2_8192_repeat\":%.1f,\"wnr1\":%.1f,\"wnr1_pre_nr2_helper\":%.1f},"
        "\"kernel_only_median_us\":{"
          "\"wnr2_4096\":{\"h2d_record\":%.1f,\"scales\":%.1f,\"weights_buckets\":%.1f,\"weights_smem_rows\":%.1f,\"total_buckets\":%.1f,\"total_smem_rows\":%.1f},"
          "\"wnr2_8192\":{\"h2d_record\":%.1f,\"scales\":%.1f,\"weights_buckets\":%.1f,\"weights_smem_rows\":%.1f,\"total_buckets\":%.1f,\"total_smem_rows\":%.1f},"
          "\"wnr1\":%.1f},"
        "\"cpu_reference_decode_median_us\":{\"wnr2_4096\":%.0f,\"wnr2_8192\":%.0f,\"wnr1\":%.0f},"
        "\"iterations\":%d,\"seconds\":{\"read\":%.1f,\"encode\":%.1f,\"exact\":%.1f,\"malformed_fuzz\":%.1f,\"wnr1_and_timing\":%.1f},"
        "\"real_wnr2_containers\":[%s],"
        "\"cross_runtime_buffers\":true,\"all_bytes_exact\":true}",
        helper.features(), real_count, experts.size() - real_count, exact, model.id, model2.id,
        raw_total, record_total[0], record_total[1], refused, fuzz, fuzz_cpu_ok, wnr1[0], wnr1[1],
        experts[0].name.c_str(), same_expert ? "true" : "false",
        r4.bytes.size(), r8.bytes.size(), w1e.total_bytes,
        call_nr2[0], call_nr2[1], call_nr2_8192_again, call_nr1, call_nr1_baseline,
        k4.h2d, k4.scales, k4.weights[0], k4.weights[1], k4.total[0], k4.total[1],
        k8.h2d, k8.scales, k8.weights[0], k8.weights[1], k8.total[0], k8.total[1], k_nr1,
        cpu_us[0], cpu_us[1], cpu_us[2],
        iterations, t_read - t_start, t_encode - t_read, t_exact - t_encode, t_reject - t_exact, t_end - t_reject,
        container_json.c_str());
    puts(json);
    if (json_path) { FILE *f = fopen(json_path, "wb"); CHECK(f); fputs(json, f); fputc('\n', f); fclose(f); }
    fclose(v1.file); fclose(v3.file);
    return 0;
}
