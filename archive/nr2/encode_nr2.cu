/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR2 lab tools (wex_nr2_tools.dll): GPU training counts and a GPU encoder
 * for the WEX expert codec WNR2, plus thin wrappers of the CPU reference in
 * wex_neural_rans.h / wex_format.h. Used by build_nr2.py and test_nr2_tools.py;
 * not linked into the engine. The CPU reference is the ground truth: the GPU
 * counts equal wex_nr2_count and the GPU records are byte-identical to
 * wex_nr2_encode (same medians, contexts, CDFs and rANS state equations).
 *
 * canonical_zero != 0 maps FP4 code 8 (-0) to 0 (+0) before counting and
 * encoding: value-exact, not byte-exact. Contexts are unchanged by it
 * (8 & 7 == 0 & 7), only the coded symbol differs.
 *
 * Return codes: 0 ok, -1 invalid input or an encoding the CPU reference would
 * also refuse, 1 CUDA unavailable/failed. wex_nr2_verify: 0 when the record
 * decodes (CPU reference) to exactly the expected raw slab, negative otherwise.
 * Not thread-safe across GPU calls by design: one process-wide workspace,
 * serialized by a mutex; CPU wrappers are reentrant.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "wex_format.h"

#define NR2_EXPORT extern "C" __declspec(dllexport)
#define NR2_WEIGHT_BINS (3u * WEX_NR2_CONTEXTS * 16u)
#define NR2_SCALE_BINS (3u * 256u)
#define NR2_MATRIX_BINS (WEX_NR2_CONTEXTS * 16u)

namespace {

struct Geometry {
    uint32_t scale_count[3], symbol_count[3];
    uint32_t scale_offset[3], weight_offset[3];
    uint32_t median[3];
    uint32_t scale_first[4], weight_first[4];   /* stream prefix sums per matrix */
    uint32_t weight_chunk, scale_chunk, scale_stride, weight_stride;
    uint32_t scale_streams, weight_streams, canonical_zero;
    unsigned long long weight_base;             /* scratch offset of weight stream 0 */
};

/* Host: geometry of a valid entry, medians from the CPU reference. Chunks 0
 * skip the stream layout (counting). */
int geometry_of(const WexEntry *entry, const unsigned char *raw_slab, int canonical_zero,
                uint32_t weight_chunk, uint32_t scale_chunk, Geometry *g, uint64_t *raw_bytes) {
    uint64_t raw, aligned;
    WexNr2Layout layout;
    if (!entry || !raw_slab || wex_entry_raw_sizes(entry, &raw, &aligned)) return -1;
    wex_nr2_layout(entry, &layout);
    std::memset(g, 0, sizeof(*g));
    for (unsigned m = 0; m < 3u; m++) {
        if (!layout.scale_count[m] || (uint64_t)layout.scale_count[m] * 32u != layout.symbol_count[m]) return -1;
        g->scale_count[m] = layout.scale_count[m];
        g->symbol_count[m] = layout.symbol_count[m];
        g->scale_offset[m] = entry->scale_offsets[m];
        g->weight_offset[m] = entry->weight_offsets[m];
        g->median[m] = wex_nr2_median(raw_slab + entry->scale_offsets[m], layout.scale_count[m]);
    }
    g->canonical_zero = canonical_zero ? 1u : 0u;
    if (weight_chunk) {
        if (!wex_nr2_pow2_in(weight_chunk, WEX_NR2_MIN_WEIGHT_CHUNK, WEX_NR2_MAX_CHUNK) ||
            !wex_nr2_pow2_in(scale_chunk, WEX_NR2_MIN_SCALE_CHUNK, WEX_NR2_MAX_CHUNK)) return -1;
        g->weight_chunk = weight_chunk; g->scale_chunk = scale_chunk;
        /* A stream of n symbols emits at most 14.003 n / 8 + 4 bytes (f >= 1
         * of 2^14), so 2 n + 8 always holds it, as the CPU's scratch does. */
        g->scale_stride = 2u * scale_chunk + 8u;
        g->weight_stride = 2u * weight_chunk + 8u;
        for (unsigned m = 0; m < 3u; m++) {
            g->scale_first[m + 1] = g->scale_first[m] + wex_nr2_streams(g->scale_count[m], scale_chunk);
            g->weight_first[m + 1] = g->weight_first[m] + wex_nr2_streams(g->symbol_count[m], weight_chunk);
        }
        g->scale_streams = g->scale_first[3];
        g->weight_streams = g->weight_first[3];
        g->weight_base = (unsigned long long)g->scale_streams * g->scale_stride;
    }
    *raw_bytes = raw;
    return 0;
}

__device__ __forceinline__ uint32_t srel_of(uint32_t scale, uint32_t median) {
    int d = (int)scale - (int)median;
    return (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3);
}

/* The 16 packed bytes (32 FP4 symbols) of one 32-weight block. */
__device__ __forceinline__ uint4 load_block(const unsigned char *p) {
    if (!((uintptr_t)p & 15u)) return __ldg((const uint4 *)p);
    uint32_t w[4];
#pragma unroll
    for (unsigned k = 0; k < 4u; k++)
        w[k] = (uint32_t)p[4u * k] | (uint32_t)p[4u * k + 1u] << 8 |
               (uint32_t)p[4u * k + 2u] << 16 | (uint32_t)p[4u * k + 3u] << 24;
    return make_uint4(w[0], w[1], w[2], w[3]);
}

/* Symbol j of a block: byte j / 2, low nibble first. */
__device__ __forceinline__ uint32_t block_symbol(const uint32_t w[4], unsigned j, uint32_t canonical_zero) {
    uint32_t x = (w[j >> 3] >> ((j & 7u) * 4u)) & 15u;
    return canonical_zero && x == 8u ? 0u : x;
}

/* Training counts of one matrix (blockIdx.y) of one expert: per-block shared
 * histograms, flushed with global atomics (per-expert counts fit in u32). */
__global__ void __launch_bounds__(512) nr2_count_kernel(const unsigned char *slab, Geometry g,
                                                        uint32_t *weight_counts, uint32_t *scale_counts) {
    __shared__ uint32_t histogram[NR2_MATRIX_BINS];
    __shared__ uint32_t scale_histogram[256];
    const unsigned m = blockIdx.y;
    for (unsigned i = threadIdx.x; i < NR2_MATRIX_BINS; i += blockDim.x) histogram[i] = 0;
    for (unsigned i = threadIdx.x; i < 256u; i += blockDim.x) scale_histogram[i] = 0;
    __syncthreads();
    const unsigned char *scales = slab + g.scale_offset[m];
    const unsigned char *weights = slab + g.weight_offset[m];
    const uint32_t median = g.median[m], blocks = g.scale_count[m], cz = g.canonical_zero;
    for (uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; b < blocks; b += gridDim.x * blockDim.x) {
        uint32_t s = scales[b], srel = srel_of(s, median), previous = 8u, maximum = 8u;
        uint4 v = load_block(weights + (size_t)b * 16u);
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
        atomicAdd(&scale_histogram[s], 1u);
#pragma unroll
        for (unsigned j = 0; j < 32u; j++) {
            uint32_t x = block_symbol(w, j, cz);
            atomicAdd(&histogram[((srel * 9u + previous) * 9u + maximum) * 16u + x], 1u);
            previous = x & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < NR2_MATRIX_BINS; i += blockDim.x)
        if (histogram[i]) atomicAdd(&weight_counts[m * NR2_MATRIX_BINS + i], histogram[i]);
    for (unsigned i = threadIdx.x; i < 256u; i += blockDim.x)
        if (scale_histogram[i]) atomicAdd(&scale_counts[m * 256u + i], scale_histogram[i]);
}

/* wex_nr2_put on the device; cursor counts down to the floor at 0. */
__device__ __forceinline__ bool nr2_put(uint32_t &state, unsigned char *slot, uint32_t &cursor,
                                        uint32_t start, uint32_t frequency) {
    const uint32_t maximum = ((WEX_NR_L >> WEX_NR2_PRECISION) << 8) * frequency;
    while (state >= maximum) {
        if (!cursor) return false;
        slot[--cursor] = (unsigned char)state;
        state >>= 8;
    }
    state = ((state / frequency) << WEX_NR2_PRECISION) + state % frequency + start;
    return true;
}

/* One thread per stream: scale streams first (matrix order), then weight
 * streams. Each encodes backwards into its own scratch slot; weight contexts
 * are computed forward per 32-weight block (they reset at every block), then
 * the block's symbols are coded in reverse. */
__global__ void __launch_bounds__(32) nr2_encode_kernel(const unsigned char *slab, const unsigned char *model,
                                                        Geometry g, unsigned char *scratch, uint32_t *lengths,
                                                        uint32_t *error) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= g.scale_streams + g.weight_streams) return;
    uint32_t state = WEX_NR_L, cursor, stride;
    unsigned char *slot;
    if (t < g.scale_streams) {
        const unsigned m = t < g.scale_first[1] ? 0u : t < g.scale_first[2] ? 1u : 2u;
        const uint32_t first = (t - g.scale_first[m]) * g.scale_chunk;
        const uint32_t count = g.scale_count[m] - first < g.scale_chunk ? g.scale_count[m] - first : g.scale_chunk;
        const uint16_t *cdf = (const uint16_t *)(model + WEX_NR2_MODEL_HEADER_BYTES + WEX_NR2_WCDF_BYTES + m * 514u);
        const unsigned char *scales = slab + g.scale_offset[m] + first;
        stride = g.scale_stride;
        slot = scratch + (size_t)t * stride;
        cursor = stride;
        for (uint32_t j = count; j-- > 0;) {
            const uint32_t s = scales[j], start = __ldg(cdf + s), frequency = __ldg(cdf + s + 1u) - start;
            if (!nr2_put(state, slot, cursor, start, frequency)) { atomicExch(error, 1u); return; }
        }
    } else {
        const uint32_t w = t - g.scale_streams;
        const unsigned m = w < g.weight_first[1] ? 0u : w < g.weight_first[2] ? 1u : 2u;
        const uint32_t first = (w - g.weight_first[m]) * g.weight_chunk;
        const uint32_t count = g.symbol_count[m] - first < g.weight_chunk ? g.symbol_count[m] - first : g.weight_chunk;
        const uint16_t *cdfs = (const uint16_t *)(model + WEX_NR2_MODEL_HEADER_BYTES) + (size_t)m * WEX_NR2_CONTEXTS * 17u;
        const unsigned char *scales = slab + g.scale_offset[m];
        const unsigned char *weights = slab + g.weight_offset[m];
        const uint32_t median = g.median[m], cz = g.canonical_zero;
        stride = g.weight_stride;
        slot = scratch + g.weight_base + (size_t)w * stride;
        cursor = stride;
        /* count is a multiple of 32: chunks and matrix sizes are. */
        for (uint32_t k = count / 32u; k-- > 0;) {
            const uint32_t b = first / 32u + k;
            const uint32_t srel = srel_of(scales[b], median);
            uint4 v = load_block(weights + (size_t)b * 16u);
            uint32_t words[4] = {v.x, v.y, v.z, v.w};
            uint32_t contexts[32];
            uint32_t previous = 8u, maximum = 8u;
#pragma unroll
            for (unsigned j = 0; j < 32u; j++) {   /* forward contexts */
                const uint32_t x = block_symbol(words, j, cz);
                contexts[j] = (srel * 9u + previous) * 9u + maximum;
                previous = x & 7u;
                if (maximum == 8u || previous > maximum) maximum = previous;
            }
#pragma unroll
            for (int j = 31; j >= 0; j--) {          /* backward coding */
                const uint32_t x = block_symbol(words, (unsigned)j, cz);
                const uint16_t *cdf = cdfs + contexts[j] * 17u;
                const uint32_t start = __ldg(cdf + x), frequency = __ldg(cdf + x + 1u) - start;
                if (!nr2_put(state, slot, cursor, start, frequency)) { atomicExch(error, 1u); return; }
            }
        }
    }
    for (unsigned b = 0; b < 4u; b++) {   /* wex_nr2_flush */
        if (!cursor) { atomicExch(error, 1u); return; }
        slot[--cursor] = (unsigned char)state;
        state >>= 8;
    }
    lengths[t] = stride - cursor;
}

/* One block per stream: move it from the end of its slot to its payload offset. */
__global__ void nr2_compact_kernel(const unsigned char *scratch, Geometry g, const uint32_t *lengths,
                                   const unsigned long long *offsets, unsigned char *payload) {
    const uint32_t t = blockIdx.x;
    const unsigned char *end = t < g.scale_streams
        ? scratch + (size_t)(t + 1u) * g.scale_stride
        : scratch + g.weight_base + (size_t)(t - g.scale_streams + 1u) * g.weight_stride;
    const uint32_t length = lengths[t];
    const unsigned char *source = end - length;
    unsigned char *destination = payload + offsets[t];
    for (uint32_t i = threadIdx.x; i < length; i += blockDim.x) destination[i] = source[i];
}

struct Workspace {
    int device = -1, sm_count = 0;
    cudaStream_t stream = nullptr;
    unsigned char *slab = nullptr, *model = nullptr, *scratch = nullptr, *payload = nullptr;
    uint32_t *counts = nullptr, *lengths = nullptr, *error = nullptr;
    unsigned long long *offsets = nullptr;
    uint32_t *host_counts = nullptr, *host_lengths = nullptr, *host_error = nullptr;
    unsigned long long *host_offsets = nullptr;
    size_t slab_bytes = 0, scratch_bytes = 0, stream_capacity = 0;

    void release_streams() {
        if (scratch) cudaFree(scratch);
        if (payload) cudaFree(payload);
        if (lengths) cudaFree(lengths);
        if (offsets) cudaFree(offsets);
        if (host_lengths) cudaFreeHost(host_lengths);
        if (host_offsets) cudaFreeHost(host_offsets);
        scratch = payload = nullptr; lengths = nullptr; offsets = nullptr;
        host_lengths = nullptr; host_offsets = nullptr;
        scratch_bytes = stream_capacity = 0;
    }
    void release() {
        if (stream) cudaStreamSynchronize(stream);
        release_streams();
        if (slab) cudaFree(slab);
        if (model) cudaFree(model);
        if (counts) cudaFree(counts);
        if (error) cudaFree(error);
        if (host_counts) cudaFreeHost(host_counts);
        if (host_error) cudaFreeHost(host_error);
        if (stream) cudaStreamDestroy(stream);
        slab = model = nullptr; counts = error = host_counts = host_error = nullptr;
        stream = nullptr; slab_bytes = 0; device = -1; sm_count = 0;
        (void)cudaGetLastError();
    }
    bool base(size_t bytes) {
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess) { (void)cudaGetLastError(); return false; }
        if (device >= 0 && device != current) release();
        if (device < 0) {
            if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, current) != cudaSuccess ||
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaMalloc((void **)&model, WEX_NR2_MODEL_BYTES) != cudaSuccess ||
                cudaMalloc((void **)&counts, (NR2_WEIGHT_BINS + NR2_SCALE_BINS) * 4u) != cudaSuccess ||
                cudaMalloc((void **)&error, 4u) != cudaSuccess ||
                cudaMallocHost((void **)&host_counts, (NR2_WEIGHT_BINS + NR2_SCALE_BINS) * 4u) != cudaSuccess ||
                cudaMallocHost((void **)&host_error, 4u) != cudaSuccess) { release(); return false; }
            device = current;
        }
        if (slab_bytes < bytes) {
            if (slab) cudaFree(slab);
            slab = nullptr; slab_bytes = 0;
            if (cudaMalloc((void **)&slab, bytes) != cudaSuccess) { release(); return false; }
            slab_bytes = bytes;
        }
        return true;
    }
    bool streams(size_t scratch_need, size_t stream_need) {
        if (scratch_bytes >= scratch_need && stream_capacity >= stream_need) return true;
        release_streams();
        if (cudaMalloc((void **)&scratch, scratch_need) != cudaSuccess ||
            cudaMalloc((void **)&payload, scratch_need) != cudaSuccess ||
            cudaMalloc((void **)&lengths, stream_need * 4u) != cudaSuccess ||
            cudaMalloc((void **)&offsets, stream_need * 8u) != cudaSuccess ||
            cudaMallocHost((void **)&host_lengths, stream_need * 4u) != cudaSuccess ||
            cudaMallocHost((void **)&host_offsets, stream_need * 8u) != cudaSuccess) { release(); return false; }
        scratch_bytes = scratch_need; stream_capacity = stream_need;
        return true;
    }
};

/* Process-wide, explicitly released: no CUDA calls from static destructors. */
Workspace workspace;
std::mutex workspace_mutex;

void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

/* Copy of a raw slab with code 8 -> 0 in every weight nibble (scales untouched). */
uint64_t canonicalize(const WexEntry *entry, unsigned char *slab) {
    uint64_t mapped = 0;
    for (unsigned m = 0; m < 3u; m++) {
        unsigned char *w = slab + entry->weight_offsets[m];
        size_t n = (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
        for (size_t i = 0; i < n; i++) {
            unsigned char b = w[i];
            if ((b & 15u) == 8u) { b &= 0xf0u; mapped++; }
            if ((b >> 4) == 8u) { b &= 0x0fu; mapped++; }
            w[i] = b;
        }
    }
    return mapped;
}

/* CPU reference views of a raw slab, optionally canonicalized into *copy. */
int cpu_views(const unsigned char *raw_slab, const WexEntry *entry, int canonical_zero, unsigned char **copy,
              const unsigned char *scales[3], const unsigned char *weights[3], WexNr2Layout *layout) {
    uint64_t raw, aligned;
    const unsigned char *base = raw_slab;
    *copy = nullptr;
    if (!raw_slab || !entry || wex_entry_raw_sizes(entry, &raw, &aligned)) return -1;
    if (canonical_zero) {
        *copy = (unsigned char *)std::malloc((size_t)raw);
        if (!*copy) return -1;
        std::memcpy(*copy, raw_slab, (size_t)raw);
        canonicalize(entry, *copy);
        base = *copy;
    }
    for (unsigned m = 0; m < 3u; m++) {
        scales[m] = base + entry->scale_offsets[m];
        weights[m] = base + entry->weight_offsets[m];
    }
    wex_nr2_layout(entry, layout);
    return 0;
}

}  // namespace

#define CUDA_OK(call) do { if ((call) != cudaSuccess) { (void)cudaGetLastError(); return 1; } } while (0)

NR2_EXPORT unsigned int wex_nr2_tools_abi(void) { return 1u; }

/* += the training counts of one expert, exactly wex_nr2_count's
 * (weight_counts[3][567][16], scale_counts[3][256]). */
NR2_EXPORT int wex_nr2_count_gpu(const unsigned char *raw_slab, const WexEntry *entry, int canonical_zero,
                                 uint64_t *weight_counts, uint64_t *scale_counts) {
    Geometry g;
    uint64_t raw;
    if (!weight_counts || !scale_counts || geometry_of(entry, raw_slab, canonical_zero, 0, 0, &g, &raw)) return -1;
    std::lock_guard<std::mutex> lock(workspace_mutex);
    Workspace &ws = workspace;
    if (!ws.base((size_t)raw)) return 1;
    CUDA_OK(cudaMemcpyAsync(ws.slab, raw_slab, (size_t)raw, cudaMemcpyHostToDevice, ws.stream));
    CUDA_OK(cudaMemsetAsync(ws.counts, 0, (NR2_WEIGHT_BINS + NR2_SCALE_BINS) * 4u, ws.stream));
    uint32_t largest = g.scale_count[0] > g.scale_count[1] ? g.scale_count[0] : g.scale_count[1];
    if (g.scale_count[2] > largest) largest = g.scale_count[2];
    uint32_t per_matrix = (uint32_t)((ws.sm_count + 2) / 3);
    uint32_t needed = (largest + 511u) / 512u;
    dim3 grid(per_matrix < needed ? per_matrix : needed, 3u);
    nr2_count_kernel<<<grid, 512, 0, ws.stream>>>(ws.slab, g, ws.counts, ws.counts + NR2_WEIGHT_BINS);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyAsync(ws.host_counts, ws.counts, (NR2_WEIGHT_BINS + NR2_SCALE_BINS) * 4u,
                            cudaMemcpyDeviceToHost, ws.stream));
    CUDA_OK(cudaStreamSynchronize(ws.stream));
    for (uint32_t i = 0; i < NR2_WEIGHT_BINS; i++) weight_counts[i] += ws.host_counts[i];
    for (uint32_t i = 0; i < NR2_SCALE_BINS; i++) scale_counts[i] += ws.host_counts[NR2_WEIGHT_BINS + i];
    return 0;
}

/* Thin wrapper of the CPU wex_nr2_model_build; also checks the blob opens. */
NR2_EXPORT int wex_nr2_build_model(const uint64_t *weight_counts, const uint64_t *scale_counts, unsigned char *blob_out) {
    WexNr2Model model;
    if (!weight_counts || !scale_counts || !blob_out) return -1;
    wex_nr2_model_build(weight_counts, scale_counts, blob_out);
    return wex_nr2_model_open(blob_out, WEX_NR2_MODEL_BYTES, &model) ? -1 : 0;
}

/* Validate a blob and return its id (FNV-1a), as wex_nr2_model_open does. */
NR2_EXPORT int wex_nr2_model_id(const unsigned char *blob, uint32_t *id) {
    WexNr2Model model;
    if (!blob || !id || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return -1;
    *id = model.id;
    return 0;
}

/* GPU encoder: the unpadded record, byte-identical to wex_nr2_encode. When the
 * record does not fit `capacity` it returns -1 and sets *written to the size
 * it needs (other failures set *written = 0). */
NR2_EXPORT int wex_nr2_encode_gpu(const unsigned char *blob, const unsigned char *raw_slab, const WexEntry *entry,
                                  int canonical_zero, uint32_t weight_chunk, uint32_t scale_chunk,
                                  unsigned char *out, size_t capacity, size_t *written) {
    WexNr2Model model;
    Geometry g;
    uint64_t raw;
    if (written) *written = 0;
    if (!blob || !out || !written || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) ||
        geometry_of(entry, raw_slab, canonical_zero, weight_chunk, scale_chunk, &g, &raw)) return -1;
    const uint32_t streams = g.scale_streams + g.weight_streams;
    const size_t meta = WEX_NR2_HEADER_BYTES + 2u * (size_t)streams;
    const size_t scratch = (size_t)g.weight_base + (size_t)g.weight_streams * g.weight_stride;
    std::lock_guard<std::mutex> lock(workspace_mutex);
    Workspace &ws = workspace;
    if (!ws.base((size_t)raw) || !ws.streams(scratch, streams)) return 1;
    CUDA_OK(cudaMemcpyAsync(ws.slab, raw_slab, (size_t)raw, cudaMemcpyHostToDevice, ws.stream));
    CUDA_OK(cudaMemcpyAsync(ws.model, blob, WEX_NR2_MODEL_BYTES, cudaMemcpyHostToDevice, ws.stream));
    CUDA_OK(cudaMemsetAsync(ws.error, 0, 4u, ws.stream));
    nr2_encode_kernel<<<(streams + 31u) / 32u, 32, 0, ws.stream>>>(ws.slab, ws.model, g, ws.scratch, ws.lengths, ws.error);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyAsync(ws.host_lengths, ws.lengths, (size_t)streams * 4u, cudaMemcpyDeviceToHost, ws.stream));
    CUDA_OK(cudaMemcpyAsync(ws.host_error, ws.error, 4u, cudaMemcpyDeviceToHost, ws.stream));
    CUDA_OK(cudaStreamSynchronize(ws.stream));
    if (*ws.host_error) return -1;
    uint64_t total = 0, scale_bytes = 0;
    for (uint32_t s = 0; s < streams; s++) {
        uint32_t length = ws.host_lengths[s];
        if (length < 4u || length > 65535u) return -1;   /* as wex_nr2_encode */
        ws.host_offsets[s] = total;
        total += length;
        if (s + 1u == g.scale_streams) scale_bytes = total;
    }
    if (meta + total > capacity) { *written = (size_t)(meta + total); return -1; }
    CUDA_OK(cudaMemcpyAsync(ws.offsets, ws.host_offsets, (size_t)streams * 8u, cudaMemcpyHostToDevice, ws.stream));
    nr2_compact_kernel<<<streams, 128, 0, ws.stream>>>(ws.scratch, g, ws.lengths, ws.offsets, ws.payload);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyAsync(out + meta, ws.payload, (size_t)total, cudaMemcpyDeviceToHost, ws.stream));
    CUDA_OK(cudaStreamSynchronize(ws.stream));
    put_u32(out, WEX_NR2_MAGIC);
    put_u32(out + 4u, model.id);
    put_u32(out + 8u, (uint32_t)scale_bytes);
    put_u32(out + 12u, (uint32_t)(total - scale_bytes));
    put_u32(out + 16u, weight_chunk);
    put_u32(out + 20u, scale_chunk);
    out[24] = (unsigned char)g.median[0]; out[25] = (unsigned char)g.median[1];
    out[26] = (unsigned char)g.median[2]; out[27] = 0;
    put_u32(out + 28u, 0);
    for (uint32_t s = 0; s < streams; s++) {
        out[WEX_NR2_HEADER_BYTES + (size_t)s * 2u] = (unsigned char)ws.host_lengths[s];
        out[WEX_NR2_HEADER_BYTES + (size_t)s * 2u + 1u] = (unsigned char)(ws.host_lengths[s] >> 8);
    }
    *written = (size_t)(meta + total);
    return 0;
}

/* CPU reference decode (wex_nr2_parse + wex_nr2_decode) of a padded record,
 * compared with the expected raw slab. 0 equal; -1 bad arguments/model,
 * -2 parse refused, -3 decode refused, -4 decoded bytes differ. */
NR2_EXPORT int wex_nr2_verify(const unsigned char *blob, const WexEntry *entry, const unsigned char *record,
                              size_t total, const unsigned char *expected_raw) {
    WexNr2Model model;
    WexNr2Layout layout;
    WexNr2Record view;
    uint64_t raw, aligned;
    unsigned char *scales[3], *weights[3];
    if (!blob || !entry || !record || !expected_raw || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) ||
        wex_entry_raw_sizes(entry, &raw, &aligned)) return -1;
    wex_nr2_layout(entry, &layout);
    if (wex_nr2_parse(record, total, &layout, model.id, &view)) return -2;
    unsigned char *decoded = (unsigned char *)std::malloc((size_t)raw);
    if (!decoded) return -1;
    std::memset(decoded, 0xa5, (size_t)raw);
    for (unsigned m = 0; m < 3u; m++) {
        scales[m] = decoded + entry->scale_offsets[m];
        weights[m] = decoded + entry->weight_offsets[m];
    }
    int result = wex_nr2_decode(&view, &layout, &model, scales, weights) ? -3
               : std::memcmp(decoded, expected_raw, (size_t)raw) ? -4 : 0;
    std::free(decoded);
    return result;
}

/* The engine's slab path: the record at wex_record_read_offset(entry) of an
 * aligned_raw slab, decoded in place by wex_decode_nr2_record (flags and
 * total_bytes from the entry), guard bytes around the slab. 0 when exact. */
NR2_EXPORT int wex_nr2_verify_slab(const unsigned char *blob, const WexEntry *entry, const unsigned char *record,
                                   const unsigned char *expected_raw) {
    WexNr2Model model;
    uint64_t raw, aligned;
    if (!blob || !entry || !record || !expected_raw || entry->flags != WEX_ENTRY_NEURAL_RANS2 ||
        wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) || wex_entry_raw_sizes(entry, &raw, &aligned) ||
        entry->total_bytes >= aligned) return -1;
    size_t at = wex_record_read_offset(entry);
    if (at != aligned - entry->total_bytes) return -1;
    unsigned char *buffer = (unsigned char *)std::malloc((size_t)aligned + 64u);
    if (!buffer) return -1;
    std::memset(buffer, 0x5a, (size_t)aligned + 64u);
    std::memcpy(buffer + 32u + at, record, entry->total_bytes);
    int result = wex_decode_nr2_record(entry, buffer + 32u, (size_t)aligned, &model) ? -3
               : std::memcmp(buffer + 32u, expected_raw, (size_t)raw) ? -4 : 0;
    for (unsigned i = 0; !result && i < 32u; i++)
        if (buffer[i] != 0x5a || buffer[32u + aligned + i] != 0x5a) result = -5;
    std::free(buffer);
    return result;
}

/* Header page + bytes [4096, table_offset) + table + file size, run through
 * wex_validate_header, wex_validate_table, wex_nr2_model_location and
 * wex_nr2_model_open (id must equal the descriptor's). 0 when valid; the
 * negative value names the failing step (-1 header .. -7 id). */
NR2_EXPORT int wex_nr2_check_container(const unsigned char *header_page, const unsigned char *pre_table,
                                       size_t pre_table_bytes, const unsigned char *table, size_t table_bytes,
                                       uint64_t file_bytes, uint32_t *model_id) {
    WexHeader header;
    WexNr2Model model;
    size_t bytes, aligned_bytes;
    uint64_t at;
    uint32_t model_bytes, id;
    if (!header_page || !pre_table || !table || !model_id) return -1;
    std::memcpy(&header, header_page, sizeof(header));
    if (wex_validate_header(&header, file_bytes, 0, 0, &bytes, &aligned_bytes)) return -1;
    if (table_bytes < bytes || pre_table_bytes != header.table_offset - WEX_ALIGN) return -2;
    if (wex_validate_table(&header, (const WexEntry *)table, file_bytes)) return -3;
    if (wex_nr2_model_location(&header, header_page, &at, &model_bytes, &id)) return -4;
    if (at - WEX_ALIGN + model_bytes > pre_table_bytes) return -5;
    if (wex_nr2_model_open(pre_table + (at - WEX_ALIGN), model_bytes, &model)) return -6;
    if (model.id != id) return -7;
    *model_id = id;
    return 0;
}

/* ---- CPU reference wrappers for tests ----------------------------------- */

NR2_EXPORT int wex_nr2_count_cpu(const unsigned char *raw_slab, const WexEntry *entry, int canonical_zero,
                                 uint64_t *weight_counts, uint64_t *scale_counts) {
    const unsigned char *scales[3], *weights[3];
    unsigned char *copy;
    WexNr2Layout layout;
    if (!weight_counts || !scale_counts || cpu_views(raw_slab, entry, canonical_zero, &copy, scales, weights, &layout))
        return -1;
    for (unsigned m = 0; m < 3u; m++)
        if (!layout.scale_count[m] || (uint64_t)layout.scale_count[m] * 32u != layout.symbol_count[m]) { std::free(copy); return -1; }
    wex_nr2_count(&layout, scales, weights, weight_counts, scale_counts);
    std::free(copy);
    return 0;
}

NR2_EXPORT int wex_nr2_encode_cpu(const unsigned char *blob, const unsigned char *raw_slab, const WexEntry *entry,
                                  int canonical_zero, uint32_t weight_chunk, uint32_t scale_chunk,
                                  unsigned char *out, size_t capacity, size_t *written) {
    const unsigned char *scales[3], *weights[3];
    unsigned char *copy;
    WexNr2Layout layout;
    WexNr2Model model;
    if (written) *written = 0;
    if (!blob || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) ||
        cpu_views(raw_slab, entry, canonical_zero, &copy, scales, weights, &layout)) return -1;
    int result = wex_nr2_encode(&model, &layout, scales, weights, weight_chunk, scale_chunk, out, capacity, written);
    std::free(copy);
    return result ? -1 : 0;
}

/* In place: weight nibbles 8 -> 0; *mapped = how many. */
NR2_EXPORT int wex_nr2_canonical_zero(const WexEntry *entry, unsigned char *raw_slab, uint64_t *mapped) {
    uint64_t raw, aligned;
    if (!entry || !raw_slab || wex_entry_raw_sizes(entry, &raw, &aligned)) return -1;
    uint64_t n = canonicalize(entry, raw_slab);
    if (mapped) *mapped = n;
    return 0;
}

NR2_EXPORT void wex_nr2_release(void) {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    workspace.release();
}
