/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Single-threaded WEX decode/load benchmark against explicit raw WEX1 bytes.
 * Usage: bench_wex_decode RAW.wex COMPRESSED.wex [iterations] [rounds] [direct]
 * Reads at most two records. CPU timings include compressed-input restore;
 * 'direct' instead measures Win32 unbuffered read plus decode. No GPU timing.
 * For lossy WEX3 supply a raw WEX1 of the quantized reference, not the original.
 */
#ifndef WEX_FORMAT_HEADER
#define WEX_FORMAT_HEADER "../wex_format.h"
#endif
#include WEX_FORMAT_HEADER
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define alloc_slab(n) _aligned_malloc(n, WEX_ALIGN)
#define free_slab(p) _aligned_free(p)
#define seek_file _fseeki64
#define tell_file _ftelli64
#else
#define alloc_slab(n) malloc(n)
#define free_slab(p) free(p)
#define seek_file fseeko
#define tell_file ftello
#endif

#define REQUIRE(ok) do { if (!(ok)) { \
    fprintf(stderr, "Benchmark failed at line %d: %s\n", __LINE__, #ok); \
    exit(1); \
} } while (0)

typedef struct {
    WexEntry entry;
    size_t capacity, prefix, offset;
    unsigned char *disk, *slab, *expected;
} Sample;

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return (double)ticks.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    REQUIRE(!clock_gettime(CLOCK_MONOTONIC, &ts));
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static WexEntry *load_table(FILE *file, WexHeader *header, uint64_t *file_bytes) {
    size_t table_bytes, aligned;
    WexEntry *table;
    REQUIRE(!seek_file(file, 0, SEEK_END));
    { int64_t size = tell_file(file); REQUIRE(size >= 0); *file_bytes = (uint64_t)size; }
    REQUIRE(!seek_file(file, 0, SEEK_SET));
    REQUIRE(fread(header, sizeof(*header), 1, file) == 1);
    REQUIRE(!wex_validate_header(header, *file_bytes, 0, 0, &table_bytes, &aligned));
    table = (WexEntry *)malloc(table_bytes);
    REQUIRE(table);
    REQUIRE(!seek_file(file, (int64_t)header->table_offset, SEEK_SET));
    REQUIRE(fread(table, 1, table_bytes, file) == table_bytes);
    return table;
}

int main(int argc, char **argv) {
    FILE *raw_file, *compressed_file;
    WexHeader raw_header, compressed_header;
    WexEntry *raw_table, *compressed_table;
    uint64_t raw_size, compressed_size, index, count;
    Sample samples[2];
    unsigned int iterations = 200, rounds = 5, n = 0, round, iteration, s;
    int direct = argc == 6;
#ifdef _WIN32
    HANDLE direct_file = INVALID_HANDLE_VALUE;
#endif
    if (argc < 3 || argc > 6 || (direct && strcmp(argv[5], "direct"))) {
        fprintf(stderr, "Usage: %s RAW.wex COMPRESSED.wex [iterations] [rounds] [direct]\n", argv[0]);
        return 1;
    }
    if (argc > 3) iterations = (unsigned int)strtoul(argv[3], NULL, 10);
    if (argc > 4) rounds = (unsigned int)strtoul(argv[4], NULL, 10);
    REQUIRE(iterations && iterations <= 100000 && rounds && rounds <= 100);
    raw_file = fopen(argv[1], "rb"); compressed_file = fopen(argv[2], "rb");
    REQUIRE(raw_file && compressed_file);
    raw_table = load_table(raw_file, &raw_header, &raw_size);
    compressed_table = load_table(compressed_file, &compressed_header, &compressed_size);
    REQUIRE(raw_header.num_layers == compressed_header.num_layers &&
            raw_header.num_experts == compressed_header.num_experts);
    count = (uint64_t)compressed_header.num_layers * compressed_header.num_experts;
    for (index = 0; index < count && n < 2; index++) {
        Sample *sample;
        uint64_t bytes, aligned;
        WexEntry *raw = &raw_table[index], *encoded = &compressed_table[index];
        if (!encoded->total_bytes) continue;
        REQUIRE(!wex_validate_entry(&raw_header, raw, raw_size) && !raw->flags);
        REQUIRE(!wex_validate_entry(&compressed_header, encoded, compressed_size));
        REQUIRE(!wex_entry_raw_sizes(encoded, &bytes, &aligned));
        REQUIRE(raw->scale_bytes == encoded->scale_bytes && raw->weight_bytes == encoded->weight_bytes);
        REQUIRE(!memcmp(raw->scale_shapes, encoded->scale_shapes,
                        sizeof(WexEntry) - offsetof(WexEntry, scale_shapes) - sizeof(encoded->flags)));
        sample = &samples[n++]; sample->entry = *encoded;
        sample->capacity = (size_t)aligned;
        sample->prefix = (encoded->flags >> WEX_ENTRY_SCALE_PAGES_SHIFT) * (size_t)WEX_ALIGN;
#if defined(WEX_ENTRY_PALETTE3)
        if (encoded->flags == WEX_ENTRY_PALETTE3) sample->prefix = encoded->total_bytes;
#endif
#if defined(WEX_ENTRY_NEURAL_RANS)
        if (encoded->flags == WEX_ENTRY_NEURAL_RANS) sample->prefix = encoded->total_bytes;
#endif
        sample->offset = wex_record_read_offset(encoded);
        sample->disk = (unsigned char *)malloc(encoded->total_bytes);
        sample->slab = (unsigned char *)alloc_slab(sample->capacity);
        sample->expected = (unsigned char *)malloc(sample->capacity);
        REQUIRE(sample->disk && sample->slab && sample->expected);
        REQUIRE(!seek_file(compressed_file, (int64_t)encoded->file_offset, SEEK_SET));
        REQUIRE(fread(sample->disk, 1, encoded->total_bytes, compressed_file) == encoded->total_bytes);
        REQUIRE(!seek_file(raw_file, (int64_t)raw->file_offset, SEEK_SET));
        REQUIRE(fread(sample->expected, 1, sample->capacity, raw_file) == sample->capacity);
        memset(sample->slab, 0, sample->capacity);
        memcpy(sample->slab + sample->offset, sample->disk, encoded->total_bytes);
        REQUIRE(!wex_decode_record(encoded, sample->slab, sample->capacity));
        REQUIRE(!memcmp(sample->slab, sample->expected, (size_t)bytes));
    }
    REQUIRE(n);
    fclose(raw_file); fclose(compressed_file); free(raw_table); free(compressed_table);
    if (direct) {
#ifdef _WIN32
        direct_file = CreateFileA(argv[2], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                  FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, NULL);
        REQUIRE(direct_file != INVALID_HANDLE_VALUE);
#else
        fprintf(stderr, "Direct I/O benchmark is currently implemented on Windows only\n");
        return 1;
#endif
    }
    printf("{\"records\":%u,\"iterations_per_record\":%u,\"microseconds_per_record\":[", n, iterations);
    for (round = 0; round < rounds; round++) {
        double start = now_seconds(), elapsed;
        for (iteration = 0; iteration < iterations; iteration++) {
            for (s = 0; s < n; s++) {
                Sample *sample = &samples[s];
                if (direct) {
#ifdef _WIN32
                    LARGE_INTEGER position;
                    DWORD bytes = 0;
                    position.QuadPart = (LONGLONG)sample->entry.file_offset;
                    REQUIRE(SetFilePointerEx(direct_file, position, NULL, FILE_BEGIN));
                    REQUIRE(ReadFile(direct_file, sample->slab + sample->offset,
                                     sample->entry.total_bytes, &bytes, NULL));
                    REQUIRE(bytes == sample->entry.total_bytes);
#endif
                } else memcpy(sample->slab + sample->offset, sample->disk, sample->prefix);
                REQUIRE(!wex_decode_record(&sample->entry, sample->slab, sample->capacity));
            }
        }
        elapsed = now_seconds() - start;
        printf("%s%.3f", round ? "," : "", elapsed * 1e6 / (iterations * n));
        for (s = 0; s < n; s++)
            REQUIRE(!memcmp(samples[s].slab, samples[s].expected,
                            samples[s].entry.scale_bytes + samples[s].entry.weight_bytes));
    }
    printf("],\"exact_reference_match\":true,\"scope\":\"%s\"}\n",
           direct ? "Win32 unbuffered synchronous read plus decode; no inference" :
                    "CPU decode plus compressed input restore; no disk I/O");
#ifdef _WIN32
    if (direct) REQUIRE(CloseHandle(direct_file));
#endif
    for (s = 0; s < n; s++) {
        free(samples[s].disk); free_slab(samples[s].slab); free(samples[s].expected);
    }
    return 0;
}
